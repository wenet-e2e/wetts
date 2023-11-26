import math
import time

import torch
from torch import nn

from model.decoders import Generator, VocosGenerator
from model.duration_predictors import (
    StochasticDurationPredictor,
    DurationPredictor
)
from model.encoders import TextEncoder, PosteriorEncoder
from model.flows import AVAILABLE_FLOW_TYPES, ResidualCouplingTransformersBlock
from utils import commons, monotonic_align


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self,
                 n_vocab,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 n_speakers=0,
                 gin_channels=0,
                 use_sdp=True,
                 vocoder_type="hifigan",
                 vocos_channels=512,
                 vocos_h_channels=1536,
                 vocos_out_channels=1026,
                 vocos_num_layers=8,
                 vocos_istft_config={
                     "n_fft": 1024,
                     "hop_length": 256,
                     "win_length": 1024,
                     "center": True,
                 },
                 is_onnx=False,
                 **kwargs):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_spk_conditioned_encoder = kwargs.get(
            "use_spk_conditioned_encoder", False)
        self.use_transformer_flows = kwargs.get("use_transformer_flows", False)
        self.transformer_flow_type = kwargs.get("transformer_flow_type",
                                                "mono_layer_post_residual")
        if self.use_transformer_flows:
            assert (
                self.transformer_flow_type in AVAILABLE_FLOW_TYPES
            ), f"transformer_flow_type must be one of {AVAILABLE_FLOW_TYPES}"
        self.use_sdp = use_sdp
        self.use_noise_scaled_mas = kwargs.get("use_noise_scaled_mas", False)
        self.mas_noise_scale_initial = kwargs.get("mas_noise_scale_initial",
                                                  0.01)
        self.noise_scale_delta = kwargs.get("noise_scale_delta", 2e-6)

        self.current_mas_noise_scale = self.mas_noise_scale_initial
        if self.use_spk_conditioned_encoder and gin_channels > 0:
            self.enc_gin_channels = gin_channels
        else:
            self.enc_gin_channels = 0
        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.enc_gin_channels,
        )
        if vocoder_type == "vocos":
            self.dec = VocosGenerator(
                inter_channels,
                vocos_channels,
                vocos_h_channels,
                vocos_out_channels,
                vocos_num_layers,
                vocos_istft_config,
                gin_channels,
                is_onnx,
            )
        else:
            self.dec = Generator(
                inter_channels,
                resblock,
                resblock_kernel_sizes,
                resblock_dilation_sizes,
                upsample_rates,
                upsample_initial_channel,
                upsample_kernel_sizes,
                gin_channels=gin_channels,
            )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingTransformersBlock(
            inter_channels,
            hidden_channels,
            5,
            1,
            4,
            gin_channels=gin_channels,
            use_transformer_flows=self.use_transformer_flows,
            transformer_flow_type=self.transformer_flow_type,
        )

        if use_sdp:
            self.dp = StochasticDurationPredictor(hidden_channels,
                                                  192,
                                                  3,
                                                  0.5,
                                                  4,
                                                  gin_channels=gin_channels)
        else:
            self.dp = DurationPredictor(hidden_channels,
                                        256,
                                        3,
                                        0.5,
                                        gin_channels=gin_channels)

        if n_speakers > 0:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(self, x, x_lengths, y, y_lengths, sid=None):
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1],
                                  keepdim=True)  # [b, 1, t_s]
            neg_cent2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2),
                s_p_sq_r)  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2),
                (m_p * s_p_sq_r))  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, [1],
                                  keepdim=True)  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            if self.use_noise_scaled_mas:
                epsilon = (torch.std(neg_cent) * torch.randn_like(neg_cent) *
                           self.current_mas_noise_scale)
                neg_cent = neg_cent + epsilon

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(
                y_mask, -1)
            attn = (monotonic_align.maximum_path(
                neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach())

        w = attn.sum(2)
        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=1.0)
            logw_ = torch.log(w + 1e-6) * x_mask
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length = torch.sum(
                (logw - logw_)**2, [1, 2]) / torch.sum(x_mask)  # for averaging

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1,
                                                          2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1),
                              logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)
        return (
            o,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            (x, logw, logw_),
        )

    def infer(
        self,
        x,
        x_lengths,
        sid=None,
        noise_scale=1,
        length_scale=1,
        noise_scale_w=1.0,
        max_len=None,
    ):
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None
        t1 = time.time()
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g)
        t2 = time.time()
        if self.use_sdp:
            logw = self.dp(x,
                           x_mask,
                           g=g,
                           reverse=True,
                           noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)
        t3 = time.time()
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None),
                                 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(
            1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        t4 = time.time()
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        t5 = time.time()
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        t6 = time.time()
        print("TextEncoder: {}s DurationPredictor: {}s Flow: {}s Decoder: {}s".
              format(
                  round(t2 - t1, 3),
                  round(t3 - t2, 3),
                  round(t5 - t4, 3),
                  round(t6 - t5, 3),
              ))
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def infer_encoder(
        self,
        x,
        x_lengths,
        sid=None,
        noise_scale=1,
        length_scale=1,
        noise_scale_w=1.0,
    ):
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None
        t1 = time.time()
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g)
        t2 = time.time()
        if self.use_sdp:
            logw = self.dp(x,
                           x_mask,
                           g=g,
                           reverse=True,
                           noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)
        t3 = time.time()
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None),
                                 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(
            1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        t4 = time.time()
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        t5 = time.time()
        print("TextEncoder: {}s DurationPredictor: {}s Flow: {}s ".
              format(
                  round(t2 - t1, 3),
                  round(t3 - t2, 3),
                  round(t5 - t4, 3),
              ))
        z = z * y_mask
        return attn, y_mask, (z, z_p, m_p, logs_p), g

    def export_forward(self, x, x_lengths, scales, sid):
        # shape of scales: Bx3, make triton happy
        audio, *_ = self.infer(
            x,
            x_lengths,
            sid,
            noise_scale=scales[0][0],
            length_scale=scales[0][1],
            noise_scale_w=scales[0][2],
        )

        return audio

    def export_encoder_forward(self, x, x_lengths, scales, sid):
        # shape of scales: Bx3, make triton happy
        attn, y_mask, (z, z_p, m_p, logs_p), g = self.infer_encoder(
            x,
            x_lengths,
            sid,
            noise_scale=scales[0][0],
            length_scale=scales[0][1],
            noise_scale_w=scales[0][2],
        )
        return z, g

    def export_decoder_forward(self, z, g):
        return self.dec(z, g=g)

    # currently vits-2 is not capable of voice conversion
    # comment - choihkk
    # Assuming the use of the ResidualCouplingTransformersLayer2 module,
    # it seems that voice conversion is possible
    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)
