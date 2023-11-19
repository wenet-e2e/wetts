import torch
from torch import nn

from model.attentions import Encoder, FFT
from model.modules import Flip, WN

AVAILABLE_FLOW_TYPES = [
    "pre_conv",
    "pre_conv2",
    "fft",
    "mono_layer_inter_residual",
    "mono_layer_post_residual",
]


class ResidualCouplingTransformersLayer2(nn.Module):  # vits2

    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.pre_transformer = Encoder(
            hidden_channels,
            hidden_channels,
            n_heads=2,
            n_layers=1,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            # window_size=None,
        )
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )

        self.post = nn.Conv1d(hidden_channels,
                              self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = h + self.pre_transformer(h * x_mask,
                                     x_mask)  # vits2 residual connection
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)
        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class ResidualCouplingTransformersLayer(nn.Module):  # vits2

    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only
        # vits2
        self.pre_transformer = Encoder(
            self.half_channels,
            self.half_channels,
            n_heads=2,
            n_layers=2,
            kernel_size=3,
            p_dropout=0.1,
            window_size=None,
        )

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        # vits2
        self.post_transformer = Encoder(
            self.hidden_channels,
            self.hidden_channels,
            n_heads=2,
            n_layers=2,
            kernel_size=3,
            p_dropout=0.1,
            window_size=None,
        )

        self.post = nn.Conv1d(hidden_channels,
                              self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        x0_ = self.pre_transformer(x0 * x_mask, x_mask)  # vits2
        x0_ = x0_ + x0  # vits2 residual connection
        h = self.pre(
            x0_) * x_mask  # changed from x0 to x0_ to retain x0 for the flow
        h = self.enc(h, x_mask, g=g)

        # vits2 - (experimental;uncomment the following 2 line to use)
        # h_ = self.post_transformer(h, x_mask)
        # h = h + h_ #vits2 residual connection

        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)
        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class FFTransformerCouplingLayer(nn.Module):  # vits2

    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        n_layers,
        n_heads,
        p_dropout=0,
        filter_channels=768,
        mean_only=False,
        gin_channels=0,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = FFT(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            isflow=True,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels,
                              self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h_ = self.enc(h, x_mask, g=g)
        h = h_ + h
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class MonoTransformerFlowLayer(nn.Module):  # vits2

    def __init__(
        self,
        channels,
        hidden_channels,
        mean_only=False,
        residual_connection=False,
        # according to VITS-2 paper fig 1B set residual_connection=True
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.residual_connection = residual_connection
        # vits2
        self.pre_transformer = Encoder(
            self.half_channels,
            self.half_channels,
            n_heads=2,
            n_layers=2,
            kernel_size=3,
            p_dropout=0.1,
            window_size=None,
        )

        self.post = nn.Conv1d(self.half_channels,
                              self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        if self.residual_connection:
            if not reverse:
                x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
                x0_ = self.pre_transformer(x0, x_mask)  # vits2
                stats = self.post(x0_) * x_mask
                if not self.mean_only:
                    m, logs = torch.split(stats, [self.half_channels] * 2, 1)
                else:
                    m = stats
                    logs = torch.zeros_like(m)
                x1 = m + x1 * torch.exp(logs) * x_mask
                x_ = torch.cat([x0, x1], 1)
                x = x + x_
                logdet = torch.sum(torch.log(torch.exp(logs) + 1), [1, 2])
                logdet = logdet + torch.log(
                    torch.tensor(2)) * (x0.shape[1] * x0.shape[2])
                return x, logdet
            else:
                x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
                x0 = x0 / 2
                x0_ = x0 * x_mask
                x0_ = self.pre_transformer(x0, x_mask)  # vits2
                stats = self.post(x0_) * x_mask
                if not self.mean_only:
                    m, logs = torch.split(stats, [self.half_channels] * 2, 1)
                else:
                    m = stats
                    logs = torch.zeros_like(m)
                x1_ = ((x1 - m) / (1 + torch.exp(-logs))) * x_mask
                x = torch.cat([x0, x1_], 1)
                return x
        else:
            x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
            x0_ = self.pre_transformer(x0 * x_mask, x_mask)  # vits2
            h = x0_ + x0  # vits2
            stats = self.post(h) * x_mask
            if not self.mean_only:
                m, logs = torch.split(stats, [self.half_channels] * 2, 1)
            else:
                m = stats
                logs = torch.zeros_like(m)
            if not reverse:
                x1 = m + x1 * torch.exp(logs) * x_mask
                x = torch.cat([x0, x1], 1)
                logdet = torch.sum(logs, [1, 2])
                return x, logdet
            else:
                x1 = (x1 - m) * torch.exp(-logs) * x_mask
                x = torch.cat([x0, x1], 1)
                return x


class ResidualCouplingTransformersBlock(nn.Module):  # vits2

    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
        use_transformer_flows=False,
        transformer_flow_type="pre_conv",
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        if use_transformer_flows:
            if transformer_flow_type == "pre_conv":
                for i in range(n_flows):
                    self.flows.append(
                        ResidualCouplingTransformersLayer(
                            channels,
                            hidden_channels,
                            kernel_size,
                            dilation_rate,
                            n_layers,
                            gin_channels=gin_channels,
                            mean_only=True,
                        ))
                    self.flows.append(Flip())
            elif transformer_flow_type == "pre_conv2":
                for i in range(n_flows):
                    self.flows.append(
                        ResidualCouplingTransformersLayer2(
                            channels,
                            hidden_channels,
                            kernel_size,
                            dilation_rate,
                            n_layers,
                            gin_channels=gin_channels,
                            mean_only=True,
                        ))
                    self.flows.append(Flip())
            elif transformer_flow_type == "fft":
                for i in range(n_flows):
                    self.flows.append(
                        FFTransformerCouplingLayer(
                            channels,
                            hidden_channels,
                            kernel_size,
                            dilation_rate,
                            n_layers,
                            gin_channels=gin_channels,
                            mean_only=True,
                        ))
                    self.flows.append(Flip())
            elif transformer_flow_type == "mono_layer_inter_residual":
                for i in range(n_flows):
                    self.flows.append(
                        ResidualCouplingLayer(
                            channels,
                            hidden_channels,
                            kernel_size,
                            dilation_rate,
                            n_layers,
                            gin_channels=gin_channels,
                            mean_only=True,
                        ))
                    self.flows.append(Flip())
                    self.flows.append(
                        MonoTransformerFlowLayer(channels,
                                                 hidden_channels,
                                                 mean_only=True))
            elif transformer_flow_type == "mono_layer_post_residual":
                for i in range(n_flows):
                    self.flows.append(
                        ResidualCouplingLayer(
                            channels,
                            hidden_channels,
                            kernel_size,
                            dilation_rate,
                            n_layers,
                            gin_channels=gin_channels,
                            mean_only=True,
                        ))
                    self.flows.append(Flip())
                    self.flows.append(
                        MonoTransformerFlowLayer(
                            channels,
                            hidden_channels,
                            mean_only=True,
                            residual_connection=True,
                        ))
        else:
            for i in range(n_flows):
                self.flows.append(
                    ResidualCouplingLayer(
                        channels,
                        hidden_channels,
                        kernel_size,
                        dilation_rate,
                        n_layers,
                        gin_channels=gin_channels,
                        mean_only=True,
                    ))
                self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class ResidualCouplingBlock(nn.Module):

    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                ))
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self):
        for i, l in enumerate(self.flows):
            if i % 2 == 0:
                l.remove_weight_norm()


class ResidualCouplingLayer(nn.Module):

    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels,
                              self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()
