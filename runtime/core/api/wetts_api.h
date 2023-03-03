// Copyright (c) 2022  Binbin Zhang (binbzha@qq.com)
//               2023  Xingchen Song(sxc19@mails.tsinghua.edu.cn)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef API_WETTS_API_H_
#define API_WETTS_API_H_

#ifdef __cplusplus
extern "C" {
#endif

/** Init tts_model from the file and returns the object
 *
 * @param model_dir: the model dir
 * @returns model object or NULL if problem occured
 */
void* wetts_init(const char* model_dir);

/** Free wetts tts_model and corresponding resource
 */
void wetts_free(void* tts_model);

/** synthesis the input text for specific speaker
 * @param text: text
 * @param sid: speaker id
 */
void wetts_synthesis(void* tts_model, const char* text, int sid);

/** Get wav result
 */
const float* wetts_get_result(void* tts_model);

/** Set language, has effect on the prepocessing
 *  @param: lang, could be chs-only now
 */
void wetts_set_language(void* tts_model, const char* lang);

/** Set log level
 *  We use glog in wetts, so the level is the glog level
 */
void wetts_set_log_level(int level);

#ifdef __cplusplus
}
#endif

#endif  // API_WETTS_API_H_
