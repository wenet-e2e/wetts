// Copyright (c) 2025 Binbin Zhang (binbzha@qq.com)
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

#ifndef UTILS_LOG_H_
#define UTILS_LOG_H_

#ifndef __ANDROID__
#include "glog/logging.h"
#else

#include <android/log.h>
#include <sstream>
#include <string>

// Android Log Tag
#ifndef LOG_TAG
#define LOG_TAG "wetts"
#endif

// Log levels mapping to Android log levels
#define ANDROID_LOG_VERBOSE 2
#define ANDROID_LOG_DEBUG 3
#define ANDROID_LOG_INFO 4
#define ANDROID_LOG_WARN 5
#define ANDROID_LOG_ERROR 6
#define ANDROID_LOG_FATAL 7

// Helper class for stream-based logging
class AndroidLogMessage {
 public:
  AndroidLogMessage(int level, const char* file, int line)
      : level_(level), file_(file), line_(line) {}

  ~AndroidLogMessage() {
    std::string message = stream_.str();
    if (!message.empty() && message.back() == '\n') {
      message.pop_back();
    }
    __android_log_print(level_, LOG_TAG, "[%s:%d] %s", file_, line_,
                        message.c_str());
  }

  std::ostream& stream() { return stream_; }

 private:
  int level_;
  const char* file_;
  int line_;
  std::ostringstream stream_;
};

// Helper class for VLOG
class AndroidVLogMessage {
 public:
  AndroidVLogMessage(int level, const char* file, int line)
      : level_(level), file_(file), line_(line) {}

  ~AndroidVLogMessage() {
    std::string message = stream_.str();
    if (!message.empty() && message.back() == '\n') {
      message.pop_back();
    }
    // VLOG maps to DEBUG level in Android
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d] VLOG(%d) %s",
                        file_, line_, level_, message.c_str());
  }

  std::ostream& stream() { return stream_; }

 private:
  int level_;
  const char* file_;
  int line_;
  std::ostringstream stream_;
};

// Helper function to get filename from path
inline const char* GetFileName(const char* path) {
  const char* filename = path;
  const char* last_slash = strrchr(path, '/');
  if (last_slash != nullptr) {
    filename = last_slash + 1;
  }
  return filename;
}

// LOG macros
#define LOG(severity)                                                        \
  AndroidLogMessage(ANDROID_LOG_##severity, GetFileName(__FILE__), __LINE__) \
      .stream()

// VLOG macros
#define VLOG(verboselevel) \
  AndroidVLogMessage(verboselevel, GetFileName(__FILE__), __LINE__).stream()

// CHECK macros
#define CHECK(condition)                                                \
  if (!(condition))                                                     \
  AndroidLogMessage(ANDROID_LOG_FATAL, GetFileName(__FILE__), __LINE__) \
          .stream()                                                     \
      << "Check failed: " #condition " "

#define CHECK_EQ(val1, val2) CHECK((val1) == (val2))
#define CHECK_NE(val1, val2) CHECK((val1) != (val2))
#define CHECK_LE(val1, val2) CHECK((val1) <= (val2))
#define CHECK_LT(val1, val2) CHECK((val1) < (val2))
#define CHECK_GE(val1, val2) CHECK((val1) >= (val2))
#define CHECK_GT(val1, val2) CHECK((val1) > (val2))

// DCHECK macros (same as CHECK in release builds)
#ifdef NDEBUG
#define DCHECK(condition) \
  while (false)           \
  AndroidLogMessage(ANDROID_LOG_FATAL, GetFileName(__FILE__), __LINE__).stream()
#define DCHECK_EQ(val1, val2) DCHECK((val1) == (val2))
#define DCHECK_NE(val1, val2) DCHECK((val1) != (val2))
#define DCHECK_LE(val1, val2) DCHECK((val1) <= (val2))
#define DCHECK_LT(val1, val2) DCHECK((val1) < (val2))
#define DCHECK_GE(val1, val2) DCHECK((val1) >= (val2))
#define DCHECK_GT(val1, val2) DCHECK((val1) > (val2))
#else
#define DCHECK(condition) CHECK(condition)
#define DCHECK_EQ(val1, val2) CHECK_EQ(val1, val2)
#define DCHECK_NE(val1, val2) CHECK_NE(val1, val2)
#define DCHECK_LE(val1, val2) CHECK_LE(val1, val2)
#define DCHECK_LT(val1, val2) CHECK_LT(val1, val2)
#define DCHECK_GE(val1, val2) CHECK_GE(val1, val2)
#define DCHECK_GT(val1, val2) CHECK_GT(val1, val2)
#endif

// Empty implementations for glog initialization functions
inline void InitGoogleLogging(const char*) {}
inline void ShutdownGoogleLogging() {}

#endif
#endif  // UTILS_LOG_H_
