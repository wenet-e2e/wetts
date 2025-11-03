// Copyright (c) 2022, Binbin Zhang (binbzha@qq.com)
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

#include "frontend/word_break.h"

#include <cctype>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "glog/logging.h"

namespace wetts {

size_t WordBreak::Utf8CharLen(unsigned char lead_byte) {
  if (lead_byte < 0x80) return 1;          // 0xxxxxxx
  if ((lead_byte >> 5) == 0x6) return 2;   // 110xxxxx
  if ((lead_byte >> 4) == 0xE) return 3;   // 1110xxxx
  if ((lead_byte >> 3) == 0x1E) return 4;  // 11110xxx
  return 1;                                // Invalid lead byte, fallback to 1
}

WordBreak::WordBreak(const std::string& lexicon_file) {
  std::ifstream is(lexicon_file);
  CHECK(is.is_open()) << "Failed to open lexicon file: " << lexicon_file;
  std::string line;
  while (getline(is, line)) {
    if (!line.empty()) {
      // Extract the first column (word), separated by space or tab
      size_t pos = line.find_first_of(" \t");
      CHECK(pos != std::string::npos)
          << "Invalid lexicon format, no space or tab found in line: " << line;
      // Extract first column
      std::string word = line.substr(0, pos);
      if (!word.empty()) {
        dictionary_.insert(word);
      }
    }
  }
  VLOG(2) << "Loaded " << dictionary_.size() << " words from lexicon file";
}

WordBreak::WordBreak(const std::unordered_set<std::string>& words)
    : dictionary_(words) {
  VLOG(2) << "Initialized with " << dictionary_.size() << " words";
}

void WordBreak::Segment(const std::string& text,
                        std::vector<std::string>* words) {
  words->clear();
  if (text.empty()) {
    return;
  }

  size_t text_len = text.length();
  size_t pos = 0;

  while (pos < text_len) {
    size_t match_len = FindLongestMatch(text, pos);
    if (match_len > 0) {
      // Found a match, add it to results
      words->emplace_back(text.substr(pos, match_len));
      pos += match_len;
    } else {
      // No match found
      // If current char is ASCII letter or digit, group consecutive
      // letters/digits as one token (split by spaces or other chars)
      unsigned char ch = static_cast<unsigned char>(text[pos]);
      if (ch < 128 && std::isalnum(ch)) {
        size_t start = pos;
        size_t end = pos;
        while (end < text_len) {
          unsigned char c = static_cast<unsigned char>(text[end]);
          if (!(c < 128 && std::isalnum(c))) break;
          end++;
        }
        words->emplace_back(text.substr(start, end - start));
        pos = end;
      } else {
        // Fallback: take one UTF-8 codepoint
        size_t len = Utf8CharLen(static_cast<unsigned char>(text[pos]));
        if (pos + len > text_len) len = 1;  // safety
        words->emplace_back(text.substr(pos, len));
        pos += len;
      }
    }
  }
}

bool WordBreak::HasWord(const std::string& word) const {
  return dictionary_.find(word) != dictionary_.end();
}

size_t WordBreak::DictSize() const { return dictionary_.size(); }

size_t WordBreak::FindLongestMatch(const std::string& text,
                                   size_t start_pos) const {
  size_t text_len = text.length();
  if (start_pos >= text_len) {
    return 0;
  }

  size_t max_match_len = 0;
  // Try all possible lengths from the longest possible (remaining text length)
  // down to 1 character
  size_t max_possible_len = text_len - start_pos;

  for (size_t len = max_possible_len; len >= 1; len--) {
    std::string candidate = text.substr(start_pos, len);
    if (HasWord(candidate)) {
      max_match_len = len;
      break;  // Found the longest match
    }
  }

  return max_match_len;
}

}  // namespace wetts
