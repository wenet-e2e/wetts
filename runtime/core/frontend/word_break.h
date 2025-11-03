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

#ifndef FRONTEND_WORD_BREAK_H_
#define FRONTEND_WORD_BREAK_H_

#include <string>
#include <unordered_set>
#include <vector>

namespace wetts {

// Maximum Forward Matching word segmentation
// This class implements the maximum forward matching algorithm for word
// segmentation (also known as longest match algorithm).
class WordBreak {
 public:
  // Constructor: initialize with a lexicon file
  // The lexicon file should contain one word per line
  explicit WordBreak(const std::string& lexicon_file);

  // Constructor: initialize with a Lexicon object
  // Extract all words from the lexicon to build the dictionary
  explicit WordBreak(const std::unordered_set<std::string>& words);

  // Segment the input text into words using maximum forward matching
  // Args:
  //   text: input text to be segmented
  //   words: output vector of segmented words
  void Segment(const std::string& text, std::vector<std::string>* words);

  // Check if a word exists in the dictionary
  bool HasWord(const std::string& word) const;

  // Get the size of the dictionary
  size_t DictSize() const;

 private:
  // Dictionary: set of valid words
  std::unordered_set<std::string> dictionary_;

  // Find the longest matching word starting from the given position
  // Returns the length of the matched word, or 0 if no match found
  size_t FindLongestMatch(const std::string& text, size_t start_pos) const;

  // Return the byte length of the UTF-8 codepoint starting at text[pos]
  // Fallback to 1 on invalid leading byte
  static size_t Utf8CharLen(unsigned char lead_byte);
};

}  // namespace wetts

#endif  // FRONTEND_WORD_BREAK_H_
