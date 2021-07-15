/*****************************************************************************
Copyright (c) 2020-2021 The Regents of the University of California
All rights reserved

Redistribution and use in source and binary forms, with or without modification 
are permitted provided that the following conditions are met:

1. Redistributions of source code and related material must retain
   this copyright notice, this list of conditions and the following disclaimer. 

2. Neither the name of the University of California at Santa Barbara nor the
   names of its contributors may be used to endorse or promote products derived 
   from this software without specific prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED 
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO 
EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/

/*
Authors: Jinjin Shao
*/

#ifndef _BINARY_INDEX_H_
#define _BINARY_INDEX_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "PostingList.h"

class BinaryIndex {
private:
  std::unordered_map<std::string, PostingList*> all_postings;

  size_t vocabulary_size;

  int kTopK;

  std::vector<std::pair<uint32_t, float>>
      RunWaveQuery(std::vector<PostingList*>& query_postings);

  std::vector<std::pair<uint32_t, float>>
      RunWAndQuery(std::vector<PostingList*>& query_postings);

  std::vector<std::pair<uint32_t, float>>
      RunBMWQuery(std::vector<PostingList*>& query_postings);

  std::vector<std::pair<uint32_t, float>>
      RunTPQuery(std::vector<PostingList*>& query_postings);

  std::vector<std::pair<uint32_t, float>>
      RunBMWQueryWithMarkedPostingBlocks(
          std::vector<PostingList*>& query_postings);

  void RunBMWQueryWithMarkedPostingBlocksAndGivenTopK(
      std::vector<PostingList*>& query_postings,
      std::vector<std::pair<uint32_t, float>>& top_k,
      std::set<uint32_t>& top_k_document_ids);

  void ComputeAllIntervals(
    std::vector<PostingList*>& query_postings,
    std::vector<std::pair<std::vector<size_t>, float>>& all_intervals);

  void SelectTopIntervals(
    std::vector<std::pair<std::vector<size_t>, float>>& all_intervals,
    float z_percentage, std::vector<int>& vec_interval_size);

public:
  explicit BinaryIndex(const char* index_file_path,
                       const char* vocabulary_file_path,
                       size_t dataset_size,
                       size_t constant_block_size,
                       const char* block_variable_size_file_path);

  BinaryIndex(const BinaryIndex&) = delete;

  BinaryIndex& operator=(const BinaryIndex&) = delete;

  ~BinaryIndex();
  
  const size_t NUM_TOTAL_DOC;

  size_t GetVocabularySize() const { return vocabulary_size; }

  // Queries the index and prints out the top-k & profiling results to stdout.
  void Query(const std::vector<std::string>& query_keywords,
             const std::vector<uint64_t>& query_keywords_frequency,
             const char* retrieval_method,
             int topK);
};

#endif
