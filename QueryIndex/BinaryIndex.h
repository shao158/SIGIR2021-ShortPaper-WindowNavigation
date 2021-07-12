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
  // Duplicates in query_keywords are checked inside this function.
  // An invalid retrieval_method results empty results.
  void Query(const std::vector<std::string>& query_keywords,
             const std::vector<uint64_t>& query_keywords_frequency,
             const char* retrieval_method,
             int topK);
};

#endif
