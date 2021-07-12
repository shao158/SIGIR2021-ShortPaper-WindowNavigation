#ifndef _POSTING_LIST_H_
#define _POSTING_LIST_H_

#include <cassert>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "SIMD-BP/compositecodec.h"
#include "SIMD-BP/variablebyte.h"
#include "SIMD-BP/simdbinarypacking.h"

struct Metadata_t {
  uint32_t first_doc_id;
  uint32_t last_doc_id;
  float block_max_score;
  bool marked_block;
};

class PostingList {
private:
  uint32_t **encoded_posting_blocks;
  Metadata_t *metadata_list;
  size_t count_posting_blocks;
  size_t count_loaded_blocks;
  size_t count_eval_documents;
  size_t count_dpm;
  size_t count_spm;
  size_t count_decoded_postings;
  uint32_t **decoded_posting_blocks;
  size_t curr_block_size;
  size_t *decoded_block_sizes;
  size_t *encoded_block_sizes;
  size_t curr_block_ind;
  size_t curr_doc_ind; // The index within the current block.
  uint32_t curr_document_id;
  uint32_t curr_document_tf;
  float curr_block_max;
  uint32_t curr_block_last_document_id;

  bool curr_block_loaded;

  std::string term_str;

  float global_max_score;
  
  const float fancy_list_percentage = 0.01;
  float fancy_list_threshold; // The tf threshold for fancy list.

  size_t document_frequency;


  const std::string binary_index_path;
  std::vector<size_t> binary_index_offset;
  std::vector<size_t> binary_index_length;
  std::vector<size_t> variable_size_blocks;

  std::set<size_t> loaded_block_ids;

  const size_t BLOCK_SIZE;

  const size_t NUM_TOTAL_DOC;

  void DeriveFancyList(uint32_t *data_holder);

  void PackIntoConstantSizedBlocks(uint32_t *data_holder);

  void PackIntoVariableSizedBlocks(uint32_t *data_holder);
  
  bool LoadPostingBlock(size_t block_id);

  float CalculateBM25(uint32_t tf, uint32_t doc_id) const;

  static const std::vector<float> norm_doc_len;

  FastPForLib::IntegerCODEC* codec;

public:
  explicit PostingList(const char* index_file_path,
                       const std::vector<std::string>& vocabulary_info,
                       const std::vector<std::string>& block_size_info,
                       size_t constant_block_size,
                       size_t num_total_doc);

  PostingList(const PostingList&) = delete;

  PostingList& operator=(const PostingList&) = delete;

  ~PostingList();

  size_t query_frequency;

  inline std::string TermStr() const { return term_str; }

  inline float GlobalMaxScore() const { return global_max_score; }
  
  inline uint32_t CurrentDocumentID() {
    if (!curr_block_loaded) {
      assert(LoadPostingBlock(curr_block_ind));
    }
    return curr_document_id;
  }

  inline uint32_t CurrentDocumentTF() {
    if (!curr_block_loaded) {
      assert(LoadPostingBlock(curr_block_ind));
    }
    return curr_document_tf;
  }

  inline size_t PostingListLength() const { return document_frequency; }

  inline float CurrentBlockMax() const {
    return curr_block_max;
  }

  inline uint32_t CurrentBlockLastDocumentID() const {
    if (curr_block_ind == count_posting_blocks) {
      return NUM_TOTAL_DOC + 1;
    }
    return curr_block_last_document_id;
  }

  inline uint32_t LastDocumentIDInBlock(size_t block_id) const {
    return metadata_list[block_id].last_doc_id;
  }

  inline uint32_t FirstDocumentIDInBlock(size_t block_id) const {
    return metadata_list[block_id].first_doc_id;
  }

  inline float BlockMaxScoreInBlock(size_t block_id) const {
    return metadata_list[block_id].block_max_score;
  }
  
  inline size_t CountPostingBlocks() const {
    return count_posting_blocks;
  }

  inline size_t CountLoadedBlocks() const {
    return count_loaded_blocks;
  }

  inline size_t CountDecodedPostings() const {
    return count_decoded_postings;
  }

  inline size_t CountEvalDocuments() const {
    return count_eval_documents;
  }

  inline size_t CountDPM() const {
    return count_dpm;
  }

  inline size_t CountSPM() const {
    return count_spm;
  }

  float EvalCurrentDoc() {
    count_eval_documents += 1;
    // assert(CurrentDocumentID() > 0);
    return CalculateBM25(CurrentDocumentTF(), CurrentDocumentID());
  }

  void MarkPostingBlockWithID(size_t block_id) {
    // assert(block_id == NUM_TOTAL_DOC + 1 || block_id < count_posting_blocks);
    if (block_id < count_posting_blocks)
      metadata_list[block_id].marked_block = true;
  }

  void ClearMarkedBlocks() {
    // assert(metadata_list != nullptr);
    for (int i = 0; i < count_posting_blocks; i++) {
      metadata_list[i].marked_block = false;
    }
  }

  void RestartForMarkedBMW();

  void Restart();

  void StartIteration();

  bool Next(uint32_t target_doc_id);

  bool NextShallow(uint32_t target_doc_id);

  bool NextFancy(uint32_t target_doc_id);

  bool NextShallowFancy(uint32_t target_doc_id);
  
  bool NextMarked(uint32_t target_doc_id);

  bool NextShallowMarked(uint32_t target_doc_id);
};

#endif
