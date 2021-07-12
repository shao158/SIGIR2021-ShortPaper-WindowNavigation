#include "BinaryIndex.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "PostingList.h"

using namespace std::chrono;

static void BubbleDownPosting(std::vector<PostingList*>& query_postings,
                              int pos) {
  for (size_t i = pos + 1; i < query_postings.size(); i++)
    if (query_postings[i]->CurrentDocumentID()
        < query_postings[i - 1]->CurrentDocumentID()) {
      std::swap(query_postings[i], query_postings[i - 1]);
    } else {
      break;
    }
}

static void BubbleDownPostingAfterPivot(
    std::vector<PostingList*>& query_postings, int pos, int pivot) {
  for (size_t i = pos + 1; i <= pivot; i++)
    std::swap(query_postings[i], query_postings[i - 1]);

  for (size_t i = pivot + 1; i < query_postings.size(); i++)
    if (query_postings[i]->CurrentDocumentID()
        < query_postings[i - 1]->CurrentDocumentID()) {
      std::swap(query_postings[i], query_postings[i - 1]);
    } else {
      break;
    }
}

static bool sortByCurrentDoc(PostingList* a, PostingList* b) {
  return (a->CurrentDocumentID() < b->CurrentDocumentID());
}

static bool sortByKeywords(PostingList* a, PostingList* b) {
  return (a->TermStr().compare(b->TermStr()) < 0);
}

static bool sortByScore(const std::pair<int32_t, float> &a,
                        const std::pair<int32_t, float> &b) {
  return (a.second > b.second);
}

static bool sortIntervalsByScore(
    const std::pair<std::vector<size_t>, float> &a,
    const std::pair<std::vector<size_t>, float> &b) {
  return (a.second > b.second);
}

BinaryIndex::BinaryIndex(const char* index_file_path,
                         const char* vocabulary_file_path,
                         size_t dataset_size,
                         size_t constant_block_size,
                         const char* block_variable_size_file_path)
    : NUM_TOTAL_DOC(dataset_size) {
  vocabulary_size = 0;

  std::ifstream index_file(index_file_path, std::ios::in | std::ios::binary);
  if (!index_file.is_open()) {
    std::cerr << "Failed to open the binary index file: "
              << index_file_path << std::endl;
    return;
  }

  std::string line;
  std::ifstream vocabulary_file(vocabulary_file_path, std::ios::in);
  if (!vocabulary_file.is_open()) {
    index_file.close();
    std::cerr << "Failed to open the vocabulary file: "
              << vocabulary_file_path
              << std::endl;
    return;
  }

  std::ifstream block_size_file(block_variable_size_file_path, std::ios::in);
  if (constant_block_size == 0 && !block_size_file.is_open()) {
    index_file.close();
    vocabulary_file.close();
    std::cerr << "Failed to open block size file: "
              << block_variable_size_file_path
              << std::endl;
    return;
  }

  // Start to build this BinaryIndex from all opened files.

  bool succ = true;
  while (getline(vocabulary_file, line)) {
    std::istringstream iss(line);
    std::vector<std::string> vocabulary_info(
          std::istream_iterator<std::string>{iss},
          std::istream_iterator<std::string>());
    PostingList* curr = nullptr;
    
    if (constant_block_size == 0) {
      if (!getline(block_size_file, line)) {
        std::cerr << "Failed to read a block size info. " << std::endl;
        succ = false;
        break;
      }
      std::istringstream iss2(line);
      std::vector<std::string> block_size_info( 
          std::istream_iterator<std::string>{iss2},
          std::istream_iterator<std::string>());
      if (!block_size_info[0].compare(vocabulary_info[0])) {
        curr = new PostingList(index_file_path,
                               vocabulary_info, block_size_info,
                               constant_block_size, NUM_TOTAL_DOC);
      } else {
        std::cerr << "Block size info has a mismatch: "
                  << block_size_info[0] << " vs. " << vocabulary_info[0]
                  << std::endl;
      }
    } else {
      curr = new PostingList(index_file_path,
                             vocabulary_info, std::vector<std::string>(),
                             constant_block_size, NUM_TOTAL_DOC);
    }

    if (curr == nullptr) {
      std::cerr << "Failed to init a PostingList. " << std::endl;
      succ = false;
      break;
    }

    all_postings.insert(std::make_pair(curr->TermStr(), curr));
  }

  if (succ) {
    vocabulary_size = all_postings.size();
  }

  index_file.close();
  vocabulary_file.close();
  block_size_file.close();
}

BinaryIndex::~BinaryIndex() {
  for (auto it : all_postings) if (it.second != nullptr) delete it.second;
}

std::vector<std::pair<uint32_t, float>> BinaryIndex::RunWaveQuery(
    std::vector<PostingList*>& query_postings) {
  int num_keywords = query_postings.size();
  std::vector<std::pair<uint32_t, float>> top_k;
  std::make_heap(top_k.begin(), top_k.end(), sortByScore); 

  uint32_t end_doc_id = NUM_TOTAL_DOC + 1;
  while (true) {
    uint32_t pivot = end_doc_id;
    for (int i = 0; i < num_keywords; i++) {
      pivot = std::min(pivot, query_postings[i]->CurrentDocumentID());
    }
    if (pivot == end_doc_id) break;

    float eval_score = 0.0;

    for (int i = 0; i < num_keywords; i++) {
      if (pivot == query_postings[i]->CurrentDocumentID()) {
        eval_score += query_postings[i]->EvalCurrentDoc();
        query_postings[i]->Next(pivot + 1);
      }
    }

    if (top_k.size() < kTopK) {
      top_k.push_back(std::make_pair(pivot, eval_score));
      std::push_heap(top_k.begin(), top_k.end(), sortByScore);
    } else {
      if (eval_score > top_k.front().second) {
        std::pop_heap(top_k.begin(), top_k.end(), sortByScore);
        top_k.pop_back();
        top_k.push_back(std::make_pair(pivot, eval_score));
        std::push_heap(top_k.begin(), top_k.end(), sortByScore);
      }
    }
  }

  std::sort_heap(top_k.begin(), top_k.end(), sortByScore);

  return top_k;
}

std::vector<std::pair<uint32_t, float>> BinaryIndex::RunWAndQuery(
    std::vector<PostingList*>& query_postings) {
  int num_keywords = query_postings.size();
  std::vector<std::pair<uint32_t, float>> top_k;
  std::make_heap(top_k.begin(), top_k.end(), sortByScore); 
  
  uint32_t end_doc_id = NUM_TOTAL_DOC + 1;
  while (true) {
    std::sort(query_postings.begin(), query_postings.end(), sortByCurrentDoc);

    if (query_postings[0]->CurrentDocumentID() == end_doc_id) break;

    int pivot = 0;
    float attempt_ranking_score = query_postings[pivot]->GlobalMaxScore();
    while (top_k.size() >= kTopK
        && attempt_ranking_score <= top_k.front().second) {
      pivot += 1;
      if (pivot >= num_keywords) break;
      attempt_ranking_score += query_postings[pivot]->GlobalMaxScore();
    }
    if (pivot >= num_keywords) break;

    uint32_t pivot_doc = query_postings[pivot]->CurrentDocumentID();
    if (pivot_doc == end_doc_id) break;

    float pivot_score = 0.0;
    for (int i = 0; i < num_keywords; i++) {
      query_postings[i]->Next(pivot_doc);
      if (query_postings[i]->CurrentDocumentID() == pivot_doc) {
        pivot_score += query_postings[i]->EvalCurrentDoc();
        query_postings[i]->Next(pivot_doc + 1);
      }
    }

    if (top_k.size() < kTopK) {
      top_k.push_back(std::make_pair(pivot_doc, pivot_score));
      std::push_heap(top_k.begin(), top_k.end(), sortByScore);
    } else {
      if (pivot_score > top_k.front().second) {
        std::pop_heap(top_k.begin(), top_k.end(), sortByScore);
        top_k.pop_back();
        top_k.push_back(std::make_pair(pivot_doc, pivot_score));
        std::push_heap(top_k.begin(), top_k.end(), sortByScore);
      }
    }
  }

  std::sort_heap(top_k.begin(), top_k.end(), sortByScore);

  return top_k;
}

std::vector<std::pair<uint32_t, float>> BinaryIndex::RunBMWQuery(
    std::vector<PostingList*>& query_postings) {
  int num_keywords = query_postings.size();
  std::vector<std::pair<uint32_t, float>> top_k;
  std::make_heap(top_k.begin(), top_k.end(), sortByScore); 

  std::sort(query_postings.begin(), query_postings.end(), sortByCurrentDoc);
  
  uint32_t end_doc_id = NUM_TOTAL_DOC + 1;
  while (true) {
    /*
    std::sort(query_postings.begin(), query_postings.end(), sortByCurrentDoc);
    */

    if (query_postings[0]->CurrentDocumentID() == end_doc_id) break;
    
    /////////////////////////////
    // Step 1: Find a pivot.
    /////////////////////////////
    int pivot = 0;
    float attempt_ranking_score = query_postings[pivot]->GlobalMaxScore();
    while (top_k.size() >= kTopK
        && attempt_ranking_score <= top_k.front().second) {
      pivot += 1;
      if (pivot >= num_keywords) break;
      attempt_ranking_score += query_postings[pivot]->GlobalMaxScore();
    }
    if (pivot >= num_keywords) break;

    uint32_t pivot_doc = query_postings[pivot]->CurrentDocumentID();
    if (pivot_doc == end_doc_id) break;

    for (size_t i = 0; i < pivot; i++) {
      query_postings[i]->NextShallow(pivot_doc);
    }

    while (pivot + 1 < query_postings.size()
           && query_postings[pivot + 1]->CurrentDocumentID() == pivot_doc) {
      pivot += 1;
    }

    ////////////////////////////////////
    // Step 2: Check Block-Max Score.
    ////////////////////////////////////
    attempt_ranking_score = 0.0;
    for (size_t i = 0; i <= pivot; i++)
      attempt_ranking_score += query_postings[i]->CurrentBlockMax();

    if (top_k.size() < kTopK
        || attempt_ranking_score > top_k.front().second) {
      /////////////////////////////////////
      // Step 3a: Let's evaluate this dude!
      /////////////////////////////////////
      if (query_postings[0]->CurrentDocumentID() != pivot_doc) {
        int next_pivot = pivot - 1;
        while (query_postings[next_pivot]->CurrentDocumentID() == pivot_doc) {
          next_pivot -= 1;
        }
        query_postings[next_pivot]->Next(pivot_doc);
        BubbleDownPosting(query_postings, next_pivot);
        continue;
      }

      float pivot_score = 0.0;
      for (size_t i = 0; i <= pivot; i++) {
        query_postings[i]->Next(pivot_doc);
        if (query_postings[i]->CurrentDocumentID() == pivot_doc) {
          pivot_score += query_postings[i]->EvalCurrentDoc();
          
          query_postings[i]->Next(pivot_doc + 1);
        }
      }

      std::sort(query_postings.begin(), query_postings.end(), sortByCurrentDoc);

      if (top_k.size() < kTopK) {
        top_k.push_back(std::make_pair(pivot_doc, pivot_score));
        std::push_heap(top_k.begin(), top_k.end(), sortByScore);
      } else {
        if (pivot_score > top_k.front().second) {
          std::pop_heap(top_k.begin(), top_k.end(), sortByScore);
          top_k.pop_back();
          top_k.push_back(std::make_pair(pivot_doc, pivot_score));
          std::push_heap(top_k.begin(), top_k.end(), sortByScore);
        }
      }
    } else {
      ///////////////////////////////////////
      // Step 3b: There is no need to evaluate further. :(
      ///////////////////////////////////////
      int target_list = 0;
      size_t smallest_length = query_postings[0]->PostingListLength();
      uint32_t next_pivot_doc = end_doc_id;
      for (size_t i = 0; i <= pivot; i++) {
        next_pivot_doc = std::min(next_pivot_doc,
            query_postings[i]->CurrentBlockLastDocumentID());

        if (query_postings[i]->PostingListLength() < smallest_length) {
          target_list = i;
          smallest_length = query_postings[i]->PostingListLength();
        }
      }
      next_pivot_doc += 1;

      assert(next_pivot_doc > query_postings[pivot]->CurrentDocumentID());

      /*
      if (next_pivot_doc <= query_postings[pivot]->CurrentDocumentID()) {
        std::cout << next_pivot_doc << " " << query_postings[pivot]->CurrentDocumentID() << std::endl; 
        for (size_t i = 0; i <= pivot; i++) {
          std::cout << query_postings[i]->CurrentDocumentID() << std::endl;
          std::cout << query_postings[i]->CurrentBlockLastDocumentID() << std::endl << std::endl;
        }
        if (pivot + 1 < query_postings.size()) {
          std::cout << query_postings[pivot + 1]->CurrentDocumentID() << std::endl;
        }
        
        next_pivot_doc = query_postings[pivot]->CurrentDocumentID() + 1;
      }
      */
      
      if (pivot + 1 < query_postings.size()) {
        next_pivot_doc = std::min(next_pivot_doc,
            query_postings[pivot + 1]->CurrentDocumentID());
      }

      if (next_pivot_doc <= query_postings[pivot]->CurrentDocumentID()) {
        next_pivot_doc = query_postings[pivot]->CurrentDocumentID() + 1;
      }
      
      query_postings[target_list]->Next(next_pivot_doc);

      BubbleDownPostingAfterPivot(query_postings, target_list, pivot);

      /*
      for (size_t i = 0; i <= pivot; i++) {
        query_postings[i]->Next(next_pivot_doc);
      }

      if (pivot + 1 < query_postings.size()) {
        query_postings[pivot + 1]->Next(next_pivot_doc);
      }
      */

      // std::sort(query_postings.begin(), query_postings.end(), sortByCurrentDoc);  
    }
  }

  std::sort_heap(top_k.begin(), top_k.end(), sortByScore);

  return top_k;
}

void BinaryIndex::RunBMWQueryWithMarkedPostingBlocksAndGivenTopK(
    std::vector<PostingList*>& query_postings,
    std::vector<std::pair<uint32_t, float>>& top_k,
    std::set<uint32_t>& top_k_document_ids) {
  int num_keywords = query_postings.size();

  std::sort(query_postings.begin(), query_postings.end(), sortByCurrentDoc);

  uint32_t end_doc_id = NUM_TOTAL_DOC + 1;
  while (true) {

    if (query_postings[0]->CurrentDocumentID() == end_doc_id) break;

    int pivot = 0;
    float attempt_ranking_score = query_postings[pivot]->GlobalMaxScore();
    while (top_k.size() >= kTopK
        && attempt_ranking_score <= top_k.front().second) {
      pivot += 1;
      if (pivot >= num_keywords) break;
      attempt_ranking_score += query_postings[pivot]->GlobalMaxScore();
    }
      
    if (pivot >= num_keywords) break;
    uint32_t pivot_doc = query_postings[pivot]->CurrentDocumentID();

    if (pivot_doc == end_doc_id) break;
    for (int i = 0; i < pivot; i++) {
      query_postings[i]->NextShallowMarked(pivot_doc);
    }

    while (pivot + 1 < query_postings.size()
           && query_postings[pivot + 1]->CurrentDocumentID() == pivot_doc) {
      pivot += 1;
    }

    attempt_ranking_score = 0.0;
    for (int i = 0; i <= pivot; i++)
      attempt_ranking_score += query_postings[i]->CurrentBlockMax();

    if (top_k.size() < kTopK
        || attempt_ranking_score > top_k.front().second) {
      if (query_postings[0]->CurrentDocumentID() != pivot_doc) {
        int next_pivot = pivot - 1;
        while (query_postings[next_pivot]->CurrentDocumentID() == pivot_doc) {
          next_pivot -= 1;
        }
        query_postings[next_pivot]->NextMarked(pivot_doc);
        BubbleDownPosting(query_postings, next_pivot);
        continue;
      }

      float pivot_score = 0.0;
      for (int i = 0; i <= pivot; i++) {
        query_postings[i]->NextMarked(pivot_doc);
        if (query_postings[i]->CurrentDocumentID() == pivot_doc) {
          pivot_score += query_postings[i]->EvalCurrentDoc();
          query_postings[i]->NextMarked(pivot_doc + 1);
        }
      }

      std::sort(query_postings.begin(), query_postings.end(), sortByCurrentDoc);

      if (top_k.size() < kTopK) {
        if (top_k_document_ids.find(pivot_doc) == top_k_document_ids.end()) {
          top_k.push_back(std::make_pair(pivot_doc, pivot_score));
          std::push_heap(top_k.begin(), top_k.end(), sortByScore);
          top_k_document_ids.insert(pivot_doc);
        } else {
          for (auto& each_top_k : top_k) {
            if (each_top_k.first != pivot_doc) continue;
            if (pivot_score > each_top_k.second) {
              each_top_k.second = pivot_score;
              std::make_heap(top_k.begin(), top_k.end(), sortByScore);
            }
            break;
          }
        }
      } else if (pivot_score > top_k.front().second) {
        if (top_k_document_ids.find(pivot_doc) == top_k_document_ids.end()) {
          std::pop_heap(top_k.begin(), top_k.end(), sortByScore);
          top_k_document_ids.erase(top_k.back().first);
          top_k.pop_back();
          top_k.push_back(std::make_pair(pivot_doc, pivot_score));
          std::push_heap(top_k.begin(), top_k.end(), sortByScore);
          top_k_document_ids.insert(pivot_doc);
        } else {
          for (auto& each_top_k : top_k) {
            if (each_top_k.first != pivot_doc) continue;
            if (pivot_score > each_top_k.second) {
              each_top_k.second = pivot_score;
              std::make_heap(top_k.begin(), top_k.end(), sortByScore);
            }
            break;
          }
        }
      }
    } else {
      int target_list = 0;
      size_t smallest_length = query_postings[0]->PostingListLength();
      uint32_t next_pivot_doc = end_doc_id;
      for (size_t i = 0; i <= pivot; i++) {
        next_pivot_doc = std::min(next_pivot_doc,
            query_postings[i]->CurrentBlockLastDocumentID());

        if (query_postings[i]->PostingListLength() < smallest_length) {
          target_list = i;
          smallest_length = query_postings[i]->PostingListLength();
        }
      }
      next_pivot_doc += 1;

      assert(next_pivot_doc > query_postings[pivot]->CurrentDocumentID());

      if (pivot + 1 < query_postings.size()) {
        next_pivot_doc = std::min(next_pivot_doc,
            query_postings[pivot + 1]->CurrentDocumentID());
      }

      if (next_pivot_doc <= query_postings[pivot]->CurrentDocumentID()) {
        next_pivot_doc = query_postings[pivot]->CurrentDocumentID() + 1;
      }
      
      query_postings[target_list]->NextMarked(next_pivot_doc);

      BubbleDownPostingAfterPivot(query_postings, target_list, pivot);
    }
  }
}

std::vector<std::pair<uint32_t, float>>
    BinaryIndex::RunBMWQueryWithMarkedPostingBlocks(
        std::vector<PostingList*>& query_postings) {
  int num_keywords = query_postings.size();
  std::vector<std::pair<uint32_t, float>> top_k;
  std::make_heap(top_k.begin(), top_k.end(), sortByScore);

  std::sort(query_postings.begin(), query_postings.end(), sortByCurrentDoc);

  uint32_t end_doc_id = NUM_TOTAL_DOC + 1;
  while (true) {

    if (query_postings[0]->CurrentDocumentID() == end_doc_id) break;

    int pivot = 0;
    float attempt_ranking_score = query_postings[pivot]->GlobalMaxScore();
    while (top_k.size() >= kTopK
        && attempt_ranking_score <= top_k.front().second) {
      pivot += 1;
      if (pivot >= num_keywords) break;
      attempt_ranking_score += query_postings[pivot]->GlobalMaxScore();
    }
    if (pivot >= num_keywords) break;

    uint32_t pivot_doc = query_postings[pivot]->CurrentDocumentID();
    if (pivot_doc == end_doc_id) break;

    for (int i = 0; i < pivot; i++) {
      query_postings[i]->NextShallowMarked(pivot_doc);
    }

    while (pivot + 1 < query_postings.size()
           && query_postings[pivot + 1]->CurrentDocumentID() == pivot_doc) {
      pivot += 1;
    }

    attempt_ranking_score = 0.0;
    for (int i = 0; i <= pivot; i++)
      attempt_ranking_score += query_postings[i]->CurrentBlockMax();

    if (top_k.size() < kTopK
        || attempt_ranking_score > top_k.front().second) {
      if (query_postings[0]->CurrentDocumentID() != pivot_doc) {
        int next_pivot = pivot - 1;
        while (query_postings[next_pivot]->CurrentDocumentID() == pivot_doc) {
          next_pivot -= 1;
        }
        query_postings[next_pivot]->NextMarked(pivot_doc);
        BubbleDownPosting(query_postings, next_pivot);
        continue;
      }

      float pivot_score = 0.0;
      for (int i = 0; i <= pivot; i++) {
        query_postings[i]->NextMarked(pivot_doc);
        if (query_postings[i]->CurrentDocumentID() == pivot_doc) {
          pivot_score += query_postings[i]->EvalCurrentDoc();
          query_postings[i]->NextMarked(pivot_doc + 1);
        }
      }

      std::sort(query_postings.begin(), query_postings.end(), sortByCurrentDoc);

      if (top_k.size() < kTopK) {
        top_k.push_back(std::make_pair(pivot_doc, pivot_score));
        std::push_heap(top_k.begin(), top_k.end(), sortByScore);
      } else {
        if (pivot_score > top_k.front().second) {
          std::pop_heap(top_k.begin(), top_k.end(), sortByScore);
          top_k.pop_back();
          top_k.push_back(std::make_pair(pivot_doc, pivot_score));
          std::push_heap(top_k.begin(), top_k.end(), sortByScore);
        }
      }
    } else {
      int target_list = 0;
      size_t smallest_length = query_postings[0]->PostingListLength();
      uint32_t next_pivot_doc = end_doc_id;
      for (size_t i = 0; i <= pivot; i++) {
        next_pivot_doc = std::min(next_pivot_doc,
            query_postings[i]->CurrentBlockLastDocumentID());

        if (query_postings[i]->PostingListLength() < smallest_length) {
          target_list = i;
          smallest_length = query_postings[i]->PostingListLength();
        }
      }
      next_pivot_doc += 1;

      assert(next_pivot_doc > query_postings[pivot]->CurrentDocumentID());

      if (pivot + 1 < query_postings.size()) {
        next_pivot_doc = std::min(next_pivot_doc,
            query_postings[pivot + 1]->CurrentDocumentID());
      }

      if (next_pivot_doc <= query_postings[pivot]->CurrentDocumentID()) {
        next_pivot_doc = query_postings[pivot]->CurrentDocumentID() + 1;
      }
      
      query_postings[target_list]->NextMarked(next_pivot_doc);

      BubbleDownPostingAfterPivot(query_postings, target_list, pivot);
    }
  }

  std::sort_heap(top_k.begin(), top_k.end(), sortByScore);

  return top_k;
}

void BinaryIndex::ComputeAllIntervals(
    std::vector<PostingList*>& query_postings,
    std::vector<std::pair<std::vector<size_t>, float>>& all_intervals) {
  int num_keywords = query_postings.size();

  uint32_t all_count_postings_blocks = 0;
  uint32_t count_posting_blocks[num_keywords];
  size_t curr_block_ind[num_keywords];
  for (int i = 0; i < num_keywords; i++) {
    all_count_postings_blocks += query_postings[i]->CountPostingBlocks();
    count_posting_blocks[i] = query_postings[i]->CountPostingBlocks();
    curr_block_ind[i] = 0;
  }

  uint32_t end_doc_id = NUM_TOTAL_DOC + 1;
  uint32_t next_pivot_doc_id = end_doc_id;
  uint32_t curr_posting_last_doc_id;
  for (size_t i = 0; i < num_keywords; i++) {
    if (curr_block_ind[i] < count_posting_blocks[i]) {
      curr_posting_last_doc_id =
          query_postings[i]->LastDocumentIDInBlock(curr_block_ind[i]); 
      if (next_pivot_doc_id > curr_posting_last_doc_id) {
        next_pivot_doc_id = curr_posting_last_doc_id;
      }
    }
  }
  
  all_intervals.reserve(all_count_postings_blocks);
  std::vector<size_t> curr_interval(num_keywords, end_doc_id);
  float interval_max_score = 0.0;
  uint32_t pivot_doc_id = end_doc_id;
  size_t i;
  while (true) {
    pivot_doc_id = next_pivot_doc_id;
    if (pivot_doc_id == end_doc_id) break;
    next_pivot_doc_id = end_doc_id; 

    interval_max_score = 0.0;
    for (i = 0; i < num_keywords; i++) {
      if (curr_block_ind[i] < count_posting_blocks[i]
          && query_postings[i]->FirstDocumentIDInBlock(curr_block_ind[i])
             <= pivot_doc_id) {

        interval_max_score +=
            query_postings[i]->BlockMaxScoreInBlock(curr_block_ind[i]);
        curr_interval[i] = curr_block_ind[i];

        curr_block_ind[i] += 
            (query_postings[i]->LastDocumentIDInBlock(curr_block_ind[i])
             == pivot_doc_id);
      }
      else {
        curr_interval[i] = end_doc_id;
      }


      if (curr_block_ind[i] < count_posting_blocks[i]) {
        curr_posting_last_doc_id =
            query_postings[i]->LastDocumentIDInBlock(curr_block_ind[i]); 
        if (next_pivot_doc_id > curr_posting_last_doc_id) {
          next_pivot_doc_id = curr_posting_last_doc_id;
        }
      }
    }

    all_intervals.emplace_back(
        std::make_pair(curr_interval, interval_max_score));
  }
}

void BinaryIndex::SelectTopIntervals(
    std::vector<std::pair<std::vector<size_t>, float>>& all_intervals,
    float z_percentage, std::vector<int>& vec_interval_size) {
  size_t base_step_size = 8;

  size_t all_intervals_size = all_intervals.size();
  
  int z_size_limit = int(z_percentage);
  if (z_percentage < 1.0) {
    z_size_limit = std::max(int(all_intervals_size * z_percentage), 1024);
  }

  // z_size_limit = 1024; // adjusted with k.

  if (all_intervals_size <= 1024) {
    int tmp = std::max(int(all_intervals_size) / 100, 1);
    std::nth_element(all_intervals.begin(),
        all_intervals.begin() + tmp - 1,
        all_intervals.end(), sortIntervalsByScore);
    vec_interval_size.emplace_back(tmp);
    return;
  }

  size_t next_intervals_size = z_size_limit;
  while (next_intervals_size * base_step_size < all_intervals_size) {
    next_intervals_size *= base_step_size;
  }

  size_t curr_intervals_size = all_intervals_size;
  while (next_intervals_size >= z_size_limit) {
    std::nth_element(all_intervals.begin(),
        all_intervals.begin() + next_intervals_size - 1,
        all_intervals.begin() + curr_intervals_size, 
        sortIntervalsByScore);
    curr_intervals_size = next_intervals_size;
    vec_interval_size.insert(vec_interval_size.begin(), next_intervals_size);
    next_intervals_size = curr_intervals_size / base_step_size;
  }
}

std::vector<std::pair<uint32_t, float>> BinaryIndex::RunTPQuery(
    std::vector<PostingList*>& query_postings) {
  size_t count_total_blocks = 0;
  for (int i = 0; i < query_postings.size(); i++) {
    count_total_blocks += query_postings[i]->CountPostingBlocks();
  }
  // If z_percentage is a real value smaller than 1.0,
  // take it as a percentage parameter in linear-based sampling.
  // If z_percentage is a real value larger than 1.0,
  // take it as a absolution number parameter in fixed sampling.
  float z_percentage = 0.01;
  /*
  int max_num_intervals = 1;
  if (z_percentage < 1.0) {
    max_num_intervals = std::max(max_num_intervals,
                                 int (z_percentage * count_total_blocks) );
  } else {
    max_num_intervals = std::max(max_num_intervals,
                                 int (z_percentage) );
  }
  */

  std::vector<int> vec_interval_size;

  ///////////////////////////////////////////////////////////////////

  high_resolution_clock::time_point t0 = high_resolution_clock::now();

  int num_keywords = query_postings.size();
  std::vector<std::pair<std::vector<size_t>, float>> all_intervals;
  ComputeAllIntervals(query_postings, all_intervals);

  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  high_resolution_clock::time_point t2 = high_resolution_clock::now();

  /* Sorting all windows is expensive!
  std::sort(all_intervals.begin(), all_intervals.end(), sortIntervalsByScore);
  std::vector<std::pair<std::vector<size_t>, double>>
      top_intervals =
          ComputeTopIntervals(query_postings, all_intervals, max_num_intervals);
  */

  SelectTopIntervals(all_intervals, z_percentage, vec_interval_size);

  high_resolution_clock::time_point t3 = high_resolution_clock::now();

  std::cout << "Number of intervals: " << all_intervals.size() << std::endl;

  // assert(!vec_interval_size.empty());

  ////////////////////////////////////////////////////////////////////

  for (int i = 0; i < num_keywords; i++) {
    query_postings[i]->ClearMarkedBlocks();
  }
  
  high_resolution_clock::time_point t4 = high_resolution_clock::now();

  for (size_t j = 0;
       j < all_intervals.size() && j < vec_interval_size[0]; j++) {
    for (int i = 0; i < num_keywords; i++) {
      query_postings[i]->MarkPostingBlockWithID(all_intervals[j].first[i]);
    }
  }

  for (int i = 0; i < num_keywords; i++) {
    query_postings[i]->RestartForMarkedBMW();
  }

  std::vector<std::pair<uint32_t, float>> top_k
      = RunBMWQueryWithMarkedPostingBlocks(query_postings);

  high_resolution_clock::time_point t5 = high_resolution_clock::now();

  size_t count_loaded_blocks = 0;
  size_t count_eval_documents = 0, count_dpm = 0, count_spm = 0;
  for (int i = 0; i < query_postings.size(); i++) {
    count_loaded_blocks += query_postings[i]->CountLoadedBlocks();
    count_eval_documents += query_postings[i]->CountEvalDocuments();
    count_dpm += query_postings[i]->CountDPM();
    count_spm += query_postings[i]->CountSPM();
  }
  
  const float epsilon = 1e-6;

  for (int i = 0; i < num_keywords; i++) {
    query_postings[i]->RestartForMarkedBMW();
  }

  // assert(top_k.size() <= kTopK);

  float approx_threshold = top_k.back().second;

  /////////////////////////////////////////////////////////

  std::set<uint32_t> top_k_document_ids;
  for (auto each_top_k : top_k) top_k_document_ids.insert(each_top_k.first);
  std::make_heap(top_k.begin(), top_k.end(), sortByScore);

  size_t count_ms_below = 0; // microseconds.
  size_t cumulative_interval_size = vec_interval_size[0];
  for (size_t i = 1; i < vec_interval_size.size(); i++) {
    // This cost should be avoided with further optimizations.
    std::sort(query_postings.begin(), query_postings.end(), sortByKeywords);

    for (int j = 0; j < num_keywords; j++) {
      query_postings[j]->ClearMarkedBlocks();
    }

    high_resolution_clock::time_point t00 = high_resolution_clock::now();
    
    if (all_intervals[cumulative_interval_size - 1].second
        + epsilon < approx_threshold) break;

    cumulative_interval_size = vec_interval_size[i];

    for (size_t j = vec_interval_size[i - 1]; j < vec_interval_size[i]; j++) {
      if (all_intervals[j].second + epsilon < approx_threshold) continue;
      for (int k = 0; k < num_keywords; k++) {
        query_postings[k]->MarkPostingBlockWithID(all_intervals[j].first[k]);
      }
    }

    for (int j = 0; j < num_keywords; j++) {
      query_postings[j]->RestartForMarkedBMW();
    }

    RunBMWQueryWithMarkedPostingBlocksAndGivenTopK(query_postings,
        top_k, top_k_document_ids);

    high_resolution_clock::time_point t01 = high_resolution_clock::now();

    std::sort_heap(top_k.begin(), top_k.end(), sortByScore);
    approx_threshold = top_k.back().second;
    std::make_heap(top_k.begin(), top_k.end(), sortByScore);

    count_ms_below += duration_cast<microseconds>(t01 - t00).count();

    for (int j = 0; j < query_postings.size(); j++) {
      count_loaded_blocks += query_postings[j]->CountLoadedBlocks();
      count_eval_documents += query_postings[j]->CountEvalDocuments();
      count_dpm += query_postings[j]->CountDPM();
      count_spm += query_postings[j]->CountSPM();
    }

    std::cout << "Interval size: " << vec_interval_size[i]
              << " tmp time: " << duration_cast<microseconds>(t01 - t00).count()
              << std::endl;
  }

  ////////////////////////////////////////////////////////////////////

  std::sort(query_postings.begin(), query_postings.end(), sortByKeywords);

  // assert(!top_k.empty());
  std::sort_heap(top_k.begin(), top_k.end(), sortByScore);
  approx_threshold = top_k.back().second;

  std::make_heap(top_k.begin(), top_k.end(), sortByScore);

  ////////////////////////////////////////////////////////////////////

  for (int i = 0; i < num_keywords; i++) {
    query_postings[i]->ClearMarkedBlocks();
  }

  high_resolution_clock::time_point t6 = high_resolution_clock::now();

  if (all_intervals[cumulative_interval_size - 1].second + epsilon
      < approx_threshold) {
    // Skipping the last phase.
  } else {
    size_t all_intervals_size = all_intervals.size();
    for (size_t i = cumulative_interval_size; i < all_intervals_size; i++) {
      if (all_intervals[i].second + epsilon < approx_threshold) continue;
      for (int j = 0; j < num_keywords; j++) {
        query_postings[j]->MarkPostingBlockWithID(all_intervals[i].first[j]);
      }
    }  

    for (int i = 0; i < num_keywords; i++) {
      query_postings[i]->RestartForMarkedBMW();
    }

    RunBMWQueryWithMarkedPostingBlocksAndGivenTopK(query_postings,
        top_k, top_k_document_ids);
  }

  high_resolution_clock::time_point t7 = high_resolution_clock::now();

  ////////////////////////////////////////////////////////////////////

  std::sort_heap(top_k.begin(), top_k.end(), sortByScore);

  size_t count_loaded_blocks_2 = 0;
  size_t count_eval_documents_2 = 0, count_dpm_2 = 0, count_spm_2 = 0;
  for (int i = 0; i < query_postings.size(); i++) {
    count_loaded_blocks_2 += query_postings[i]->CountLoadedBlocks();
    count_eval_documents_2 += query_postings[i]->CountEvalDocuments();
    count_dpm_2 += query_postings[i]->CountDPM();
    count_spm_2 += query_postings[i]->CountSPM();
  }

  std::cout << "Time cost total: "
            << duration_cast<milliseconds>(t1 - t0).count()
                + duration_cast<milliseconds>(t3 - t2).count()
                + duration_cast<milliseconds>(t5 - t4).count()
                + duration_cast<milliseconds>(t7 - t6).count()
                + count_ms_below / 1000
            << " ms. " 
            << "In detail: "
            << duration_cast<milliseconds>(t1 - t0).count()
            << " " << duration_cast<milliseconds>(t3 - t2).count()
            << " " << duration_cast<milliseconds>(t5 - t4).count()
            << " " << duration_cast<milliseconds>(t7 - t6).count()
            << " " << count_ms_below / 1000
            << std::endl;

  std::cout << "Loaded postings blocks: "
            << count_loaded_blocks + count_loaded_blocks_2
            << " of " << count_total_blocks
            << ". In detail: " << count_loaded_blocks
            << " " << count_loaded_blocks_2 << std::endl;

  std::cout << "Number of eval: "
            << count_eval_documents + count_eval_documents_2
            << ". In detail: " << count_eval_documents
            << " " << count_eval_documents_2 << std::endl;

  std::cout << "dpm and spm: " << count_dpm + count_dpm_2
            << " " << count_spm + count_spm_2
            << ". In detail: " << count_dpm << " + " << count_dpm_2
            << " " << count_spm << " + " << count_spm_2 << std::endl;
  
  return top_k;
}

void BinaryIndex::Query(
    const std::vector<std::string>& query_keywords,
    const std::vector<uint64_t>& query_keywords_frequency,
    const char* retrieval_method,
    int top_k) {
  std::cout << "Query";
  int fre_ind = 0;
  for (std::string each_keyword : query_keywords) {
    std::cout << ":" << each_keyword
              << "," << query_keywords_frequency[fre_ind++];
  }
  std::cout << std::endl;

  kTopK = top_k;

  std::vector<PostingList*> query_postings;
  for (int i = 0; i < query_keywords.size(); i++) {
    if (all_postings.find(query_keywords[i]) == all_postings.end()) {
      std::cout << "No posting available: "
                 << query_keywords[i] << std::endl;
      return;
    }
    query_postings.emplace_back(all_postings[query_keywords[i]]);
    query_postings[i]->query_frequency = query_keywords_frequency[i];
    query_postings[i]->StartIteration();
  }

  std::sort(query_postings.begin(), query_postings.end(), sortByKeywords);
  
  std::vector<std::pair<uint32_t, float>> top_k_accumulator;

  high_resolution_clock::time_point t0 = high_resolution_clock::now();

  std::cout << "Start query now. " << std::endl;

  if (!std::strcmp(retrieval_method, "wave")) {
    top_k_accumulator = RunWaveQuery(query_postings);
  } else if (!std::strcmp(retrieval_method, "wand")) {
    top_k_accumulator = RunWAndQuery(query_postings);
  } else if (!std::strcmp(retrieval_method, "bmw")) {
    top_k_accumulator = RunBMWQuery(query_postings);
  } else if (!std::strcmp(retrieval_method, "tp")) {
    top_k_accumulator = RunTPQuery(query_postings);
  }

  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  for (auto iter : top_k_accumulator) {
    std::cout << iter.first << " : " << iter.second << std::endl;
  }

  size_t count_total_blocks = 0, count_loaded_blocks = 0;
  size_t count_eval_documents = 0, count_dpm = 0, count_spm = 0;
  size_t count_decoded_postings = 0;
  for (int i = 0; i < query_keywords.size(); i++) {
    count_loaded_blocks += query_postings[i]->CountLoadedBlocks();
    count_total_blocks += query_postings[i]->CountPostingBlocks();
    count_eval_documents += query_postings[i]->CountEvalDocuments();
    count_dpm += query_postings[i]->CountDPM();
    count_spm += query_postings[i]->CountSPM();
    count_decoded_postings += query_postings[i]->CountDecodedPostings();
  }

  if (std::strcmp(retrieval_method, "tp")) {
    std::cout << "Time cost: "
              << duration_cast<milliseconds>(t1 - t0).count()
              << " ms. " << std::endl;
    std::cout << "Loaded postings blocks: " << count_loaded_blocks
              << " of " << count_total_blocks << std::endl;
    std::cout << "Number of eval: " << count_eval_documents << std::endl;
    std::cout << "dpm and spm: " << count_dpm << " " << count_spm << " " << count_decoded_postings << std::endl;
  }
}
