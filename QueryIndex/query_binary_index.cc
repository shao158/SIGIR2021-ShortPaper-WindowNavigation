#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>

#include "BinaryIndex.h"

static constexpr size_t kTotalNumDocClueweb = 33836981;

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "Usage: "
              << "./query_binary_index "
              << "binary_index_file "
              << "vocabulary_file "
              << "query_file "
              << "retrieval_method "
              << "constant_block_size "
              << "block_size_file "
              << "top_k" << std::endl;
    return 0;
  }

  BinaryIndex *my_index = new BinaryIndex(
      /*index_file=*/argv[1],
      /*vocabulary_file=*/argv[2],
      /*dataset_size=*/kTotalNumDocClueweb,
      /*constant_block_size=*/std::stoll(argv[5]),
      /*block_size_file=*/argv[6]);
  if (my_index->GetVocabularySize() == 0) {
    std::cerr << "Failed to init a BinaryIndex. " << std::endl;
    delete my_index;
    return 0;
  }
  std::cerr << "Init a BinaryIndex with vocabulary size being "
            << my_index->GetVocabularySize()
            << std::endl;

  std::ifstream query_file(argv[3], std::ios::in);
  if (!query_file.is_open()) {
    std::cerr << "Failed to open the query file. " << std::endl;
    delete my_index;
    return 0;
  }

  std::string line;
  while (getline(query_file, line)) {
    std::istringstream iss(line);
    std::vector<std::string> query_keywords(
        std::istream_iterator<std::string>{iss},
        std::istream_iterator<std::string>());

    std::vector<std::string> dedup_query_keywords;
    std::vector<uint64_t> query_keywords_frequency;
    std::map<std::string, uint64_t> count_frequency;
    for (std::string query_keyword : query_keywords) {
      if (count_frequency.find(query_keyword) == count_frequency.end()) {
        count_frequency[query_keyword] = 1;
      } else {
        count_frequency[query_keyword] += 1;
      }
    }

    for (auto count_iter = count_frequency.begin();
        count_iter != count_frequency.end(); ++count_iter) {
      dedup_query_keywords.push_back(count_iter->first);
      query_keywords_frequency.push_back(count_iter->second);
    }
    
    my_index->Query(/*query_keywords=*/dedup_query_keywords,
                    /*query_keywords_frequency=*/query_keywords_frequency,
                    /*retrieval_method=*/argv[4],
                    /*top_k=*/std::stoi(argv[7]));
  }

  query_file.close();
  delete my_index;
  return 0;
}
