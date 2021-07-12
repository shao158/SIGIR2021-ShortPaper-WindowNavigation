#include <cassert>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>

#include "indri/QueryEnvironment.hpp"
#include "indri/Repository.hpp"
#include "SIMD-BP/compositecodec.h"
#include "SIMD-BP/variablebyte.h"
#include "SIMD-BP/simdbinarypacking.h"

using namespace FastPForLib;

// Delta-encode every other value starting at index 0 in data_buffer.
void deltaEncoding(uint32_t* data_buffer, int len) {
  assert(len % 2 == 0);
  for (int i = len - 2; i >= 2;  i -= 2)
    data_buffer[i] = data_buffer[i] - data_buffer[i - 2];
}

bool validKeyword(std::string s) {
  for (size_t i = 0; i < s.size(); i++) if (!isalnum(s[i])) return false;
  // Filter out more bad terms.
  return !(isdigit(s[0]) && (!isdigit(s[s.size()-1]) || s.size() > 4));
}

/*
 * Build binary index containing posting lists for all terms.
 *
 * ./buildBinaryIndex indri_index_path binary_index_path > vocabulary_info_path
 */
int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Please list one indri index path and one binary index path. "
              << std::endl;
    return 0;
  }

  if (!indri::collection::Repository::exists(argv[1])) {
    std::cerr << "Invalid indri index path. "
              << std::endl;
    return 0;
  }

  IntegerCODEC* codec = new CompositeCodec<SIMDBinaryPacking, VariableByte>();
  
  indri::collection::Repository* repository =
      new indri::collection::Repository();

  repository->openRead(argv[1], nullptr);

  if (repository->indexes().get()) {
    std::cerr << "Found " << repository->indexes()->size()
              << " indexes in "
              << argv[1] << std::endl;

    indri::index::Index* cur_index = repository->indexes()->front();

    std::cerr << "There are " << cur_index->uniqueTermCount()
              << " unique terms." << std::endl;

    std::ofstream index_file(argv[2], std::ios::out | std::ios::binary);
    uint32_t *data_buffer = new uint32_t[2 * 1000 * 1000];
    uint32_t *encoded_data = new uint32_t[2 * 1000 * 1000 + 1024]; // N + 1024

    indri::index::VocabularyIterator* termIter =
        cur_index->vocabularyIterator();
    termIter->startIteration();
    while (!termIter->finished()) {
      // termID starts with 1.
      lemur::api::TERMID_T cur_term_id = termIter->currentEntry()->termID;

      if (!validKeyword(termIter->currentEntry()->termData->term)) {
        std::cerr << "Skip(0): "
                  << termIter->currentEntry()->termData->term << std::endl;
        termIter->nextEntry();
        continue;
      }

      if (termIter->currentEntry()->termData->corpus.documentCount < 100) {
        std::cerr << "Skip(1): "
                  << termIter->currentEntry()->termData->term << std::endl;
        termIter->nextEntry();
        continue;
      }

      std::cout << termIter->currentEntry()->termData->term
                << " " << cur_term_id << " "
                << termIter->currentEntry()->termData->corpus.documentCount;

      {
        int count_buffer = 0;
        indri::index::DocListIterator* docIter =
            cur_index->docListIterator(cur_term_id);
        docIter->startIteration();

        data_buffer[0] = docIter->currentEntry()->document;
        data_buffer[1] = docIter->currentEntry()->positions.size();
        count_buffer += 2; 
      
        docIter->nextEntry();
        while (!docIter->finished()) {
          data_buffer[count_buffer] = docIter->currentEntry()->document;
          data_buffer[count_buffer + 1] =
              docIter->currentEntry()->positions.size();

          count_buffer += 2; 

          if (count_buffer == 2 * 1000 * 1000) {
            deltaEncoding(data_buffer, count_buffer);
            size_t encoded_len = 2LL * 1000LL * 1000LL + 1024;
            codec->encodeArray(
                data_buffer, count_buffer, encoded_data, encoded_len);
            int64_t start_pos = index_file.tellp();
            index_file.write((char*)encoded_data, encoded_len * 4);

            std::cout << " " << start_pos << " " << encoded_len * 4;

            count_buffer = 0;
          }
          docIter->nextEntry();
        }

        if (count_buffer > 0) {
          deltaEncoding(data_buffer, count_buffer);
          size_t encoded_len = 2LL * 1000LL * 1000LL + 1024;
          codec->encodeArray(
              data_buffer, count_buffer, encoded_data, encoded_len);
          int64_t start_pos = index_file.tellp();
          index_file.write((char*)encoded_data, encoded_len * 4);

          std::cout << " " << start_pos << " " << encoded_len * 4;
        }
      
        delete docIter;
      }
      
      std::cout << std::endl;

      termIter->nextEntry();
    }
    index_file.close();

    delete termIter;
    delete [] data_buffer;
    delete [] encoded_data;
  } else {
    std::cerr << "Cannot find indexes in " << argv[1] << std::endl;
  }

  delete repository;
  delete codec;
  return 0;
}
