#include "PostingList.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

using namespace FastPForLib;

static std::vector<float> InitializeClueWebNormDocLen() {
  std::vector<float> res(33836981, 0);
  std::string tmp_file_path = 
      "/home/shao158/Documents/datasets/clueweb_33M_doc_normalized_len.txt";
  std::ifstream doc_len_file(tmp_file_path.c_str(), std::ios::in);
  assert(doc_len_file.is_open());

  std::string line;
  int i = 0;
  while (getline(doc_len_file, line)) res[i++] = std::stof(line);
  std::cerr << "Load doc len: " << i << std::endl;
  doc_len_file.close();

  return res;
}

static bool sortByTF(float a, float b) {
  return (a > b);
}

const std::vector<float> PostingList::norm_doc_len
  = InitializeClueWebNormDocLen();

PostingList::PostingList(const char* index_file_path,
                         const std::vector<std::string>& vocabulary_info,
                         const std::vector<std::string>& block_size_info,
                         size_t constant_block_size,
                         size_t num_total_doc)
    : binary_index_path (index_file_path),
      BLOCK_SIZE (constant_block_size),
      NUM_TOTAL_DOC (num_total_doc) {
  assert(vocabulary_info.size() % 2 == 1 && vocabulary_info.size() >= 5);
  term_str = vocabulary_info[0];
  document_frequency = std::stoll(vocabulary_info[2].c_str());
  for (size_t i = 3; i < vocabulary_info.size(); i+=2) {
    binary_index_offset.push_back(std::stoll(vocabulary_info[i]));
    binary_index_length.push_back(std::stoll(vocabulary_info[i + 1]));
  }

  variable_size_blocks.clear();
  if (!block_size_info.empty()) {
    variable_size_blocks.resize(block_size_info.size() - 1);
    for (size_t i = 1; i < block_size_info.size(); i++) {
      variable_size_blocks[i -1] = std::stoll(block_size_info[i]);
    }
  }

  codec = new CompositeCodec<SIMDBinaryPacking, VariableByte>();

  metadata_list = nullptr;

  encoded_posting_blocks = nullptr;
  decoded_posting_blocks = nullptr;

  decoded_block_sizes = nullptr;
  encoded_block_sizes = nullptr;
}

PostingList::~PostingList() {
  if (metadata_list != nullptr) {
    delete [] metadata_list;
  }

  if (encoded_posting_blocks != nullptr) {
    for (size_t i = 0; i < count_posting_blocks; i++)
      delete [] encoded_posting_blocks[i];
    delete [] encoded_posting_blocks;
  }

  if (encoded_block_sizes != nullptr) {
    delete [] encoded_block_sizes;
  }

  if (decoded_posting_blocks != nullptr) {
    for (size_t i = 0; i < count_posting_blocks; i++)
      delete [] decoded_posting_blocks[i];
    delete [] decoded_posting_blocks;
  }

  if (decoded_block_sizes != nullptr) {
    delete [] decoded_block_sizes;
  }

  delete codec;
}

void PostingList::StartIteration() {
  assert(binary_index_offset.size() == binary_index_length.size()
      && !binary_index_offset.empty());

  std::ifstream index_file(binary_index_path, std::ios::in | std::ios::binary);
  assert(index_file.is_open());

  size_t ind = 0;
  uint32_t *compressed_data = new uint32_t[2 * 1000 * 1000 + 1024];
  uint32_t *uncompressed_data = new uint32_t[2 * 1000 * 1000];
  uint32_t *tmp_posting_holder = new uint32_t[document_frequency * 2];

  // std::cout << "Posting: ";
  for (size_t i = 0; i < binary_index_offset.size(); i++) {
    index_file.seekg(binary_index_offset[i]);
    index_file.read((char *)compressed_data, binary_index_length[i]);

    size_t n_uncompressed_data = 2 * 1000 * 1000;
    codec->decodeArray(compressed_data, binary_index_length[i] / 4,
                       uncompressed_data, n_uncompressed_data);

    for (size_t j = 0; j < n_uncompressed_data; j++) {
      tmp_posting_holder[ind + j] = uncompressed_data[j];
      if (j % 2 == 0 && j > 0) {
        tmp_posting_holder[ind + j] += tmp_posting_holder[ind + j - 2];
      }
      // std::cout << tmp_posting_holder[ind + j] << " ";
    }

    ind += n_uncompressed_data;
  }

  // std::cout << std::endl;

  assert(ind == document_frequency * 2);
  if (variable_size_blocks.empty()) {
    PackIntoConstantSizedBlocks(tmp_posting_holder);
  } else {
    PackIntoVariableSizedBlocks(tmp_posting_holder);
  }

  delete [] compressed_data;
  delete [] uncompressed_data;
  delete [] tmp_posting_holder;

  index_file.close();
  count_loaded_blocks = 0;
  count_decoded_postings = 0;
  count_eval_documents = 0;
  count_dpm = 0;
  count_spm = 0;
  curr_block_loaded = false;
  loaded_block_ids.clear();
  assert(LoadPostingBlock(0));
}

void PostingList::Restart() {
  count_loaded_blocks = 0;
  count_decoded_postings = 0;
  count_eval_documents = 0;
  count_dpm = 0;
  count_spm = 0;
  curr_block_loaded = false;
  loaded_block_ids.clear();

  curr_block_ind = 0;
  assert(LoadPostingBlock(curr_block_ind));
}

void PostingList::RestartForMarkedBMW() {
  count_loaded_blocks = 0;
  count_decoded_postings = 0;
  count_eval_documents = 0;
  count_dpm = 0;
  count_spm = 0;
  curr_block_loaded = false;
  loaded_block_ids.clear();

  curr_block_ind = 0;
  while (curr_block_ind < count_posting_blocks
      && !metadata_list[curr_block_ind].marked_block) {
    curr_block_ind += 1;
  }
  if (curr_block_ind < count_posting_blocks) {
    assert(LoadPostingBlock(curr_block_ind));
  } else {
    curr_document_id = NUM_TOTAL_DOC + 1;
    curr_block_loaded = true;
  }
}

bool PostingList::Next(uint32_t target_doc_id) {
  if (target_doc_id <= curr_document_id) return true;

  count_dpm += 1;

  if (!curr_block_loaded) {                                                   
    assert(LoadPostingBlock(curr_block_ind));
  }

  size_t target_block_ind = curr_block_ind;
  while (target_block_ind < count_posting_blocks
      && metadata_list[target_block_ind].last_doc_id < target_doc_id) {
    target_block_ind += 1;
  }

  if (target_block_ind == count_posting_blocks) {
    curr_block_ind = count_posting_blocks;
    curr_document_id = NUM_TOTAL_DOC + 1;
    curr_block_loaded = true;
    return false;
  }

  if (target_block_ind != curr_block_ind) {
    LoadPostingBlock(target_block_ind);
  }

  while (curr_doc_ind < curr_block_size
      && decoded_posting_blocks[curr_block_ind][curr_doc_ind * 2]
          < target_doc_id) {
    curr_doc_ind += 1;
  }
  
  // assert(curr_doc_ind < curr_block_size);
  curr_document_id =
      decoded_posting_blocks[curr_block_ind][curr_doc_ind * 2];
  curr_document_tf =
      decoded_posting_blocks[curr_block_ind][curr_doc_ind * 2 + 1];
  return true;
}

bool PostingList::NextShallow(uint32_t target_doc_id) {
  if (target_doc_id <= curr_document_id) return true;

  size_t target_block_ind = curr_block_ind;
  while (target_block_ind < count_posting_blocks
      && metadata_list[target_block_ind].last_doc_id < target_doc_id) {
    target_block_ind += 1;
  }

  if (target_block_ind == count_posting_blocks) {
    curr_block_ind = count_posting_blocks;
    curr_document_id = NUM_TOTAL_DOC + 1;
    curr_block_loaded = true;
    return false;
  }

  if (target_block_ind == curr_block_ind) {
    return true;
  }

  count_spm += 1;

  curr_block_ind = target_block_ind;
  curr_block_size = 0;
  curr_block_max = metadata_list[curr_block_ind].block_max_score;

  curr_block_last_document_id = metadata_list[curr_block_ind].last_doc_id;

  curr_doc_ind = 0;
  curr_document_id = 0;
  curr_document_tf = 0;

  curr_block_loaded = false;

  return true;
}

bool PostingList::NextFancy(uint32_t target_doc_id) {
  if (target_doc_id <= curr_document_id) return true;

  count_dpm += 1;

  if (!curr_block_loaded) {                                                   
    assert(LoadPostingBlock(curr_block_ind));                                 
  }

  size_t target_block_ind = curr_block_ind;
  while (target_block_ind < count_posting_blocks
      && (metadata_list[target_block_ind].block_max_score + 1e-6
              < fancy_list_threshold
          || metadata_list[target_block_ind].last_doc_id < target_doc_id)) {
    target_block_ind += 1;
  }

  if (target_block_ind == count_posting_blocks) {
    curr_block_ind = count_posting_blocks;
    curr_document_id = NUM_TOTAL_DOC + 1;
    curr_block_loaded = true;
    return false;
  }

  if (target_block_ind != curr_block_ind) {
    LoadPostingBlock(target_block_ind);
  }

  while (curr_doc_ind < curr_block_size
      && (decoded_posting_blocks[curr_block_ind][curr_doc_ind * 2 + 1]
              < fancy_list_threshold
          || decoded_posting_blocks[curr_block_ind][curr_doc_ind * 2]
              < target_doc_id)) {
    curr_doc_ind += 1;
  }
  
  if (curr_doc_ind < curr_block_size) {
    curr_document_id =
        decoded_posting_blocks[curr_block_ind][curr_doc_ind * 2];
    curr_document_tf =
        decoded_posting_blocks[curr_block_ind][curr_doc_ind * 2 + 1];
  } else {
    assert(curr_doc_ind == curr_block_size);
    NextFancy(
        decoded_posting_blocks[curr_block_ind][curr_doc_ind * 2 - 2] + 1);
  }

  return true;
}

bool PostingList::NextShallowFancy(uint32_t target_doc_id) {
  if (target_doc_id <= curr_document_id) return true;

  size_t target_block_ind = curr_block_ind;
  while (target_block_ind < count_posting_blocks
      && (metadata_list[target_block_ind].block_max_score + 1e-6
              < fancy_list_threshold
          || metadata_list[target_block_ind].last_doc_id < target_doc_id)) {
    target_block_ind += 1;
  }

  if (target_block_ind == count_posting_blocks) {
    curr_block_ind = count_posting_blocks;
    curr_document_id = NUM_TOTAL_DOC + 1;
    curr_block_loaded = true;
    return false;
  }

  if (target_block_ind == curr_block_ind) {
    return true;
  }

  count_spm += 1;

  curr_block_ind = target_block_ind;
  curr_block_size = 0;
  curr_block_max = metadata_list[curr_block_ind].block_max_score;

  curr_block_last_document_id = metadata_list[curr_block_ind].last_doc_id;

  curr_doc_ind = 0;
  curr_document_id = 0;
  curr_document_tf = 0;

  curr_block_loaded = false;

  return true;
}

bool PostingList::NextMarked(uint32_t target_doc_id) {
  if (target_doc_id <= curr_document_id) return true;

  count_dpm += 1;

  if (!curr_block_loaded) {                                                   
    LoadPostingBlock(curr_block_ind);
  }

  size_t target_block_ind = curr_block_ind;
  while (target_block_ind < count_posting_blocks
      && (!metadata_list[target_block_ind].marked_block
          || metadata_list[target_block_ind].last_doc_id < target_doc_id)) {
    target_block_ind += 1;
  }

  if (target_block_ind == count_posting_blocks) {
    curr_block_ind = count_posting_blocks;
    curr_document_id = NUM_TOTAL_DOC + 1;
    curr_block_loaded = true;
    return false;
  }

  if (target_block_ind != curr_block_ind) {
    LoadPostingBlock(target_block_ind);
  }

  while (curr_doc_ind < curr_block_size
      && decoded_posting_blocks[curr_block_ind][curr_doc_ind * 2]
          < target_doc_id) {
    curr_doc_ind += 1;
  }
  
  // assert(curr_doc_ind < curr_block_size);
  curr_document_id =
      decoded_posting_blocks[curr_block_ind][curr_doc_ind * 2];
  curr_document_tf =
      decoded_posting_blocks[curr_block_ind][curr_doc_ind * 2 + 1];
  return true;
}

bool PostingList::NextShallowMarked(uint32_t target_doc_id) {
  if (target_doc_id <= curr_document_id) return true;

  size_t target_block_ind = curr_block_ind;
  while (target_block_ind < count_posting_blocks
      && (!metadata_list[target_block_ind].marked_block
          || metadata_list[target_block_ind].last_doc_id < target_doc_id)) {
    target_block_ind += 1;
  }

  if (target_block_ind == count_posting_blocks) {
    curr_block_ind = count_posting_blocks;
    curr_document_id = NUM_TOTAL_DOC + 1;
    curr_block_loaded = true;
    return false;
  }

  if (target_block_ind == curr_block_ind) {
    return true;
  }

  count_spm += 1;

  curr_block_ind = target_block_ind;
  curr_block_size = 0;
  curr_block_max = metadata_list[curr_block_ind].block_max_score;

  curr_block_last_document_id = metadata_list[curr_block_ind].last_doc_id;

  curr_doc_ind = 0;
  curr_document_id = 0;
  curr_document_tf = 0;

  curr_block_loaded = false;

  return true;
}

void PostingList::DeriveFancyList(uint32_t *data_holder) {
  int max_num_fancy = fancy_list_percentage * document_frequency;
  max_num_fancy = std::max(max_num_fancy, 1);
  std::vector<float> fancy_list_tf;
  std::make_heap(fancy_list_tf.begin(), fancy_list_tf.end(), sortByTF); 

  for (size_t i = 0; i < document_frequency; i++) {
    float curr_bm25_score = CalculateBM25(data_holder[i * 2 + 1],
                                          data_holder[i * 2]);
    if (fancy_list_tf.size() < max_num_fancy) {
      fancy_list_tf.push_back(curr_bm25_score);
      std::push_heap(fancy_list_tf.begin(), fancy_list_tf.end(), sortByTF);
    } else if (curr_bm25_score > fancy_list_tf.front()) {
      std::pop_heap(fancy_list_tf.begin(), fancy_list_tf.end(), sortByTF);
      fancy_list_tf.pop_back();
      fancy_list_tf.push_back(curr_bm25_score);
      std::push_heap(fancy_list_tf.begin(), fancy_list_tf.end(), sortByTF);
    }
  }

  std::sort_heap(fancy_list_tf.begin(), fancy_list_tf.end(), sortByTF);

  fancy_list_threshold = fancy_list_tf.back();
}

void PostingList::PackIntoConstantSizedBlocks(uint32_t *data_holder) {
  if (metadata_list != nullptr) {
    delete [] metadata_list;
    metadata_list = nullptr;
  }

  if (encoded_posting_blocks != nullptr) {
    for (size_t i = 0; i < count_posting_blocks; i++)
      delete [] encoded_posting_blocks[i];
    delete [] encoded_posting_blocks;

    encoded_posting_blocks = nullptr;
  }

  if (encoded_block_sizes != nullptr) {
    delete [] encoded_block_sizes;
    encoded_block_sizes = nullptr;
  }

  if (decoded_posting_blocks != nullptr) {
    for (size_t i = 0; i < count_posting_blocks; i++)
      delete [] decoded_posting_blocks[i];
    delete [] decoded_posting_blocks;
  
    decoded_posting_blocks = nullptr;
  }

  if (decoded_block_sizes != nullptr) {
    delete [] decoded_block_sizes;
    decoded_block_sizes = nullptr;
  }

  count_posting_blocks = document_frequency / BLOCK_SIZE;
  if (document_frequency % BLOCK_SIZE != 0) count_posting_blocks += 1;
  encoded_posting_blocks = new uint32_t*[count_posting_blocks];
  encoded_block_sizes = new size_t[count_posting_blocks];
  decoded_posting_blocks = new uint32_t*[count_posting_blocks];
  decoded_block_sizes = new size_t[count_posting_blocks];
  for (size_t i = 0; i < count_posting_blocks; i++) {
    decoded_posting_blocks[i] = nullptr;
    decoded_block_sizes[i] = 0;
  }
  metadata_list = new Metadata_t[count_posting_blocks];

  float global_max_tf = 0;

  uint32_t *data_buffer = new uint32_t[BLOCK_SIZE * 2];
  uint32_t *encoded_data = new uint32_t[BLOCK_SIZE * 2 + 1024]; // keep 1024.
  for (size_t i = 0, block_id = 0;
      i < document_frequency;
      i += BLOCK_SIZE, block_id += 1) {
    float curr_max_tf = 0;
    size_t j = 0;
    for (; j < BLOCK_SIZE && j + i < document_frequency; j++) {
      data_buffer[j * 2 + 0] = data_holder[(i + j) * 2 + 0];
      if (j > 0) data_buffer[j * 2 + 0] -= data_holder[(i + j - 1) * 2 + 0];
      data_buffer[j * 2 + 1] = data_holder[(i + j) * 2 + 1];

      curr_max_tf = std::max(float(curr_max_tf),
          CalculateBM25(data_holder[(i + j) * 2 + 1],
                        data_holder[(i + j) * 2]));
    }

    size_t encoded_len = 2LL * 1000LL * 1000LL + 1024;
    codec->encodeArray(data_buffer, j * 2, encoded_data, encoded_len);
    decoded_block_sizes[block_id] = j;
    encoded_block_sizes[block_id] = encoded_len;
    encoded_posting_blocks[block_id] = new uint32_t[encoded_len];
    std::memmove(encoded_posting_blocks[block_id],
                 encoded_data, encoded_len * sizeof(uint32_t));

    /*
    uint64_t encoded_len = simple9_encode(data_buffer, j * 2, encoded_data);
    encoded_posting_blocks[block_id] = new uint8_t[encoded_len];
    std::memmove(encoded_posting_blocks[block_id], encoded_data, encoded_len);
    */

    global_max_tf = std::max(global_max_tf, curr_max_tf);

    metadata_list[block_id].first_doc_id = data_holder[i * 2];
    metadata_list[block_id].last_doc_id = data_holder[(i + j - 1) * 2];
    metadata_list[block_id].block_max_score = curr_max_tf;
    metadata_list[block_id].marked_block = false;
  }

  global_max_score = global_max_tf;

  delete [] data_buffer;
  delete [] encoded_data;
}

void PostingList::PackIntoVariableSizedBlocks(uint32_t *data_holder) {
  if (metadata_list != nullptr) {
    delete [] metadata_list;
    metadata_list = nullptr;
  }

  if (encoded_posting_blocks != nullptr) {
    for (size_t i = 0; i < count_posting_blocks; i++)
      delete [] encoded_posting_blocks[i];
    delete [] encoded_posting_blocks;

    encoded_posting_blocks = nullptr;
  }

  if (encoded_block_sizes != nullptr) {
    delete [] encoded_block_sizes;
    encoded_block_sizes = nullptr;
  }

  if (decoded_posting_blocks != nullptr) {
    for (size_t i = 0; i < count_posting_blocks; i++)
      delete [] decoded_posting_blocks[i];
    delete [] decoded_posting_blocks;
  
    decoded_posting_blocks = nullptr;
  }

  if (decoded_block_sizes != nullptr) {
    delete [] decoded_block_sizes;
    decoded_block_sizes = nullptr;
  }

  count_posting_blocks = variable_size_blocks.size();

  encoded_posting_blocks = new uint32_t*[count_posting_blocks];
  encoded_block_sizes = new size_t[count_posting_blocks];
  decoded_posting_blocks = new uint32_t*[count_posting_blocks];
  decoded_block_sizes = new size_t[count_posting_blocks];
  for (size_t i = 0; i < count_posting_blocks; i++) {
    decoded_posting_blocks[i] = nullptr;
    decoded_block_sizes[i] = 0;
  }
  metadata_list = new Metadata_t[count_posting_blocks];

  float global_max_tf = 0;

  size_t i = 0;
  uint32_t *data_buffer = nullptr;
  uint32_t *encoded_data = nullptr;
  for (size_t block_id = 0; block_id < count_posting_blocks; block_id++) {
    data_buffer = new uint32_t[variable_size_blocks[block_id] * 2];
    // Keep 1024 as suggested.
    if (variable_size_blocks[block_id] > 128) {
      encoded_data = new uint32_t[variable_size_blocks[block_id] * 2 + 1024];
    } else {
      encoded_data = new uint32_t[1024 + 1024];
    }

    float curr_max_tf = 0;
    size_t j = 0;
    for (; j < variable_size_blocks[block_id]; j++) {
      data_buffer[j * 2 + 0] = data_holder[(i + j) * 2 + 0];
      if (j > 0) data_buffer[j * 2 + 0] -= data_holder[(i + j - 1) * 2 + 0];
      data_buffer[j * 2 + 1] = data_holder[(i + j) * 2 + 1];

      curr_max_tf = std::max(float(curr_max_tf),
          CalculateBM25(data_holder[(i + j) * 2 + 1],
                        data_holder[(i + j) * 2]));
    }

    size_t encoded_len = 2LL * 1000LL * 1000LL + 1024;
    codec->encodeArray(data_buffer, j * 2, encoded_data, encoded_len);
    decoded_block_sizes[block_id] = j;
    encoded_block_sizes[block_id] = encoded_len;
    encoded_posting_blocks[block_id] = new uint32_t[encoded_len];
    std::memmove(encoded_posting_blocks[block_id],
                 encoded_data, encoded_len * sizeof(uint32_t));

    /*
    uint64_t encoded_len = simple9_encode(data_buffer, j * 2, encoded_data);
    assert(encoded_len <= variable_size_blocks[block_id] * 2 * 4
        || encoded_len <= 1024);
    encoded_posting_blocks[block_id] = new uint8_t[encoded_len];
    std::memmove(encoded_posting_blocks[block_id], encoded_data, encoded_len);
    */

    global_max_tf = std::max(global_max_tf, curr_max_tf);

    metadata_list[block_id].first_doc_id = data_holder[i * 2];
    metadata_list[block_id].last_doc_id = data_holder[(i + j - 1) * 2];
    metadata_list[block_id].block_max_score = curr_max_tf;
    metadata_list[block_id].marked_block = false;

    i += variable_size_blocks[block_id];

    delete [] data_buffer;
    delete [] encoded_data;
  }
  global_max_score = global_max_tf;
}

bool PostingList::LoadPostingBlock(size_t block_id) {
  // assert(block_id >= 0 && block_id < count_posting_blocks);

  // std::cout << "Posting block: ";

  if (decoded_posting_blocks[block_id] == nullptr) {
    decoded_posting_blocks[block_id] =
        new uint32_t[decoded_block_sizes[block_id] * 2];
    size_t n_uncompressed_data = 2 * 1000 * 1000;
    codec->decodeArray(encoded_posting_blocks[block_id],
                       encoded_block_sizes[block_id],
                       decoded_posting_blocks[block_id], n_uncompressed_data);

    // std::cout << decoded_block_sizes[block_id] << " " << n_uncompressed_data << std::endl;

    for (size_t j = 2; j < decoded_block_sizes[block_id] * 2; j += 2) {
      decoded_posting_blocks[block_id][j] +=
          decoded_posting_blocks[block_id][j - 2];
    }

    /*
    for (size_t j = 0; j < decoded_block_sizes[block_id] * 2; j++) {
      std::cout << decoded_posting_blocks[block_id][j] << " ";
    }
    */
    // decoded_block_sizes[block_id] = n_uncompressed_data / 2;

    // assert(loaded_block_ids.find(block_id) == loaded_block_ids.end());
    loaded_block_ids.insert(block_id);
    count_loaded_blocks += 1;
    count_decoded_postings += decoded_block_sizes[block_id]; 
  }

  // std::cout << std::endl;

  curr_block_ind = block_id;
  curr_block_size = decoded_block_sizes[block_id];
  curr_block_max = metadata_list[curr_block_ind].block_max_score;
  curr_block_last_document_id = metadata_list[curr_block_ind].last_doc_id;

  curr_doc_ind = 0;
  curr_document_id = decoded_posting_blocks[block_id][0];
  curr_document_tf = decoded_posting_blocks[block_id][1];

  curr_block_loaded = true;

  return true;
}

float PostingList::CalculateBM25(uint32_t tf, uint32_t doc_id) const {
  static constexpr float k1 = 1.2; // 1.2
  static constexpr float b = 0.5;

  // assert(doc_id > 0 && doc_id < norm_doc_len.size() + 1);
  float ftf = tf * 1.0f;
  float fdf = document_frequency * 1.0f;
  float tf_weight = query_frequency * ftf * (k1 + 1.0f)
      / (ftf + k1 * (1.0f - b + b * norm_doc_len[doc_id - 1]));
  float idf_weight = std::max(1.0E-6,
      log((NUM_TOTAL_DOC * 1.0f - fdf + 0.5f) / (fdf + 0.5f)));
  return tf_weight * idf_weight;
}
