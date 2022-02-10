import json
import sys
import gzip
from wsgiref.validate import validator
import numpy as np
import math
from nltk.tokenize import word_tokenize
from numpy.core.defchararray import encode
from pyfastpfor import getCodec
from tqdm import tqdm

from transformers import BertTokenizer

def deltaEncoding(arr):
    assert(len(arr) % 2 == 0)
    for i in range(len(arr) - 2, 0, -2):
        arr[i] = arr[i] - arr[i - 2]

if __name__ == "__main__":
    # doc_len = []

    # f = open('/share/msmarco/msmarco_passage_v1.normlen')
    # for line in f:
    #     doc_len.append(float(line.strip()))
    # TOT = len(doc_len)

    json_path = sys.argv[1]

    posting = {}

    length = []

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    for i in range(100):
        print(i)
        for line in gzip.open("%s%02d.jsonl.gz" % (json_path, i)):
            doc_dict = json.loads(line)
            id = doc_dict['id']
            # contents = doc_dict['contents'].lower()
            # contents = tokenizer.convert_ids_to_tokens(tokenizer(contents)['input_ids'])
            # length.append(len(contents))

            # count_dict = {}
            # for w in contents:
            #     if w not in count_dict:
            #         count_dict[w] = 0
            #     count_dict[w] += 1

            vector = doc_dict['vector']

            for k in vector:
                if k == '[CLS]' or k == '[SEP]':
                    continue
                # if k not in count_dict:
                #     continue
                if k not in posting:
                    posting[k] = []

                posting[k] += [id, vector[k]]

    # fout_length = open(sys.argv[2] + ".length", 'w')
    # avglength = np.sum(length) / len(length)
    # for l in length:
    #     fout_length.write(str(l / avglength) + '\n')

    term_id = {}
    id = 0
    for k in posting:
        term_id[k] = id
        id += 1

    with open(sys.argv[2] + '.id', 'w') as f:
        json.dump(term_id, f)
        
    STEP = 1000 * 1000
    codec = getCodec('simdbinarypacking')

    fout_data = open(sys.argv[2] + ".data", 'wb')
    fout_raw = open(sys.argv[2] + ".raw", 'w')
    fout_info = open(sys.argv[2] + ".info", 'w')


    for k in tqdm(posting):
        fout_raw.write(k + ' ' + str(term_id[k]) + ' ' + ' '.join([str(i) for i in posting[k]]) + '\n')
        fout_info.write(k + ' ' + str(term_id[k]) + ' ' + str(len(posting[k]) // 2))

        for i in range(0, len(posting[k]), 2 * STEP):
            p_bytes = np.array(posting[k][i : i + 2 * STEP], dtype=np.uint32, order='C')
            deltaEncoding(p_bytes)
            size = len(p_bytes)
            encoded = np.zeros(size + 1024, dtype = np.uint32, order='C')
            encoded_size = codec.encodeArray(p_bytes, size, encoded, size + 1024)

            start_pos = fout_data.tell()
            encoded_bytes = bytes(encoded[:encoded_size])
            fout_data.write(encoded_bytes)

            fout_info.write(' ' + str(start_pos) + ' ' + str(encoded_size * 4))
            
        fout_info.write('\n')
            