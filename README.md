# SIGIR2021-ShortPaper-WindowNavigation

Demonstrate the effectiveness of Window Navigation with Adaptive Probing when executing Block-Max WAND (BMW) or VBMW.

Paper for reference: https://dl.acm.org/doi/10.1145/3404835.3463109

Index:
```python BuildIndexPy/buildIndex.py docpath_prefix output_prefix score[BM25, DeepImpact]```

Search:
```
cd QueryIndex
make
./query_binary_index norm_doc_len_path binary_index_file vocabulary_file query_file retrieval_method constant_block_size top_k score[BM25, DeepImpact]
```
