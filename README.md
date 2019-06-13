# HashtagMaster: Segmentation tool for hashtags

This repository contains the code and resources from the following [paper](https://mounicam.github.io/HashtagMaster_ACL_camera_ready_v2.pdf)


## Repo Structure: 
1. ```word_breaker```: Code for word-breaker beam search.

1. ```neural_ranker```: Code for our neural pairwise ranker models. (4 variants) 

1. ```data```: Task datasets and other feature files. All the features files for the experiment are added except the language models. We provided a small sample of the language models. Please email us for the whole language model. 

## Instructions: 
1. First, run the "Word Breaker" to get the top-k candidates: 

    ```python word_breaker/main.py --k 10 --lm data/small_gt.bin --out train_topk.tsv --input data/our_dataset/train_corrected.tsv```
    
   ```python word_breaker/main.py --k 10 --lm data/small_gt.bin --out test_topk.tsv --input data/our_dataset/test_corrected.tsv```

1. Rerank the top-k candidates: 

   ```python3 neural_ranker/main.py --train data/our_dataset/train_corrected.tsv --train_topk train_topk.tsv  --test data/our_dataset/test_corrected.tsv --test_topk test_topk.tsv --out output.tsv```


## Citation
Please cite if you use the above resources for your research
```
@InProceedings{ACL-2019-Maddela,
  author = 	"Maddela, Mounica and Xu, Wei and Preoţiuc-Pietro, Daniel",
  title = 	"Multi-task Pairwise Neural Ranking for Hashtag Segmentation",
  booktitle = 	"Proceedings of the Association for Computational Linguistics (ACL)",
  year = 	"2019",
}
```

Please use Python 3 to run the code.
