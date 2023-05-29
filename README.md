# A Causal Inspired Explanations for Understanding Black-box Models

This project is for the paper: A Causal Inspired Explanations for Understanding Black-box Models, Proceedings of the
29th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2023.

The code was developed on Python 3.8 and Pytorch 1.12.1

## Usage

### 1. run train_bert.py: training black-box model Bert
```
usage: python train_bert.py [--device GPU_ID] [--dataset DATASET_NAME]

arguments:
  --device GPU_ID
                        GPU ID, default is 0.
  --dataset DATASET_NAME
                        support: clickbait, hate, yelp, imdb.
```

### 2. run CIMI.py: training the interpreter in CIMI
```
usage: python CIMI.py [--device GPU_ID] [--dataset DATASET_NAME] [--batch_size BATCH_SIZE] --train_stack

arguments:
  --device GPU_ID
                        GPU ID, default is 0.
  --dataset DATASET_NAME
                        support: clickbait, hate, yelp, imdb.
  --batch_size BATCH_SIZE
                        batch size, default is 8.
```

### 3. run eval.py: evaluating CIMI's performance
```
usage: python eval.py [--device GPU_ID] [--dataset DATASET_NAME]

arguments:
  --device GPU_ID
                        GPU ID, default is 0.
  --dataset DATASET_NAME
                        support: clickbait, hate, yelp, imdb.
```
