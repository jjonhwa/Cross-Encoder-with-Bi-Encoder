# Cross-Encoder-with-Bi-Encoder
For Tok-k passages that have passed through the Bi-Encoder Retrival, ReRank is performed using CrossEncoder.

## What can we do to improve the performance of Retriever?
### 1. Explore the data set production process.
- **Sparse Embedding may be better in tasks for viewing Passage and creating a question (if there is an annotation bias), such as SQuAD.**
    - A briefly summarized it in Korean. -> [Dense Passage Retrieval for Open Domain Question Answering Review](https://github.com/jjonhwa/Paper_Review/blob/main/Dense%20Passage%20Retrieval%20for%20Open-Domain%20Question%20Answering.pdf) (3. Passage Retrieval - Main Results)
- **In most other data, documents can be extracted with higher accuracy if Dense Passage Retreat is used.**

### 2. Sparse Embedding & Dense Embedding
- Most of the content was knowledge obtained by referring to [Paper](https://arxiv.org/abs/2004.04906), and based on this, it led to improvement in Retriever performance.
- Prior to the application of DPR, in the case of **'KLUE MRC database' in which datasets were configured in the same manner as SQuAD, it would be better to utilize techniques such as Sparse embedding technique BM25 compared to DPR.**
- Actually, until ReRank Strategy was applied, **the highest performance was achieved with elastic search based on BM25.**
- **When only biencoder was used, Retrieval accuracy was far below elastic search** in the 'KLUE MRC competition'
- Retrieval Accuracy in our Data

||Top-5|Top-50|Top-100|
|---|---|---|---|
|Elastic Search|0.852|0.945|0.962|
|DPR Bi-Encoder|-|0.775|0.85|

### 3. **ReRank Strategy with CrossEncoder**
- Our purpose is to bring high performance from KLUE MRC competition to End-to-End from Retrieval to Reader. From this, **the ReRank strategy using Cross Encoder was used.**
- After extracting the Retrival Passage of the Top-500 using the Bi-Encoder, **only a small number of Passages are finally extracted by returning to the Cross Encoder.**
- Retrieval Accuracy in our Data

||Top-5|Top-50|Top-100|
|---|---|---|---|
|Elastic Search|0.852|0.945|0.962|
|DPR without CrossEncoder|-|0.775|0.85|
|DPR with CrossEncoder|0.825|0.95|-|

### 4. Ensemble
- In this process, the contents of CrossEncoder were mainly written, and the contents of Ensemble were omitted.
- **An experiment was conducted assuming that performance improvement would be achieved from different types of Retrival combinations by conducting Ensemble using Sparse Embedding and Dense Embedding.**
- Top-100 was selected using Elastic Search and Top-100 was selected using DPR and Cross Encoder, and the final output score was calculated by combining them 1 to 1 and normalizing them.
- When the final Reader model was tested, when Top-5 was input, the performance was the best, so the experiment was conducted after limiting the number of passages to be returned to five.
- **Actually, the performance has improved significantly, and the retrival accuracy is as follows.**

||Top-5|Top-50|Top-100|
|---|---|---|---|
|Elastic Search|0.852|0.945|0.962|
|DPR with CrossEncoder|0.825|0.95|-|
|Ensemble|0.9082|-|-|

## Train CrossEncoder & BiEncoder
- Learn crossencoder and biencoder and store them.
- Modify only the data path to match your data. (find "your_dataset_path")
```
# If you want to train cross encoder, you can input the configuration for the encoder
python train.py --encoder 'cross' --output_directory './save_directory/'

or 

python train.py --encoder 'bi' --output_directory './save_directory/'
```

## Run ReRank
- It precedes creating an encoder using crossencoder and biencoder.
- Modify only the data path to match your data. (find "your_dataset_path")
```
python rerank.py --input_directory './save_directory/'
```

## Run Retriever Demo
- Top 500 Passages are Retrieved from about 60000 data using Biencoder, and Top 5 is finally retrieved using CrossEncoder.
- Passage Embedding about wiki data, Cross Encoder and Bi-Encoder can be downloaded and utilized, but the data used for learning and raw wiki data cannot be disclosed, so you can understand the process through Demo and modify it to suit your data.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qkVMPM8Hw8n4gGs2_-Wacp8oKMVvAokS?usp=sharing)

