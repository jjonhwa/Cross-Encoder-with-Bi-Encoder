# Cross-Encoder-with-Bi-Encoder
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://github.com/jjonhwa/Retrieval_Streamlit_Demo)

For Tok-k passages that have passed through the Bi-Encoder Retrival, ReRank is performed using CrossEncoder.

## Data

**Data used by "Open-Domain Question Answering Competition" hosted by Aistages, and copyrights can be used under CC-BY-2.0.**

```
+- data
|   +- train_dataset
    |   +- train
        |   +- dataset.arrow
        |   +- dataset_info.json
        |   +- indices.arrow
        |   +- state.json
    |   +- validataion
        |   +- dataset.arrow
        |   +- dataset_info.json
        |   +- indices.arrow
        |   +- state.json
    |   +- dataset_dict.json
|   +- test_dataset
    |   +- validation
        |   +- dataset.arrow
        |   +- dataset_info.json
        |   +- indices.arrow
        |   +- state.json
    |   +- dataset_dict.json
|   +- wikipedia_documents.json
```

- Wikipedia data can be uploaded to the folder location above and used.

```
!git clone https://github.com/jjonhwa/Cross-Encoder-with-Bi-Encoder.git # git clone
% cd ./Cross-Encoder-with-Bi-Encoder/_data                              # change directory (input your own path)

!gdown --id 1OSKOeZxVmjRWokMCNiFtpOQuFDArNINf # wiki data upload        # download wikipedia data
```

## Setup

### Dependencies

- `datasets==1.5.0`
- `transformers==4.5.0`
- `tqdm==4.41.1`
- `pandas==1.1.4`
- `CUDA==11.0`

### Install Requirements

```python
bash install_requirements.sh
```

### Hardware

- `GPU : Tesla V100 (32GB)`

### Note

- You can check the code in the Colab environment using Demo.
- It does not work in Colab Basic.

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

|                | Top-5 | Top-50 | Top-100 |
| -------------- | ----- | ------ | ------- |
| Elastic Search | 0.852 | 0.945  | 0.962   |
| DPR Bi-Encoder | -     | 0.775  | 0.85    |

### 3. **ReRank Strategy with CrossEncoder (In-Batch_Negative Samples)**

- Our purpose is to bring high performance from KLUE MRC competition to End-to-End from Retrieval to Reader. From this, **the ReRank strategy using Cross Encoder was used.**
- In addition, when implementing Cross Encoder, **the key point is to extract a negative sample within Batch and use it to calculate loss.**
- After extracting the Retrival Passage of the Top-500 using the Bi-Encoder, **only a small number of Passages are finally extracted by returning to the Cross Encoder.**
- Retrieval Accuracy in our Data

|                          | Top-5 | Top-50 | Top-100 |
| ------------------------ | ----- | ------ | ------- |
| Elastic Search           | 0.852 | 0.945  | 0.962   |
| DPR without CrossEncoder | -     | 0.775  | 0.85    |
| DPR with CrossEncoder    | 0.825 | 0.95   | -       |

### 4. Ensemble

- In this process, the contents of CrossEncoder were mainly written, and the contents of Ensemble were omitted.
- **An experiment was conducted assuming that performance improvement would be achieved from different types of Retrival combinations by conducting Ensemble using Sparse Embedding and Dense Embedding.**
- Top-100 was selected using Elastic Search and Top-100 was selected using DPR and Cross Encoder, and the final output score was calculated by combining them 1 to 1 and normalizing them.
- When the final Reader model was tested, when Top-5 was input, the performance was the best, so the experiment was conducted after limiting the number of passages to be returned to five.
- **Actually, the performance has improved significantly, and the retrival accuracy is as follows.**

|                       | Top-5  | Top-50 | Top-100 |
| --------------------- | ------ | ------ | ------- |
| Elastic Search        | 0.852  | 0.945  | 0.962   |
| DPR with CrossEncoder | 0.825  | 0.95   | -       |
| Ensemble              | 0.9082 | -      | -       |

## Train CrossEncoder & BiEncoder

- Learn crossencoder and biencoder and store them.
- Modify only the data path to match your data. (find "your_dataset_path")

```python
python train.py --encoder 'cross' --output_directory './save_directory/'
```

or

```python
python train.py --encoder 'bi' --output_directory './save_directory/'
```

## Run ReRank

- It precedes creating an encoder using crossencoder and biencoder. (Before Run ReRank, you have to run 'train.py' to make)
- Modify only the data path to match your data. (find "your_dataset_path")

```python
python rerank.py --input_directory './save_directory/'
```


~~## Run Retriever Demo~~

- ~~Top 500 Passages are Retrieved from about 60000 data using Biencoder, and Top 5 is finally retrieved using CrossEncoder.~~
- ~~Passage Embedding about wiki data, Cross Encoder and Bi-Encoder can be downloaded and utilized~~
