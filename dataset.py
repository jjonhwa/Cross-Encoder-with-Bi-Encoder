import numpy as np
import random
import torch

from transformers import AutoTokenizer

from torch.utils.data import TensorDataset, Dataset


class BiEncoder_Dataset_Original(Dataset):
    """
    If the question and passage are tokenized,
     simply cut them into max_length and input them.

    ex) Q: 김연아는 어느나라 사람이니?
        P: 김연아는 대한민국의 대표로서 올림픽에서 당당히 금메달을 차지했다.
        -> Q김연아는 어느나라 사람이니? / P(김연아는 대한민국의 대표로서)
    """

    def __init__(self, queries, passages, tokenizer):
        self.queries = queries
        self.passages = passages
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.queries)

    def _return_train_dataset(self):
        q_seqs = self.tokenizer(
            self.queries, padding="max_length", truncation=True, return_tensors="pt"
        )
        p_seqs = self.tokenizer(
            self.passages, padding="max_length", truncation=True, return_tensors="pt"
        )

        train_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )

        return train_dataset


class BiEncoder_Dataset_Overflow(Dataset):
    """
    When the question and passage are tokenized,
     additional data is generated for sentences that are truncated with max_length,
     question and the cut passage is made into q-p pairs
     
    ex) Q: 김연아는 어느나라 사람이니?
        P: 김연아는 대한민국의 대표로서 올림픽에서 당당히 금메달을 차지했다.
        -> Q김연아는 어느나라 사람이니? / P(김연아는 대한민국의 대표로서)
        -> Q김연아는 어느나라 사람이니? / P(대표로서 올림픽에서 당당히 금메달을)
        -> Q김연아는 어느나라 사람이니? / P(금메달을 차지했다. Pad Pad Pad)
        
    In the case of Korean corpus, this code was used
     because there are many losses of information
     if the passage is tokenized and most of the tokenized passage length exceeds max_length.
    
    Return:
        Dataset format
    """

    def __init__(self, queries, passages, tokenizer):
        self.queries = queries
        self.passages = passages
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.queries)

    def _return_train_dataset(self):
        
        q_seqs = {'input_ids': [],
                  'token_type_ids': [],
                  'attention_mask': []}
        p_seqs = {'input_ids': [],
                  'token_type_ids': [],
                  'attention_mask': []}
        
        for i in range(len(self.queries)):
            q_seq = self.tokenizer(
                self.queries[i],
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            p_seq = self.tokenizer(
                self.passages[i],
                truncation=True,
                stride=128,
                padding='max_length',
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_tensors='pt'
            )

            p_seq.pop("overflow_to_sample_mapping")
            p_seq.pop("offset_mapping")

            for k in q_seq.keys():
                q_seq[k] = q_seq[k].tolist()
                p_seq[k] = p_seq[k].tolist()

            # Add query and passage together to suit the number of cut passes
            for j in range(len(p_seq['input_ids'])):
                q_seqs['input_ids'].append(q_seq['input_ids'][0])
                q_seqs['token_type_ids'].append(q_seq['token_type_ids'][0])
                q_seqs['attention_mask'].append(q_seq['attention_mask'][0])
                p_seqs['input_ids'].append(p_seq['input_ids'][j])
                p_seqs['token_type_ids'].append(p_seq['token_type_ids'][j])
                p_seqs['attention_mask'].append(p_seq['attention_mask'][j])

        for k in q_seqs.keys():
            q_seqs[k] = torch.tensor(q_seqs[k])
            p_seqs[k] = torch.tensor(p_seqs[k])

        train_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            # If you use the roberta model, annotate 'token_type_ids'.
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            # If you use the roberta model, annotate 'token_type_ids'.
            q_seqs["token_type_ids"],
        )
        return train_dataset


class CrossEncoder_Dataset(Dataset):
    """
    The question and Passage are grouped and overflowed to form an input form of CrossEncoder.

    ex) Q: 김연아는 어느나라 사람이니?
        P: 김연아는 대한민국의 대표로서 올림픽에서 당당히 금메달을 차지했다.
        -> [CLS] 김연아는 어느나라 사람이니? [SEP] 김연아는 대한민국의 대표로서
        -> [CLS] 김연아는 어느나라 사람이니? [SEP] 대표로서 올림픽에서 당당히 금메달을
        -> [CLS] 김연아는 어느나라 사람이니? [SEP] 금메달을 차지했다. [SEP] [PAD] ..
    """

    def __init__(self, queries, passages, tokenizer):
        self.queries = queries
        self.passages = passages
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.queries)

    def _return_train_dataset(self):
        tokenized_examples = self.tokenizer(
            self.queries,
            self.passages,
            truncation="only_second",
            max_length=512,
            strid=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            # return_token_type_ids=False,  # if you want to use roberta tokenizer, you release this code
            padding="max_length",
            return_tensors="pt",
        )

        train_dataset = TensorDataset(
            tokenized_examples["input_ids"],
            tokenized_examples["attention_mask"],
            tokenized_examples[
                "token_type_ids"
            ],  # When you use RoBertaModel, annotate it
        )

        return train_dataset
