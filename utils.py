import random
import json
import torch

from torch.utils.data import Sampler
from tqdm import tqdm


class CustomSampler(Sampler):
    """
    When creating a DataLoader, make sure
    that three consecutive indexes do not included in one batch

    This CustomSampler assumes that one q-p pair is split into three.
    If it splits more,
    you need to modify "abs(s-f) <= 2" in the code below to fit the length

    you don't have to use this code
    But, if you don't use this code, you have to insert 'shuffle=True' in your DataLoader
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        n = len(self.data_source)
        index_list = []
        while True:
            out = True
            for i in range(self.batch_size):  # Creat an index list of batch_size
                tmp_data = random.randint(0, n - 1)
                index_list.append(tmp_data)
            for f, s in zip(index_list, index_list[1:]):
                if (
                    abs(s - f) <= 2
                ):  # If splits more, modify this code like 'abs(s-f) <= 3'
                    out = False
            if out == True:
                break

        while True:  # Insert additional index data according to condition and length
            tmp_data = random.randint(0, n - 1)
            if (tmp_data not in index_list) and (
                abs(tmp_data - index_list[-i])
                > 2  # If splits more, modify this code like 'abs(tmp_data - index_list[-i]) > 3'
                for i in range(1, self.batch_size + 1)
            ):
                index_list.append(tmp_data)
            if len(index_list) == n:
                break
        return iter(index_list)

    def __len__(self):
        return len(self.data_source)


class Passage_Embedding:
    """
    It receives wiki_path and p_encoder and returns the embedding value for the passage.

    get_corpus:
        Based on wiki data, return the corpus that has been duplicated.

    get_passage_embedding:
        Using p_encoder, all sentences existing in corpus are embedding and returned.
    """

    def __init__(self, wiki_path, p_encoder):
        self.wiki_path = wiki_path
        self.p_encoder = p_encoder

        self.corpus = self.get_corpus()

    def get_corpus(self):
        with open(self.wiki_path, "r", encoding="utf-8") as f:
            wiki = json.load()

        corpus = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # Remove overlapping passages to wiki and bring them in the form of a list.

        return corpus

    def get_passage_embedding(self, tokenizer):
        with torch.no_grad():
            self.p_encoder.eval()

            p_embs = []
            for p in tqdm(self.corpus):
                p = tokenizer(
                    p, padding="max_length", truncation=True, return_tensors="pt"
                )

                if torch.cuda.is_available():
                    p = p.to("cuda")
                p_emb = self.p_encoder(**p).to("cpu").numpy()
                p_embs.append(p_emb)
        p_embs = torch.Tensor(p_embs).squeeze()

        # # If you want to save passage embedding, use the code below by unannotated.
        # import pickle
        # file_path = 'save_directory/passage_embedding.bin'
        # with open(file_path, 'wb') as file :
        #     pickle.dump(p_embs, file)

        return p_embs
