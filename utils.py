import random

from torch.utils.data import Sampler


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
