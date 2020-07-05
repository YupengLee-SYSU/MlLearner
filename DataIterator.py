import numpy as np
from numpy import ndarray


class DataIterator:
    def __init__(self, data: ndarray, batch_size: int, is_auto_reset=False):
        self.is_auto_reset = is_auto_reset
        self.data, self.batch_size = data, batch_size
        self.batched_data, self.data_no = None, data.shape[0]
        self.iter_cnt, self.batch_no = 0, 0
        self.fit()

    def fit(self):
        data_list = [self.data[i: min(i+self.batch_size, self.data_no), :] for i in
                     range(0, self.data_no, self.batch_size)]
        self.iter_cnt = 0
        self.batch_no = len(data_list)
        self.batched_data = iter(data_list)

    def shuffle(self):
        np.random.shuffle(self.data)
        return self.fit()

    def reset(self):
        return self.fit()

    def next(self):
        if self.iter_cnt == self.batch_no:
            if self.is_auto_reset:
                self.fit()
                self.iter_cnt += 1
                return next(self.batched_data)
            else:
                return None
        else:
            self.iter_cnt += 1
            return next(self.batched_data)

    def has_next(self):
        return self.iter_cnt < self.batch_no

if __name__ == "__main__":
    # dataIter = DataIterator(data, 2)
    # dataIter.fit()
    # for a in dataIter.batched_data:
    #     print(a)
    # print(dataIter.batched_data)

    data = np.array([[1, 27, 3], [2, 26, 3], [3, 25, 3], [4, 24, 3],
                     [5, 23, 3], [6, 22, 3], [7, 21, 3]])
    np.random.shuffle(data)

    print(data)
