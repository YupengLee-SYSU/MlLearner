import numpy as np
from sklearn import datasets
from LR.LogisticRegression import LogisticRegression
from base_model.DataIterator import DataIterator
from base_model import DataUtils
from criteria.ROC import Roc

cancer = datasets.load_breast_cancer()
cancer_data = cancer['data']
cancer_target = cancer['target']

in_data = DataUtils.combine_data_target(DataUtils.normalize(cancer_data), cancer_target)

# print(in_data.shape)

data_iter = DataIterator(data=in_data, batch_size=in_data.shape[0])
data_iter.shuffle()

data_all = data_iter.next()
data_train = data_all[0: int(np.ceil(data_all.shape[0]*4/5)), :]
data_test = data_all[int(np.ceil(data_all.shape[0]*4/5)): data_all.shape[0], :]

data_iter = DataIterator(data=data_train, batch_size=data_train.shape[0])
data_iter.shuffle()

model = LogisticRegression(n_input=30, n_output=1, i_iter=5, learn_rate=0.0001)
epoch = 20
for i in range(epoch):
    print("epoch: "+str(i+1))
    while data_iter.has_next():
        i_data = data_iter.next()
        model.fit(in_data=i_data[:, 0: 30], target=i_data[:, 30])
        model.train()
    data_iter.shuffle()
        # break
y_pred = model.predict(in_data=data_test[:, 0: 30])
y_label_test = data_test[:, 30]
print(y_pred)
roc = Roc(-y_pred.reshape([y_pred.size, ]), y_label_test.reshape([y_label_test.size, ]))

print(roc.get_auc())
