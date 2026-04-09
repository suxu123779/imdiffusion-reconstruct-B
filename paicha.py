import pickle
import numpy as np

y = pickle.load(open("data/CODERED_test_label.pkl","rb"))
print("unique labels:", np.unique(y))
print("sum:", np.sum(y), "len:", len(y))
y = pickle.load(open("data/CODERED_train_label.pkl","rb"))
print(np.unique(y), np.sum(y))
