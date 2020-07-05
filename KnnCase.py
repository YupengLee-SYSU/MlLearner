import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets

if __name__ == "__main__":
    n_neighbors = 11

    # 导入数据
    iris = datasets.load_iris()
    x = iris.data[:, :2]  # 只采用前两个feature,方便画图在二维平面显示
    y = iris.target

    print(type(x))
    print(type(y))