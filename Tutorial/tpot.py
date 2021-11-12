
# %%
import ipyenv as uu
uu.chdir(__file__)
uu.curdir

# %%
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

# %% 鸢尾花案例
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                   train_size=0.75, test_size=0.25, random_state=42)

# %% 查看数据
print(iris.keys())
print(iris.feature_names, iris.target_names)
print(iris.data[:5])  # #显示前6行数据
print(iris.target.shape)

print(iris.data.shape)
print(iris.target)  # 用0、1和2三个整数分别代表了花的三个品种
print(X_train.shape, X_test.shape)

# %% 查看数据关联
import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 4, figsize=(21,21))
plt.suptitle("iris_pairplot")
for i in range(4):
    for j in range(3):
        ax[j, i].scatter(X_train[:,i], X_train[:,j], c=y_train)
plt.show()

# %% 基于TPOT
from tpot import TPOTClassifier
tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=1)
tpot.fit(X_train, y_train)

print("Test score:", tpot.score(X_test, y_test))
tpot.export("tpot_iris_pipeline.py")

#####################################################################

# %% 手写体识别案例
mnist = datasets.load_digits()
print(type(mnist.data))

# %% 查看数据的组成
print(mnist.keys())
print(mnist.feature_names, mnist.target_names)
print(mnist.data[0].shape)  # (64,), 即为8x8的图像

# %% 查看图像
import matplotlib.pyplot as plt
n = 6
fig, ax = plt.subplots(n, 1)
for i in range(n):
    im = mnist.data[i].reshape(8,8)
    ax[i].imshow(im)
plt.show()

# %%
from tpot import TPOTClassifier

X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, train_size=0.8)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

tpot = TPOTClassifier(generations=20, population_size=50,
                      verbosity=2, early_stop=1, n_jobs=-2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export(uu.rpath("tpot_mnist_pipeline.py"))

#####################################################################

# %% 波士顿房价预测
from tpot import TPOTRegressor

housing = datasets.load_boston()
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target,
        train_size=0.8, random_state=1)

model = TPOTRegressor(early_stop=1, n_jobs=-2, verbosity=2)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
tpot.export(uu.rpath("tpot_boston_pipeline.py"))
