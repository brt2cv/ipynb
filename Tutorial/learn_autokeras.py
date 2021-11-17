# %%
import ipyenv as uu
uu.chdir(__file__)

# %%
# 数据集：https://www.kaggle.com/c/digit-recognizer/data

import pandas as pd
train = pd.read_csv(uu.rpath("digit-recognizer/train.csv"))
test = pd.read_csv(uu.rpath("digit-recognizer/test.csv"))

# %%
y_train = train["label"]
X_train = train.drop(labels=["label"], axis=1)

# %%
import seaborn as sns
g = sns.countplot(y_train)
y_train.value_counts()

# %% Check for null and missing values
X_train.isnull().any().describe()
# There is no missing values in the train and test dataset. So we can safely go ahead.

# %% Normalize the data
# We perform a grayscale normalization to reduce the effect of illumination's differences.
# Moreover the CNN converg faster on 0..1 data than on 0..255.
X_train = X_train / 255
test = test / 255

# %%
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1, 28, 28, 1)
print(">>", X_train.shape)

# %%
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=10)
print(">>", y_train.shape)

# %%
# > [github: autokeras/examples/mnist.py](https://github.com/keras-team/autokeras/blob/master/examples/mnist.py)
import autokeras as ak
clf = ak.ImageClassifier(max_trials=3)

# %%
clf.fit(X_train, y_train, epochs=3)

# print(f"Accuracy: {clf.evaluate(test)}")
# %%
