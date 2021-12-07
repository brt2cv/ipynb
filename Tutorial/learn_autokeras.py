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
X_train = X_train.values.reshape(-1,28,28)
test = test.values.reshape(-1, 28, 28)
print(">>", X_train.shape)

# %%
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=10)
print(">>", y_train.shape, type(y_train))

# %%
y_train = train["label"].to_numpy()

# %%
# > [github: autokeras/examples/mnist.py](https://github.com/keras-team/autokeras/blob/master/examples/mnist.py)
import autokeras as ak
clf = ak.ImageClassifier(max_trials=3)

# %%
clf.fit(X_train, y_train, epochs=10)

# print(f"Accuracy: {clf.evaluate(test)}")


# %% Titanic
import tensorflow as tf
import autokeras as ak

TRAIN_DATA_URL = r"file:///D:/Home/workspace/ipynb/Tutorial/titanic/train.csv"
TEST_DATA_URL = r"file:///D:/Home/workspace/ipynb/Tutorial/titanic/test.csv"

# Initialize the classifier.
train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

# %%
clf = ak.StructuredDataClassifier(
    max_trials=10, directory="tmp_dir", overwrite=True
)

# %%
import timeit

start_time = timeit.default_timer()
# x is the path to the csv file. y is the column name of the column to predict.
clf.fit(train_file_path, "Survived")
stop_time = timeit.default_timer()
print(
    "Total time: {time} seconds.".format(time=round(stop_time - start_time, 2))
)

# %%
# Evaluate the accuracy of the found model.
accuracy = clf.evaluate(test_file_path, "Survived")[1]
print("Accuracy: {accuracy}%".format(accuracy=round(accuracy * 100, 2)))

# %%
