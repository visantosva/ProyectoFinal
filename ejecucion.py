import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import missingno as msno
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import levene
from scipy.stats import shapiro
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import scale
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, MaxPooling2D
from keras import layers
import tensorflow as tf
import os
import os.path
from pathlib import Path
import cv2
import tensorflow.keras.preprocessing.image
from keras.utils.np_utils import to_categorical

filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=UserWarning)
Meat_Data_Data = Path("C:\Users\David\Documents\PROYECTO\CarneDataset\train\CLASS_02")
# main path
JPG_Path = list(Meat_Data_Data.glob(r"*/*.jpg"))
# jpg path
print(JPG_Path[0:5])
Labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], JPG_Path))
# splitting fresh and spoiled
print(Labels.count("Fresh"))
print(Labels.count("Spoiled"))
File_Path = pd.Series(JPG_Path, name="JPG").astype(str)
print(File_Path.head())
Label_Name = pd.Series(Labels, name="CATEGORY")
print(Label_Name.head())
print(Label_Name.value_counts())
Main_Data = pd.concat([File_Path, Label_Name], axis=1)
Main_Data = Main_Data.sample(frac=1).reset_index(drop=True)
print(Main_Data.head())
print(Main_Data["CATEGORY"].value_counts())
Fresh_Meat = Main_Data[Main_Data["CATEGORY"] == "Fresh"]
Spoiled_Meat = Main_Data[Main_Data["CATEGORY"] == "Spoiled"]
print(Fresh_Meat.head())
print(Spoiled_Meat.head())
figure = plt.figure(figsize=(8, 8))
plt.imshow(plt.imread(Main_Data["JPG"][10]))
plt.show()
figure = plt.figure(figsize=(8, 8))
plt.imshow(plt.imread(Main_Data["JPG"][2]))
plt.show()
figure = plt.figure(figsize=(8, 8))
plt.imshow(plt.imread(Fresh_Meat["JPG"][10]))
plt.show()
figure = plt.figure(figsize=(8, 8))
plt.imshow(plt.imread(Spoiled_Meat["JPG"][1]))
plt.show()
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(8, 8),
                         subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(Main_Data["JPG"][i]))
    ax.set_title(Main_Data["CATEGORY"][i])
plt.tight_layout()
plt.show()
Train_Data, Test_Data = train_test_split(Main_Data, train_size=0.8, shuffle=True, random_state=42)
print(Train_Data.shape)
Data_Generator = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
Train_Gen = Data_Generator.flow_from_dataframe(dataframe=Train_Data,
                                               x_col="JPG",
                                               y_col="CATEGORY",
                                               shuffle=True, seed=42,
                                               color_mode="rgb",
                                               class_mode="categorical",
                                               batch_size=32,
                                               subset="training")
print(Train_Gen.classes[0:20])
print(Train_Gen.split)
Test_Gen = Data_Generator.flow_from_dataframe(dataframe=Test_Data,
                                              x_col="JPG",
                                              y_col="CATEGORY",
                                              shuffle=False, seed=42,
                                              color_mode="rgb",
                                              class_mode="categorical",
                                              batch_size=32)
print(Test_Gen.classes[0:20])
print(Test_Gen.split)
Validation_Gen = Data_Generator.flow_from_dataframe(dataframe=Train_Data,
                                                    x_col="JPG",
                                                    y_col="CATEGORY",
                                                    shuffle=True, seed=42,
                                                    color_mode="rgb",
                                                    class_mode="categorical",
                                                    batch_size=32,
                                                    subset="validation")
print(Validation_Gen.classes[0:20])
print(Validation_Gen.split)
model = tf.keras.models.Sequential([

    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
    tf.keras.layers.Flatten(input_shape=(113,)),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(2, activation="softmax")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
Call_Back = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=2)
ANN_Model = model.fit(Train_Gen,
                      validation_data=Validation_Gen,
                      epochs=10, batch_size=5,
                      callbacks=Call_Back)
Model_Results = model.evaluate(Test_Gen, verbose=False)
print("LOSS:  " + "%.4f" % Model_Results[0])
print("ACCURACY:  " + "%.2f" % Model_Results[1])
print(model.summary())
plt.plot(ANN_Model.history["accuracy"])
plt.plot(ANN_Model.history["val_accuracy"])
plt.ylabel("ACCURACY")
plt.legend()
plt.show()
HistoryDict = ANN_Model.history

val_losses = HistoryDict["val_loss"]
val_acc = HistoryDict["val_accuracy"]
acc = HistoryDict["accuracy"]
losses = HistoryDict["loss"]
epochs = range(1, len(val_losses) + 1)
plt.plot(epochs, val_losses, "k-", label="LOSS")
plt.plot(epochs, val_acc, "r", label="ACCURACY")
plt.title("LOSS & ACCURACY")
plt.xlabel("EPOCH")
plt.ylabel("Loss & Acc")
plt.legend()
plt.show()
plt.plot(epochs, acc, "k-", label="ACCURACY")
plt.plot(epochs, val_acc, "r", label="ACCURACY VAL")
plt.title("ACCURACY & ACCURACY VAL")
plt.xlabel("EPOCH")
plt.ylabel("ACCURACY & ACCURACY VAL")
plt.legend()
plt.show()
plt.plot(epochs, losses, "k-", label="LOSS")
plt.plot(epochs, val_losses, "r", label="LOSS VAL")
plt.title("LOSS & LOSS VAL")
plt.xlabel("EPOCH")
plt.ylabel("LOSS & LOSS VAL")
plt.legend()
plt.show()
Dict_Summary = pd.DataFrame(ANN_Model.history)
Dict_Summary.plot()
Model_Predict = model.predict(Test_Gen)
Model_Predict = np.argmax(Model_Predict, axis=1)
Predict_Label = Test_Gen.class_indices
Predict_Label = dict((v, k) for k, v in Predict_Label.items())
Model_Predict = [Predict_Label[k] for k in Model_Predict]
print(Model_Predict[:10])
Test_Results = list(Test_Data["CATEGORY"])
Class_Report = classification_report(Test_Results, Model_Predict)
print(Class_Report)
Conf_Report = confusion_matrix(Test_Results, Model_Predict, normalize="true")
figure = plt.figure(figsize=(10, 10))
sns.heatmap(Conf_Report, vmax=1, center=0, vmin=-1, annot=True)
plt.show()
fig, axes = plt.subplots(nrows=8,
                         ncols=8,
                         figsize=(15, 15),
                         subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(Test_Data["JPG"].iloc[i]))
    ax.set_title(f"TEST:{Test_Data.CATEGORY.iloc[i]}\n PREDICTION:{Model_Predict[i]}")
plt.tight_layout()
plt.show()
