# ProyectoFinal
Participantes en el proyecto de la materia de tratamiento de datos son: Alvaro Misael Criollo Rodas, David Octavio Briones Rodriguez, Victor Avelino Santos Valarezo,  Karla Jacqueline Calapi Mena
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
#from scipy.stats.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
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
from sklearn.linear_model import ElasticNetCV
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
#from lightgbm import LGBMRegressor, LGBMClassifier
#from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, MaxPooling2D
from keras import models
from keras import layers
import tensorflow as tf
import os
import os.path
from pathlib import Path
#import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical


# In[ ]:
print("RUTA de IMAGENeS")

filterwarnings("ignore", category=DeprecationWarning) 
filterwarnings("ignore", category=FutureWarning) 
filterwarnings("ignore", category=UserWarning)


# In[ ]:


Meat_Data_Data = Path("C:/Users/David/Documents/PROYECTO/CarneDataset/train/CLASS_02")
# main path


# In[ ]:

JPG_Path = list(Meat_Data_Data.glob(r"*/*.jpg"))
# jpg path


# In[ ]:

print(JPG_Path[0:5])


# In[ ]:
Labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], JPG_Path))
# splitting fresh and spoiled


# In[ ]:


print(Labels.count("Fresh"))


# In[ ]:


print(Labels.count("Spoiled"))


# In[ ]:


File_Path = pd.Series(JPG_Path, name="JPG").astype(str)


# In[ ]:


print(File_Path.head())


# In[ ]:


Label_Name = pd.Series(Labels, name="CATEGORY")


# In[ ]:


print(Label_Name.head())


# In[ ]:


print(Label_Name.value_counts())


# In[ ]:


Main_Data = pd.concat([File_Path, Label_Name], axis=1)


# In[ ]:


Main_Data = Main_Data.sample(frac=1).reset_index(drop=True)


# In[ ]:


print(Main_Data.head())


# In[ ]:


print(Main_Data["CATEGORY"].value_counts())


# In[ ]:


Fresh_Meat = Main_Data[Main_Data["CATEGORY"] == "Fresh"]
Spoiled_Meat = Main_Data[Main_Data["CATEGORY"] == "Spoiled"]


# In[ ]:


print(Fresh_Meat.head())


# In[ ]:


print(Spoiled_Meat.head())


# In[ ]:


figure = plt.figure(figsize=(8, 8))
plt.imshow(plt.imread(Main_Data["JPG"][1]))
plt.show()


# In[ ]:


#figure = plt.figure(figsize=(8,8))
#plt.imshow(plt.imread(Main_Data["JPG"][2]))
#plt.show()


# In[ ]:


#figure = plt.figure(figsize=(8,8))
#plt.imshow(plt.imread(Fresh_Meat["JPG"][10]))
#plt.show()


# In[ ]:


#figure = plt.figure(figsize=(8,8))
#plt.imshow(plt.imread(Spoiled_Meat["JPG"][1]))
#plt.show()


# In[ ]:


fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(8, 8), subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(Main_Data["JPG"][i]))
    ax.set_title(Main_Data["CATEGORY"][i])
plt.tight_layout()
plt.show()


# In[ ]:


Train_Data, Test_Data = train_test_split(Main_Data, train_size=0.8, shuffle=True, random_state=42)


# In[ ]:


print(Train_Data.shape)


# In[ ]:


Data_Generator = ImageDataGenerator(rescale=1./255,validation_split=0.2)


# In[ ]:


Train_Gen = Data_Generator.flow_from_dataframe(dataframe=Train_Data,
                                               x_col="JPG",
                                               y_col="CATEGORY",
                                               shuffle=True,seed=42,
                                               color_mode="rgb",
                                               class_mode="categorical",
                                               batch_size=32,
                                               subset="training")


# In[ ]:


print(Train_Gen.classes[0:20])


# In[ ]:


print(Train_Gen.split)


# In[ ]:


Test_Gen = Data_Generator.flow_from_dataframe(dataframe=Test_Data,
                                               x_col="JPG",
                                               y_col="CATEGORY",
                                               shuffle=False,seed=42,
                                               color_mode="rgb",
                                               class_mode="categorical",
                                               batch_size=32)


# In[ ]:


print(Test_Gen.classes[0:20])


# In[ ]:


print(Test_Gen.split)


# In[ ]:


Validation_Gen = Data_Generator.flow_from_dataframe(dataframe=Train_Data,
                                               x_col="JPG",
                                               y_col="CATEGORY",
                                               shuffle=True,seed=42,
                                               color_mode="rgb",
                                               class_mode="categorical",
                                               batch_size=32,
                                               subset="validation")


# In[ ]:


print(Validation_Gen.classes[0:20])


# In[ ]:


print(Validation_Gen.split)


# In[ ]:


model = tf.keras.models.Sequential([
  
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Flatten(input_shape=(113,)),
  
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  
  tf.keras.layers.Dense(2,activation="softmax")
])


# In[ ]:


model.compile(optimizer="rmsprop",
             loss="binary_crossentropy",
             metrics=["accuracy"])


# In[ ]:


Call_Back = tf.keras.callbacks.EarlyStopping(monitor="loss",patience=2)


# In[ ]:


ANN_Model = model.fit(Train_Gen,
                     validation_data=Validation_Gen,
                     epochs=10,batch_size=5,
                     callbacks=Call_Back)


# In[ ]:


Model_Results = model.evaluate(Test_Gen,verbose=False)
print("LOSS:  " + "%.4f" % Model_Results[0])
print("ACCURACY:  " + "%.2f" % Model_Results[1])


# In[ ]:


print(model.summary())


# In[ ]:


plt.plot(ANN_Model.history["accuracy"])
plt.plot(ANN_Model.history["val_accuracy"])
plt.ylabel("ACCURACY")
plt.legend()
plt.show()


# In[ ]:


HistoryDict = ANN_Model.history

val_losses = HistoryDict["val_loss"]
val_acc = HistoryDict["val_accuracy"]
acc = HistoryDict["accuracy"]
losses = HistoryDict["loss"]
epochs = range(1,len(val_losses)+1)


# In[ ]:


plt.plot(epochs,val_losses,"k-",label="LOSS")
plt.plot(epochs,val_acc,"r",label="ACCURACY")
plt.title("LOSS & ACCURACY")
plt.xlabel("EPOCH")
plt.ylabel("Loss & Acc")
plt.legend()
plt.show()


# In[ ]:


plt.plot(epochs,acc,"k-",label="ACCURACY")
plt.plot(epochs,val_acc,"r",label="ACCURACY VAL")
plt.title("ACCURACY & ACCURACY VAL")
plt.xlabel("EPOCH")
plt.ylabel("ACCURACY & ACCURACY VAL")
plt.legend()
plt.show()


# In[ ]:


plt.plot(epochs, losses, "k-", label="LOSS")
plt.plot(epochs, val_losses, "r", label="LOSS VAL")
plt.title("LOSS & LOSS VAL")
plt.xlabel("EPOCH")
plt.ylabel("LOSS & LOSS VAL")
plt.legend()
plt.show()


# In[ ]:


Dict_Summary = pd.DataFrame(ANN_Model.history)
Dict_Summary.plot()


# In[ ]:


Model_Predict = model.predict(Test_Gen)


# In[ ]:


Model_Predict = np.argmax(Model_Predict,axis=1)


# In[ ]:


Predict_Label = (Test_Gen.class_indices)
Predict_Label = dict((v, k) for k, v in Predict_Label.items())


# In[ ]:


Model_Predict = [Predict_Label[k] for k in Model_Predict]


# In[ ]:


print(Model_Predict[:10])


# In[ ]:


Test_Results = list(Test_Data["CATEGORY"])


# In[ ]:


Class_Report = classification_report(Test_Results,Model_Predict)
print(Class_Report)


# In[ ]:


Conf_Report = confusion_matrix(Test_Results,Model_Predict, normalize="true")
figure = plt.figure(figsize=(10,10))
sns.heatmap(Conf_Report,vmax=1,center=0,vmin=-1,annot=True)
plt.show()


# In[ ]:


fig, axes = plt.subplots(nrows=8,
                         ncols=8,
                         figsize=(15, 15),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(Test_Data["JPG"].iloc[i]))
    ax.set_title(f"TEST:{Test_Data.CATEGORY.iloc[i]}\n PREDICTION:{Model_Predict[i]}")
plt.tight_layout()
plt.show()


# In[ ]:


get_ipython().system(' pip install xgboost')


# In[ ]:


get_ipython().system(' pip3 install xgboost')


# In[ ]:


get_ipython().system(' conda install -c conda-forge xgboost')


# In[ ]:


import xgboost


# In[ ]:
