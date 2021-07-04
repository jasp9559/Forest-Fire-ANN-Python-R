# Assignment for ANN with Fire forest data
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Layer, Lambda

forestfires = pd.read_csv("C:/Data Science/Data Science/Assignments/Ass22. ANN/fireforests.csv")

#As dummy variables are already created, we will remove the month and alsoday columns
forestfires.drop(["month", "day"], axis = 1, inplace = True)

forestfires["area"] = np.where(forestfires["area"] > 50, 1, 0)

forestfires["area"].value_counts()
forestfires.isnull().sum()
forestfires.describe()

#Normalization being done.
def norm_func(i):
     x = (i - i.min()) / (i.max() -	i.min())
     return (x)

predictors = forestfires.iloc[ :, 0:8]
target = forestfires.iloc[ :, 8]

predictors1 = norm_func(predictors)
#data = pd.concat([predictors1,target],axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(predictors1, target, test_size = 0.2, stratify = target)

def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1, len(hidden_dim) - 1):
        if (i == 1):
            model.add(Dense(hidden_dim[i], input_dim = hidden_dim[0], activation = "relu"))
        else:
            model.add(Dense(hidden_dim[i], activation = "relu"))
    model.add(Dense(hidden_dim[-1], kernel_initializer = "normal", activation = "sigmoid"))
    model.compile(loss = "binary_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])
    return model  

#y_train = pd.DataFrame(y_train)
    
first_model = prep_model([8, 50, 40, 20, 1])
first_model.fit(np.array(x_train), np.array(y_train), epochs = 750)
pred_train = first_model.predict(np.array(x_train))

#Converting the predicted values to series 
pred_train = pd.Series([i[0] for i in pred_train])

size = ["small", "large"]
pred_train_class = pd.Series(["small"]*413)
pred_train_class[[i > 0.5 for i in pred_train]] = "large"

train = pd.concat([x_train, y_train], axis = 1)
train["area"].value_counts()

# Cheking with prediction for training data
from sklearn.metrics import confusion_matrix
train["original_class"] = "small"
train.loc[train["area"] == 1, "original_class"] = "large"
train.original_class.value_counts()
confusion_matrix(pred_train_class, train["original_class"])
np.mean(pred_train_class == pd.Series(train["original_class"]).reset_index(drop = True)) #98.54%
pd.crosstab(pred_train_class,pd.Series(train["original_class"]).reset_index(drop = True))

#For test data
pred_test = first_model.predict(np.array(x_test))
pred_test = pd.Series([i[0] for i in pred_test])
pred_test_class = pd.Series(["small"]*104)
pred_test_class[[i>0.5 for i in pred_test]] = "large"
test =pd.concat([x_test, y_test], axis = 1)
test["original_class"] = "small"
test.loc[test["area"] == 1, "original_class"] = "large"

test["original_class"].value_counts()

np.mean(pred_test_class==pd.Series(test["original_class"]).reset_index(drop = True)) # 95.19%
confusion_matrix(pred_test_class,test["original_class"])
pd.crosstab(pred_test_class,pd.Series(test["original_class"]).reset_index(drop = True))
