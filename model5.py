# -*- coding: utf-8 -*-
"""

@author: LUCIA
"""
import pandas as pd
import numpy as np
# Import `Sequential` from `keras.models`
from keras.models import Sequential
# Import `Dense` from `keras.layers`
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
#for the confusion matrix
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


"""The first step: Reading data
"""

filename_test =  'phishing_dataset_test.csv'
filename_train = 'phishing_dataset_training.csv'
filename_val = 'phishing_dataset_validation.csv'

data_test = pd.read_csv(filename_test, header=0)
data_train = pd.read_csv(filename_train, header=0)
data_val = pd.read_csv(filename_val, header=0)


"""Select the set of feature you want to try
"""
feature_selection_01 = range(30)
feature_selection_02 = [0,1,3,5,6,11,12,13,14,15,17,18,20,21,23,24,25]
feature_selection_03 = [0,1,5,6,11,12,13,23,25]
feature_selection_04 = [0,1,5,6,7,12,13,23,25]
feature_selection_05 = [7,23]

selection=feature_selection_01#********************************************************************************

X_test =data_test.iloc[:,selection]
y_test =np.ravel(data_test.Result)

X_train =data_train.iloc[:,selection]
y_train =np.ravel(data_train.Result)

X_val =data_val.iloc[:,selection]
y_val =np.ravel(data_val.Result)

print('Please enter a name to store the epoch with best accuracy on validation set: ', end ='' )
filepath=input()+'.hdf5'

"""Define the architecture of the neural net
"""
model3 = Sequential()
input_len = len(selection)
input_len_iter = (input_len,)
# Add an input layer 
model3.add(Dense(2*input_len, activation='relu', input_shape = input_len_iter))
# Add one hidden layer 
model3.add(Dense(90, activation='relu'))
model3.add(Dense(60, activation='relu'))
model3.add(Dense(10, activation='relu'))
model3.add(Dense(5, activation='relu'))
model3.add(Dense(10, activation='relu'))
# Add an output layer 
model3.add(Dense(1, activation='sigmoid'))

"""Train the neural net
"""
model3.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model3.fit(X_train, y_train,
           epochs = 500, #**************************************************************************************
           batch_size = 1, 
           callbacks = callbacks_list, 
           verbose = 1,
           validation_data = (X_val, y_val))

"""Recover the neural net with the highest accuracy on VALIDATION dataset
"""
model_selected = load_model(filepath)
y_pred_val = model_selected.predict_classes(X_val)
print('Accuract on validation set:')
score_val = model_selected.evaluate(X_val, y_val,verbose=0)
print(score_val)

"""Show performance on Validation dataset
"""
y_pred_test = model_selected.predict_classes(X_test)
#Data for confusion Matrix
cm_labels = [0,1]
cm =confusion_matrix(y_val, y_pred_val, labels = cm_labels)
cm_plot_labels = ['Phishy','Legitimate']
plot_confusion_matrix(cm, cm_plot_labels, normalize=True, title = 'Confusion Matrix for Features: sel_01')#*********************