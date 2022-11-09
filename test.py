from mimetypes import init
import SintromDosage as sd
import pandas as pd
import datetime
import calendar
import tensorflow as tf
import keras_tuner as kt
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    test_X = pd.DataFrame({"TP" : [31.2],
    "INR" : [3.1],
    "Category" : [2],
    "Duration" : [5],
    "NewTP" : [47.6],
    "NewINR" : [2.00],
    "NewCategory": [1]
    })
    test_Y = pd.DataFrame({"NewCategory": [1]})

    file_path = "./checkpoints/checkpoint2022-11-09-11-55-15"
    #model = tf.keras.models.load_model(file_path)

    model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(test_X.shape[1])),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(6),
    tf.keras.layers.Activation(activation=tf.nn.softmax)
    ])
    #model.load_model(file_path)
    model.load_weights(file_path).expect_partial()

    
    result = pd.DataFrame(model(np.array(test_X)))#, index=range(5), columns=[0,.25,.5,.75,1])#probability_model(np.array(test_X))
    
    categories = ["1/2 everyday",# Done
                        "1/4 everyday",# Done
                        "1/4 every other day",# Done
                        "One day 1/4, one day 1/2",# Done
                        #"1/2 everyday, except Monday and thursday 1/4",# Done
                        #"1/4 everyday, except Monday and thursday 1/2",# Done
                        #"1/4 everyday, except Monday and thursday 0",# Done
                        #"1/2 everyday, except Monday and thursday 3/4",# Done
                        "Three days 1/4, Two days 0",#Sorte of done
                        "Two days 1/2, one day 1/4",#Sorte of done
                        "Other"]
    print("result:\n",[categories[e] for e in result.idxmax(axis=1)], "\n----------------\n")
    print([categories[e] for e in test_Y.loc[:,'NewCategory']], "\n----------------\n")
