from mimetypes import init
import SintromDosage as sd
import pandas as pd
import datetime
import calendar
import tensorflow as tf
import keras_tuner as kt
import numpy as np
from matplotlib import pyplot as plt

def prep_data():
    '''
    df = pd.read_csv(r'./dataset.csv')
    df['weekday'] = [calendar.day_name[datetime.datetime.strptime(dt, '%Y-%m-%d').date().weekday()] for dt in df.Date]
    print(df)
    df.to_csv(r"test.csv")
    '''
    df = pd.read_csv(r'./test.csv', header=0, index_col=0)
    
    NaN_indx = pd.isnull(df).any(1).to_numpy().nonzero()[0]
    print(NaN_indx,  "\n------------------\n")
    print(df.iloc[NaN_indx, :],  "\n------------------\n")
    for i in NaN_indx:
        df.loc[i, 'DosageToday'] = df.loc[i, df.loc[i,'Weekday']]
    print(df)
    df.to_csv(r'data.csv')


def first_try(train_X, train_Y,test_X, test_Y):
    #Create ML Model
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(train_X)

    model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(13,)),
    normalizer,
    tf.keras.layers.Dense(52, activation='relu'),
    tf.keras.layers.Dense(52, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(8)
    ])


    model.compile(optimizer='adam',
              loss='mean_absolute_error',
              metrics=['accuracy'])
    
    model.fit(train_X, train_Y, epochs=1000)

    model.evaluate(test_X,  test_Y, verbose=2)

    result = model(test_X)
    print(pd.DataFrame(result), "\n----------------\n")
    print(pd.DataFrame(test_Y), "\n----------------\n")


def second_try(train_X, train_Y,test_X, test_Y):
#Create ML Model
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(train_X)

    model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(5,)),
    normalizer,
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
    ])


    model.compile(optimizer='adam',
              loss='mean_absolute_error',
              metrics=['accuracy'])
    
    model.fit(train_X, train_Y, epochs=1000)

    model.evaluate(test_X,  test_Y, verbose=2)

    result = model(test_X)
    print(pd.DataFrame(result), "\n----------------\n")
    print(pd.DataFrame(test_Y), "\n----------------\n")

class MyHyperModel(kt.HyperModel):
    def __init__(self, train_X, name=None, tunable=True):
        super().__init__(name, tunable)
        self.train_X = train_X

    def build(self, hp):
        #normalizer = tf.keras.layers.Normalization(axis=-1)
        #normalizer.adapt(self.train_X)

        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(5,)))
        #model.add(normalizer)

        # Tune the number of units in the first Dense layer
        # Choose an optimal value between 32-512
        hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
        model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))
        # model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(5))

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])


def third_try(train_X, train_Y,test_X, test_Y):
#Create ML Model
    hp = MyHyperModel(train_X)
    tuner = kt.Hyperband(hp,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='Sintrom')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(train_X, train_Y, epochs=50, validation_split=0.2, callbacks=[stop_early])

    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_X, train_Y, epochs=50, validation_split=0.2)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    hypermodel.fit(train_X, train_Y, epochs=best_epoch, validation_split=0.2)

    eval_result = hypermodel.evaluate(test_X, test_Y)
    print("[test loss, test accuracy]:", eval_result)



def fourth_try(train_X, train_Y,test_X, test_Y):
#Create ML Model
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(train_X)

    model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(5,)),
    normalizer,
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5)
    ])


    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    model.fit(train_X, train_Y, epochs=50)

    model.evaluate(test_X,  test_Y, verbose=2)

    probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
    ])

    result = pd.DataFrame(probability_model(np.array(test_X)), index=range(5), columns=[0,.25,.5,.75,1])#probability_model(np.array(test_X))
    prediction = pd.DataFrame()
    prediction["Predicted"] = result.idxmax(axis=1)
    prediction["Probability"] = result.max(axis=1)
    print("result:\n",prediction, "\n----------------\n")
    print(pd.DataFrame(test_Y), "\n----------------\n")
    
def fifth_try(train_X, train_Y,test_X, test_Y, epochs=500):
#Create ML Model
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(train_X)

    model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(train_X.shape[1])),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(6),
    tf.keras.layers.Activation(activation=tf.nn.softmax)
    ])

    file_path = './checkpoints/checkpoint' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    input(f'\nCheckpoit file : {file_path}\n')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=file_path,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=0,
        save_best_only=True)


    model.compile(tf.keras.optimizers.Adam(learning_rate=1e-3),#optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    history = model.fit(train_X, train_Y, validation_split = 0.2, epochs=epochs, batch_size=2, callbacks=[model_checkpoint_callback])
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train accuracy', 'val accuracy'], loc='upper left')

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'val'], loc='upper left')

    fig.savefig(file_path+'.png')
    fig.show()
    '''
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.show()
    '''

    model.evaluate(test_X,  test_Y, verbose=2)

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


def mnist_ml():
    #Creating model
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test,  y_test, verbose=2)

    probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
    ])

    df = pd.DataFrame(pd.DataFrame(probability_model(x_test[:5])).idxmax(axis=1))
    df[1]=y_test[:5]
    print(df)


if __name__ == '__main__':
    data_df = pd.read_csv(r'./data.csv', header=0, index_col=0)
    #print(data_df.info())
    #print(data_df.head())

    durations=[
        (datetime.datetime.strptime(data_df.loc[i+1,'Date'], '%Y-%m-%d').date()
         - datetime.datetime.strptime(data_df.loc[i,'Date'], '%Y-%m-%d').date()).days
        for i in range(data_df.shape[0] - 1)
    ]
    '''
    durations.append((datetime.datetime.today().date()
         - datetime.datetime.strptime(data_df.loc[data_df.shape[0]-1,'Date'], '%Y-%m-%d').date()).days)
    '''
    data_df['Duration'] = durations +[0]
    data_df.drop(data_df.tail(1).index,inplace=True)

    #print(data_df, "\n----------------\n")

    train_df = data_df.drop(columns=['Date', 'LastSaturday', 'DiffrentToday', 'Weekday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','DosageToday'])
    train_df['NewTP'] = [train_df.loc[i+1,'TP'] for i in range(train_df.shape[0]-1)] + [0]
    train_df['NewINR'] = [train_df.loc[i+1,'INR'] for i in range(train_df.shape[0]-1)] + [0]
    train_df['NewCategory'] = [train_df.loc[i+1,'Category'] for i in range(train_df.shape[0]-1)] + [0]
    
    train_df.drop(train_df.tail(1).index,inplace=True)
    #print(train_df.info())
    #print(train_df.head())
    
    print(train_df, "\n----------------\n")
    
    X = train_df.iloc[:,[0,1,2,3,4,5,6]]
    Y = train_df.loc[:,['NewCategory']]
    Y.reset_index(drop=True, inplace=True)
    
    print(X, "\n----------------\n")
    print(Y, "\n----------------\n")
    
    X['Category'].replace(8,4,inplace=True)
    X['Category'].replace(9,5,inplace=True)
        
    X['NewCategory'].replace(8,4,inplace=True)
    X['NewCategory'].replace(9,5,inplace=True)
        
    Y['NewCategory'].replace(8,4,inplace=True)
    Y['NewCategory'].replace(9,5,inplace=True)
    
    mean = X.mean()
    std = X.std()
    print(mean)
    print(std)
    X = X - mean
    X = X / std

    '''
    fig, ax = plt.subplots()
    print(X.info())
    print(Y.value_counts())
    VP = ax.boxplot(X)
    plt.xticks([i for i in range(1,len(X.columns)+1)], [e for e in X.columns])
    plt.show()
    
    input()
    '''
    seed = 14982#200
    train_X = X.sample(frac=0.8,random_state=seed) #random state is a seed value
    test_X = X.drop(train_X.index)

    print(train_X, "\n----------------\n")
    print(test_X, "\n----------------\n")

    train_Y = Y.sample(frac=0.8,random_state=seed)
    test_Y = Y.drop(train_Y.index)

    print(train_Y, "\n----------------\n")
    print(test_Y, "\n----------------\n")

    #print(train_X.info())
    #print(train_Y.info())
    
    
    fifth_try(train_X, train_Y, test_X, test_Y)
    input()
    
    ''' 
    train_Xs = {}
    test_Xs = {}
    train_Ys = {}
    test_Ys = {}
    
    for c in train_Y.columns:
        #print(c, ' : ')
        train_Xs[c] = train_X.loc[:,['TP','INR',c,'NewTP','NewINR']]
        test_Xs[c] = test_X.loc[:,['TP','INR',c,'NewTP','NewINR']]
        #print(train_Xs[c], "\n----------------\n")
        #print(test_Xs[c], "\n----------------\n")
        
        train_Ys[c] = train_Y.loc[:,[c]]
        test_Ys[c] = test_Y.loc[:,[c]]
        #print(train_Ys[c], "\n----------------\n")
        #print(test_Ys[c], "\n----------------\n")

    for c in train_Ys.keys():
        print("\n----------------\n", c, ' : ')
        #print("train_Xs[c] :")
        #print(train_Xs[c])
        #print("train_Ys[c] :")
        #print(train_Ys[c])
        #second_try(train_Xs[c], train_Ys[c], test_Xs[c], test_Ys[c])
        fourth_try(train_Xs[c], train_Ys[c], test_Xs[c], test_Ys[c])
        input()
    '''

