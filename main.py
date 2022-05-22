#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
# import k fold cross validation
from sklearn.model_selection import KFold


# In[3]:


dataframe = pd.read_csv('dataset.csv')
dataframe.head()


# In[4]:


# remove the airport column from the dataframe
dataframe.drop(['airport'], axis=1, inplace=True)


# In[5]:


# for each row in the dataframe, get each column value
data = dataframe.values
x = []
y = []

x_counter = 0
for i_index, i  in enumerate(data):
    for j in range(0, 8):
        data_sample = i[j:j+5]
        x.append(data_sample[:4])
        y.append(data_sample[4])


# In[6]:


len(x), len(y)


# In[7]:


x = np.array(x)
y = np.array(y)


# In[8]:


# create a neural network with 4 inputs, 6 hidden neurons and 1 regression output
model = tf.keras.Sequential([
    tf.keras.layers.Dense(6, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(1, activation='linear')
])


# In[9]:


# show visual node diagram of the model
model.summary()


# In[10]:


# train the model using 5 folds
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

fold_counter = 1
for train_index, test_index in kfold.split(x, y):
    x_train_k, x_test_k = x[train_index], x[test_index]
    y_train_k, y_test_k = y[train_index], y[test_index]

    # create a neural network with 4 inputs, 6 hidden neurons and 1 regression output
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(6, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    # compile the model
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss=['mape'], metrics=['mape', tf.keras.metrics.RootMeanSquaredError(name='rmse'), 'accuracy', 'mae', 'msle', tf.keras.metrics.CosineSimilarity(name='cosine_similarity')])

    # initialize tensorboard logs for each fold
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs/fold_{}'.format(fold_counter), histogram_freq=1)

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_counter} ...')

    # fit the model
    model.fit(x_train_k, y_train_k, epochs=100, batch_size=32, callbacks=[tensorboard_callback])

    # evaluate the model
    loss_metrics = model.evaluate(x_test_k, y_test_k)
    print(loss_metrics)

    fold_counter += 1


# In[ ]:





# In[ ]:





# In[ ]:




