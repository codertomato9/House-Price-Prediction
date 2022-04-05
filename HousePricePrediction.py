#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from utils import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from tensorflow.keras.models import Sequential

get_ipython().run_line_magic('matplotlib', 'inline')
tf.logging.set_verbosity(tf.logging.ERROR)

print('The libraries have been imported!')


# In[5]:


# Import Data
df = pd.read_csv('data.csv', names = column_names) 
df.head()


# In[ ]:


# Data Normalisation
df = df.iloc[:,1:]
df_norm = (df - df.mean()) / df.std()
df_norm.head()


# In[ ]:


# Importing the Label Values

y_mean = df['price'].mean()
y_std = df['price'].std()

def convert_label_value(pred):
    return int(pred * y_std + y_mean)

print(convert_label_value(0.350088))


# In[ ]:


# Selecting Features and Labels
X = df_norm.iloc[:, :6]
X.head()

Y = df_norm.iloc[:, -1]
Y.head()


# In[ ]:


X_arr = X.values
Y_arr = Y.values

print('X_arr shape: ', X_arr.shape)
print('Y_arr shape: ', Y_arr.shape)


# In[ ]:


# Splitting Data


X_train, X_test, y_train, y_test = train_test_split(X_arr, Y_arr, test_size = 0.05, shuffle = True, random_state=0)

print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)


# In[ ]:


# Creating Model
def get_model():
    
    model = Sequential([
        Dense(10, input_shape = (6,), activation = 'relu'),
        Dense(20, activation = 'relu'),
        Dense(5, activation = 'relu'),
        Dense(1)
    ])

    model.compile(
        loss='mse',
        optimizer='adadelta'
    )
    
    return model

model = get_model()
model.summary()


# In[ ]:


# Training model
early_stopping = EarlyStopping(monitor='val_loss', patience = 5)

model = get_model()

preds_on_untrained = model.predict(X_test)

history = model.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    epochs = 1000,
    callbacks = [early_stopping]
)


# In[6]:


# Training and Validation loss
plot_loss(history)


# In[7]:


# Comparing Predictions
preds_on_trained = model.predict(X_test)

compare_predictions(preds_on_untrained, preds_on_trained, y_test)


# In[ ]:


# Plotting Price Predictions


price_on_untrained = [convert_label_value(y) for y in preds_on_untrained]
price_on_trained = [convert_label_value(y) for y in preds_on_trained]
price_y_test = [convert_label_value(y) for y in y_test]

compare_predictions(price_on_untrained, price_on_trained, price_y_test)


# In[ ]:




