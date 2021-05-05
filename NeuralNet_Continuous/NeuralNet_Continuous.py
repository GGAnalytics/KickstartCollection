### Path setup ###
##################
import os
os.chdir("C:/Users")


# Import
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split



### Data setup ###
##################
data = pd.read_csv("SineTraining.csv")

XY = data.copy()
Y = XY.pop("Y").to_numpy()
X = XY.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)



### Model setup ###
###################
model = keras.Sequential()
model.add(layers.Dense(100, input_shape=(1,), activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(1, activation="linear"))

# Compile model
model.compile(loss="mse", optimizer=Adam(lr=0.01))



### Model run ###
#################
model.fit(X_train, y_train, epochs=500, validation_split=0.1)



### Evaluate model ###
######################
model.evaluate(X_test, y_test)



### Plot results (predict) ###
##############################
test_data = np.linspace(0, 6.28, num=200)  # making some test data

f0 = plt.figure(1, figsize=(9, 9))
plt.plot(test_data, np.sin(test_data), linewidth=5.)  # plotting estimated values
plt.plot(test_data, model.predict(test_data))  # plotting true values for sin



