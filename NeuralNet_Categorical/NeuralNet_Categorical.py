### Path setup ###
##################
import os
os.chdir("D:/Temp/ThingsToHave/Python Scripts/ML/Collection 27042021/SineTraining")


# Import
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental import preprocessing



### Data setup ###
##################`
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# OneHot encode labels if needed
#enc = OneHotEncoder(handle_unknown='ignore')
#Y = enc.fit_transform(y_train.reshape(60000,1)).toarray()



### Model setup ###
###################`
model = keras.Sequential([
    keras.Input(shape=(28, 28)),
    preprocessing.Rescaling(scale = 1./255),  
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='sigmoid'),
    layers.Dense(10, activation='sigmoid')
])

# Compile model
# Note: If you want to provide labels using one-hot representation, please use CategoricalCrossentropy loss
model.compile(optimizer=Adam(lr=0.01),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



### Model run ###
#################
model.fit(x_train, y_train, batch_size=100, epochs=10)



### Evaluate model ###
######################
model.evaluate(x_test, y_test)



### Predict ###
###############
Preds = [np.argmax(x) for x in model.predict(x_test)]



##############################################
### Alternative preprocessing run for GPUs ###
##############################################

# Option 1: Make them part of the model, like this:

# inputs = keras.Input(shape=input_shape)
# x = preprocessing_layer(inputs)
# outputs = rest_of_the_model(x)
# model = keras.Model(inputs, outputs)

# With this option, preprocessing will happen on device, synchronously with the rest of the model execution, 
# meaning that it will benefit from GPU acceleration. If you're training on GPU, 
# this is the best option for the Normalization layer, 
# and for all image preprocessing and data augmentation layers.




