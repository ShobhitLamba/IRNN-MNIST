# An IRNN implemented over MNIST dataset
# Author: Shobhit Lamba
# e-mail: slamba4@uic.edu

# Importing the libraries
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN
from keras import initializers
from keras.optimizers import RMSprop
from keras.datasets import mnist

batch_size = 32
num_classes = 10
epochs = 200

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1, 1)
x_test = x_test.reshape(x_test.shape[0], -1, 1)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Building the network
model = Sequential()
model.add(SimpleRNN(100,
                    kernel_initializer=initializers.RandomNormal(stddev = 0.001),
                    recurrent_initializer=initializers.Identity(gain = 1.0),
                    activation = "relu",
                    input_shape = x_train.shape[1:]))
model.add(Dense(num_classes))
model.add(Activation("softmax"))
rmsprop = RMSprop(lr = 1e-6)
model.compile(loss = "categorical_crossentropy",
              optimizer = rmsprop,
              metrics = ["accuracy"])

model.fit(x_train, y_train,
          batch_size = batch_size,
          epochs = epochs,
          verbose = 1,
          validation_data = (x_test, y_test))

scores = model.evaluate(x_test, y_test, verbose = 0)
print("Score:", scores[0])
print("Accuracy:", scores[1])