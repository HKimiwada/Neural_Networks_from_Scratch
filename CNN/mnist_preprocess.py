# Preprocessing MNIST Dataset (using keras just to load dataset)
import keras
from keras.utils import to_categorical

def preprocess_data(x, y, limit):
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype('float32') / 255
    y = to_categorical(y, 10)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]


