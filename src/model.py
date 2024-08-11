from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.activations import relu, softmax

def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation=relu),
        Dense(10, activation=softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


