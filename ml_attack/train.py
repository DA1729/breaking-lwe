# STEP 1: Upload the file
from google.colab import files
uploaded = files.upload()

# STEP 2: Load the dataset
import numpy as np
data = np.load("lwe_dataset.npz")
X = data["X"]
B = data["B"]
secret = data["secret"]
print("X shape:", X.shape)

# STEP 3: Define and train models
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score

n = 512
q = 12289
p = 4
delta = q // p

class LWEAttacker:
    def __init__(self, n=512, q=12289):
        self.n = n
        self.q = q
        self.models = []

    def create_bit_model(self):
        model = keras.Sequential([
            keras.Input(shape=(self.n,)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_models(self, X, secret, epochs=30):
        self.models = []
        accuracies = []

        for i in range(self.n):
            y_bit = np.full(len(X), secret[i])
            model = self.create_bit_model()
            model.fit(X, y_bit, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)
            self.models.append(model)
            predictions = (model.predict(X, verbose=0) > 0.5).astype(int).flatten()
            acc = accuracy_score(y_bit, predictions)
            accuracies.append(acc)
            print(f"Bit {i+1}/{self.n} accuracy: {acc:.3f}", end='\r')

        print("\n✅ All models trained.")
        print(f"Average accuracy: {np.mean(accuracies):.3f}")
        return accuracies

attacker = LWEAttacker()
attacker.train_models(X, secret, epochs=30)

