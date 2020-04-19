from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import seaborn as sns
import pickle
import numpy as np
import datetime

df = sns.load_dataset("iris")

X = df.iloc[:,:4].values
y = df.iloc[:,-1:].values

feature_scaler = StandardScaler()
X = feature_scaler.fit_transform(X)
with open("feature_scaler.pickle", "wb") as file:
    pickle.dump(feature_scaler, file, protocol=pickle.HIGHEST_PROTOCOL)

label_encoder = LabelBinarizer()
y = label_encoder.fit_transform(y)
with open("label_encoder.pickle", "wb") as file:
    pickle.dump(label_encoder, file, protocol=pickle.HIGHEST_PROTOCOL)

lr = 2e-3
dense1 = 16
dense2 = 8
drop1=0.2
drop2=0.2

model = Sequential()

model.add(Dense(dense1, activation="relu"))
model.add(Dropout(drop1))

model.add(Dense(dense2, activation="relu"))
model.add(Dropout(drop2))

model.add(Dense(3, activation="softmax"))

model.compile(optimizer=Adam(lr=lr), loss="categorical_crossentropy", metrics=["accuracy"])

log_dir = "logs\\fit\\" + f"lr={lr} dense ({dense1}, {dense2}) drop ({drop1}, {drop2})"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
model_saver = tf.keras.callbacks.ModelCheckpoint('saved_model', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

model.fit(X, y, validation_split=0.10, epochs=400, callbacks=[tensorboard_callback, model_saver])