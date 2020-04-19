from tensorflow.keras.models import load_model
import seaborn as sns
import pickle

df = sns.load_dataset("iris")

X = df.iloc[:,:4].values
y = df.iloc[:,-1:].values

with open("feature_scaler.pickle", "rb") as file:
    feature_scaler = pickle.load(file)
    X = feature_scaler.transform(X)

with open("label_encoder.pickle", "rb") as file:
    label_encoder = pickle.load(file)
    y = label_encoder.transform(y)

model = load_model("saved_model")

print(f"\n\nAccuracy = {model.evaluate(X, y, verbose=0)[1]*100:.2f}%")
