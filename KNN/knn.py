from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle

df = sns.load_dataset("iris")

X = df.iloc[:,:4].values
y = df.iloc[:,-1:].values

feature_scaler = StandardScaler()
X = feature_scaler.fit_transform(X)
with open("feature_scaler.pickle", "wb") as file:
    pickle.dump(feature_scaler, file, protocol=pickle.HIGHEST_PROTOCOL)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y.reshape(y.shape[0],))
with open("label_encoder.pickle", "wb") as file:
    pickle.dump(label_encoder, file, protocol=pickle.HIGHEST_PROTOCOL)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

knn = KNeighborsClassifier(n_neighbors=5, n_jobs=4)

knn.fit(X_train, y_train)

print(f"Accuracy = {knn.score(X_val, y_val)*100:.2f}%")
