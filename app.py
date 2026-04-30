from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

app = FastAPI()

iris = load_iris()
X = iris.data
y = iris.target

model = DecisionTreeClassifier()
model.fit(X, y)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/predict")
def predict(sl: float = None, sw: float = None, pl: float = None, pw: float = None):
    if None in (sl, sw, pl, pw):
        return {"error": "Missing query parameters"}

    prediction = model.predict([[sl, sw, pl, pw]])[0]
    class_name = iris.target_names[prediction]

    return {
        "prediction": int(prediction),
        "class_name": class_name
    }