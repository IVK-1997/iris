from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

app = FastAPI()

iris = load_iris()
X = iris.data
y = iris.target

model = DecisionTreeClassifier()
model.fit(X, y)

# ✅ HEALTH ENDPOINT (must be simple, no params)
@app.get("/health")
def health():
    return {"status": "ok"}

# ✅ PREDICT ENDPOINT
@app.get("/predict")
def predict(sl: float, sw: float, pl: float, pw: float):
    prediction = model.predict([[sl, sw, pl, pw]])[0]
    class_name = iris.target_names[prediction]

    return {
        "prediction": int(prediction),
        "class_name": class_name
    }