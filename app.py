from flask import Flask,render_template,request
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

app = Flask(__name__)


def preprocess_name(name):
    name = name.lower()
    name = list(name)
    name_length = 50
    name = (name + [' ']*name_length)[:name_length]
    name = [max(0, ord(char.lower()) - ord('a') + 1) for char in name]
    return name

@app.route("/")
def home():
    return render_template("main_page.html")

@app.route("/predict",methods=["GET","POST"])
def predict():
    name = request.form['name']
    preprocessed_name = preprocess_name(name)
    X_pred = np.reshape(preprocessed_name, (1, -1))
    model = pickle.load(open('CNN.pkl','rb'))
    prediction = model.predict(X_pred)
    if prediction>=0.5:
        result = name+" is a Male name!"
        return render_template("result-male.html",result = result)        
    else:
        result = name+" is a Female name!"
        return render_template("result-female.html",result = result)


if __name__ == "__main__":
    app.run(debug=True)





