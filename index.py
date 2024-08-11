from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crop_prediction')
def crop_prediction():
    return render_template('prediction.html')

@app.route('/prediction', methods=["POST"])
def prediction():
    temp = request.form.get("temp")
    humd = request.form.get("humd")
    ph = request.form.get("ph")
    rain = request.form.get("rain")

    # Check if any of the inputs are None
    if not temp or not humd or not ph or not rain:
        return "Error: All form fields must be filled out"

    df = pd.read_csv("crop_dataset.csv")

    y_train = df["label"]
    df = df.drop(columns=["label"])

    x_train = df

    clf_knn = KNeighborsClassifier(n_neighbors=3)
    clf_knn.fit(x_train, y_train)

    x_test = [[float(temp), float(humd), float(ph), float(rain)]]
    pre_res = clf_knn.predict(x_test)

    return render_template('result1.html', prediction=pre_res[0])


@app.route('/soil_prediction')
def soil_prediction():
    return render_template('prediction1.html')

@app.route('/prediction1', methods=["POST"])
def prediction1():
    n = request.form.get("N")
    p = request.form.get("P")
    k = request.form.get("K")
    fert = request.form.get("fert")

    # Check if any of the inputs are None
    if not n or not p or not k or not fert:
        return "Error: All form fields must be filled out"

    df = pd.read_csv("soil_dataset.csv")

    y_train = df["label"]
    df = df.drop(columns=["label"])

    x_train = df

    clf_knn = KNeighborsClassifier(n_neighbors=3)
    clf_knn.fit(x_train, y_train)

    x_test = [[float(n), float(p), float(k), float(fert)]]
    pre_res = clf_knn.predict(x_test)

    return render_template('result2.html', prediction1=pre_res[0])



if __name__ == '__main__':
    app.run(host="localhost", port=1166, debug=True)


