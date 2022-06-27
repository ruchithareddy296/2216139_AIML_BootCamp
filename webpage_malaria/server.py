from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
df = pd.read_csv("outbreak_detect.csv")

app = Flask(__name__)

# Deserialization
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template("index.html")  # due to this function we are able to send our webpage to client(browser)-GET


@app.route('/predict', methods=['POST', 'GET'])  # gets input data from client(browser) to Flask server-to give to ml model
def predict():
    features = [x for x in request.form.values()]
    print(features)
    final = [np.array(features)]
    # Our model was trained on scaled data
    x = df.iloc[:, 0:4]
    sc = StandardScaler().fit(x)
    output = model.predict(sc.transform(final))
    print(output)
    if output[0] == 0:
        return render_template('index.html', pred=f'person is not having malaria disease')
    else:
        return render_template('index.html', pred=f'person is  having malaria disease')


if __name__ == '__main__':
    app.run(debug=True)