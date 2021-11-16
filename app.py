from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
# Load from file
with open('mymodel.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        input_data = request.form.to_dict()
        try:
            input_data = dict([a, int(x)] for a, x in input_data.items())
        except:
            return "Please enter numeric data ONLY!!!"
        result = get_prediction(input_data)
        return result
    return render_template('index.html')


def get_prediction(input_data):
    data = np.array([list(input_data.values())])
    if np.all(data > 0):
        return "Predicted Crop is {}".format(model.predict(data)[0].upper())
    else:
        return "Please enter positive values!!!"

if __name__ == "__main__":
    app.run(debug=True)
