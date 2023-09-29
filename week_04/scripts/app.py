import numpy as np
import pickle

from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open("model/rf.pkl", "rb"))

@app.route('/')

def index():
	return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
	int_features = [int(x) for x in request.form.values()]
	final_features = [np.array(int_features)]
	prediction = model.predict(final_features)

	output = round(prediction[0], 2)

	return render_template('index.html', prediction_text='The selling price of the house should be $ {}'.format(output))



if __name__ == "__main__":
	app.run(port=5000, debug= True)