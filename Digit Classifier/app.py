
from flask import Flask, render_template, request

from scipy.misc import imread, imresize, imshow
from sklearn.neighbors import KNeighborsClassifier

import numpy as np

import base64

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('mnist_train.csv')

y  = data['label']
x = data.drop(['label'],axis=1)
x=x.values
y=y.values

#clf=RandomForestClassifier(n_estimators=100)

#clf.fit(x,y)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x, y)

app = Flask(__name__)

	
def convertImage(imgData):
	with open("imageToSave.png", "wb") as fh:
		fh.write(base64.decodebytes(imgData))


@app.route("/")
def index():
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
	imgData = request.get_data()
	convertImage(imgData[22:])
	z = imread('imageToSave.png',mode='L')
	z = np.invert(z)
	z = imresize(z,(28,28))
	plt.imshow(z,cmap='gray')
	plt.show()
	z = z.reshape(1,784)
	print(knn.predict(z))
	return str(knn.predict(z)[0])
	

if __name__ == "__main__":
	app.run(debug=True)