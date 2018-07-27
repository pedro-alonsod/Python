import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []

def getData(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	return

def predictPrices(dates, prices, x):
	dates = np.reshape(dates, (len(dates), 1))

	svrLin = SVR(kernel='linear', C=1e3)
	svrPoly = SVR(kernel='poly', C=1e3, degree=2)
	svrRbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
	
	svrLin.fit(dates, prices)
	svrPoly.fit(dates, prices)
	svrRbf.fit(dates, prices)

	plt.scatter(dates, prices, color='black', label='data')
	plt.plot(dates, svrRbf.predict(dates), color='red', label='RBF')
	plt.plot(dates, svrLin.predict(dates), color='green', label='Linear')
	plt.plot(dates, svrPoly.predict(dates), color='blue', label='Poly')

	plt.xlabel('Date')
	plt.ylabel('Prices')
	plt.title('Support vector regression')
	plt.legend()
	plt.show()

	return svrLin.predict(x)[0], svrPoly.predict(x)[0], svrRbf.predict(x)[0]


getData('aapl.csv')

predictedPrice = predictPrices(dates, prices, 29)

print(predictedPrice)





