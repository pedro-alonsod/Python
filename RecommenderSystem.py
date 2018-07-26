import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch data and format it

data =  fetch_movielens(min_rating=4.0)

#print trainig and testing
print(repr(data['train']))
print(repr(data['test']))

#create model
model = LightFM(loss='warp')

#train model
model.fit(data['train'], epochs=30, num_threads=2)

def sampleRecommendation(model, data, userIds):

	 #number of users and movies in training
	 nUsers, nItems = data['train'].shape

	 #generaterecommendation for each user we input
	 for userI in userIds:

	 	#movies they already like
	 	knownPositives = data['item_labels'][data['train'].tocsr()[userI].indices]

	 	#movies our model predict they will like
	 	scores = model.predict(userI, np.arange(nItems)) 

	 	##rank them form high to low
	 	topItems = data['item_labels'][np.argsort(-scores)]

	 	print("User %s" % userI)
	 	print("       Known positives: ")

	 	for x in knownPositives[:3]:
	 		print(".      %s" % x)

	 	print("        Recommended:")
	 	for x in topItems[:3]:
	 		print(".       %s" % x)

sampleRecommendation(model, data, [3, 25, 400])