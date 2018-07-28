from textblob import TextBlob
import tweepy

wiki  = TextBlob("Siraj is angry that he never gets good matches in Tinder")

print wiki.tags
print wiki.words
print wiki.sentiment.polarity

consumerKey = ''
consumerSecret = ''

accessToken = ''
accessTokenSecret = ''

auth = tweepy.OAuthHandler(consumerKey, consumerSecret)

auth.set_access_token(accessToken, accessTokenSecret)

api = tweepy.API(aut)


publicTweets = api.search('Trump')

for tweet in publicTweets:
	print(tweet.text)
	analysis = TextBlob(tweet.text)
	print(analysis.sentiment)


