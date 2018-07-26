from textblob import TextBlob


wiki  = TextBlob("Siraj is angry that he never gets good matches in Tinder")

print wiki.tags
print wiki.words
print wiki.sentiment.polarity