'''
There are a slew of advanced topics and applications of NLP, many of which rely on deep learning and neural networks.

Naive Bayes classifiers are supervised machine learning algorithms that leverage a probabilistic theorem to make predictions and classifications. They are widely used for sentiment analysis (determining whether a given block of language expresses negative or positive feelings) and spam filtering.

We’ve made enormous gains in machine translation, but even the most advanced translation software using neural networks and LSTM still has far to go in accurately translating between languages.

Some of the most life-altering applications of NLP are focused on improving language accessibility for people with disabilities. Text-to-speech functionality and speech recognition have improved rapidly thanks to neural language models, making digital spaces far more accessible places.

NLP can also be used to detect bias in writing and speech. Feel like a political candidate, book, or news source is biased but can’t put your finger on exactly how? Natural language processing can help you identify the language at issue.
'''

from reviews import counter, training_counts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Add your review:
review = "I kinda liked it, but there is a room for improvement"
review_counts = counter.transform([review])

classifier = MultinomialNB()
training_labels = [0] * 1000 + [1] * 1000

classifier.fit(training_counts, training_labels)
  
neg = (classifier.predict_proba(review_counts)[0][0] * 100).round()
pos = (classifier.predict_proba(review_counts)[0][1] * 100).round()

if pos > 50:
  print("Thank you for your positive review!")
elif neg > 50:
  print("We're sorry this hasn't been the best possible lesson for you! We're always looking to improve.")
else:
  print("Naive Bayes cannot determine if this is negative or positive. Thank you or we're sorry?")

  
print("\nAccording to our trained Naive Bayes classifier, the probability that your review was negative was {0}% and the probability it was positive was {1}%.".format(neg, pos))
