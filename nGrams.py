'''
For parsing entire phrases or conducting language prediction, you will want to use a model that pays attention to each word’s neighbors. Unlike bag-of-words, the n-gram model considers a sequence of some number (n) units and calculates the probability of each unit in a body of language given the preceding sequence of length n. Because of this, n-gram probabilities with larger n values can be impressive at language prediction.

There are a couple problems with the n gram model:

How can your language model make sense of the sentence “The cat fell asleep in the mailbox” if it’s never seen the word “mailbox” before? During training, your model will probably come across test words that it has never encountered before (this issue also pertains to bag of words). A tactic known as language smoothing can help adjust probabilities for unknown words, but it isn’t always ideal.

For a model that more accurately predicts human language patterns, you want n (your sequence length) to be as large as possible. That way, you will have more natural sounding language, right? Well, as the sequence length grows, the number of examples of each sequence within your training corpus shrinks. With too few examples, you won’t have enough data to make many predictions.

Enter neural language models (NLM)! Much recent work within NLP has involved developing and training neural networks to approximate the approach our human brains take towards language. This deep learning approach allows computers a much more adaptive tack to processing human language.
'''

import nltk, re
from nltk.tokenize import word_tokenize
# importing ngrams module from nltk
from nltk.util import ngrams
from collections import Counter
from looking_glass import looking_glass_full_text

cleaned = re.sub('\W+', ' ', looking_glass_full_text).lower()
tokenized = word_tokenize(cleaned)

# if n value is 2:
looking_glass_bigrams = ngrams(tokenized, 2)
looking_glass_bigrams_frequency = Counter(looking_glass_bigrams)

# if n value is 3:
looking_glass_trigrams = ngrams(tokenized, 3)
looking_glass_trigrams_frequency = Counter(looking_glass_trigrams)

# if n value is a number greater than 3:
looking_glass_ngrams = ngrams(tokenized, 15)
looking_glass_ngrams_frequency = Counter(looking_glass_ngrams)

print("Looking Glass Bigrams:")
print(looking_glass_bigrams_frequency.most_common(10))

print("\nLooking Glass Trigrams:")
print(looking_glass_trigrams_frequency.most_common(10))

print("\nLooking Glass n-grams:")
print(looking_glass_ngrams_frequency.most_common(10))
