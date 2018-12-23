Kaggle Toxic Comment Classification Challenge
===============================


This respository contains my solution to the Kaggle
[Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).
The competition's goal was to train a model to detect toxic comments like threats,
obsenity, insults, and identity-base hate. The data set consisted of comments from Wikipedia's
talk page edits.

I wrote my solution in Python using
[Tensorflow](http://www.tensorflow.org),
[spaCy](https://spacy.io), and
[Gensim](https://radimrehurek.com/gensim/).

This was a difficult competition as it was fairly easy to achieve a high score and the text
contained a lot of symbol swearing, misspelled curses, etc. I experimented with a variety of
approxes but settled on this pipeline.

1. Tokenized and lematized the data using spaCy
2. Learned the vocabulary using Gensim
3. Identified significant tokens using Chi2 tests

1. Cleaned up the text to remove markup, network addresses, and re-write frequent symbol swears
  with english equivalents.

I trained the model for 8 epochs at a batch size of 128. 

My final score was an AUC ROC of 0.9804 .. ranked 2443 out of 4551 teams (53%). 