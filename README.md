Kaggle Toxic Comment Classification Challenge
===============================


This respository contains my solution to the Kaggle
[Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).
The competition's goal was to train a model to detect toxic comments like threats,
obsenity, insults, and identity-base hate. The data set consisted of comments 
from Wikipedia's talk page edits.

My final solution was a bi-directional [RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network) 
with 80 [GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit) units. I wrote my solution in Python using 
[Tensorflow](http://www.tensorflow.org),
[spaCy](https://spacy.io), and
[Gensim](https://radimrehurek.com/gensim/),
and [scikit-learn](https://scikit-learn.org/stable/). I also used 
pre-trained [FastText](https://fasttext.cc/docs/en/english-vectors.html) 
embedding vectors.

Before training the RNN, I preprocessed the data by,

1. Tokenizing and lematizing the data (spaCy)
2. Learning the vocabulary (Gensim)
3. Creating TF-IDF vector models of each comment (Gensim)
4. Scoring each vocabulary term's toxic/non-toxic discrimination using Chi2 (sklearn) and [Delta TFIDF](https://www.aaai.org/ocs/index.php/ICWSM/09/paper/view/187/504) metrics.
5. Manually correcting a small number discriminating non-dictionary words

  
The following diagram illustrates my final network design. Each line is a tensor 
annotated with it's dimensions (excluding batch size). Each box is a simplified
representation of operations. 

<pre>
                                                 Logits                      
                                                    ▲                        
                                                    │ 1x6                    
                                                    │                        
                                         ┌─────────────────────┐             
                                         │   Dense Layer (6)   │             
                                         └─────────────────────┘             
                                                    ▲                        
                                                    │ 1x334                  
                                                    │                        
                                         ┌─────────────────────┐             
                                         │       Concat        │             
                                         └─────────────────────┘             
                                                    ▲                        
                    ┌───────────────────────────────┤                        
               1x14 │                               │ 1x320                  
           ┌────────────────┐            ┌─────────────────────┐             
           │   Reduce Max   │            │       Concat        │             
           └────────────────┘            └─────────────────────┘             
                    ▲                               ▲                        
                    │                  ┌────────────┴────────────┐           
                    │             1x160│                         │ 1x160     
                    │                  │                         │           
                    │       ┌─────────────────────┐   ┌─────────────────────┐
                    │       │   Avg Pooling 1D    │   │   Max Pooling 1D    │
                    │       └─────────────────────┘   └─────────────────────┘
                    │                  ▲                         ▲           
                    │                  │                         │           
                    │                  └────────────┬────────────┘           
                    │                               │ 150x160                
                    │                               │                        
                    │                    ┌─────────────────────┐             
                    │                    │       Concat        │             
                    │                    └─────────────────────┘             
                    │                               ▲                        
                    │                  ┌────────────┴────────────┐           
                    │           150x80 │                         │150x80     
                    │                  │                         │           
                    │       ┌─────────────────────┐   ┌─────────────────────┐
                    │       │  Forward GRU (80)   │   │  Backward GRU (80)  │
      ┌─────────────┘       └─────────────────────┘   └─────────────────────┘
      │                                ▲                         ▲           
      │                                │                         │           
      │                                └────────────┬────────────┘           
      │                                             │                        
      │                                             │ 150x300                
      │                                             │                        
      │                                ┌─────────────────────────┐           
      │                                │         Dropout         │           
      │                                └─────────────────────────┘           
      │                                             ▲                        
      │                                             │ 150x300                
      │                                             │                        
      │                                ┌─────────────────────────┐           
      │               ┌───────────────▶│   Embedding Weighting   │           
      │               │                └─────────────────────────┘           
      │               │                             ▲                        
      │               │ 150x1                       │ 150x300                
      │               │                             │                        
      │  ┌─────────────────────────┐   ┌─────────────────────────┐           
      │  │     1D Convolution      │   │   FastText Embeddings   │           
      │  └─────────────────────────┘   └─────────────────────────┘           
      │               ▲                             ▲                        
      └───────────────┤ 150x14                      │ 150x1                  
                      │                             │                        
                                                                             
                Term Scores                     Term IDs                     
                                                                             
</pre>

The model's inputs were:

- The comment's first 150 preprocessed tokens
- The Chi2 and Delta-IDF score for each token and label

The term scores were used in two ways:

- To weight the FastText embeddings via a 1D convolutional layer that
  merged the 14 scores into one scalar weight
- As features for the final dense layer, after a reduce max operation
  to take the highest score for each term

Weighting the embeddings was inspired by [previous experiments](/2018/02/19/tfidf_vectors.html) 
using TF-IDF weighted embeddings. I don't recall how much weighting
the embeddings helped but I believe it had a positive effect.
  
Another novel thing I tried was weighting the losses for each
category by their logs [odds ratio](https://en.wikipedia.org/wiki/Odds_ratio).
The rationale was to use boosting to address class imbalance. Again, I don't recall how much this helped
but I must have had good reason to keep it!

I trained the model for 8 epochs at a batch size of 128 on my OpenSuse Linux
box with a Core i7 6850K (6 cores), 32GB DRAM, and Nvidia Titan X (Pascal) GPU.
My final score was an AUC ROC of 0.9804, which is normally great. However, I
only ranked 2443 out of 4551 teams (53%). 

Source code contains the preprocessing and final model. Also included
are unused models that I tried during the competition.
  
