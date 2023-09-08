# text_classifier.py
import argparse
import pickle
import numpy as np
import math
import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings("ignore")

# Get the English stop words
stop_words = set(stopwords.words('english'))

# Initialize the WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_review(review):
    '''
    Input:
        review: a string containing a review.
    Output:
        review_cleaned: a processed review. 
    '''

    # Lowercase the review
    review = review.lower()

    # Remove links
    review = re.sub(r"http\S+|www\S+|https\S+", '', review, flags=re.MULTILINE)

    # Remove punctuations
    review = review.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the review
    tokens = word_tokenize(review)

    # Remove stopwords and perform lemmatization
    review_cleaned = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    # Join the cleaned tokens back into a string
    review_cleaned = ' '.join(review_cleaned)

    return review_cleaned

def find_occurrence(frequency, word, label):
    '''
    Params:
        frequency: a dictionary with the frequency of each pair (or tuple)
        word: the word to look up
        label: the label corresponding to the word
    Return:
        n: the number of times the word with its corresponding label appears.
    '''
    pair = (word, label)
    n = frequency.get(pair, 0)
    
    return n

def review_counter(output_occurrence, reviews, positive_or_negative):
    '''
    Params:
        output_occurrence: a dictionary that will be used to map each pair to its frequency
        reviews: a list of reviews
        positive_or_negative: a list corresponding to the sentiment of each review (either 0 or 1)
    Return:
        output: a dictionary mapping each pair to its frequency
    '''
    ## Steps :
    # define the key, which is the word and label tuple
    # if the key exists in the dictionary, increment the count
    # else, if the key is new, add it to the dictionary and set the count to 1
    
    for label, review in zip(positive_or_negative, reviews):
      split_review = clean_review(review).split()
      for word in split_review:
        key = (word, label)

        if key in output_occurrence:
            output_occurrence[key] += 1
        else:
            output_occurrence[key] = 1
   
    return output_occurrence

def naive_bayes_predict(review, logprior, loglikelihood):
    '''
    Params:
        review: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Return:
        total_prob: the sum of all the loglikelihoods of each word in the review (if found in the dictionary) + logprior (a number)

    '''
    # process the review to get a list of words
    word_l = clean_review(review).split()

    # initialize probability to zero
    total_prob = 0
    
    # add the logprior
    total_prob += logprior

    for word in word_l:
        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            total_prob += loglikelihood[word]

    # classify as 1 (negative) if total_prob >= 0, else 0 (positive)
    return 1 if total_prob >= 0 else 0

def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    """
    Input:
        test_x: A list of reviews
        test_y: the corresponding labels for the list of reviews
        logprior: the logprior
        loglikelihood: a dictionary with the loglikelihoods for each word
    Output:
        accuracy: (# of reviews classified correctly)/(total # of reviews)
    """
    accuracy = 0  

    y_hats = []
    for review in test_x:
        # if the prediction is > 0
        if naive_bayes_predict(review, logprior, loglikelihood) > 0:
            # the predicted class is 1
            y_hat_i = 1
        else:
            # otherwise the predicted class is 0
            y_hat_i = 0

        # append the predicted class to the list y_hats
        y_hats.append(y_hat_i)

    # error is the average of the absolute values of the differences between y_hats and test_y
    error = sum(abs(y_hat_i - y) for y_hat_i, y in zip(y_hats, test_y)) / len(test_y)

    accuracy = 1 - error

    return y_hats, accuracy

def main():
    # Load the model parameters
    with open('model.pkl', 'rb') as f:
        logprior, loglikelihood = pickle.load(f)

    while True:
        # Get user input
        review = input("Enter a review (or 'X' to quit): ")

        # Quit if the user enters 'X'
        if review.upper() == 'X':
            break

        # Preprocess the review
        cleaned_review = clean_review(review)

        # Predict sentiment
        sentiment = naive_bayes_predict(cleaned_review, logprior, loglikelihood)

        # Print the result
        if sentiment == 0:
            print("The review is positive.")
        else:
            print("The review is negative.")


if __name__ == "__main__":
    main()



