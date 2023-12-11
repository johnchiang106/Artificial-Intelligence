# naive_bayes.py
# ---------------
# Licensing Information: You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for MP3. You should only modify code
within this file and the last two arguments of line 34 in mp3.py
and if you want-- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import math

def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter=1.0, pos_prior=0.8):
    """
    train_set - List of lists of words corresponding to each movie review
    train_labels - List of labels corresponding to train_set
    dev_set - List of lists of words corresponding to each review that we are testing on
    smoothing_parameter - The smoothing parameter (Laplace) (1.0 by default)
    pos_prior - The prior probability that a word is positive (you do not need to change this value).
    """
    # TODO: Write your code here
    # Return predicted labels of development set
    positive = {}
    negative = {}
    pos_size = 0
    neg_size = 0

    for i in range(len(train_set)):
        for word in train_set[i]:
            if train_labels[i] == 1:
                temp = positive.get(word)
                if temp is None:
                    positive[word] = 1
                else:
                    positive[word] = temp + 1
                pos_size += 1
            else:
                temp = negative.get(word)
                if temp is None:
                    negative[word] = 1
                else:
                    negative[word] = temp + 1
                neg_size += 1

    label = []
    for review in dev_set:
        pos_probability = math.log10(pos_prior)
        neg_probability = math.log10(1 - pos_prior)

        for word in review:
            temp = positive.get(word)
            temp2 = negative.get(word)

            if temp is None:
                pos_probability += math.log10(smoothing_parameter / (pos_size + smoothing_parameter * (len(positive) + 1)))
            else:
                pos_probability += math.log10((temp + smoothing_parameter) / (pos_size + smoothing_parameter * (len(positive) + 1)))

            if temp2 is None:
                neg_probability += math.log10(smoothing_parameter / (neg_size + smoothing_parameter * (len(negative) + 1)))
            else:
                neg_probability += math.log10((temp2 + smoothing_parameter) / (neg_size + smoothing_parameter * (len(negative) + 1)))

        if pos_probability >= neg_probability:
            label.append(1)
        else:
            label.append(0)

    return label

def bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter=1.0, bigram_smoothing_parameter=0.001, bigram_lambda=0.3, pos_prior=0.8):
    """
    train_set - List of lists of words corresponding to each movie review
    train_labels - List of labels corresponding to train_set
    dev_set - List of lists of words corresponding to each review that we are testing on
    unigram_smoothing_parameter - The smoothing parameter for the unigram model (Laplace) (1.0 by default)
    bigram_smoothing_parameter - The smoothing parameter for the bigram model (1.0 by default)
    bigram_lambda - Determines what fraction of your prediction is from the bigram model and what fraction is from the unigram model (default is 0.5)
    pos_prior - The prior probability that a word is positive (you do not need to change this value).
    """
    # TODO: Write your code here
    # Return predicted labels of the development set using a bigram model

    positive = {}
    negative = {}
    uni_positive = {}
    uni_negative = {}
    pos_size = 0
    neg_size = 0
    uni_pos_size = 0
    uni_neg_size = 0

    for i in range(len(train_set)):
        for j in range(len(train_set[i]) - 1):
            bigram = train_set[i][j] + train_set[i][j + 1]

            if train_labels[i] == 1:
                temp = positive.get(bigram)
                if temp is None:
                    positive[bigram] = 1
                else:
                    positive[bigram] = temp + 1
                pos_size += 1
            else:
                temp = negative.get(bigram)
                if temp is None:
                    negative[bigram] = 1
                else:
                    negative[bigram] = temp + 1
                neg_size += 1

        for word in train_set[i]:
            if train_labels[i] == 1:
                temp = uni_positive.get(word)
                if temp is None:
                    uni_positive[word] = 1
                else:
                    uni_positive[word] = temp + 1
                uni_pos_size += 1
            else:
                temp = uni_negative.get(word)
                if temp is None:
                    uni_negative[word] = 1
                else:
                    uni_negative[word] = temp + 1
                uni_neg_size += 1

    label = []

    for review in dev_set:
        pos_probability = math.log10(pos_prior)
        neg_probability = math.log10(1 - pos_prior)
        uni_pos_prob = math.log10(pos_prior)
        uni_neg_prob = math.log10(1 - pos_prior)

        for i in range(len(review) - 1):
            bigram = review[i] + review[i + 1]
            temp = positive.get(bigram)
            temp2 = negative.get(bigram)

            if temp is None:
                pos_probability += math.log10(bigram_smoothing_parameter / (pos_size + bigram_smoothing_parameter * (len(positive) + 1)))
            else:
                pos_probability += math.log10((temp + bigram_smoothing_parameter) / (pos_size + bigram_smoothing_parameter * (len(positive) + 1)))

            if temp2 is None:
                neg_probability += math.log10(bigram_smoothing_parameter / (neg_size + bigram_smoothing_parameter * (len(negative) + 1)))
            else:
                neg_probability += math.log10((temp2 + bigram_smoothing_parameter) / (neg_size + bigram_smoothing_parameter * (len(negative) + 1)))

        for word in review:
            temp = uni_positive.get(word)
            temp2 = uni_negative.get(word)

            if temp is None:
                uni_pos_prob += math.log10(unigram_smoothing_parameter / (uni_pos_size + unigram_smoothing_parameter * (len(uni_positive) + 1)))
            else:
                uni_pos_prob += math.log10((temp + unigram_smoothing_parameter) / (uni_pos_size + unigram_smoothing_parameter * (len(uni_positive) + 1)))

            if temp2 is None:
                uni_neg_prob += math.log10(unigram_smoothing_parameter / (uni_neg_size + unigram_smoothing_parameter * (len(uni_negative) + 1)))
            else:
                uni_neg_prob += math.log10((temp2 + unigram_smoothing_parameter) / (uni_neg_size + unigram_smoothing_parameter * (len(uni_negative) + 1)))

        pos_probability = (1 - bigram_lambda) * uni_pos_prob + bigram_lambda * pos_probability
        neg_probability = (1 - bigram_lambda) * uni_neg_prob + bigram_lambda * neg_probability

        if pos_probability >= neg_probability:
            label.append(1)
        else:
            label.append(0)

    return label
