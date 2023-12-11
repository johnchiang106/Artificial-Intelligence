# bigram_naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter


'''
utils for printing values
'''
def print_values(unigram_laplace, pos_prior):
    print(f"Unigram unigram_laplace: {unigram_laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram unigram_laplace: {unigram_laplace}")
    print(f"Bigram unigram_laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the unigram_laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigramBayes(dev_set, train_set, train_labels, unigram_laplace=1.0, bigram_laplace=0.001, bigram_lambda=0.3, pos_prior=0.5, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)
    
    wordDict = [{}, {}, {}, {}] # uni pos, uni neg, bi pos, bi neg
    wordSize = [0, 0, 0, 0] # uni pos, uni neg, bi pos, bi neg

    for i in range(len(train_set)):
        for j in range(len(train_set[i]) - 1):
            bigram = train_set[i][j] + train_set[i][j + 1]
            idx = 3 - train_labels[i]
            wordSize[idx] += 1
            if wordDict[idx].get(bigram) is None:
                wordDict[idx][bigram] = 1
            else:
                wordDict[idx][bigram] += 1

        for word in train_set[i]:
            idx = 1 - train_labels[i]
            wordSize[idx] += 1
            if wordDict[idx].get(word) is None:
                wordDict[idx][word] = 1
            else:
                wordDict[idx][word] += 1

    yhats = []

    for review in dev_set:
        prob = [math.log10(pos_prior), math.log10(1 - pos_prior)]
        # uni pos, uni neg, bi pos, bi neg
        prob = prob * 2

        pos_d = math.log10((wordSize[2] + bigram_laplace * (len(wordDict[2]) + 1)))
        neg_d = math.log10((wordSize[3] + bigram_laplace * (len(wordDict[3]) + 1)))

        for i in range(len(review) - 1):
            bigram = review[i] + review[i + 1]
            pos_count = wordDict[2].get(bigram)
            neg_count = wordDict[3].get(bigram)

            if pos_count is None:
                prob[2] += math.log10(bigram_laplace) - pos_d
            else:
                prob[2] += math.log10(pos_count + bigram_laplace) - pos_d

            if neg_count is None:
                prob[3] += math.log10(bigram_laplace) - neg_d
            else:
                prob[3] += math.log10(neg_count + bigram_laplace) - neg_d

        pos_d = math.log10((wordSize[0] + unigram_laplace * (len(wordDict[0]) + 1)))
        neg_d = math.log10((wordSize[1] + unigram_laplace * (len(wordDict[1]) + 1)))

        for word in review:
            pos_count = wordDict[0].get(word)
            neg_count = wordDict[1].get(word)

            if pos_count is None:
                prob[0] += math.log10(unigram_laplace) - pos_d
            else:
                prob[0] += math.log10(pos_count + unigram_laplace) - pos_d

            if neg_count is None:
                prob[1] += math.log10(unigram_laplace) - neg_d
            else:
                prob[1] += math.log10(neg_count + unigram_laplace) - neg_d

        pos_prob = (1 - bigram_lambda) * prob[0] + bigram_lambda * prob[2]
        neg_prob = (1 - bigram_lambda) * prob[1] + bigram_lambda * prob[3]

        if pos_prob >= neg_prob:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats