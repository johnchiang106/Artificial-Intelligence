# naive_bayes.py
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
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"pos_dict prior: {pos_prior}")

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
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the pos_dict label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naiveBayes(dev_set, train_set, train_labels, laplace=1.0, pos_prior=0.5, silently=False):
    print_values(laplace,pos_prior)

    pos_dict = {}
    neg_dict = {}
    pos_size = 0
    neg_size = 0

    for i in range(len(train_set)):
        for word in train_set[i]:
            if train_labels[i] == 1:
                pos_size += 1
                if pos_dict.get(word) is None:
                    pos_dict[word] = 1
                else:
                    pos_dict[word] += 1
            else:
                neg_size += 1
                if neg_dict.get(word) is None:
                    neg_dict[word] = 1
                else:
                    neg_dict[word] += 1
    
    yhats = []
    for review in tqdm(dev_set):
        pos_prob = math.log10(pos_prior)
        neg_prob = math.log10(1 - pos_prior)

        pos_denominator = math.log10(pos_size + laplace * (len(pos_dict) + 1))
        neg_denominator = math.log10(neg_size + laplace * (len(neg_dict) + 1))
        # tuning constant
        pos_alpha = math.log10(laplace) - pos_denominator
        neg_alpha = math.log10(laplace) - neg_denominator

        for word in review:
            pos_count = pos_dict.get(word)
            neg_count = neg_dict.get(word)

            if pos_count is None:
                pos_prob += pos_alpha
            else:
                pos_prob += math.log10(pos_count + laplace) - pos_denominator

            if neg_count is None:
                neg_prob += neg_alpha
            else:
                neg_prob += math.log10(neg_count + laplace) - neg_denominator

        if pos_prob >= neg_prob:
            yhats.append(1)
        else:
            yhats.append(0)
    # for doc in tqdm(dev_set, disable=silently):
    #     yhats.append(1)

    return yhats
