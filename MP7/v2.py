"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect


def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    # Count occurrences of tags, tag pairs, and tag/word pairs    
    tag_count = defaultdict(int)
    tag_pair_count = defaultdict(lambda: defaultdict(int))
    tag_word_count = defaultdict(lambda: defaultdict(int))

    for sentence in sentences:
        prev_tag = 'START'  # Initialize prev_tag as 'START'
        for word, tag in sentence:
            tag_count[tag] += 1
            emit_prob[tag][word] += 1
            trans_prob[prev_tag][tag] += 1
            tag_pair_count[prev_tag][tag] += 1
            tag_word_count[word][tag] += 1
            prev_tag = tag

    all_tags = set(tag_count.keys())
    for tag in all_tags:
        for word in emit_prob[tag]:
            emit_prob[tag][word] = (emit_prob[tag][word] + epsilon_for_pt) / (tag_count[tag] + epsilon_for_pt * (len(emit_prob[tag]) + 1))
        for next_tag in all_tags:
            trans_prob[tag][next_tag] = (trans_prob[tag][next_tag] + epsilon_for_pt) / (tag_count[tag] + epsilon_for_pt * len(all_tags))

    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.
    if i == 0:
        for tag in emit_prob:
            log_prob[tag] = math.log(prev_prob[tag]) + math.log(emit_prob[tag].get(word, emit_epsilon))
            predict_tag_seq[tag] = [tag]
    else:
        for tag in emit_prob:
            max_prob = float('-inf')
            best_prev_tag = None
            for prev_tag in emit_prob:
                prob = prev_prob[prev_tag] + math.log(trans_prob[prev_tag][tag]) + math.log(emit_prob[tag].get(word, emit_epsilon))
                if prob > max_prob:
                    max_prob = prob
                    best_prev_tag = prev_tag
            log_prob[tag] = max_prob
            predict_tag_seq[tag] = prev_predict_tag_seq[best_prev_tag] + [tag]
    
    return log_prob, predict_tag_seq

def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)
    
    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob)
            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        # Backtracking to find the best tag sequence
        best_final_tag = max(log_prob, key=lambda tag: log_prob[tag])
        best_tag_sequence = predict_tag_seq[best_final_tag]
        
        tagged_sentence = [(word, tag) for word, tag in zip(sentence, best_tag_sequence)]
        predicts.append(tagged_sentence)
        
    return predicts




