import math
import operator

def viterbi_1(train, test):
    tag_ct = {}  # counts total occurrences of each tag
    tag_pair_ct = {}  # for each tag, holds a dict counting the # of occurrences it followed each tag
    tag_word_ct = {}  # counts the number of occurrences of a word with each tag

    # Do the counts
    for sentence in train:
        for i in range(len(sentence)):
            tw_pair = sentence[i]
            # count occurrences of tag/word pairs
            cur_word = tw_pair[0]
            cur_tag = tw_pair[1]

            cur_tw_dict = tag_word_ct.setdefault(cur_word, {})
            cur_tw_ct = cur_tw_dict.setdefault(cur_tag, 0)
            cur_tw_dict[cur_tag] = cur_tw_ct + 1
            tag_word_ct[cur_word] = cur_tw_dict

            # count occurrences of tags
            cur_tag_ct = tag_ct.setdefault(cur_tag, 0)
            tag_ct[cur_tag] = cur_tag_ct + 1

            # count occurrences of tag pairs
            if i == len(sentence) - 1:
                continue

            next_tw = sentence[i + 1]
            next_tag = next_tw[1]
            next_tag_p = tag_pair_ct.setdefault(next_tag, {})
            cur_tag_p_ct = next_tag_p.setdefault(cur_tag, 0)
            next_tag_p[cur_tag] = cur_tag_p_ct + 1
            tag_pair_ct[next_tag] = next_tag_p

    # Compute the smoothed probabilities
    k = 10 ** -10
    p_tb_ta = {}
    unique_tb_ct = len(tag_ct.keys())

    for tb in tag_ct.keys():
        for ta in tag_ct.keys():
            tb_ta_ct = tag_pair_ct.get(tb, {}).get(ta, 0)
            ta_ct = tag_ct[ta]
            p_tb_ta[(tb, ta)] = math.log(tb_ta_ct + k) - math.log(ta_ct + k * (unique_tb_ct + 1))

    p_w_t = {}
    unique_w_ct = len(tag_word_ct.keys())

    for word in tag_word_ct.keys():
        for t in tag_ct.keys():
            w_t_ct = tag_word_ct.get(word, {}).get(t, 0)
            t_ct = tag_ct[t]
            p_w_t[(word, t)] = math.log(w_t_ct + k) - math.log(t_ct + k * (unique_w_ct + 1))

    tags = tag_ct.keys()
    log_p_zero = -10000000  # use this for P = 0 since log(0) is undefined
    test_labels = []

    for sentence in test:
        trellis_states = []
        trellis_bptr = []

        trellis_states.append({})
        trellis_bptr.append({})

        for tag in tags:
            trellis_bptr[0][tag] = None

            if tag == 'START':
                trellis_states[0][tag] = math.log(1)
            else:
                trellis_states[0][tag] = log_p_zero

        for i in range(1, len(sentence)):
            trellis_states.append({})
            trellis_bptr.append({})
            cur_word = sentence[i]

            for new_tag in tags:
                max_p = None
                max_p_prev_tag = None

                for prev_tag in tags:
                    if (cur_word, new_tag) not in p_w_t.keys():
                        p_prev_word_tag = math.log(k) - math.log(tag_ct[new_tag] + k * (unique_w_ct + 1))
                    else:
                        p_prev_word_tag = p_w_t[(cur_word, new_tag)]

                    cur_p = trellis_states[i - 1][prev_tag] + p_tb_ta[(new_tag, prev_tag)] + p_prev_word_tag

                    if max_p is None:
                        max_p = cur_p
                        max_p_prev_tag = prev_tag
                    elif cur_p > max_p:
                        max_p = cur_p
                        max_p_prev_tag = prev_tag

                trellis_states[i][new_tag] = max_p
                trellis_bptr[i][new_tag] = max_p_prev_tag

        sentence_list = []
        state_idx = len(trellis_bptr) - 1
        highest_p_state = max(trellis_states[state_idx].items(), key=operator.itemgetter(1))[0]

        while highest_p_state is not None:
            sentence_list.append((sentence[state_idx], highest_p_state))
            highest_p_state = trellis_bptr[state_idx][highest_p_state]
            state_idx -= 1

        sentence_list.reverse()
        test_labels.append(sentence_list)

    return test_labels

# Example usage:
# train_data = [your training data]
# test_data = [your test data]
# labeled_test_data = viterbi_1(train_data, test_data)
