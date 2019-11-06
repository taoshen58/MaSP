from collections import defaultdict
import numpy as np


def get_transition_params_from_sequence_list(seq_list, sorted_label_list=None, sort_fn=None, reverse=False):
    # get sorted label list
    if sorted_label_list is None:
        label_set = set()
        for seq in seq_list:
            label_set.update(set(seq))
        if sort_fn is None:
            sorted_label_list = sorted(list(label_set), reverse=reverse)
        else:
            sorted_label_list = sorted(list(label_set), key=sort_fn, reverse=reverse)

    # map generate
    map_label2id = dict((label, idx) for idx, label in enumerate(sorted_label_list))

    # count pair
    pair_counter = defaultdict(lambda: 0)
    for seq in seq_list:
        for i in range(len(seq)-1):
            pair = (map_label2id[seq[i]], map_label2id[seq[i+1]])
            pair_counter[pair] += 1

    transition_count = np.zeros([len(sorted_label_list), len(sorted_label_list)], dtype='int32')
    for row in range(len(sorted_label_list)):
        for col in range(len(sorted_label_list)):
            transition_count[row, col] = pair_counter[(row, col)]

    transition_count_sum = np.sum(transition_count, axis=-1)  # row
    transition_count[transition_count_sum == 0] = 1
    transition_count = transition_count.astype("float32")

    transition_params = transition_count / np.sum(transition_count, axis=-1, keepdims=True)

    return transition_params


if __name__ == '__main__':
    seq_list = [0, 2, 1, 2, 3]

    print(get_transition_params_from_sequence_list([seq_list, ], ))



