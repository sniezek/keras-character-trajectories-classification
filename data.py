import copy
import random

import numpy as np
from keras.utils import np_utils

# input
zero_point_that_can_be_skipped = '0,0,0'
single_sequence_end = ',,'
padding_vector = [0.0, 0.0, 0.0]
longest_sequence_length_with_trimmed_zeros = 182
longest_sequence_length = 205
shortest_sequence_length = 109
# output
number_of_character_classes = 20  # a b c d e g h l m n o p q r s u v w y z


def get_data(test_fraction):
    x = get_input_data()
    y = get_output_data()

    x_y = list(zip(x, y))
    random.shuffle(x_y)
    x, y = zip(*x_y)

    test_count = int(test_fraction * len(x))
    return np.array(x[test_count:]), np.array(y[test_count:]), np.array(x[:test_count]), np.array(y[:test_count])


def get_input_data():
    x = []
    with open('data/input.csv') as f:
        single_sequence = []
        for point in f:
            if zero_point_that_can_be_skipped in point:
                continue

            if single_sequence_end in point:
                for i in range(longest_sequence_length_with_trimmed_zeros - len(single_sequence)):
                    single_sequence.insert(0, padding_vector)

                x.append(copy.deepcopy(single_sequence))

                single_sequence = []
                continue

            single_sequence.append([])
            for point_element in point.split(','):
                single_sequence[-1].append(float(point_element))
    return x


def get_output_data():
    y = []
    with open('data/output.txt') as f:
        for character_class in f.readlines()[0].split('|'):
            y.append(int(character_class) - 1)

    return np_utils.to_categorical(y, number_of_character_classes)
