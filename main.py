import copy

import numpy as np
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.models import Sequential
from keras.utils import np_utils

longest_sequence_length = 205
longest_sequence_length_with_trimmed_zeros = 182
shortest_sequence_length = 109

number_of_character_classes = 20

zero_point_line_that_can_be_skipped = "0,0,0"
single_sequence_end_line = ",,"
padding_vector = [0.0, 0.0, 0.0]

X_train = []
y_train = []

with open('data/classes.txt') as f:
    character_classes = f.readlines()
for character_class in character_classes[0].split('|'):
    y_train.append(int(character_class) - 1)

y_train = np_utils.to_categorical(y_train, number_of_character_classes)

with open('data/sequences.csv') as f:
    x_y_pressure_points = f.readlines()

single_sequence = []

for point in x_y_pressure_points:
    if zero_point_line_that_can_be_skipped in point:
        continue

    if single_sequence_end_line in point:
        for i in range(longest_sequence_length_with_trimmed_zeros - len(single_sequence)):
            single_sequence.insert(0, padding_vector)

        X_train.append(copy.deepcopy(single_sequence))

        single_sequence = []
        continue

    single_sequence.append([])
    for point_element in point.split(','):
        single_sequence[-1].append(float(point_element))

X_train = np.array(X_train)
print("X_train shape", X_train.shape)
y_train = np.array(y_train)
print("y_train shape", y_train.shape)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(longest_sequence_length_with_trimmed_zeros, 3)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(number_of_character_classes))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.15)
