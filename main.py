from keras.layers import LSTM, Dense, Activation, Dropout
from keras.models import Sequential
from keras.utils import plot_model

import data

x_train, y_train, x_test, y_test = data.get_data(test_fraction=0.1)

print('x_train shape: ' + str(x_train.shape))
print('x_test shape: ' + str(x_test.shape))
print('y_train shape: ' + str(y_train.shape))
print('y_test shape: ' + str(y_test.shape))

model = Sequential()
model.add(LSTM(input_shape=(x_train.shape[1], x_train.shape[2]), units=100, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
model.add(LSTM(units=200, dropout=0.1, recurrent_dropout=0.1))
model.add(Dense(units=data.number_of_character_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True)

model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
loss, accuracy = model.evaluate(x_test, y_test)
print('test loss: ' + str(loss) + ', test accuracy: ' + str(accuracy))
