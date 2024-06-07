from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
label_map = {label:num for num, label in enumerate(actions)}
# print(label_map)
sequences, labels = [], []
# for action in actions:
#     for sequence in range(no_sequences):
#         window = []
#         for frame_num in range(sequence_length):
#             res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)
#             window.append(res)
#         sequences.append(window)
#         labels.append(label_map[action])

# max_frame_length = 63  # Adjust according to your data

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)
            # Ensure each frame has the same length
            # padded_res = np.zeros(max_frame_length)  # Assuming max_frame_length is the maximum length of your frames
            # padded_res[:len(res)] = res[:max_frame_length]  # Truncate if longer, pad with zeros if shorter
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


# X = np.array(sequences)
# y = to_categorical(labels).astype(int)

X = np.array(sequences)
# print("X shape:", X.shape)  # Check the shape here

y = to_categorical(labels, num_classes=len(actions)).astype(int)
# print("y shape:", y.shape)  # Check the shape of labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
res = [.7, 0.2, 0.1]

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
model.summary()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')

# model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, max_frame_length)))
# model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(LSTM(64, return_sequences=False, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(len(actions), activation='softmax'))  # Adjust according to number of classes

# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# # Split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# # Train the model
# model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

# # Save the model
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# model.save('model.h5')
