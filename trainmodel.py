import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

with open("lib/samples/valid-sample.txt", "r", encoding="utf-8") as f:
    valid = [line.strip() for line in f if line.strip()]

with open("lib/samples/negative-sample.txt", "r", encoding="utf-8") as f:
    invalid = [line.strip() for line in f if line.strip()]

urls = valid + invalid
labels = [1] * len(valid) + [0] * len(invalid)

tokenizer = Tokenizer(char_level=True, lower=False)
tokenizer.fit_on_texts(urls)
sequences = tokenizer.texts_to_sequences(urls)
max_len = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_len, padding='post')
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vocab_size = len(tokenizer.word_index) + 1

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=32, input_length=max_len),
    Conv1D(64, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32)

loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

model.save("lib/models/cnndata.h5")
with open("lib/models/cnndatatokenized.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Model and tokenizer saved.")
