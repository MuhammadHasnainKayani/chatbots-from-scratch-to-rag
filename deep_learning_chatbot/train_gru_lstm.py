
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset from JSON
with open('intent_dataset.json') as file:
    data = json.load(file)

# Prepare training data
training_sentences = []
training_labels = []
classes = []
responses = {}

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    if intent['tag'] not in classes:
        classes.append(intent['tag'])
    responses[intent['tag']] = intent['responses']

# Encode labels
label_encoder = LabelEncoder()
training_labels = label_encoder.fit_transform(training_labels)

# Tokenize the sentences
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, padding='post')

# Load pre-trained word embeddings (e.g., GloVe)
embedding_index = {}
with open(r'glove.6B.100d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        embedding_vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = embedding_vector

embedding_dim = 100
vocab_size = len(word_index) + 1

# Create embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, index in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# Convert labels to categorical
num_classes = len(classes)
y = tf.keras.utils.to_categorical(training_labels, num_classes=num_classes)


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=padded_sequences.shape[1], trainable=False),
    Bidirectional(LSTM(128, return_sequences=True)),
    Bidirectional(GRU(64)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )

model.summary()
# Train the model
# history = model.fit(padded_sequences, y, epochs=100, batch_size=16, verbose=1)
history = model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1, validation_data=(X_val, y_val))

# Save the model and tokenizer configuration
model.save('model/chatbot_model_advanced_GRU_LSTM.h5')
tokenizer_json = tokenizer.to_json()
with open('model/tokenizer.json', 'w') as f:
    f.write(tokenizer_json)

# Save the label encoder classes
label_encoder_classes = label_encoder.classes_.tolist()
with open('model/label_encoder.json', 'w') as f:
    json.dump(label_encoder_classes, f)

# Evaluate the model
loss, accuracy = model.evaluate(padded_sequences, y)
print(f'Accuracy: {accuracy}')
