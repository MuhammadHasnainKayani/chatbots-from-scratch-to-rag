import json
import numpy as np
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2Model
from sklearn.preprocessing import LabelEncoder

# Load dataset from JSON
with open('intentfileBiit.json') as file:
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

# Tokenize the sentences
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Set padding token
tokenizer.pad_token = tokenizer.eos_token
# Tokenize the sentences
tokenized_inputs = tokenizer(training_sentences, return_tensors="tf", padding=True, truncation=True, max_length=64)

# Extract input tensors
input_ids = tokenized_inputs['input_ids']
attention_mask = tokenized_inputs['attention_mask']

# Load pre-trained GPT-2 model
gpt2_model = TFGPT2Model.from_pretrained("gpt2")

# Freeze the GPT-2 weights
gpt2_model.trainable = True

# Convert labels to categorical
label_encoder = LabelEncoder()
training_labels = label_encoder.fit_transform(training_labels)
num_classes = len(classes)
y = tf.keras.utils.to_categorical(training_labels, num_classes=num_classes)

# Define a classification model
input_ids_input = tf.keras.Input(shape=(input_ids.shape[1],), dtype=tf.int32, name="input_ids")
attention_mask_input = tf.keras.Input(shape=(input_ids.shape[1],), dtype=tf.int32, name="attention_mask")

gpt2_outputs = gpt2_model(input_ids=input_ids_input, attention_mask=attention_mask_input)
last_hidden_state = gpt2_outputs.last_hidden_state

# Use the last hidden state for classification
outputs = last_hidden_state[:, -1, :]
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(outputs)
model = tf.keras.Model(inputs=[input_ids_input, attention_mask_input], outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x={'input_ids': input_ids, 'attention_mask': attention_mask}, y=y, epochs=5, batch_size=16, verbose=1)

# Save the model and tokenizer configuration
model.save('model/chatbot_model_advanced_GPT2.h5')
tokenizer.save_pretrained('model/gpt2_tokenizer')

# Save the label encoder classes
label_encoder_classes = label_encoder.classes_.tolist()
with open('model/label_encoder.json', 'w') as f:
    json.dump(label_encoder_classes, f)

# Evaluate the model
loss, accuracy = model.evaluate(x={'input_ids': input_ids, 'attention_mask': attention_mask}, y=y)
print(f'Accuracy: {accuracy}')