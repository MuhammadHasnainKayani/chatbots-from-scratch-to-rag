import json
import numpy as np
import tensorflow as tf
from transformers import AlbertTokenizer, TFAlbertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load dataset from JSON
with open('/content/drive/My Drive/intentfileBiit.json') as file:
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

# Train-test split
train_sentences, val_sentences, train_labels, val_labels = train_test_split(training_sentences, training_labels, test_size=0.2)

# Tokenize sentences
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
train_encodings = tokenizer(train_sentences, truncation=True, padding=True, return_tensors="tf")
val_encodings = tokenizer(val_sentences, truncation=True, padding=True, return_tensors="tf")

# Convert labels to numpy arrays
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels)).shuffle(1000).batch(16)  # Increased batch size
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels)).batch(16)

# Load ALBERT model
albert = TFAlbertModel.from_pretrained('albert-base-v2')

# Define custom model with dropout
input_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
albert_output = albert(input_ids, attention_mask=attention_mask)[0]
dropout = tf.keras.layers.Dropout(0.1)(albert_output[:, 0, :])  # Add dropout layer
output = tf.keras.layers.Dense(len(classes), activation='softmax')(dropout)

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

# Compile model with a smaller learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

# Train model
model.fit(train_dataset, epochs=15, validation_data=val_dataset)  # Reduced number of epochs
# Evaluate model on the validation dataset
val_loss, val_accuracy = model.evaluate(val_dataset)

print(f'Validation Accuracy: {val_accuracy}')
# Save model and tokenizer
model_save_path = '/content/drive/My Drive/albert_model'
albert.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
with open(f'{model_save_path}/label_encoder.json', 'w') as f:
    json.dump(label_encoder.classes_.tolist(), f)

def predict_intent(sentence):
    inputs = tokenizer(sentence, return_tensors="tf", truncation=True, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    outputs = model.predict([input_ids, attention_mask])  # Use model.predict to get the predicted probabilities
    predicted_label = np.argmax(outputs, axis=1)  # Get the index of the maximum probability
    intent = label_encoder.inverse_transform(predicted_label)[0]
    return intent

# Function to get a response based on the predicted intent
def get_response(intent):
    return np.random.choice(responses[intent])

# Interactive chatting function
def chat():
    print("Start chatting with the bot (type 'exit' to stop):")
    while True:
        sentence = input("You: ")
        if sentence.lower() == 'exit':
            break
        intent = predict_intent(sentence)
        response = get_response(intent)
        print(f"Bot: {response}")

# Start the chatting session
chat()
