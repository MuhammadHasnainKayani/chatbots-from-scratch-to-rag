import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from sklearn.preprocessing import LabelEncoder

# Load tokenizer
with open('model/tokenizer.json') as f:
    tokenizer_config = json.load(f)
    tokenizer = tokenizer_from_json(json.dumps(tokenizer_config))  # Fixing the JSON loading issue

# Load label encoder
with open('model/label_encoder.json') as f:
    label_encoder_classes = json.load(f)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(label_encoder_classes)

# Load trained model
model = load_model('model/chatbot_model_advanced_GRU_LSTM.h5')

# Load dataset
with open('intentfileBiit.json') as file:
    data = json.load(file)
    responses = {intent['tag']: intent['responses'] for intent in data['intents']}

# Function to predict response
def predict_response(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=20, padding='post')
    predicted = model.predict(padded_sequence)[0]
    predicted_index = np.argmax(predicted)
    tag = label_encoder.inverse_transform([predicted_index])[0]
    return tag

# Testing loop
print("Start talking with the chatbot (type 'quit' to stop)!")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    tag = predict_response(user_input)
    print(f"Chatbot: {np.random.choice(responses[tag])}")