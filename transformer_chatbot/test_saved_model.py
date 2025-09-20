

import json
import numpy as np
import tensorflow as tf
from transformers import AlbertTokenizer, TFAlbertModel
from sklearn.preprocessing import LabelEncoder

# Load the model, tokenizer, and label encoder
model_save_path = 'albert_model'

# Register the custom object when loading the model
with tf.keras.utils.custom_object_scope({'TFAlbertModel': TFAlbertModel}):
    model = tf.keras.models.load_model(f'{model_save_path}/tf_model.h5')

tokenizer = AlbertTokenizer.from_pretrained(model_save_path)

with open(f'{model_save_path}/label_encoder.json') as f:
    label_encoder_classes = json.load(f)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(label_encoder_classes)

# Load responses
with open('intentfileBiit.json') as file:
    data = json.load(file)

responses = {}
for intent in data['intents']:
    responses[intent['tag']] = intent['responses']

# Function to predict the intent of a sentence
def predict_intent(sentence):
    inputs = tokenizer(sentence, return_tensors="tf", truncation=True, padding=True, max_length=64)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    outputs = model.predict([input_ids, attention_mask])
    predicted_label = np.argmax(outputs, axis=1)
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