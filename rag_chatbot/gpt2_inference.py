import json
import numpy as np
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2Model
from sklearn.preprocessing import LabelEncoder

# Load the custom object scope for TFGPT2Model
custom_objects = {'TFGPT2Model': TFGPT2Model}

# Load the saved model with the custom object scope
model = tf.keras.models.load_model('gpt_model_finetuned/chatbot_model_advanced_GPT2.h5', custom_objects=custom_objects)

# Load the saved tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt_model_finetuned/gpt2_tokenizer')

# Load the saved label encoder classes
with open('gpt_model_finetuned/label_encoder.json', 'r') as f:
    label_encoder_classes = json.load(f)

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(label_encoder_classes)

# Load the responses
with open('intentfileBiit.json') as file:
    data = json.load(file)

responses = {}
for intent in data['intents']:
    responses[intent['tag']] = intent['responses']


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = np.random.choice(i['responses'])
            break
    return result


def classify_intent(text):
    # Tokenize the input text
    tokenized_input = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=64)
    input_ids = tokenized_input['input_ids']
    attention_mask = tokenized_input['attention_mask']

    # Predict the intent
    predictions = model.predict({'input_ids': input_ids, 'attention_mask': attention_mask})
    predicted_label = np.argmax(predictions, axis=1)
    intent = label_encoder.inverse_transform(predicted_label)[0]

    return intent


def chat():
    print("Start talking with the bot (type 'quit' to stop)!")
    while True:
        text = input("You: ")
        if text.lower() == 'quit':
            print("Goodbye!")
            break

        intent = classify_intent(text)
        response = get_response([{'intent': intent}], data)

        print(f"Bot: {response}")


# Start the chat
chat()