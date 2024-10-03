import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer   # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import random
import json
from sklearn.utils.class_weight import compute_class_weight

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Preprocess the data
def preprocess(data):
    tokens = nltk.word_tokenize(data)
    tokens = [word.lower() for word in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)  # Join tokens back into a string

# Load data from JSON file
with open('HR_Quarries.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# Extract patterns and intents from the raw data
processed_data = []
labels = []
for item in raw_data:
    for pattern in item['patterns']:
        processed_data.append(preprocess(pattern))  # Preprocess the patterns
        labels.append(item['intent'])

# Create a label mapping dictionary
label_mapping = {label: idx for idx, label in enumerate(set(labels))}

# Create the training labels array using the label mapping
training_labels = np.array([label_mapping[label] for label in labels])

# Define the model parameters
vocab_size = 5000
embedding_dim = 64
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

# Tokenization and padding
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(processed_data)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(processed_data)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Calculate class weights to balance the data
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(training_labels),
    y=training_labels
)

class_weights_dict = dict(enumerate(class_weights))

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(label_mapping), activation='softmax')  # Adjust output layer size
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
num_epochs = 50
history = model.fit(padded_sequences, training_labels, epochs=num_epochs, class_weight=class_weights_dict, verbose=2)

# Function to predict an answer
def predict_answer(model, tokenizer, question, raw_data):
    try:
        processed_question = preprocess(question)

        if not processed_question:
            return "I'm sorry, I didn't understand that. Could you please rephrase?"

        # Tokenize and pad the input question
        sequence = tokenizer.texts_to_sequences([processed_question])
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)

        # Predict the intent
        pred = model.predict(padded_sequence)[0]
        print("Prediction probabilities:", pred)  # Log the prediction probabilities

        idx = np.argmax(pred)
        print("Predicted index:", idx)  # Log the predicted index

        # Get the predicted intent label
        predicted_intent = None
        for intent, label_idx in label_mapping.items():
            if label_idx == idx:
                predicted_intent = intent
                break

        print("Predicted intent:", predicted_intent)  # Log the predicted intent

        # Fetch the associated responses from the raw_data based on the predicted intent
        for item in raw_data:
            if item['intent'] == predicted_intent:
                # Pick a random response from the responses list for this intent
                return random.choice(item['responses'])

        return "I'm not sure how to respond to that."

    except Exception as e:
        return "An error occurred: " + str(e)

# Main loop for user interaction
while True:
    question = input('You: ')
    answer = predict_answer(model, tokenizer, question, raw_data)
    print('Chatbot:', answer)