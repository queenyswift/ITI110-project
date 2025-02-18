import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# Load the pre-trained sentiment analysis model (.h5)
model = load_model(r'./sentiment_lstm_model.h5')

# Load tokenizer 
tokenizer = Tokenizer()  


st.header('📝 Sentiment Analysis for Text Input')

with st.expander('Analyze Text', icon=":material/text_fields:"):
    text = st.text_input('Enter text here: ')
    
    if text:
        # Preprocess the input text (same steps as during training)
        sequences = tokenizer.texts_to_sequences([text])
        padded_sequences = pad_sequences(sequences, maxlen=10000)  # Adjust maxlen based on your model's input size
        
        # Predict sentiment using the loaded model
        prediction = model.predict(padded_sequences)
        
        # Convert the prediction into a sentiment label for multi-class classification
        sentiment_classes = ["Negative", "Neutral", "Positive"]
        predicted_class_index = np.argmax(prediction[0])  # Get the index of the highest value
        sentiment = sentiment_classes[predicted_class_index]  # Get the corresponding sentiment label
        
        st.write(f"Sentiment Analysis: {sentiment}")
