import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import requests
from bs4 import BeautifulSoup

# Load the pre-trained sentiment analysis model (.h5)
model = load_model(r'./sentiment_lstm_model.h5')

# Load tokenizer 
tokenizer = Tokenizer()

# Function to fetch text from a webpage
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for request errors
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract all text from paragraph tags using BeautifulSoup
        paragraphs = [p.get_text() for p in soup.find_all("p") if p.get_text().strip()]
        return paragraphs if paragraphs else None  # Return list of paragraphs
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching webpage: {e}")
        return None

# Sentiment analysis function
def predict_sentiment(text):
    # Preprocess the input text (same steps as during training)
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=10000)  # Adjust maxlen based on your model's input size
    
    # Predict sentiment using the loaded model
    prediction = model.predict(padded_sequences)
    
    # For multi-class classification (3 classes: Negative, Neutral, Positive)
    sentiment_classes = ["Negative", "Neutral", "Positive"]
    predicted_class_index = np.argmax(prediction[0])  # Get the index of the highest probability
    sentiment = sentiment_classes[predicted_class_index]  # Get the corresponding sentiment label
    
    return sentiment # Return sentiment

# Streamlit UI
st.header('📝 Sentiment Analysis for Website')

# Input URL
url = st.text_input("🌍 Enter Webpage URL:")

if url:
    st.markdown(f"Click [here]({url}) to view the webpage.")
    
    # Allow user to fetch text from the webpage
    if st.button('Extract Text for Analysis'):
        st.write("Fetching and analyzing text...")
        extracted_text = extract_text_from_url(url)
        
        if extracted_text:
            # Convert the extracted text to a DataFrame
            df = pd.DataFrame({"text": extracted_text})
            
            # Apply sentiment analysis
            sentiments = []
            for text in extracted_text:
                sentiment, probability = predict_sentiment(text)
                sentiments.append(sentiment)
                
            
            # Add sentiment and probabilities to the DataFrame
            df['analysis'] = sentiments
            
            #add today date to the file with the time to 00:00:00
            df['date'] = pd.to_datetime('today').normalize().strftime('%Y-%m-%d %H:%M:%S') 
            
            # Display results
            st.write("Extracted Text with Sentiment Analysis:")
            st.dataframe(df)

            # Provide CSV download option
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", data=csv, file_name="webpage_sentiment.csv", mime="text/csv")
        else:
            st.error("No text found on the webpage.")
