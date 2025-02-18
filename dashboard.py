import pandas as pd
import streamlit as st
from docx import Document
import PyPDF2
import re
from bs4 import BeautifulSoup
import requests
from PIL import Image
import easyocr
import numpy as np
import matplotlib.pyplot as plt
import io

def save_plot_as_image(fig):
    # Save the plot as a BytesIO object (image file in memory)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf

# File uploader for multiple files
st.header('📊 Sentiment Analysis Dashboard')
uploaded_files = st.file_uploader(
    "Upload one file to show charts for the result of sentiment analyze",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name.lower()
        st.subheader(f"📄 Processing: {uploaded_file.name}")

        try:
            # Reading CSV or Excel files
            if file_name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                st.write("✅ CSV file loaded successfully!")
            elif file_name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
                st.write("✅ Excel file loaded successfully!")
            else:
                st.error(f"❌ Unsupported file type for {uploaded_file.name}")
                continue

            # Check for available text columns
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            if not text_columns:
                st.error(f"❌ The file {uploaded_file.name} does not contain any valid text columns for analysis.")
                continue

            # Use the first available text column for sentiment analysis
            text_column = text_columns[0]
            df[text_column] = df[text_column].astype(str)

            # Check if the 'date' column exists and convert it to datetime
            if 'date' not in df.columns:
                st.error(f"❌ The file {uploaded_file.name} does not contain a 'date' column for time-based analysis.")
                continue

            #convert the datetime value into 2024-02-19 00:00:00
            #If a value cannot be converted to a datetime, it replaces it with NaT (Not a Time, similar to NaN for dates).
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

            # Allow the user to select a time range
            min_date = df['date'].min()
            max_date = df['date'].max()
            start_date, end_date = st.date_input("Select Time Range", [min_date, max_date], min_value=min_date, max_value=max_date)

            # Filter data based on selected time range
            filtered_df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

            # Display filtered data 
            st.write(f"Filtered Data from {start_date} to {end_date}")
            st.write(filtered_df)

            # Allow the user to choose how to group data
            group_option = st.radio("Group data by:", ("Day", "Month", "Year"))

            if group_option == "Day":
                filtered_df['grouping'] = filtered_df['date'].dt.date  # Keep as date
            elif group_option == "Month":
                filtered_df['grouping'] = filtered_df['date'].dt.to_period('M').astype(str)  # Convert to month string
            elif group_option == "Year":
                filtered_df['grouping'] = filtered_df['date'].dt.year.astype(str)  # Convert to string


            
            # Group data based on selected option and count occurrences of each sentiment category
            sentiment_counts = filtered_df.groupby(['grouping', 'analysis']).size().unstack(fill_value=0)

          # Display sentiment count table
            st.write(f"Sentiment Counts by {group_option}")
            st.write(sentiment_counts)
            
            # Allow user to select which charts to display
            show_line_chart = st.checkbox('Show Line Chart', value=True)
            show_bar_chart = st.checkbox('Show Bar Chart', value=True)

            # Generate the Line Chart
            if show_line_chart:
                st.subheader("Line Chart - Sentiment Counts Over Time ({group_option})")
                fig, ax = plt.subplots(figsize=(10, 6))
                sentiment_counts.plot(ax=ax, kind='line', marker='o')
                plt.title("Sentiment Counts by {group_option}", fontsize=16, family='Arial')
                plt.ylabel("Count", fontsize=12, family='Arial',labelpad=10)
                plt.xlabel("Date", fontsize=12, family='Arial',labelpad=10)
                
                ax.tick_params(axis='x', labelsize=10, labelrotation=90, width=5)
                ax.tick_params(axis='y', labelsize=10, width=1)                
                
                st.pyplot(fig)

                # Allow user to download the line chart image
                line_chart_image = save_plot_as_image(fig)
                st.download_button(
                    label="📥 Download Line Chart as Image",
                    data=line_chart_image,
                    file_name="sentiment_line_chart.png",
                    mime="image/png"
                )

            # Generate the Bar Chart
            if show_bar_chart:
                st.subheader("Bar Chart -Sentiment Counts Over Time ({group_option})")
                fig, ax = plt.subplots(figsize=(10, 6))
                sentiment_counts.plot(ax=ax, kind='bar')
                plt.title("Sentiment Counts by {group_option}", fontsize=16, family='Arial')
                plt.ylabel("Count", fontsize=12, family='Arial',labelpad=10)
                plt.xlabel("Date", fontsize=12, family='Arial',labelpad=10)

                ax.tick_params(axis='x', labelsize=10, labelrotation=0, width=10)
                ax.tick_params(axis='y', labelsize=10, width=1)                          
                
                st.pyplot(fig)

                # Allow user to download the bar chart image
                bar_chart_image = save_plot_as_image(fig)
                st.download_button(
                    label="📥 Download Bar Chart as Image",
                    data=bar_chart_image,
                    file_name="sentiment_bar_chart.png",
                    mime="image/png"
                )

        except Exception as e:
            st.error(f"⚠️ Error processing {uploaded_file.name}: {e}")
