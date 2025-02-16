#C:\Users\qiqiy\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\streamlit.exe run app_2.py
#pip install streamlit beautifulsoup4 requests textblob pandas
#https://www.webfx.com/tools/emoji-cheat-sheet/
#https://fonts.google.com/icons?icon.query=text&icon.size=24&icon.color=%23e8eaed&icon.platform=web
#https://www.geeksforgeeks.org/deploy-a-machine-learning-model-using-streamlit-library/
#python -m pip install -U matplotlib
#pip install --upgrade pip
# For GPU users
#pip install tensorflow[and-cuda]
# For CPU users
#pip install tensorflow
#https://www.youtube.com/watch?v=JIBsJx7U0Xw to enable longpath. otherwise cannot download tensorflow as the path too long

from textblob import TextBlob
import pandas as pd
import streamlit as st
from docx import Document
import PyPDF2
import re
from bs4 import BeautifulSoup
import requests
from PIL import Image
import easyocr




#website tab setup
#wide means use the entire screen
st.set_page_config(page_title="Tech-Duo",page_icon="🧬",layout="wide")


#logo
#st.sidebar.image(r"C:\Users\qiqiy\Desktop\NYP\ITI110\tech_logo.png", caption="Text Sentiment Analyze")
    
# PAGE SETUP
page_1 = st.Page(
    page=r"C:\Users\qiqiy\Desktop\NYP\ITI110\enter_text.py",
    title="Enter text",
    icon=":material/add_circle:",
    default=True,
)

page_2 = st.Page(
    page=r"C:\Users\qiqiy\Desktop\NYP\ITI110\upload_file.py",
    title="Upload file",
    icon=":material/add_circle:",   
)

page_3= st.Page(
    page=r"C:\Users\qiqiy\Desktop\NYP\ITI110\web_analyze.py",
    title="Web Scraping",
    icon=":material/add_circle:", 
)

page_4= st.Page(
    page=r"C:\Users\qiqiy\Desktop\NYP\ITI110\dashboard.py",
    title="Dashboard",
    icon=":material/add_circle:", 
)

page =st.navigation(
    {
        "Home": [page_1, page_2, page_3],
        "Data Visualization": [page_4],
    }
)

page.run()  
