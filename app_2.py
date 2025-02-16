
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

    
# PAGE SETUP
page_1 = st.Page(
    page=r"./enter_text.py",
    title="Enter text",
    icon=":material/add_circle:",
    default=True,
)

page_2 = st.Page(
    page=r"./upload_file.py",
    title="Upload file",
    icon=":material/add_circle:",   
)

page_3= st.Page(
    page=r"./web_analyze.py",
    title="Web Scraping",
    icon=":material/add_circle:", 
)

page_4= st.Page(
    page=r"./dashboard.py",
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
