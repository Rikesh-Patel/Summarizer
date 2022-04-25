

import base64
from summa import summarizer
import streamlit as st
import pandas as pd

# Page title and name
st.set_page_config(
    page_title='Project Summarizer'
)
st.markdown("""
        <h1 style='text-align: center; color: #FFFFFF; margin-bottom: -30px;'>
        Basic Text Summarizer
        </h1>
    """, unsafe_allow_html=True
    )
    
st.caption("""
        <p style='text-align: center'>
        by <a href='https://www.rikeshpatel.io/'>Rikesh Patel</a>
        </p>
    """, unsafe_allow_html=True
    )

# Add custom background    
@st.cache(allow_output_mutation=True)
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = """
        <style>
        .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        }
        </style>
    """ % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_bg('assets/background.png')


# User input text request
input_sent = st.text_area("Input Text", "", height=200)
# User input for summarization percent request
ratio = st.slider(
"Summarization factor", min_value=0.0, max_value=1.0, value=0.3, step=0.01
)
summarized_text = summarizer.summarize(
input_sent, ratio=ratio, language="english", split=True, scores=True
)
# Print out the results.

st.markdown("""<style>.big-font {    font-size:10px !important;
}</style>""", unsafe_allow_html=True)

for sentence, score in summarized_text:
    st.markdown('<p class="big-font">'+ sentence +'</p>', unsafe_allow_html=True)
    
