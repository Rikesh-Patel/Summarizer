import base64
from summa import summarizer
import streamlit as st
import pandas as pd
import requests
from geopy.geocoders import Nominatim
import json
import folium
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

# Page title and name
st.set_page_config(page_title='Welp')



st.markdown("""
        <h1 style='text-align: center; color: #FFFFFF; margin-bottom: -30px;'>
        Welp: Reviews and Ratings
        </h1>
    """, unsafe_allow_html=True
    )
    
st.caption("""
        <p style='text-align: center; color: #FFFFFF;'>
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
        color=#FFFFFF;
        }
        </style>
    """ % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_bg('assets/background.png')





# Create a text input field
input_text=''
input_text = st.text_input('Location:', input_text)
selected_option = ''

def autocomplete_geolocation(query):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "format": "json",
        "q": query,
        "limit": 4,
        "addressdetails": 1
    }
    response = requests.get(url, params=params)
    return response.json()



# Update the options of the radio buttons based on the text input
if input_text:
  options = [address['display_name'] for address in autocomplete_geolocation(input_text)]
  selected_option = st.radio('', options)
else:
  options = []


if selected_option:
    location = geolocator.geocode(selected_option)
    
    # Get the latitude and longitude from the location object
    lat = location.latitude
    lng = location.longitude
    
    # Display the latitude and longitude in the Streamlit app
    st.write(f'Latitude: {lat}')
    st.write(f'Longitude: {lng}')
    
    
    
    
    

# User input text request
input_sent = st.text_area("", "Input Text", height=200)
# User input for summarization percent request

summarized_text = summarizer.summarize(
input_sent, ratio=0.2, language="english", split=True, scores=True
)
# Print out the results

st.markdown("""<style>.big-font {    font-size:10px !important;color: #FFFFFF;
}</style>""", unsafe_allow_html=True)

for sentence, score in summarized_text:
    st.markdown('<p class="big-font" style="color: #FFFFFF;">'+ sentence +'</p>', unsafe_allow_html=True)
    
