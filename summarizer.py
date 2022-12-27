

import base64
from summa import summarizer
import streamlit as st
import pandas as pd
import requests

# Page title and name
st.set_page_config(
    page_title='Project Summarizer'
)

st.sidebar.markdown("""
        <p style='text-align: center; color: #FFFFFF; background-color: white;
  color: black;
  border: 2px solid #e7e7e7; color: black;'>
        <a href='https://www.rikeshpatel.io/'>Return to Portfolio</a>
        </p>
    """, unsafe_allow_html=True
    )

# Create sidebar for possible user inputs
st.sidebar.header('User Input Features')

ratio =  st.sidebar.slider("Summarization factor", min_value=0.0, max_value=1.0, value=0.3, step=0.01 
)

st.markdown("""
        <h1 style='text-align: center; color: #FFFFFF; margin-bottom: -30px;'>
      Text Summarizer
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



from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent='my_application')
location = geolocator.geocode('my location')

st.write('Latitude:', location.latitude)
st.write('Longitude:', location.longitude)



st.write()



# Create a text input field
input_text=''
input_text = st.text_input('Enter text:', input_text)


# Create a set of radio buttons with placeholder options
selected_option = st.radio('Select an option:', ['Option 1', 'Option 2'])

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
  selected_option = st.radio('Select an option:', options)
else:
  options = ['Option 1', 'Option 2']









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
input_sent, ratio=ratio, language="english", split=True, scores=True
)
# Print out the results

st.markdown("""<style>.big-font {    font-size:10px !important;color: #FFFFFF;
}</style>""", unsafe_allow_html=True)

for sentence, score in summarized_text:
    st.markdown('<p class="big-font" style="color: #FFFFFF;">'+ sentence +'</p>', unsafe_allow_html=True)
    
