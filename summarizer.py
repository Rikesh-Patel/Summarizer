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


search=0


# Create a text input field
input_text=''
input_text = st.text_input('Search Bar', input_text)
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

geolocator = Nominatim(user_agent='my_application')

if selected_option:
    location = geolocator.geocode(selected_option)
    
    # Get the latitude and longitude from the location object
    lat = location.latitude
    lng = location.longitude
    
    # Display the latitude and longitude in the Streamlit app
    st.write(f'Latitude: {lat}')
    st.write(f'Longitude: {lng}')
    search = st.button("Search")

if search:
    # Yelp
    # Get Business ID
    import requests
    #&latitude=latitude&longitude=longtiude&radius=radius
    query = "Pizza"
    location = "culver city, ca"
    url = f"https://api.yelp.com/v3/businesses/search?latitude={lat}&longitude={lng}&term={query}&categories=&sort_by=best_match&limit=20"

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer A_1nx-eEZxP4IQ-fT7r32jAHjBIU1gQqNzMM5hkc-XGtQFSbeRJr5FNXyXxVBsEA7z5r47W_7rGMK6-hc2OkoUJE_bpgtZ4Oq2zndySrIjBCvS0kH2EWcmxSyx2fY3Yx"
    }

    response = requests.get(url, headers=headers)
    if 'error' in response.json():
        print(response.json()['error']['code'])
    
    import json
    import pandas as pd
    df = pd.json_normalize(response.json(), 'businesses')
    df = df.sort_values("distance")

    def extract_list(json_obj):
        # flat_df = pd.json_normalize(json_obj)
        # return flat_df['name'].tolist()
        return [json['title'] for json in json_obj]

    # Apply the function to each row in the DataFrame
    df['categories'] = df['categories'].apply(extract_list)
    df = df.loc[:,df.columns.isin(['id', 'name', 'image_url', 'is_closed', 'url', 'review_count',
        'categories', 'rating', 'transactions', 'price', 'display_phone',
        'distance', 'coordinates.latitude', 'coordinates.longitude',
        'location.display_address'])]
    df[df['is_closed']==True]
    df = df.loc[:,df.columns.isin(['name', 'url', 'review_count',
        'categories', 'rating', 'transactions', 'price', 'display_phone',
        'distance','location.display_address'])]
       
    
    # Allow the user to sort the data based on any column
    sort_column = st.selectbox('Sort by column', df.columns)
    
    st.dataframe(df.sort_values(by=sort_column)) 

    
    

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
    
