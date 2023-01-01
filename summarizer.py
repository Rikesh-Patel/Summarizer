import base64
from summa import summarizer
import streamlit as st
import pandas as pd
import numpy as np
import requests
from geopy.geocoders import Nominatim
import json
import folium
import nltk
from streamlit_folium import st_folium, folium_static
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
step2=0
step3=0









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
        return [json['title'] for json in json_obj]

    # Apply the function to each row in the DataFrame
    df['categories'] = df['categories'].apply(extract_list)
    # df = df[df['is_closed']=="false"]
    df['location.display_address'] = df['location.display_address'].apply(lambda l: ', '.join(l))
    df['transactions'] = df['transactions'].apply(lambda l: ', '.join(l))
    df['categories'] = df['categories'].apply(lambda l: ', '.join(l))
    df['price'] = df['price'].fillna('')
    df[['rating','distance']] = df[['rating','distance']].apply(pd.to_numeric, errors='coerce')
    df['rating'] =df['rating'].apply(str).str.replace('000','').apply(float)
    df['distance'] = np.round(0.000621371192*df['distance'].astype('int'), decimals = -2)
    df['name']= df.apply(lambda row: f'<a target="_blank" href="{row["url"]}"> {row["name"]}</a>', axis=1)


    df['image'] = df.apply(lambda row: f'<img src="{row["image_url"]}" width="60"', axis=1)
    # df = df.loc[:,df.columns.isin(['id', 'name', 'image_url', 'is_closed', 'url', 'review_count',
    #     'categories', 'rating', 'transactions', 'price', 'display_phone',
    #     'distance', 'coordinates.latitude', 'coordinates.longitude',
    #     'location.display_address'])]
    step2=1
    
       
    

if step2:
    # Allow the user to sort the data based on any column
    # sort_column = st.selectbox('Sort by column', df.loc[:,df.columns.isin(['name', 'url', 'review_count',
    #     'categories', 'rating', 'transactions', 'price', 'display_phone',
    #     'distance','location.display_address'])].columns)
    df_display = df.loc[:,df.columns.isin(['name', 'image', 'url', 'review_count',
        'categories', 'rating', 'transactions', 'price', 'display_phone',
        'distance','location.display_address'])]
    df_display.columns = ['Name', 'Image', 'Reviews',
        'Type', 'Rating', 'Transactions', 'Price', 'Phone',
        'Distance','Address'] 
    df_display = df_display.reindex(columns= ['Name', 'Image', 'Rating','Reviews','Miles','Price',
        'Type', 'Transactions', 'Phone',
        'Address'] )

    st.write(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.write()
    # create a map centered at the average latitude and longitude of the restaurants
    map = folium.Map(location=[lat, lng], zoom_start=13,  scrollWheelZoom=False)








    folium.Marker( location=[lat, lng], icon=folium.Icon(color='red') , popup="Current Location").add_to(map)

    def get_color(value):
        # Map the value to a color scale from yellow to green
        value = (value - 1.0) / (5.0 - 1.0)
        r = int(255 * (1 - value))
        g = int(255 * value)
        b = 0
        return f'#{r:02x}{g:02x}{b:02x}'

    # add a marker for each restaurant
    def create_marker(row):
        # Create a marker at the latitude and longitude specified in the row
        marker = folium.Marker( location=[row['coordinates.latitude'], 
                                        row['coordinates.longitude']], 
                            popup=f"<a href={row['url']}>{row['name']}</a>",
                            icon=folium.DivIcon(
                                    icon_size=(36,36),
                                    icon_anchor=(18,36),
                                    html='<div style="display: flex; align-items: center;">'
                                        f'<i class="fa fa-map-marker" style="font-size: 30pt;text-shadow: 2px 1px black; color: {get_color(row["rating"])}"></i>'
                                        
                                        f'<div style="font-size: 12pt; font-weight: bold; color: white;text-align:left; margin-left: -22px;text-shadow: 2px 1px black;" > {row["rating"]}</div>'
                                        f'<div style="font-size: 8pt; font-weight: bold; line-height: 1; margin-left: 8px; text-shadow: 2px 1px white;">{row["name"]}</div>'
                                        '</div>'
            
        ))
        return marker.add_to(map)


    df.apply(create_marker, axis=1)

    st_map = folium_static(map, width=700, height=450)

    # Create a restaurant with a dropdown menu
    selected_r = st.selectbox('Select a restaurant', df.name.tolist())

    # Display the selected text
    st.write('Selected restaurant:', selected_r)
    
    step3 = st.button('Details')
step4 =0
if step3:
    st.write('', df[df['name']==selected_r])
    step4 = 1

if step4:
    # Foursquare
    import requests

    query = "LaRocco's Pizza 310 837-8345	"
    location = "Culver City California"
    ll = "34.024858,-118.394843"
    min = 1
    max = 3
    # &open_now=true&near={location}
    url = f"https://api.foursquare.com/v3/places/search?query={query}&ll={ll}&radius=100&min_price={min}&max_price={max}&sort=RELEVANCE&limit=20"

    headers = {
        "accept": "application/json",
        "Authorization": "fsq3FRAOl0xYdG0DAHJpfsoq8kcnDmt3JiiV08t5Cpcyj6g="
    }

    response = requests.get(url, headers=headers)

    #https://location.foursquare.com/developer/reference/place-details

    import json
    import pandas as pd
    df = pd.json_normalize(response.json(), 'results')
    # df = df.sort_values("distance")

    def extract_list(json_obj):
    # flat_df = pd.json_normalize(json_obj)
    # return flat_df['name'].tolist()
    return [json['name'] for json in json_obj]

    # Apply the function to each row in the DataFrame
    df['categories'] = df['categories'].apply(extract_list)
    df
