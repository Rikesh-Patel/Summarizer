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
step4=0
step31=0
selected_r = 0




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



button1 = st.button('Search')

if selected_option:
#     location = geolocator.geocode(selected_option)
    
#     # Get the latitude and longitude from the location object
#     lat = location.latitude
#     lng = location.longitude
    
#     # Display the latitude and longitude in the Streamlit app
#     st.write(f'Latitude: {lat}')
#     st.write(f'Longitude: {lng}')
   


# if st.session_state.get('button') != True:

#     st.session_state['button'] = button1

# if st.session_state['button'] == True:
#     if not selected_option:
#         st.session_state['button'] = False
#         st.write("Nothing in search")
#     if selected_option:

    
#         # Yelp
#         # Get Business ID
#         import requests
#         #&latitude=latitude&longitude=longtiude&radius=radius
#         query = "Pizza"

#         url = f"https://api.yelp.com/v3/businesses/search?latitude={lat}&longitude={lng}&term={query}&categories=&sort_by=best_match&limit=20"

#         headers = {
#             "accept": "application/json",
#             "Authorization": "Bearer A_1nx-eEZxP4IQ-fT7r32jAHjBIU1gQqNzMM5hkc-XGtQFSbeRJr5FNXyXxVBsEA7z5r47W_7rGMK6-hc2OkoUJE_bpgtZ4Oq2zndySrIjBCvS0kH2EWcmxSyx2fY3Yx"
#         }

#         response = requests.get(url, headers=headers)
#         if 'error' in response.json():
#             print(response.json()['error']['code'])
        
#         import json
#         import pandas as pd
#         df = pd.json_normalize(response.json(), 'businesses')
#         df = df.sort_values("distance")
#         def extract_list(json_obj):
#             return [json['title'] for json in json_obj]

#         # Apply the function to each row in the DataFrame
#         df['categories'] = df['categories'].apply(extract_list)
#         # df = df[df['is_closed']=="false"]
#         df['location.display_address'] = df['location.display_address'].apply(lambda l: ', '.join(l))
#         df['transactions'] = df['transactions'].apply(lambda l: ', '.join(l))
#         df['categories'] = df['categories'].apply(lambda l: ', '.join(l))
#         df['price'] = df['price'].fillna('')
#         df[['rating','distance']] = df[['rating','distance']].apply(pd.to_numeric, errors='coerce')
#         df['rating'] =df['rating'].apply(str).str.replace('000','').apply(float)
#         df_display = df.copy()
#         df_display['distance'] = np.round(0.000621371192*df_display['distance'].astype(float), decimals = 2)
#         df_display['name']= df_display.apply(lambda row: f'<a target="_blank" href="{row["url"]}"> {row["name"]}</a>', axis=1)


#         df_display['image'] = df_display.apply(lambda row: f'<img src="{row["image_url"]}" width="60"', axis=1)
#         # df = df.loc[:,df.columns.isin(['id', 'name', 'image_url', 'is_closed', 'url', 'review_count',
#         #     'categories', 'rating', 'transactions', 'price', 'display_phone',
#         #     'distance', 'coordinates.latitude', 'coordinates.longitude',
#         #     'location.display_address'])]
#         step2=1
        
        
        

#     if step2:
#         # Allow the user to sort the data based on any column
#         # sort_column = st.selectbox('Sort by column', df.loc[:,df.columns.isin(['name', 'url', 'review_count',
#         #     'categories', 'rating', 'transactions', 'price', 'display_phone',
#         #     'distance','location.display_address'])].columns)
#         df_display = df_display.loc[:,df_display.columns.isin(['name', 'image', 'review_count','categories', 'rating', 'transactions', 'price', 'display_phone','distance','location.display_address'])]
#         df_display = df_display[['name', 'image', 'review_count','categories', 'rating', 'transactions', 'price', 'display_phone','distance','location.display_address'] ]
#         df_display.columns =    ['Name', 'Image', 'Reviews','Type', 'Rating', 'Transactions', 'Price', 'Phone','Miles','Address'] 
#         st.write(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
#         st.write('')
#         # create a map centered at the average latitude and longitude of the restaurants
#         map = folium.Map(location=[lat, lng], zoom_start=13,  scrollWheelZoom=False)








#         folium.Marker( location=[lat, lng], icon=folium.Icon(color='red') , popup="Current Location").add_to(map)

#         def get_color(value):
#             # Map the value to a color scale from yellow to green
#             value = (value - 1.0) / (5.0 - 1.0)
#             r = int(255 * (1 - value))
#             g = int(255 * value)
#             b = 0
#             return f'#{r:02x}{g:02x}{b:02x}'

#         # add a marker for each restaurant
#         def create_marker(row):
#             # Create a marker at the latitude and longitude specified in the row
#             marker = folium.Marker( location=[row['coordinates.latitude'], 
#                                             row['coordinates.longitude']], 
#                                 popup=f"<a href={row['url']}>{row['name']}</a>",
#                                 icon=folium.DivIcon(
#                                         icon_size=(36,36),
#                                         icon_anchor=(18,36),
#                                         html='<div style="display: flex; align-items: center;">'
#                                             f'<i class="fa fa-map-marker" style="font-size: 30pt;text-shadow: 2px 1px black; color: {get_color(row["rating"])}"></i>'
                                            
#                                             f'<div style="font-size: 12pt; font-weight: bold; color: white;text-align:left; margin-left: -22px;text-shadow: 2px 1px black;" > {row["rating"]}</div>'
#                                             f'<div style="font-size: 8pt; font-weight: bold; line-height: 1; margin-left: 8px; text-shadow: 2px 1px white;">{row["name"]}</div>'
#                                             '</div>'
                
#             ))
#             return marker.add_to(map)


#         df.apply(create_marker, axis=1)

#         st_map = folium_static(map, width=700, height=450)

#         # Create a restaurant with a dropdown menu
#         selected_r = st.selectbox('Select a restaurant', df.name.tolist())
#         # Display the selected text
#         st.write('Selected restaurant:', selected_r)
        
       


        if st.button('Details'):

            # selected = df[df['name']==selected_r]
            selected_r = "Frankie's Downtown"
            # st.dataframe(selected)

            # Foursquare
            import requests
            ll = "32.7805,-96.8009"
            # ll = f"{selected.iloc[0]['coordinates.latitude']},{selected.iloc[0]['coordinates.longitude']}"
            url = f"https://api.foursquare.com/v3/places/search?query={selected_r}&ll={ll}&radius=200&sort=RELEVANCE&limit=1"
            headers = {
                "accept": "application/json",
                "Authorization": "fsq3FRAOl0xYdG0DAHJpfsoq8kcnDmt3JiiV08t5Cpcyj6g="
            }

            response = requests.get(url, headers=headers)
            df_fsq = pd.json_normalize(response.json(), 'results')

            def extract_list(json_obj):
                return [json['name'] for json in json_obj]

            # Apply the function to each row in the DataFrame
            df_fsq['categories'] = df_fsq['categories'].apply(extract_list)
            st.write(df_fsq)

            def id_reviews(id):
                url = f"https://api.foursquare.com/v3/places/{id}/tips?limit=50"

                headers = {
                    "accept": "application/json",
                    "Authorization": "fsq3FRAOl0xYdG0DAHJpfsoq8kcnDmt3JiiV08t5Cpcyj6g="
                }

                response = requests.get(url, headers=headers)
                return [json['text'] for json in response.json()]
  
            texts = id_reviews(df_fsq.iloc[0]['fsq_id'])

            corpus = '  \n'.join(texts)
            reviews = pd.DataFrame(texts, columns=['text'])

            from textblob import TextBlob

            # Define a function that classifies the sentiment of a review as positive, negative, or neutral
            def classify_sentiment(review):
                # Use TextBlob to classify the sentiment of the review
                sentiment = TextBlob(review).sentiment
                # Classify the sentiment as positive, negative, or neutral based on the polarity
                if sentiment.polarity > 0:
                    return 'positive'
                elif sentiment.polarity < 0:
                    return 'negative'
                else:
                    return 'neutral'
            # nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            # reviews['sentiment'] =
            reviews['sentiment'] = reviews['text'].apply(classify_sentiment)
            reviews
            
            #Summarized Tips
            # !pip3 install pysummarization
            # from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
            # from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
            # from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor

            # # Object of automatic summarization.
            # auto_abstractor = AutoAbstractor()
            # # Set tokenizer.
            # auto_abstractor.tokenizable_doc = SimpleTokenizer()
            # # Set delimiter for making a list of sentence.
            # auto_abstractor.delimiter_list = ['\n']

            # # Object of abstracting and filtering document.
            # abstractable_doc = TopNRankAbstractor()
            # auto_abstractor.set_top_sentences(4)
            # # abstractable_doc.num_of_sentences = 4
            # # # Summarize document
            # result_dict = auto_abstractor.summarize(corpus, abstractable_doc)

            # # Get top 3 after sorting the list of scoring_data tuples in descending order by the second element of the tuple (the scoring_data value)
            # sorted_scoring_data = sorted(result_dict['scoring_data'], key=lambda x: x[1], reverse=True)
            # st.write(sorted_scoring_data)
            # # Get the indices of the top 3 summary results
            # top_3_indices = [tuple[0] for tuple in sorted_scoring_data]
            # st.write(top_3_indices)
            # st.write(result_dict['summarize_result'])
            # # Get the top 3 summary results
            # top_3_results = [result_dict['summarize_result'][index] for index in top_3_indices]

            # st.write(' '.join(top_3_results))
            

            from nltk.corpus.reader.reviews import ReviewsCorpusReader
            #Emotion 
            from wordcloud import WordCloud
            import nltk
            import re
            from nltk.corpus import stopwords
            from nltk.stem import SnowballStemmer
            from nltk.stem import WordNetLemmatizer
            from string import punctuation
            # from autocorrect import spell
            from matplotlib import pyplot as plt
            import nltk
            nltk.download('stopwords')
            nltk.download('punkt')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            # !pip3 install nrclex
            import spacy
            from nltk.corpus import stopwords
            from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer 
            # from nltk.stem import PorterStemmer, LancasterStemmer
            # # from sklearn.feature_extraction.text import CountVectorizer
            # from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
            import gensim
            import numpy as np
            import seaborn as sns

            tokenizer = RegexpTokenizer(r'\w+')
            for sentiment in ['positive', 'neutral', 'negative']:
                sentiment_corpus = ' '.join(reviews[reviews['sentiment'] == sentiment]['text'])
                if sentiment_corpus == '':
                    break
                sentiment_text = sentiment_corpus.lower()
                cleaned_text = re.sub('\W', ' ', sentiment_text)
                stopword = stopwords.words("english")
                wnl = WordNetLemmatizer()
                snowball_stemmer = SnowballStemmer("english")
                word_tokens = nltk.word_tokenize(cleaned_text)
                stemmed_word = [wnl.lemmatize(word) if wnl.lemmatize(word).endswith(('e','ous')) else  snowball_stemmer.stem(word) for word in word_tokens]
                processed_text = [word for word in stemmed_word if word not in stopword]
                text_string=(" ").join(processed_text)
                #make word cloud
                wc = WordCloud(colormap='tab20c',max_words=30,margin=10).generate(text_string)
                #applies colors from your image mask into your word cloud
                # fig = plt.figure(figsize=(15,8))
                fig = plt.figure(figsize=(15,8))
                plt.set_title(sentiment)
                fig.axis("off")
                fig.imshow(wc)
                st.pyplot()
                

            st.header("All Reviews")
            st.write(corpus)
            st.session_state['button'] = False
    
# # '''