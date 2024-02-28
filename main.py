import re
import json
import nltk
import requests
import numpy as np
import pandas as pd
from io import StringIO 
from fastapi import FastAPI
from textblob import TextBlob 
from fuzzywuzzy import process
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize 
from nltk.tokenize import word_tokenize 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from geopy.geocoders import Nominatim
import asyncio

import logging
logging.basicConfig(level=logging.INFO,filename='log.log',filemode='w',format='%(asctime)s %(levelname)s %(message)s')

app = FastAPI()

class Item(BaseModel):
    file: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Specify the origin of your Next.js app
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


file_url_storage = None

# @app.post('/')
# async def root(item:Item):
#     response = requests.get(item.file, allow_redirects=True)
#     if response.status_code == 200:
#         if response.text.strip():
#             # global df
#             df = pd.read_csv(StringIO(response.text), encoding='utf-8')
#             # json_data = peoandmonth(df)
#             # return JSONResponse(json_data)
#             global file_url_storage
#             file_url_storage = item.file
#             # print(df.head())
#         else:
#             print("CSV content is empty.")
#     else:
#         logging.error(f"Failed to get file {response.status_code}",exc_info=True)
#     return item.file

@app.post('/')
async def root(item:Item):
    response = requests.get(item.file, allow_redirects=True)
    if response.status_code == 200:
            global file_url_storage
            file_url_storage = item.file
            logging.info(f"Downloading file link got {item.file}")
    else:
        print(f"Failed to get file link: {response.status_code}")
        logging.error(f"Failed to get file {response.status_code}",exc_info=True)
    return item.file

@app.get('/data/people')
async def get_people():
    response = requests.get(file_url_storage, allow_redirects=True)
    if response.status_code == 200:
        if response.text.strip():
            df = pd.read_csv(StringIO(response.text), encoding='utf-8')
            json_data = peoandmonth(df)
            return JSONResponse(json_data)
        else:
            print("CSV content is empty.")
    else:
        logging.error(f"Failed to get file {response.status_code}",exc_info=True)


@app.get('/data/other')
async def get_other():
    response = requests.get(file_url_storage, allow_redirects=True)
    if response.status_code == 200:
        if response.text.strip():
            df = pd.read_csv(StringIO(response.text), encoding='utf-8')
            print(df.head())
        else:
            print("CSV content is empty.")
    else:
        logging.error(f"Failed to get file {response.status_code}",exc_info=True)


def convert_to_json(data):
    data = data.to_dict(orient='records')
    json_data = json.dumps(data, indent=4)
    return json_data

def clean_text(text):
    if isinstance(text, str):
        text=re.sub(r'[^a-zA-Z0-9-\s]','',text)
        text=text.lower()
        return text
    else:
        return text


def clean_text1(text):
    if isinstance(text, str):
        text=re.sub(r'[^a-zA-Z-\s]','',text)
        return text
    else:
        return text
    
def peoandmonth(df):
    df1=df.copy()
    df1=df1.fillna('-')
    df1['Date'].replace('07.04,22','07.04.22',inplace=True)
    df1=df1[df1[ 'Date']!='-']
    df1[ 'Date']= pd.to_datetime(df1[ 'Date'] , format='%d.%m.%y')
    df1[ 'Designation'] = df1[ 'Designation'].str.strip().replace('Co-founder', 'Co-Founder')
    df1[ 'Designation'].value_counts().head(50)
    df1[ 'Designation']=df1[ 'Designation'].apply(clean_text)
    df1[ 'Place']=df1[ 'Place'].apply(clean_text)
    df1[ 'Areas of interest']=df1[ 'Areas of interest'].apply(clean_text)
    df1[ 'Insights']=df1[ 'Insights'].apply(clean_text)
    df1[ 'Platform']=df1[ 'Platform'].apply(clean_text)
    df1[ 'Insights']=df1[ 'Insights'].apply(clean_text1)
    df1[ 'Platform'].value_counts()
    df1=df1[df1['Platform'] != '-']
    df1['Month'] = df1[ 'Date'].dt.strftime('%B')
    df1['year'] = df1[ 'Date'].dt.year
    month_analysis=df1.groupby(['Month']).count()['Name'].reset_index()
    month_analysis.rename(columns={'Name':'Number of people'},inplace=True)
    return convert_to_json(month_analysis)


@app.get('/api/data')
async def get_data():
    data = {
        'key1': 'value1',
        'key2': 'value2',
    }
    return data

@app.get('/data/search')
async def get_search(item: str):
    response = requests.get(file_url_storage, allow_redirects=True)
    if response.status_code == 200:
        if response.text.strip():
            df = pd.read_csv(StringIO(response.text), encoding='utf-8')
            search_results = search_data(df, item)
            return search_results
        else:
            print("CSV content is empty.")
    else:
        logging.error(f"Failed to get file {response.status_code}",exc_info=True)

@app.get('/data/map')
async def get_map():
    response = requests.get(file_url_storage, allow_redirects=True)
    if response.status_code == 200:
        if response.text.strip():
            df1 = pd.read_csv(StringIO(response.text), encoding='utf-8')
            json_data=mapgetter(df1)
            return json_data
        else:
            print("CSV content is empty.")
    else:
        logging.error(f"Failed to get file {response.status_code}",exc_info=True)
    

@app.get('/data/year')
async def get_year():
    response = requests.get(file_url_storage, allow_redirects=True)
    if response.status_code == 200:
        if response.text.strip():
            df1 = pd.read_csv(StringIO(response.text), encoding='utf-8')
            df1=df1.fillna('-')
            df1[df1['Date'].str.contains(r'[^0-9.-]')==True]
            df1['Date'].replace('07.04,22','07.04.22',inplace=True)
            df1=df1[df1[ 'Date']!='-']
            df1[ 'Date']= pd.to_datetime(df1[ 'Date'] , format='%d.%m.%y')
            df1['Month'] = df1[ 'Date'].dt.strftime('%B')
            df1['Date']= pd.to_datetime(df1['Date'] , format='%d.%m.%y')
            df1['year'] = df1[ 'Date'].dt.year
            year_analysis=df1.groupby(['year']).count()['Name'].reset_index()
            year_analysis.rename(columns={'Name':'Number of people'},inplace=True)
            return convert_to_json(year_analysis)
        else:
            print("CSV content is empty.")
    else:
        logging.error(f"Failed to get file {response.status_code}",exc_info=True)



@app.get('/data/senti')
async def get_senti():
    response = requests.get(file_url_storage, allow_redirects=True)
    if response.status_code == 200:
        if response.text.strip():
            df1 = pd.read_csv(StringIO(response.text), encoding='utf-8')
            df1['Sentiment'] = df1['Insights'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
            df1['Sentiment'] = pd.to_numeric(df1['Sentiment'])
            df1['Sentiment']=np.where(df1['Sentiment'] > 0,'Positive',np.where(df1['Sentiment'] < 0,'Negative',np.where(df1['Sentiment'] == 0,'Neutral','Undefined')))
            senti_count=df1.groupby('Sentiment',as_index=True).count()['Name']
            senti_count.reset_index()
            senti_count_df=pd.DataFrame(senti_count)
            senti_count_df.rename(columns={'Name':'Numbers'},inplace=True)
            return convert_to_json(senti_count_df)
        else:
            print("CSV content is empty.")
    else:
        logging.error(f"Failed to get file {response.status_code}",exc_info=True)



@app.get('/data/inter')
async def get_interaction():
    response = requests.get(file_url_storage, allow_redirects=True)
    if response.status_code == 200:
        if response.text.strip():
            df1 = pd.read_csv(StringIO(response.text), encoding='utf-8')
            df1['Platform'].replace('lunchclub','lunch club',inplace=True)
            df1['Platform'].replace('suhas ref','suhas',inplace=True)
            platform_list=df1.groupby(['Platform']).count()['Name'].reset_index()
            platform_list.rename(columns={'Name':'Number of people'},inplace=True)
            return convert_to_json(platform_list)
        else:
            print("CSV content is empty.")
    else:
        logging.error(f"Failed to get file {response.status_code}",exc_info=True)

@app.get('/data/table')
async def get_table():
    response = requests.get(file_url_storage, allow_redirects=True)
    if response.status_code == 200:
        if response.text.strip():
            df1 = pd.read_csv(StringIO(response.text), encoding='utf-8')
            top_10_clients = df1.head(10)[[ 'Date', 'Company', 'Place', 'Platform']]
            top_10_clients_reset = top_10_clients.reset_index(drop=True)
            top_10_clients_reset.index += 1
            return convert_to_json(top_10_clients_reset)
        else:
            print("CSV content is empty.")
    else:
        logging.error(f"Failed to get file {response.status_code}",exc_info=True)

@app.get('/data/table2')
async def get_table2():
    response = requests.get(file_url_storage, allow_redirects=True)
    if response.status_code == 200:
        if response.text.strip():
            df1 = pd.read_csv(StringIO(response.text), encoding='utf-8')
            df1_sorted = df1.sort_values(by='Date', ascending=False)
            top_clients = df1_sorted[['Date', 'Company', 'Name', 'Learnings']]
            top_clients[ 'Learnings'] = top_clients[ 'Learnings'].replace('-', 'No insights available')
            top_clients_reset = top_clients.reset_index(drop=True)
            top_clients_reset.index += 1
            return convert_to_json(top_clients_reset)
        else:
            print("CSV content is empty.")
    else:
        logging.error(f"Failed to get file {response.status_code}",exc_info=True)


@app.get('/data/desig')
async def get_designation():
    response = requests.get(file_url_storage, allow_redirects=True)
    if response.status_code == 200:
        if response.text.strip():
            df1 = pd.read_csv(StringIO(response.text), encoding='utf-8')
            stop_words = ['and', 'the', 'in', 'on', 'a', 'is', 'there', 'of', 'head']
            def remove_stop_word(text):
                if isinstance(text, str):
                    words = text.split()
                    filter_words = [word for word in words if word.lower() not in stop_words]
                    return ' '.join(filter_words)
                else:
                    return str(text)
            df1['Designation'] = df1['Designation'].apply(remove_stop_word)
            word_counts = df1['Designation'].str.split(expand=True).stack().value_counts()
            freq = pd.DataFrame({'Designation': word_counts.index, 'Count': word_counts.values})
            freq = freq.sort_values(by='Count', ascending=False)
            freq=freq.head(10)
            return convert_to_json(freq)
        else:
            print("CSV content is empty.")
    else:
        logging.error(f"Failed to get file {response.status_code}",exc_info=True)


@app.get('/data/recent')
async def get_recent():
    response = requests.get(file_url_storage, allow_redirects=True)
    if response.status_code == 200:
        if response.text.strip():
            df = pd.read_csv(StringIO(response.text), encoding='utf-8')
            df['last_interacted'] = pd.to_datetime(df['Date'])
            df_sorted = df.sort_values(by='last_interacted', ascending=False)
            top_10 = df_sorted.head(10)
            json_data = df.to_json(orient='records')
            return json_data
        else:
            print("CSV content is empty.")
    else:
        logging.error(f"Failed to get file {response.status_code}",exc_info=True)

def search_data(df1, search_keyword):
    def search_data1(search_keyword):
        df1['Areas of interest'] = df1['Areas of interest'].fillna('')
        filtered_data = df1[df1['Areas of interest'].str.contains(search_keyword, case=False)]
        return filtered_data
    df1=search_data1(search_keyword)
    df_search = df1[
    (df1['Name'].str.contains(search_keyword, case=False)) |
    (df1['Company'].str.contains(search_keyword, case=False)) |
    (df1['Designation'].str.contains(search_keyword, case=False)) |
    (df1['Areas of interest'].str.contains(search_keyword, case=False))
]
    columns_to_display = ['Name', 'Date', 'Designation', 'Company', 'Areas of interest','Linkedin']
    search_results = df_search[columns_to_display]
    return search_results


async def mapgetter(df1):
    async def get_coordinates_async(place, geolocator):
        location = await geolocator.geocode(place)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None

    async def process_row_async(row, geolocator):
        place_data = await get_coordinates_async(row["Place"], geolocator)
        return place_data

    async def main_async(df):
        async with aiohttp.ClientSession() as session:
            geolocator = Nominatim(user_agent="geo_visualization", timeout=10, session=session)
            tasks = [process_row_async(row, geolocator) for index, row in df.iterrows()]
            places_data = await asyncio.gather(*tasks)
            df[['Latitude', 'Longitude']] = pd.DataFrame(places_data, columns=['Latitude', 'Longitude'])
            df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
            return df

    def convert_df_coordinates(df):
       place_mapping = {
        'bangalore': 'Bangalore',
        'whitefield bengaluru': 'Bangalore',
        'mangalore bengaluru': 'Bangalore',
        'bombay': 'Mumbai',
        'malad mumbai': 'Mumbai',
        'andheri mumbai': 'Mumbai',
        'borivali mumbai': 'Mumbai',
        'govandimumbai': 'Mumbai',
        'new delhi': 'Delhi',
        'bengaluru but originally from hyderabad': 'Hyderabad',
        'hyderabad orissa': 'Hyderabad',
        'pune mumbai': 'Pune',
        'vadodara gujarat': 'Gujarat',
        'ahmedabad': 'Gujarat',
      }

       df['Place'] = df['Place'].str.lower().map(place_mapping).fillna(df['Place'])
       return df

    def fetch_coordinates(df):
        result_df = asyncio.run(main_async(df))
        return result_df

    df1 = pd.DataFrame({'Place': ['Bangalore', 'Mumbai', 'New Delhi', 'Hyderabad']})
    df1 = convert_df_coordinates(df1)
    result_df = fetch_coordinates(df1)
    return result_df
