import re
import json
import requests
import numpy as np
import pandas as pd
import os
from io import StringIO 
# from openai import OpenAI
from datetime import datetime
from fastapi import FastAPI
from textblob import TextBlob 
from collections import Counter
from pydantic import BaseModel
from nltk.tokenize import sent_tokenize 
from nltk.tokenize import word_tokenize 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


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
            # json_data = peoandmonth(df)
            # return JSONResponse(json_data)
        else:
            print("CSV content is empty.")
    else:
        logging.error(f"Failed to get file {response.status_code}",exc_info=True)


@app.get('/data/other')
async def get_other():
    response = requests.get(file_url_storage, allow_redirects=True)
    if response.status_code == 200:
        if response.text.strip():
            df1 = pd.read_csv(StringIO(response.text), encoding='utf-8')
            data = month_and_year_counter(df1)
            return data
        else:
            print("CSV content is empty.")
    else:
        logging.error(f"Failed to get file {response.status_code}",exc_info=True)


def month_and_year_counter(df1):
    df1 = preprocess_column_headings(df1)
    column_headings = df1.columns.tolist()
    date_coloumn_lst = df1['date'].tolist()
    dates_parsed = [datetime.strptime(date, '%d.%m.%y') for date in date_coloumn_lst]
    months_yearly = Counter(date.strftime('%Y-%m') for date in dates_parsed)
    years = Counter(date.strftime('%Y') for date in dates_parsed)
    monthly_counts_combined = Counter(date.strftime('%m') for date in dates_parsed)
    
    month_wise_counter_dict= {}
    for month, count in sorted(months_yearly.items()):
        month_wise_counter_dict.update({month: count})

    year_wise_count_dict = {}
    for year, count in sorted(years.items()):
        year_wise_count_dict.update({year: count})

    month_wise_yearly_dict = {}
    for month in range(1, 13):
        total_count = sum(1 for date in dates_parsed if date.month == month)
        month_wise_yearly_dict.update({str(month).zfill(2): total_count})

    result = {"timeline_count":
                {"month_wise_counter": month_wise_counter_dict,
                "year_wise_counter": year_wise_count_dict,
                "month_wise_yearly_counter": month_wise_yearly_dict}}
    return result

def preprocess_column_headings(df1):
    df1.columns = df1.columns.str.lower().str.replace(' ', '_').str.strip()
    return(df1)


def platformcount(df1):
    df1 = preprocess_column_headings(df1)
    column_headings = df1.columns.tolist()
    platform_column_lst = df1['platform'].tolist()
    platform_count_lst = [str(item).lower().replace(' ', '').strip() for item in platform_column_lst]
    string_counts = Counter(platform_count_lst)
    platform_column_dict = dict(string_counts)
    sorted_platform_column_dict = dict(sorted(platform_column_dict.items(), key=lambda item: item[1], reverse=True))
    result = {"platform_count": sorted_platform_column_dict}
    return (result)


def search_keyword(df, keyword):
    if df.empty:
        return "DataFrame is empty."
    df = df.fillna('')
    df = df.astype(str)
    queries_df = df[df.apply(lambda x: x.str.contains(keyword, case=False)).any(axis=1)]
    queries_df = queries_df.dropna()
    if not queries_df.empty:   
        result = queries_df.to_json(orient='records', lines=True)
        return result
    else:
        return "Keyword '{}' not found in any column.".format(keyword)





def displayer(df1):
    df1 = preprocess_column_headings(df1)
    column_headings = df1.columns.tolist()
    serial_number_lst = df1["serial_no"].tolist()
    template_list_dict = {
        "network": {
            "Name": dict(zip(serial_number_lst, df1["name"].tolist())),
            "Date": dict(zip(serial_number_lst, df1["date"].tolist())),
            "Company": dict(zip(serial_number_lst, df1["company"].tolist())),
        }
    }
    overall_list_dict_template = {
        "network": {
            "Designation": dict(zip(serial_number_lst, df1["designation"].tolist())),
            "Areas of Interest": dict(zip(serial_number_lst, df1["areas_of_interest"].tolist())),
            "Insights": dict(zip(serial_number_lst, df1["insights"].tolist())),
            "Place": dict(zip(serial_number_lst, df1["place"].tolist()))
        }
    }
    overall_list_dict = {**overall_list_dict_template, **template_list_dict}
    return overall_list_dict





# def productivity(df1):
#     df1.columns = df1.columns.str.lower().str.replace(' ', '_').str.strip()    
    
#     serial_number_lst = df1["serial_no"].tolist()
#     template_list_dict = {
#         "network": {
#             "Name": dict(zip(serial_number_lst, df1["name"].tolist())),
#             "Date": dict(zip(serial_number_lst, df1["date"].tolist())),
#             "Company": dict(zip(serial_number_lst, df1["company"].tolist())),
#         }
#     }
#     # Overall data with all the parameters
#     overall_list_dict_template = {
#         "network": {
#             "Designation": dict(zip(serial_number_lst, df1["designation"].tolist())),
#             "Areas of Interest": dict(zip(serial_number_lst, df1["areas_of_interest"].tolist())),
#             "Insights": dict(zip(serial_number_lst, df1["insights"].tolist())),
#             "Place": dict(zip(serial_number_lst, df1["place"].tolist()))
#         }
#     }
#     overall_list_dict = {**overall_list_dict_template, **template_list_dict}
#     feature1_template_dict = {
#         "network": {
#             "Insights": dict(zip(serial_number_lst, df1["insights"].tolist())),
#         }
#     }
#     if "network" in template_list_dict and "network" in feature1_template_dict:
#         feature1_template = template_list_dict.copy()
#         feature1_template["network"].update(feature1_template_dict["network"])
#     else:
#         result = {"status":"fail",
#                 "reason":"both dictionaries must have the 'network' key."}

#     insight_lst_test = df1["insights"].to_list()
#     insight_lst = insight_lst_test[0:5]
#     f1_Prompt_str = '''
#     Here are my learnings after a networking call, I want to improve my communication skills from on my learnings. 
#     Don't give generic advice. Give questions I should have asked. 
#     Please keep the top 3 (not a strict number) important questions if you have sufficient learning data based on the level of high impact on my communication. Keep it short and crisp.
#     If the learning has No data or enough data, Give "Not enough data" as output. But never make Not enough data as a separate point.
#     '''

#     f2_Prompt_str = "Please write a 150 character followup message for text message without hashtags based on the data"

#     counter = 0
#     new_insight_lst = []
#     for i in range(0,len(insight_lst)):
#         if insight_lst[i] == "-" or insight_lst[i].isdigit():
#             no_data = "No data"
#             counter += 1
#             new_insight_lst.append(no_data)
#         else:
#             new_insight_lst.append(insight_lst[i])
#     f1_lst = []
#     f2_lst = []

#     for c in new_insight_lst:
#         session = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[{
#                         "role": "user",
#                         "content": f1_Prompt_str + c
#                     }],
#                     temperature=0.0
#         )
#         response = session.choices[0].message.content
#         f1_lst.append(response)

#         session = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[{
#                         "role": "user",
#                         "content": f2_Prompt_str + response
#                     }],
#             temperature=0.0
#         )
#         response = session.choices[0].message.content
#         f2_lst.append(response)

#     f1_dict = {
#         "network": {
#             "Improve_communication": dict(zip(serial_number_lst, f1_lst)),
#             "Follow_up": dict(zip(serial_number_lst, f2_lst))
#         }
#     }

#     if "network" in feature1_template and "network" in f1_dict:
#         improve_communication = feature1_template.copy()
#         improve_communication["network"].update(f1_dict["network"])
#         follow_up = feature1_template.copy()
#         follow_up["network"].update(f1_dict["network"])
        
#     else:
#         result = {"status":"fail",
#                 "reason":"both dictionaries must have the 'network' key."}
    
#     result = {"status":"pass",
#                 "response":improve_communication}
    
#     return (result)




# def openaii(df):
#     client = OpenAI(
#         api_key="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",
#     )
#     insight_lst = df["Insights"].to_list()
#     serial_number = (df.iloc[:, 0]).to_list()
#     f1_Prompt_str = '''
#     Here are my learnings after a networking call, I want to improve my communication skills from on my learnings. 
#     Don't give generic advice. Give questions I should have asked. 
#     Please keep the top 3 important questions based on the level of high impact on my communication. Keep it short and crisp.
#     If the learning has No data, Give one "No data" as output
#     '''
#     counter = 0
#     new_insight_lst = []
#     for i in range(0,len(insight_lst)):
#         if insight_lst[i] == "-" or insight_lst[i].isdigit():
#             no_data = "No data"
#             counter += 1
#             new_insight_lst.append(no_data)
#         else:
#             new_insight_lst.append(insight_lst[i])

#     f1_lst = []
#     for c in new_insight_lst:
#         session = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[{
#                         "role": "user",
#                         "content": f1_Prompt_str + c
#                     }]
#         )
#         response = session.choices[0].message.content
#         f1_lst.append(response)
#     insight_result_dict = dict(zip(serial_number, insight_lst))
#     improve_communication_result_dict = dict(zip(serial_number, f1_lst))
#     insights_dict = { "Insights" : insight_result_dict, "Improve communication": improve_communication_result_dict}
#     json_data = json.dumps(insights_dict)
#     return json_data






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
    


@app.get('/data/search')
async def get_search(item: str):
    response = requests.get(file_url_storage, allow_redirects=True)
    if response.status_code == 200:
        if response.text.strip():
            df = pd.read_csv(StringIO(response.text), encoding='utf-8')
            search_results = search_data(df, item)
            return search_results
        else:
            return {
                    'status_code': 204,
                    'error': 'CSV content is empty.'
                }
    else:
        logging.error(f"Failed to get file {response.status_code}",exc_info=True)
    

@app.get('/data/year')
async def get_year():
    response = requests.get(file_url_storage, allow_redirects=True)
    if response.status_code == 200:
        if response.text.strip():
            df1 = pd.read_csv(StringIO(response.text), encoding='utf-8')
            # data = productivity(df1)
            # return data
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
            data = platformcount(df1)
            return data
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
