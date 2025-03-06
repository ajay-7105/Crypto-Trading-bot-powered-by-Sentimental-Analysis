import asyncio
from http.client import HTTPResponse
import time
from urllib import response
from fastapi import FastAPI,BackgroundTasks,Request
from pydantic import basemodel
import praw
from datetime import datetime, timedelta
from langdetect import detect
import pandas as pd
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import torch
import warnings
from py3cw.request import Py3CW
from coinsort import retrieve_top_coins
from constants import update_api, update_secret
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse,StreamingResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")
reddit = praw.Reddit(
    client_id='Q98big5njgjw5erO9U7k5w',
    client_secret='b_PCr27yf273Zm0f7Oo3mWAuIt872w',
    user_agent='SentiBot by /u/Quirky_Raisin6035',
    username='Quirky_Raisin6035',
    password='M$neyM$trix@2023',
    check_for_async=False
)




def scrape_reddit(df):
    posts = []
    all_data = []
    limit_per_symbol = 75

    for index, row in df.iterrows():
        name = row['Name']
        symbol = row['Symbol']
        print(name)
        query = f"{name} OR {symbol}"

        try:
            subreddit = reddit.subreddit(query)
            posts = reddit.subreddit('all').search(query, sort='hot', limit=limit_per_symbol)
        except:
            print(f"Subreddit '{query}' not found. Skipping...")
            continue

        post_data = []
        current_time = datetime.now()

        for post in posts:
            utc_time = datetime.utcfromtimestamp(post.created_utc)
            ist_time = utc_time + timedelta(hours=5, minutes=30)

            # Replace MoreComments instances to retrieve all comments
            post.comments.replace_more(limit=100)
            comments = []
            for comment in post.comments.list():
                comments.append(comment.body)
            comments = ','.join(comments)

            title = post.title + post.selftext

            # Detect language of the title
            try:
                title_lang = detect(title)
            except:
                title_lang = ''

            if title_lang == 'en':
                post_data.append({
                    'name': name,
                    'Symbol': symbol,
                    'Time Created (IST)': ist_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'Title': title,
                    'Comments': comments,
                    'Upvotes': post.ups,
                    'Score': post.score
                })

        all_data.extend(post_data)

    post_df = pd.DataFrame(all_data)
    post_csv = post_df.to_csv('/Users/ajay/Sentiment-Analysis bot produciton/scrapped_tweets/scrape.csv')
    return post_df


def pre_process(reddit_data):
    processed_data = []  # List to store processed data

    for index, row in reddit_data.iterrows():
        title = str(row['Title'])
        symbol = str(row['Symbol'])
        name = str(row['name'])

        # Remove links from the title
        remove_links_title = re.sub(r'http\S+|www\S+', '', title)

        # Remove links from the comments (if applicable)

        comments = str(row['Comments'])
        remove_links_comments = re.sub(r'http\S+|www\S+', '', comments)

        remove_newlines = remove_links_title.replace('\n', '')
        remove_newlines_comments = remove_links_comments.replace('\n', '')
        remove_special_chars = re.sub(r'[$#>*|(){}\[\]\\_]', '', remove_newlines)

        remove_special_chars_comments = re.sub(r'[$#>*|(){}\[\]\\_]', '', remove_newlines_comments)
        row['Title'] = remove_special_chars
        row['Comments'] = remove_special_chars_comments

        symbol_regex = r'(?:^|\s|[$#])' + re.escape(symbol)
        name_regex = r'\b' + re.escape(name) + r'\b'

        if re.search(symbol_regex, remove_special_chars, flags=re.IGNORECASE) or \
                re.search(name_regex, remove_special_chars, flags=re.IGNORECASE):
            processed_data.append(row)

        if re.search(symbol_regex, remove_special_chars_comments, flags=re.IGNORECASE) or \
                re.search(name_regex, remove_special_chars_comments, flags=re.IGNORECASE):
            row['Comments'] = remove_special_chars_comments
            processed_data.append(row)

    processed_df = pd.DataFrame(processed_data)  # Convert processed data to DataFrame
    processed_df.dropna(inplace=True)  # Drop null values
    processed_df.drop_duplicates(subset='Title', inplace=True)
    processed_df.drop_duplicates(subset='Comments', inplace=True)  # Drop duplicates based on both columns
    print(processed_df)
    # processed_csv = processed_df.to_csv('scrapped_tweets/pre_processed_reddit.csv', index=False)
    return processed_df


def predict_sentiment_scores(df):
    try:
        warnings.filterwarnings("ignore")
        symbols = df['Symbol'].unique()
        dfs = {}

        # Iterate over the symbols to create separate dataframes for each symbol
        for symbol in symbols:
            dfs[symbol] = df[df['Symbol'] == symbol]

        dfs_list = list(dfs.values())
        model = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForSequenceClassification.from_pretrained(model)
        tokenizer.model_max_length = 512
        for index, df in enumerate(dfs_list):
            # Calculate sentiment scores for the title column
            encoded_title = tokenizer(list(df['Title']), truncation=True, padding=True, return_tensors='pt')
            title_outputs = model(**encoded_title)
            title_logits = title_outputs.logits
            title_scores = softmax(title_logits.detach().numpy(), axis=1)

            df['Title_Sentiment'] = title_scores[:, 2]
            # Positive sentiment scores for title

            # Calculate sentiment scores for the comments column
            df['Comments'].fillna('', inplace=True)  # Fill empty comments with an empty string
            encoded_comments = tokenizer(list(df['Comments']), truncation=True, padding=True, return_tensors='pt')
            comments_outputs = model(**encoded_comments)
            comments_logits = comments_outputs.logits
            comments_scores = softmax(comments_logits.detach().numpy(), axis=1)

            df['Comments_Sentiment'] = comments_scores[:, 2]  # Positive sentiment scores for comments

            # Calculate total sentiment score as the sum of title and comments scores
            df['Sentiment_Total'] = df['Title_Sentiment'] + df['Comments_Sentiment']

            # Assign sentiment score of 0 if comments are empty
            df.loc[df['Comments'] == '', 'Sentiment_Total'] = 0

        # Perform mean normalization on sentiment scores
        overall_scores = {}

        for df in dfs_list:
            symbol_name = f"{df['Symbol'].iloc[0]}"
            sentiment_score = df['Sentiment_Total'].mean()

            overall_scores[symbol_name] = {
                'Sentiment Score': sentiment_score,
            }

        # Sort by sentiment scores
        sorted_scores = sorted(overall_scores.items(), key=lambda x: x[1]['Sentiment Score'], reverse=True)[:3]

        # Convert to DataFrame
        df_senti = pd.DataFrame(sorted_scores, columns=["Symbol", "Sentiment Score"])
        df_senti["Symbol"] = "USDT_" + df_senti["Symbol"]

        return df_senti 
    except RuntimeError as e:
        print("Error occurred during sentiment score prediction:")
        print(str(e))
        # Handle the error gracefully (e.g., log the error, provide an alternative action, etc.)
        # Optionally, you can return a default or empty value to indicate the failure
        return pd.DataFrame()  # Return an empty DataFrame in case of an error


def update_pairs(df_senti):
    pairs = ','.join(df_senti['Symbol'].apply(str))
    print(pairs)

    # Initialize the Py3CW instance with your API key and secret
    api_key = update_api
    api_secret = update_secret
    p3cw = Py3CW(key=api_key, secret=api_secret)

    bot_id = "11424707"  # Replace with your bot ID

    # Update the bot pairs using the Py3CW wrapper
    response = p3cw.request(
        entity="bots",
        action="update",
        action_id=bot_id,
        payload={
            "name": "Bot1",
            "base_order_volume": 100,
            "take_profit": 2.0,
            "safety_order_volume": 200,
            "max_active_deals": 3,
            "martingale_volume_coefficient": 1.0,
            "martingale_step_coefficient": 1.0,
            "max_safety_orders": 10,
            "active_safety_orders_count": 10,
            "safety_order_step_percentage": 1.0,
            "take_profit_type": "total",
            "strategy_list": "",
            "pairs": pairs,
            "bot_id": "11424707"
        }
    )
    return pairs



is_bot_running = False


#@app.get("/")
#async def read_root():
#    return {"Money Matrix-Bot v1.0 "}

updated_pairs_str = ""
updated_pairs_list = []
updated_pairs_event = asyncio.Event()  # Event to notify the endpoint
websocket_connections = set()
@app.get("/updated-pairs")
async def get_updated_pairs():
    async def stream():
        while True:
            await updated_pairs_event.wait()  # Wait for the event to be set
            yield f"data: {updated_pairs_str}\n\n"
            updated_pairs_event.clear()  # Clear the event after sending

    return StreamingResponse(stream(), media_type="text/event-stream")  # Clear the event after sending

    
@app.post("/start-bot")
def start_bot_endpoint(background_tasks: BackgroundTasks):
    global is_bot_running
    if not is_bot_running:
        is_bot_running = True
        updated_pairs_list.clear()  # Clear the updated_pairs list
        background_tasks.add_task(startbot)
         # Initialize updated_pairs_str with an empty string
        return {"message": "Bot started successfully,", "updated_pairs": updated_pairs_str}
    else:
         # Initialize updated_pairs_str with an empty string
        return {"message": "Bot is already running", "updated_pairs": updated_pairs_str}
async def startbot():
    
    while is_bot_running:
        start_time_retrieve = time.time()

        df_top_5 = retrieve_top_coins()

        print("Scraping posts for Sentiment Analysis")

        post_df = scrape_reddit(df_top_5)

    # Pre-process the DataFrame
        pre_processed = pre_process(post_df)

    # Predict sentiment scores
        senti_Scores = predict_sentiment_scores(pre_processed)
        updated_pairs=update_pairs(senti_Scores)
        updated_pairs_list.append(updated_pairs)
        
        updated_pairs_str = ", ".join(updated_pairs_list)
        updated_pairs_str =+ updated_pairs_str
        updated_pairs_event.set() 
        print("Waiting for the next coin retrieval")
        elapsed_time_retrieve = time.time() - start_time_retrieve

        remaining_time_hour = max(0, 3600 - elapsed_time_retrieve)
        print(remaining_time_hour / 60, "minutes left for the next coin retrieval")
        
        await asyncio.sleep(remaining_time_hour)

@app.post("/stop-bot")
def stop_bot_endpoint():
    global is_bot_running
    if is_bot_running:
        is_bot_running = False
        return {"message": "Bot stopped successfully",}
    else:
        return {"message": "Bot is not running"}
@app.get("/")
async def get_html():
    try:

        html_content = """
    <html>
    <head>
        <title>Money Matrix Bot</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f2f2f2;
            }
            h1 {
                color: #333;
            }
            form {
                margin-top: 20px;
            }
            button {
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        <h1>Money Matrix Bot v1.0</h1>
        <form action="/start-bot" method="post">
            <button type="submit">Start Bot</button>
        </form>
        <form action="/stop-bot" method="post">
            <button type="submit">Stop Bot</button>
        </form>
    </body>
    </html>
    """
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        return {"error": str(e)}
    

app.mount("/static", StaticFiles(directory="static"), name="static")

            
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)



