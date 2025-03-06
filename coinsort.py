import time
import requests
import pandas as pd
import warnings

from datetime import datetime
import os
from constants import API_KEY

# CoinMarketCap API endpoint
API_URL = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'

# CoinMarketCap API key
API_KEY = API_KEY

def retrieve_top_coins():
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    headers = {
        'X-CMC_PRO_API_KEY': API_KEY,
    }

    parameters = {
        'limit': 20,  # Fetch a larger number of coins to sort and filter later
    }

    try:
        response = requests.get(API_URL, headers=headers, params=parameters)

        if response.status_code == 200:
            data = response.json()
            coins = data['data']
            sorted_coins = sorted(coins, key=lambda x: x['quote']['USD']['percent_change_1h'], reverse=True)

            skip_symbols = ['LEO', 'OKB', 'CRO','TON','USDP','USDT','USDC','WBTC','DAI','BUSD','TUSD']  # List of symbols to skip
            top_coins = []
            for coin in sorted_coins:
                symbol = coin['symbol']
                if symbol not in skip_symbols:
                    name = coin['name']
                    market_cap = coin['quote']['USD']['market_cap']
                    percent_change_1h = coin['quote']['USD']['percent_change_1h']
                    top_coins.append({
                        'Name': name,
                        'Symbol': symbol,
                        'Market Cap': market_cap,
                        'Percent Change 1h': percent_change_1h,
                        'Timestamp': timestamp_str
                    })

            df = pd.DataFrame(top_coins)
            #df.to_csv('/Users/ajay/Sentiment-Analysis bot produciton/test/top20.csv', index=False)
            df_top_5 = df.head()
            #df_top_5.to_csv('/Users/ajay/Sentiment-Analysis bot produciton/test/top5.csv', index=False)
            return df_top_5

        else:
            print('Failed to retrieve top coins. Status Code:', response.status_code)
            print('Response:', response.text)
            # Log the error to a file
            with open('/path/to/error.log', 'a') as f:
                f.write('Failed to retrieve top coins. Status Code: {}\n'.format(response.status_code))
                f.write('Response: {}\n'.format(response.text))

    except requests.exceptions.RequestException as e:
        print('An error occurred during the API request:', e)
        # Log the error to a file
        

retrieve_top_coins()
