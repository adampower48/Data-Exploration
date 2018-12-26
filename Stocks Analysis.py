import requests
import json

import pandas as pd

# # Stock data
def read_json(filename):
    with open(filename) as f:
        return json.loads(f.read())


cred_filename = "./alphavantage_credentials.json"
alphavantage_api_key = read_json(cred_filename)["alphavantage_api_key"]

symbol = "GOOGL"
interval = "1min"

url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={}&interval={}&apikey={}&outputsize=full&datatype=csv".format(
        symbol, interval, alphavantage_api_key)


# Read stock prices
stock_df = pd.read_csv(url)
stock_df["timestamp"] = pd.to_datetime(stock_df["timestamp"], infer_datetime_format=True)
stock_df.set_index("timestamp", inplace=True)
stock_df.sort_index(inplace=True)

stock_df.to_csv("../Datasets/stocks_google.csv")

