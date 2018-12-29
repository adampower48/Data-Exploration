import requests
import json
import base64
import time
import datetime
from dateutil.relativedelta import relativedelta as tdelta

import pandas as pd
import numpy as np

from nltk.sentiment.vader import SentimentIntensityAnalyzer


def read_json(filename):
    with open(filename) as f:
        return json.loads(f.read())

        
def file_len(fname):
    with open(fname) as f:
        i = 0
        for i, l in enumerate(f, 1):
            pass
    return i
    

def try_except(cmd, args, kwargs, errors, tries=5, timeout=1):
    for _ in range(tries):
        try:
            return cmd(*args, **kwargs)
        except errors as e:
            print("Error:", e)
            time.sleep(timeout)
    
    print("Giving up...")
    return None

    
# # Authentication
def get_bearer_token(consumer_key, consumer_secret):
    # Not sure why this needs all the encoding.
    key_secret = "{}:{}".format(consumer_key, consumer_secret)
    b64_encoded_key = base64.b64encode(key_secret.encode("ascii")).decode("ascii")

    headers={
        "Authorization": 'Basic {}'.format(b64_encoded_key), 
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
    }

    resp = requests.post(auth_url, headers=headers, data={'grant_type': 'client_credentials'})
    token = resp.json()["access_token"]
    
    return token


# # Get Tweets
def search_tweets(query, count, result_type="recent", filter_retweets=True, filter_replies=True):
    search_headers = {
        'Authorization': 'Bearer {}'.format(access_token)    
    }

    search_params = {
        'q': (query 
                + " AND -filter:retweets" * filter_retweets
                + " AND -filter:replies" * filter_replies
             ),
        'count': count,
        'result_type': result_type,
        "include_entities": True,
        "lang": "en",
        "tweet_mode": "extended",
    }


    
    # Try to get data 5 times before giving up
    js = try_except(requests.get, [search_url], dict(headers=search_headers, params=search_params), (requests.exceptions.SSLError, requests.exceptions.ConnectionError))
    if js:
        js = js.json()["statuses"]
    else:
        return pd.DataFrame(columns=["date", "id", "lang", "retweets", "favorites", "text", "user", "url"])
    
    
    tweets = pd.DataFrame(js, columns=["created_at", "id", "lang", "retweet_count", "favorite_count"])
    tweets["text"] = [t["retweeted_status"]["full_text"] if "retweeted_status" in t else t["full_text"] for t in js]
    tweets["user"] = [t["user"]["screen_name"] for t in js]
    tweets["created_at"] = pd.to_datetime(tweets["created_at"], infer_datetime_format=True)
    tweets["url"] = "https://twitter.com/i/web/status/" + tweets["id"].astype(str)
    
    tweets.rename(columns={"created_at": "date", "retweet_count": "retweets", "favorite_count": "favorites"}, inplace=True)
    
    return tweets


# # Sentiment
def get_sentiment(text):
    sent = sia.polarity_scores(text)
    
    _df = pd.Series({
        "sent_neg": sent["neg"],
        "sent_neu": sent["neu"],
        "sent_pos": sent["pos"],
        "sent_comp": sent["compound"],
    })
    
    return _df


# # Automation
def sleep_until(dt):
    now = datetime.datetime.today()
    sleep_sec = (dt - now).total_seconds()
    
    if sleep_sec > 0:
        print("Sleeping for", round(sleep_sec), "seconds...")
        time.sleep(sleep_sec)
        print("Done.")
    
    
def sleep_for(time_delta):
    sleep_until(datetime.datetime.today() + time_delta)

    
def export_tweets(df, filename):
    with open(filename, 'a') as f:
        df[export_columns].sort_values("date").to_csv(f, header=False, index=False)
        
        
def read_last_tweet(filename):
    len_file = file_len(filename)
    
    # Empty
    if len_file == 0:
        # Fill in headers
        pd.DataFrame(columns=export_columns).to_csv(filename, index=False)
        return read_last_tweet(filename)
    
    
    # Get headers
    headers_df = pd.read_csv(filename, header=0, nrows=0, parse_dates=True, infer_datetime_format=True)
    
    # Only headers
    if len_file == 1:
        return headers_df
    
    
    # Read last line
    lines_df = pd.read_csv(filename, header=None, skiprows=len_file - 1,
                           parse_dates=True, infer_datetime_format=True)
    lines_df.columns = headers_df.columns
    lines_df.index.name = headers_df.index.name
    
    
    return pd.concat([headers_df, lines_df])
    
    
def harvest_tweets(topic, tweets_filename, seconds_between_calls):
    tweets_per_call = 100


    tweets_df = read_last_tweet(tweets_filename)
    if tweets_df is None or len(tweets_df) < 1:
        last_id = 0
    else:
        last_id = tweets_df.loc[0, "id"]


    last_checked = None

    while True:
        now = datetime.datetime.today()
        
        # get new tweets
        tmp_tweets = search_tweets(topic, tweets_per_call)
        new_tweets = tmp_tweets[tmp_tweets["id"] > last_id]
        print("Found {} new tweets.".format(len(new_tweets)))
        
        # Skip if no new tweets found
        if len(new_tweets) == 0:
            sleep_until(now + tdelta(seconds=seconds_between_calls))
            continue
            
        
        # Sentiment
        new_tweets[['sent_neg', 'sent_neu', 'sent_pos', 'sent_comp']] = new_tweets["text"].apply(get_sentiment)
        
        
        # Save tweets
        export_tweets(new_tweets, tweets_filename)

        
        print(new_tweets[["date", "id", "sent_comp"]])
        
        
        last_checked = now
        last_id = new_tweets["id"].max()
        sleep_until(now + tdelta(seconds=seconds_between_calls))
        

cred_filename = "./twitter_credentials.json"
credentials = read_json(cred_filename)
search_url = "https://api.twitter.com/1.1/search/tweets.json"
auth_url = "https://api.twitter.com/oauth2/token"
access_token = get_bearer_token(credentials["CONSUMER_KEY"], credentials["CONSUMER_SECRET"])

sia = SentimentIntensityAnalyzer()
export_columns = ["date", "id", "sent_neg", "sent_neu", "sent_pos", "sent_comp", "retweets", "favorites"]
        

if __name__ == "__main__":
    harvest_tweets("@Google", "../Datasets/tweets_google_nums.csv", 30)



