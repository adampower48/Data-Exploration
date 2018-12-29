# * API calls
import requests
import json
from bs4 import BeautifulSoup
import base64
# * Timekeeping
import time
import datetime
from dateutil.relativedelta import relativedelta as tdelta
# * Sentiment Analysis
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# * Other
from string import punctuation
import pandas as pd


# # General Helpers

def read_json(filename):
    with open(filename) as f:
        return json.loads(f.read())


def file_len(fname):
    with open(fname) as f:
        i = 0
        for i, l in enumerate(f, 1):
            pass
    return i


def sleep_until(dt):
    now = datetime.datetime.today()
    sleep_sec = (dt - now).total_seconds()
    
    if sleep_sec > 0:
        print("Sleeping for", round(sleep_sec), "seconds...")
        time.sleep(sleep_sec)
        print("Done.")
    
    
def sleep_for(time_delta):
    sleep_until(datetime.datetime.today() + time_delta)


def try_except(cmd, args, kwargs, errors, tries=5, timeout=1):
    for _ in range(tries):
        try:
            return cmd(*args, **kwargs)
        except errors as e:
            print("Error:", e)
            time.sleep(timeout)
    
    print("Giving up...")
    return None


# # Twitter Helpers
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
    js = try_except(requests.get, [twitter_search_url], dict(headers=search_headers, params=search_params), 
                    (requests.exceptions.SSLError, requests.exceptions.ConnectionError))
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


# # News Article Helpers
def get_content(url):
    resp = try_except(requests.get, [url], {}, (requests.exceptions.ConnectionError, ))
    if not resp:
        return ""
    
    soup = BeautifulSoup(resp.text, "lxml")
    paragraphs = soup.findAll("p")
    clean_paras = [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
    content = "\n".join(clean_paras)
    
    return content.strip()


def search_articles(query, count, sort_by="publishedAt", from_timestamp=None):
    search_params = {
        'q': query,
        "from": from_timestamp,
        'pageSize': count,
        'sortBy': sort_by,
        "include_entities": True,
        "language": "en",
        "apiKey": newsapi_key,
    }

    
    # Try to get data 5 times before giving up
    js = try_except(requests.get, [newsapi_search_url], dict(params=search_params),
                    (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError))
    if js:
        js = js.json()
    else:
        return pd.DataFrame(columns=["date", "id", "title", "author", "source", "url"])
    
    
    articles = []
    for article in js["articles"]:
        # Extract from json
        source = article["source"]["name"]
        date = article["publishedAt"]
        url = article["url"]
        title = article["title"]
        author = article["author"]
        _id = abs(hash(url))
        
        articles.append([date, _id, title, author, source, url])
        
        
    articles_df = pd.DataFrame(articles, columns=["date", "id", "title", "author", "source", "url"])
    # Parse date
    articles_df["date"] = pd.to_datetime(articles_df["date"], infer_datetime_format=True)


    return articles_df


# # Sentiment
sia = SentimentIntensityAnalyzer()
def get_sentiment(text):
    sent = sia.polarity_scores(text)
    
    _df = pd.Series({
        "sent_neg": sent["neg"],
        "sent_neu": sent["neu"],
        "sent_pos": sent["pos"],
        "sent_comp": sent["compound"],
    })
    
    return _df


to_filter = set(punctuation) | set(stopwords.words("english"))
pstem = PorterStemmer()
def get_topic_freq(text, topic):
    # Parse article
    words = word_tokenize(text)
    filtered_words = [w for w in words if w not in to_filter]
    stemmed_words = [pstem.stem(w) for w in filtered_words]

    fd = FreqDist(stemmed_words)
    
    # Frequency of topic word in article
    topic_freq = fd[pstem.stem(topic)]
    try:
        topic_density = topic_freq / fd.N()
    except ZeroDivisionError:
        topic_density = 0
    
    return pd.Series({"topic_freq": topic_freq, "topic_density": topic_density})


# # I/O
def export_records(df, filename, columns, sort_by="date"):
    with open(filename, 'a') as f:
        df[columns].sort_values(sort_by).to_csv(f, header=False, index=False)


def read_last_record(filename, default_columns):
    len_file = file_len(filename)
    
    # Empty
    if len_file == 0:
        # Fill in headers
        pd.DataFrame(columns=default_columns).to_csv(filename, index=False)
        return read_last_record(filename)
    
    
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


# # Gathering Functions
def gather_new_tweets(topic, last_id, tweets_filename, tweets_per_call=100):
    # get new tweets
    tmp_tweets = search_tweets(topic, tweets_per_call)
    new_tweets = tmp_tweets[tmp_tweets["id"] > last_id]
    print("Found {} new tweets.".format(len(new_tweets)))
    
    # Skip if no new tweets found
    if len(new_tweets) == 0:
        return None
    
     # Sentiment
    new_tweets[['sent_neg', 'sent_neu', 'sent_pos', 'sent_comp']] = new_tweets["text"].apply(get_sentiment)

    # Save tweets
    export_records(new_tweets, tweets_filename, twitter_export_columns)


    print(new_tweets[["date", "id", "sent_comp"]])
    
    # Return latest tweet id
    return new_tweets["id"].max()


def gather_new_articles(topic, _from, articles_filename, articles_per_call=100):
    # get new articles
    tmp_articles = search_articles(topic, articles_per_call)
    new_articles = tmp_articles[tmp_articles["date"] > _from]
    print("Found {} new articles.".format(len(new_articles)))

    # Skip if no new articles found
    if len(new_articles) == 0:
        return None


    # Sentiment
    new_articles["content"] = new_articles["url"].apply(get_content)
    new_articles[['sent_neg', 'sent_neu', 'sent_pos', 'sent_comp']] = new_articles["content"].apply(get_sentiment)
    new_articles[["topic_freq", "topic_density"]] = new_articles["content"].apply(get_topic_freq, args=(topic,))


    # Save articles
    export_records(new_articles, articles_filename, articles_export_columns)


    print(new_articles[["date", "id", "sent_comp"]])

    # Return date of newest article
    return new_articles["date"].max()


# # Main Loop
def start_gathering(topics, filenames, seconds_between_calls=30):
    # Index for topics/filenames:
    # 0: news articles
    # 1: tweets
    
    articles_df = read_last_record(filenames[0], articles_export_columns)
    tweets_df = read_last_record(filenames[1], twitter_export_columns)

    # Publish date of last checked article
    if articles_df is None or len(articles_df) < 1:
        last_article_date = datetime.datetime.today() - tdelta(weeks=1)
    else:
        last_article_date = articles_df.loc[0, "date"]

    # ID of last checked tweet
    if tweets_df is None or len(tweets_df) < 1:
        last_tweet_id = 0
    else:
        last_tweet_id = tweets_df.loc[0, "id"]


    while True:
        now = datetime.datetime.today()

        new_article_date = gather_new_articles(topics[0], last_article_date, filenames[0])
        new_tweet_id = gather_new_tweets(topics[1], last_tweet_id, filenames[1])

        # Update tracking variables if newer records have been found
        if new_article_date is not None:
            last_article_date = new_article_date
        if new_tweet_id is not None:
            last_tweet_id = new_tweet_id


        sleep_until(now + tdelta(seconds=seconds_between_calls))





articles_export_columns = ["date", "id", "sent_neg", "sent_neu", "sent_pos", "sent_comp", "topic_freq", "topic_density"]
twitter_export_columns = ["date", "id", "sent_neg", "sent_neu", "sent_pos", "sent_comp", "retweets", "favorites"]


newsapi_search_url = "http://newsapi.org/v2/everything"
twitter_search_url = "https://api.twitter.com/1.1/search/tweets.json"
auth_url = "https://api.twitter.com/oauth2/token"

# # Credentials
# NewsAPI
newsapi_cred_filename = "newsapi_credentials.json"
newsapi_key = read_json(newsapi_cred_filename)["newsapi_key"]
# Twitter
twitter_cred_filename = "./twitter_credentials.json"
twitter_credentials = read_json(twitter_cred_filename)
access_token = get_bearer_token(twitter_credentials["CONSUMER_KEY"], twitter_credentials["CONSUMER_SECRET"])


# [articles, tweeets]
topics = ["Google", "@Google"]
filenames = ["../Datasets/articles_google_nums.csv", "../Datasets/tweets_google_nums.csv"]

if __name__ == "__main__":
    start_gathering(topics, filenames, 120)

