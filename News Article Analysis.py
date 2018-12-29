# Manipulating
import requests
import json
from bs4 import BeautifulSoup
import time
import datetime
from dateutil.relativedelta import relativedelta as tdelta

import nltk
import pandas as pd

# Sentiment Analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Topic Modelling
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from string import punctuation


def read_json(filename):
    with open(filename) as f:
        return json.loads(f.read())


# # Extract article content from links
def get_content(url):
    resp = try_except(requests.get, [url], [], [requests.exceptions.ConnectionError])
    if not resp:
        return ""
    
    
    soup = BeautifulSoup(resp.text, "lxml")
    paragraphs = soup.findAll("p")
    clean_paras = [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
    content = "\n".join(clean_paras)
    
    return content.strip()

def try_except(cmd, args, kwargs, errors, tries=5, timeout=1):
    for _ in range(tries):
        try:
            return cmd(*args, **kwargs)
        except errors as e:
            print("Error:", e)
            time.sleep(timeout)
    
    print("Giving up...")
    return None


# # Sentiment Analysis
def get_sentiment(text):
    sent = sia.polarity_scores(text)
    
    _df = pd.Series({
        "sent_neg": sent["neg"],
        "sent_neu": sent["neu"],
        "sent_pos": sent["pos"],
        "sent_comp": sent["compound"],
    })
    
    return _df


# # Article Relevance / Topic Modelling
# * Tokenize
# * Remove stopwords
# * Stem words
# * Find frequency of topic word in article
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
    
    
def sleep_until(dt):
    now = datetime.datetime.today()
    sleep_sec = (dt - now).total_seconds()
    
    if sleep_sec > 0:
        print("Sleeping for", round(sleep_sec), "seconds...")
        time.sleep(sleep_sec)
        print("Done.")
    
    
def sleep_for(time_delta):
    sleep_until(datetime.datetime.today() + time_delta)

    
def export_articles(df, filename):
    with open(filename, 'a') as f:
        df[export_columns].sort_values("date").to_csv(f, header=False, index=False)
        
        
def read_last_article(filename):
    len_file = file_len(filename)
    
    # Empty
    if len_file == 0:
        # Fill in headers
        pd.DataFrame(columns=export_columns).to_csv(filename, index=False)
        return read_last_article(filename)
    
    
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
    
    
def file_len(fname):
    with open(fname) as f:
        i = 0
        for i, l in enumerate(f, 1):
            pass
    return i

    
def harvest_articles(topic, articles_filename, seconds_between_calls):
    articles_per_call = 100
    
    
    articles_df = read_last_article(articles_filename)
    if articles_df is None or len(articles_df) < 1:
        last_checked = datetime.datetime.today() - tdelta(weeks=1)
    else:
        last_checked = articles_df.loc[0, "date"]


    while True:
        now = datetime.datetime.today()
        
        # get new articles
        tmp_articles = search_articles(topic, articles_per_call)
        new_articles = tmp_articles[tmp_articles["date"] > last_checked]
        print("Found {} new articles.".format(len(new_articles)))
        
        # Skip if no new articles found
        if len(new_articles) == 0:
            sleep_until(now + tdelta(seconds=seconds_between_calls))
            continue
            
        
        # Sentiment
        new_articles["content"] = new_articles["url"].apply(get_content)
        new_articles[['sent_neg', 'sent_neu', 'sent_pos', 'sent_comp']] = new_articles["content"].apply(get_sentiment)
        new_articles[["topic_freq", "topic_density"]] = new_articles["content"].apply(get_topic_freq, args=(topic,))
        
        
        # Save articles
        export_articles(new_articles, articles_filename)
                
        
        print(new_articles[["date", "id", "sent_comp"]])
        
        
        last_checked = now
        sleep_until(now + tdelta(seconds=seconds_between_calls))
    
    
# Internal
to_filter = set(punctuation) | set(stopwords.words("english"))
pstem = PorterStemmer()
sia = SentimentIntensityAnalyzer()
export_columns = ["date", "id", "sent_neg", "sent_neu", "sent_pos", "sent_comp", "topic_freq", "topic_density"]
    
# NewsAPI Credentials
cred_filename = "newsapi_credentials.json"
newsapi_key = read_json(cred_filename)["newsapi_key"]
search_url = "https://newsapi.org/v2/everything"


if __name__ == "__main__":
    harvest_articles("Google", "../Datasets/articles_google_nums.csv", 100)


