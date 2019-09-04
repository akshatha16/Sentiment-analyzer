#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import numpy as np
import seaborn as sns
from time import time
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import API
from tweepy import Cursor
import twitter_credentials as tc
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import gensim
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')
import nltk 
import string
import re
from textblob import TextBlob
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_colwidth', 100)
nltk.download('stopwords')
nltk.download('wordnet')


# In[80]:


#client
class TwitterClients:
    def __init__(self,twitter_user=None):
        self.auth=TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client=API(self.auth)
        self.twitter_user=twitter_user
        
    def get_tweets(self,num_tweets):
        msgs = []
        msg =[]
        for tweet in Cursor(api.search, q='#sanitarypads', rpp=100).items(num_tweets):
            msg = [tweet.id, tweet.text]                     
            msgs.append(msg)
        return msgs
        
    def get_friend_list(self,num_friends):
        friend_list=[]
        for friend in Cursor(self.twitter_client.friends).items(num_friends):
            friend_list.append(friend)
            return friend_list
        
    def get_twitter_client_api(self):
        return self.twitter_client
        
        
#authentication
class TwitterAuthenticator():
    def authenticate_twitter_app(self):
        auth = OAuthHandler(tc.CONSUMER_KEY, tc.CONSUMER_SECRET)
        auth.set_access_token(tc.ACCESS_TOKEN, tc.ACCESS_TOKEN_SECRET)
        return auth
    
    
#Class for streaming and processing live tweets.   
class TwitterStreamer():
    def __init__(self):
        self.twitter_autenticator = TwitterAuthenticator()    

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
        # This handles Twitter authetification and the connection to Twitter Streaming API
        listener = TwitterListener(fetched_tweets_filename)
        auth = self.twitter_autenticator.authenticate_twitter_app() 
        stream = Stream(auth, listener)

        # This line filter Twitter Streams to capture data by the keywords: 
        stream.filter(track=hash_tag_list)
        
        
# TWITTER STREAM LISTENER
class TwitterListener(StreamListener):
    #This is a basic listener that just prints received tweets to stdout.
   
    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            print(data)
            with open(self.fetched_tweets_filename, 'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True
    def on_error(self, status):
        if(status==420):
            return false
        print(status)
        
#analysis     
class TweetsAnalyzer():
    def tweets_to_df(self,tweets):
        df=pd.DataFrame(data=[tweet[0] for tweet in tweets],columns=['Id'])
        df["SentimentText"]=np.array([tweet[1] for tweet in tweets])
        return df
    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    def analyze_sentiment(self,tweet):
        analysis=TextBlob(self.clean_tweet(tweet))
        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1

if __name__ == '__main__':
    tweet_analyzer=TweetsAnalyzer()
    twitter_client=TwitterClients()
    tweets=twitter_client.get_tweets(6000)
    df= tweet_analyzer.tweets_to_df(tweets)
    df['sentiment'] = np.array([tweet_analyzer.analyze_sentiment(tweet) for tweet in df['SentimentText']])
    print(df.head())


# In[81]:


def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text
df['Tweet_punct'] = df['SentimentText'].apply(lambda x: remove_punct(x))

def tokenization(text):
    text = re.split('\W+', text)
    return text
df['Tweet_tokenized'] = df['Tweet_punct'].apply(lambda x: tokenization(x.lower()))

stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text    
df['Tweet_nonstop'] = df['Tweet_tokenized'].apply(lambda x: remove_stopwords(x))

ps = nltk.PorterStemmer()
def stemming(text):
    text = [ps.stem(word) for word in text]
    return text
df['Tweet_stemmed'] = df['Tweet_nonstop'].apply(lambda x: stemming(x))

wn = nltk.WordNetLemmatizer()
def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text

df['Tweet_lemmatized'] = df['Tweet_nonstop'].apply(lambda x: lemmatizer(x))
df.head()


# In[82]:


countVectorizer = CountVectorizer(analyzer=clean_text) 
countVector = countVectorizer.fit_transform(df['SentimentText'])
print('{} Number of tweets has {} words'.format(countVector.shape[0], countVector.shape[1]))


# In[83]:


count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names())
count_vect_df.head()
len(count_vect_df.columns)
df.sentiment.value_counts()


# In[76]:


Sentiment_count=df.groupby('sentiment').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['SentimentText'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()


# In[77]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
tf=TfidfVectorizer()
text_tf= tf.fit_transform(df['SentimentText'])
X_train, X_test, y_train, y_test = train_test_split(
    text_tf, df['sentiment'], test_size=0.3, random_state=123)


# In[78]:


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("Predicted Value:", predicted)
cm=confusion_matrix(y_test,predicted)
print(cm)
print(classification_report(y_test,predicted))
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))


# In[101]:


classes = ["Negatives","Neutral","Positives"]
plt.figure(figsize = (10,7))
#sns.heatmap(cm, annot=True)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Sentiment Analysis")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

text_format = 'd'
thresh = cm.max() / 2.
for row, column in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(column, row, format(cm[row, column], text_format),
             horizontalalignment="center",
             color="white" if cm[row, column] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()

plt.show()


# In[ ]:




