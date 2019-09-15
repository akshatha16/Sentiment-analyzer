#!/usr/bin/env python
# coding: utf-8

# In[110]:


import pandas as pd
import numpy as np
import csv
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
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_colwidth', 100)
nltk.download('stopwords')
nltk.download('wordnet')


# In[111]:


def load_data():
    data = pd.read_csv('C:/twitter/train111.csv',encoding='latin-1')
    return data


# In[112]:


tweet_df = load_data()
tweet_df.head()


# In[113]:


print('Dataset size:',tweet_df.shape)
print('Columns are:',tweet_df.columns)


# In[114]:


tweet_df.info()
sns.countplot(x = 'Sentiment', data = tweet_df)
#visualization

# In[115]:


df  = pd.DataFrame(tweet_df[['ItemID','Sentiment', 'SentimentText']])


# In[116]:


from wordcloud import WordCloud, STOPWORDS , ImageColorGenerator
# Start with one review:
df_pos = tweet_df[tweet_df['Sentiment']==1]
df_neg = tweet_df[tweet_df['Sentiment']==0]
tweet_All = " ".join(review for review in df.SentimentText)
tweet_pos = " ".join(review for review in df_pos.SentimentText)
tweet_neg = " ".join(review for review in df_neg.SentimentText)

fig, ax = plt.subplots(3, 1, figsize  = (30,30))
# Create and generate a word cloud image:
wordcloud_ALL = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_All)
wordcloud_pos = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_pos)
wordcloud_neg = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_neg)

# Display the generated image:
ax[0].imshow(wordcloud_ALL, interpolation='bilinear')
ax[0].set_title('All Tweets', fontsize=30)
ax[0].axis('off')
ax[1].imshow(wordcloud_pos, interpolation='bilinear')
ax[1].set_title('Tweets under positive Class',fontsize=30)
ax[1].axis('off')
ax[2].imshow(wordcloud_neg, interpolation='bilinear')
ax[2].set_title('Tweets under negative Class',fontsize=30)
ax[2].axis('off')


# In[117]:


def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

df['Tweet_punct'] = df['SentimentText'].apply(lambda x: remove_punct(x))
df.head(10)


# In[118]:


def tokenization(text):
    text = re.split('\W+', text)
    return text
df['Tweet_tokenized'] = df['Tweet_punct'].apply(lambda x: tokenization(x.lower()))


# In[119]:


stopword = nltk.corpus.stopwords.words('english')


# In[120]:


def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
    
df['Tweet_nonstop'] = df['Tweet_tokenized'].apply(lambda x: remove_stopwords(x))


# In[121]:


ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

df['Tweet_stemmed'] = df['Tweet_nonstop'].apply(lambda x: stemming(x))


# In[122]:


wn = nltk.WordNetLemmatizer()
df['Tweet_lemmatized'] = df['Tweet_stemmed'].apply(lambda x: ' '.join([wn.lemmatize(word,'v')for word in x]))
df.head()


# In[123]:


countVectorizer = CountVectorizer(analyzer=clean_text) 
countVector = countVectorizer.fit_transform(df['Tweet_lemmatized'])
print('{} Number of tweets has {} words'.format(countVector.shape[0], countVector.shape[1]))

# In[124]:


count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names())
count_vect_df.head()


# In[125]:


len(count_vect_df.columns)
df.Sentiment.value_counts()


# In[126]:


Sentiment_count=df.groupby('Sentiment').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['Tweet_lemmatized'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()


# In[127]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(df['Tweet_lemmatized'])


# In[128]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
text_tf= tf.fit_transform(df['Tweet_lemmatized'])


# In[129]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_tf, df['Sentiment'], test_size=0.3, random_state=123)


# In[130]:


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


# In[131]:


# PLOTTING CONFUSION MATRIX
import itertools
classes = ["Negatives", "Positives"]

plt.figure()
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


# In[132]:


auth = OAuthHandler(tc.CONSUMER_KEY, tc.CONSUMER_SECRET)
auth.set_access_token(tc.ACCESS_TOKEN, tc.ACCESS_TOKEN_SECRET)
api = API(auth,wait_on_rate_limit=True)
msg=[]
msgs=[]
for tweet in Cursor(api.search,q="#sanitarypads",count=100,
                           lang="en",
                           since="2017-04-03").items():
    msg = [tweet.id, tweet.text]                     
    msgs.append(msg)


# In[133]:


test_data=pd.DataFrame(data=[tweet[0] for tweet in msgs],columns=['Id'])
test_data["SentimentText"]=np.array([tweet[1] for tweet in msgs])
test_data.head()


# In[134]:


#CLEANING TEST DATA
def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text
test_data['Tweet_punct'] = test_data['SentimentText'].apply(lambda x: remove_punct(x))

def tokenization(text):
    text = re.split('\W+', text)
    return text
test_data['Tweet_tokenized'] = test_data['Tweet_punct'].apply(lambda x: tokenization(x.lower()))

stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text    
test_data['Tweet_nonstop'] = test_data['Tweet_tokenized'].apply(lambda x: remove_stopwords(x))

ps = nltk.PorterStemmer()
def stemming(text):
    text = [ps.stem(word) for word in text]
    return text
test_data['Tweet_stemmed'] = test_data['Tweet_nonstop'].apply(lambda x: stemming(x))

wn = nltk.WordNetLemmatizer()
test_data['Tweet_lemmatized'] = test_data['Tweet_stemmed'].apply(lambda x: ' '.join([wn.lemmatize(word,'v')for word in x]))
test_data.head()


# In[135]:


## for transforming the whole train data ##
train_tf = tf.fit_transform(df['Tweet_lemmatized'])
## for transforming the test data ##
test_tf = tf.transform(test_data['Tweet_lemmatized'])
## fitting the model on the transformed train data ##
clf.fit(train_tf,df['Sentiment'])
## predicting the results ##
predictions = clf.predict(test_tf)


# In[136]:


final_result = pd.DataFrame({'id':test_data['Id'],'label':predictions})
final_result.to_csv('C:/twitter/output.csv',index=False)


# In[143]:


Sentiment_count=final_result.groupby('label').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['id'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()


# In[ ]:




