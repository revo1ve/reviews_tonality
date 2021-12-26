import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
import pickle
import re

import pymorphy2 as pm

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

stop_words = stopwords.words("english")
stop_words.extend(stopwords.words("russian"))
stop_words = set(stop_words)

morph = pm.MorphAnalyzer()

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

stop_words = set(stopwords.words('russian'))
def mytokenize(text, stop_words=stop_words):
    text = remove_html(text)
    text = remove_URL(text)
    text = text.lower()
    text = word_tokenize(text)
    text = [word for word in text if word.isalpha()]
    text = [word for word in text if not word in stop_words]
    text = [morph.normal_forms(word)[0] for word in text]
    
    return text

def prepare_data(data):
    data['Tokens'] = data.Text.apply(mytokenize)
    data['joined_tokens'] = data.Tokens.apply(lambda x: ' '.join(x))

    return data
