# Utils

# Import necessary libraries
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import unicodedata

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to clean text (lowercasing, remove link, remove punctuation, mentions, hashtags, digits, words with length < 2)
def clean_text(text):
    text = text.lower()  # Lowercase
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
    text = re.sub(r'http\S+', '', text)           # Remove links
    text = re.sub(r'[^\w\s]', '', text)           # Remove punctuation
    text = re.sub(r'@\w+', '', text)              # Remove mentions
    text = re.sub(r'#\w+', '', text)              # Remove hashtags
    text = re.sub(r'\d+', '', text)               # Remove digits
    text = re.sub(r'\s+', ' ', text)              # Collapse multiple whitespace
    text = text.strip()                           # Trim leading/trailing spaces
    text = ' '.join([word for word in text.split() if len(word) > 1])  # Remove single character words
    return text

# POS function
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'


# Function to tokenize and remove stepwords
def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in tagged_tokens]
    tokens = [word for word in tokens if word not in stop_words]

    return tokens
