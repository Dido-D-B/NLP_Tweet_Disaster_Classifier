import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

# Download required NLTK data 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in tagged_tokens]
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    Tokenizer = tf.keras.preprocessing.text.Tokenizer
    import pickle

    # Load full dataset
    df_full = pd.read_csv("full_data.csv")  # Must contain 'text' and 'target' columns

    # Split into train and test
    X = df_full["text"]
    y = df_full["target"]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create train/test DataFrames with raw text
    train_df = pd.DataFrame({"text": X_train_raw, "target": y_train})
    test_df = pd.DataFrame({"text": X_test_raw, "target": y_test})

    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    # Clean training text and fit tokenizer
    cleaned_texts = [" ".join(tokenize_and_remove_stopwords(text.lower())) for text in X_train_raw]

    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(cleaned_texts)