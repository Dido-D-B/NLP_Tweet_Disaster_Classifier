import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import random
import os

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("app/model/cnn_disaster_classifier.keras")
    return model

@st.cache_data
def load_tokenizer():
    with open("app/model/tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

model = load_model()
tokenizer = load_tokenizer()

# Preprocessing Helper Block
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
import unicodedata


def _safe_nltk_download(pkg):
    try:
        if pkg == "punkt":
            nltk.data.find("tokenizers/punkt")
        elif pkg == "punkt_tab":
            # Newer NLTK splits punkt models into 'punkt_tab/<lang>'
            nltk.data.find("tokenizers/punkt_tab/english")
        elif pkg == "stopwords":
            nltk.corpus.stopwords.words("english")
        elif pkg == "averaged_perceptron_tagger":
            nltk.data.find("taggers/averaged_perceptron_tagger")
        elif pkg == "averaged_perceptron_tagger_eng":
            # Newer NLTK name for the English perceptron tagger
            nltk.data.find("taggers/averaged_perceptron_tagger_eng")
        elif pkg == "wordnet":
            nltk.data.find("corpora/wordnet")
        return
    except LookupError:
        nltk.download(pkg, quiet=True)

for pkg in ["punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "wordnet"]:
    _safe_nltk_download(pkg)

stop_words = set(stopwords.words("english"))
# Add custom stopwords used during training
custom_stopwords = {
    'im', 'get', 'like', 'go', 'u', 'via', 'one', 'people', 'say', 'dont',
    'make', 'time', 'come', '2', 'see', 'amp', 'news'
}
stop_words.update(custom_stopwords)
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

def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in tagged_tokens]
    tokens = [word for word in tokens if word not in stop_words]

    return tokens

# Text preprocessing function for model input
def preprocess_text(text, tokenizer, max_len=100):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded

# Streamlit interface
st.set_page_config(page_title="Disaster Tweet Classifier", layout="centered", page_icon="🌪️")

# Sidebar header
st.sidebar.header("Disaster Tweet Classifier")
st.sidebar.write("This app classifies tweets as related to real disasters or not.")
st.sidebar.write("Use the sidebar to navigate between different functionalities.")

# collapsible sidebar section
with st.sidebar.expander("How to Use"):
    st.write("""
                    - **Classify a Tweet**: Enter your own tweet for classification.
                    - **Random Tweet Test**: Classify a random tweet from the test dataset.
                    - **EDA & Word Analysis**: Explore word clouds and tweet statistics.
                    - **Model Summary**: View model performance metrics and read more about the classification model.
                 """)
                     
# Page selection
page = st.sidebar.radio("Navigate", [
    "Classify a Tweet",
    "Random Tweet Test",
    "EDA & Word Analysis",
    "Model Summary"
])

if page == "Classify a Tweet":
    st.title("Disaster Tweet Classifier")
    st.write("Enter a tweet below and see whether it's likely about a disaster or not.")
    st.write("Check out the preprocessing steps to understand how the tweet is processed before classification.")

    user_input = st.text_area("✍️ Your Tweet", height=150)

    if st.button("Classify Tweet"):
        if user_input.strip() == "":
            st.warning("Please enter a tweet to classify.")
        else:
            st.subheader("Original Tweet")
            st.write(user_input)

            # Preprocess the input tweet
            cleaned = clean_text(user_input)
            lemmatized_tokens = tokenize_and_remove_stopwords(cleaned)

            # Vectorize once so we can display it and reuse for prediction
            joined_text = " ".join(lemmatized_tokens)
            token_sequence = tokenizer.texts_to_sequences([joined_text])
            processed = pad_sequences(token_sequence, maxlen=100, padding='post', truncating='post')

            with st.expander("🔍 Preprocessing Steps"):
                st.subheader("Cleaned Tweet")
                st.write(cleaned)

                st.subheader("Lemmatized & Stopword-Removed")
                st.write(" ".join(lemmatized_tokens))

                st.subheader("Tokenized Sequence (word tokens)")
                st.write(lemmatized_tokens)

                st.subheader("Token-Index Mapping")
                token_word_map = {
                    word: tokenizer.word_index.get(word, "[OOV]") for word in lemmatized_tokens
                }
                st.write(token_word_map)

                st.subheader("Vectorized Sequence (token IDs)")
                st.write(token_sequence[0])

                st.subheader("Padded Sequence (length = 100)")
                st.write(processed[0].tolist())

            # Use the already computed `processed` for prediction
            prediction = model.predict(processed)[0][0]
            label = "🌪️ Disaster" if prediction >= 0.6 else "✅ Not a Disaster"
            confidence = round(float(prediction if prediction >= 0.6 else 1 - prediction) * 100, 2)
 
            # Display the prediction result
            st.markdown(f"### Prediction: **{label}**")
            st.markdown(f"Confidence: **{confidence}%**")

elif page == "Random Tweet Test":
    st.title("Random Tweet Classifier")
    st.write("This page generates a random tweet from the test dataset and classifies it, showing the preprocessing steps and prediction.")
    st.write("Check out the preprocessing steps to understand how the tweet is processed before classification.")

    @st.cache_data
    def load_test_data():
        candidates = ["app/data/test_sample.csv", "app/data/test.csv"]
        for path in candidates:
            if os.path.exists(path):
                return pd.read_csv(path)
        st.info("No test data found. Add **app/data/test_sample.csv** (preferred) or **app/data/test.csv** to enable the Random Tweet page.")
        return pd.DataFrame(columns=["text", "target"])

    test_df = load_test_data()
    if test_df.empty:
        st.stop()

    if st.button("Generate and Classify Random Tweet"):
        random_row = test_df.sample(1).iloc[0]
        original_text = random_row["text"]

        st.subheader("Original Tweet")
        st.write(original_text)

        # Preprocess the random tweet
        cleaned = clean_text(original_text)
        lemmatized_tokens = tokenize_and_remove_stopwords(cleaned)

        # Vectorize once so we can display it and reuse for prediction
        joined_text = " ".join(lemmatized_tokens)
        token_sequence = tokenizer.texts_to_sequences([joined_text])
        processed = pad_sequences(token_sequence, maxlen=100, padding='post', truncating='post')

        with st.expander("🔍 Preprocessing Steps"):
            st.subheader("Cleaned Tweet")
            st.write(cleaned)

            st.subheader("Lemmatized & Stopword-Removed")
            st.write(" ".join(lemmatized_tokens))

            st.subheader("Tokenized Sequence (word tokens)")
            st.write(lemmatized_tokens)

            st.subheader("Token-Index Mapping")
            token_word_map = {
                word: tokenizer.word_index.get(word, "[OOV]") for word in lemmatized_tokens
            }
            st.write(token_word_map)

            st.subheader("Vectorized Sequence (token IDs)")
            st.write(token_sequence[0])

            st.subheader("Padded Sequence (length = 100)")
            st.write(processed[0].tolist())

        # Use the already computed `processed` for prediction
        prediction = model.predict(processed)[0][0]
        label = "🌪️ Disaster" if prediction >= 0.6 else "👍 Not a Disaster"
        confidence = round(float(prediction if prediction >= 0.6 else 1 - prediction) * 100, 2)

        # Display the prediction result
        st.markdown(f"### Prediction: **{label}**")
        st.markdown(f"Confidence: **{confidence}%**")

        # Compare with true label and show correctness
        true_label = random_row["target"]
        predicted_label = 1 if prediction >= 0.6 else 0
        correctness = "✅ Correct" if predicted_label == true_label else "❌ Incorrect"
        st.markdown(f"### **{correctness}**")

elif page == "EDA & Word Analysis":
    st.title("EDA & Word Analysis")

    st.markdown(
        """
        Explore the dataset's language patterns. Below you'll find combined and class-specific word clouds, the most frequent words per class,
        and a quick look at tweet length distribution.
        """
    )

    # Wordclouds Section
    st.header("Wordclouds")
    st.write("Word clouds visualize the most frequent words in the tweets, both combined and per class (disaster vs non-disaster).")

    st.subheader("All Tweets (Combined)")
    st.image("app/images/wordcloud.png", use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Disaster Tweets")
        st.image("app/images/wordcloud_disaster.png", use_container_width=True)
    with col2:
        st.subheader("Non-Disaster Tweets")
        st.image("app/images/wordcloud_non_disaster.png", use_container_width=True)

    # Word Frequency Analysis Section
    st.header("Word Frequency Analysis")
    st.write("This section shows the most frequent words in disaster and non-disaster tweets.")

    st.subheader("Disaster Words")
    st.image("app/images/top10_disaster_words.png", use_container_width=True)

    st.subheader("Non-Disaster Words")
    st.image("app/images/top10_non_disaster_words.png", use_container_width=True)

    st.header("Target Variable Distribution")
    st.write("This section shows the distribution of the target variable (disaster vs non-disaster) in the dataset.")
    st.image("app/images/dist_target.png", use_container_width=True)

    st.header("Tweet Length Distribution")
    st.write("This section shows the distribution of tweet lengths in the dataset.")
    st.image("app/images/dist_text_length.png", use_container_width=True)

elif page == "Model Summary":
    st.title("Model Summary")
    st.markdown("""
    This app was built as part of a **Natural Language Processing project** to classify tweets as related to real disasters or not.
                
    Multiple classification models were tested, including Logistic Regression and Convolutional Neural Networks (CNNs). The CNN model was selected for deployment due to its superior performance.            

    * **Model**: Convolutional Neural Network (CNN)  
    * **Input Representation**: Tokenized text with Keras Tokenizer and padded sequences  
    * **Training Data**: 10,000 labeled tweets from a public dataset
    * **Metrics**: Accuracy, Precision, Recall, F1-score

    The model was trained on a cleaned version of the tweets using lowercasing, lemmatization, and removal of stopwords and other noise. The dataset was split into training and testing sets, and vectorized using Keras Tokenizer. 
    """)

    # Model Performance Section
    st.header("Accuracy Comparison")
    st.write("This section compares the accuracy of the training and testing datasets.")
    st.image("app/images/train_vs_test_accuracy.png", use_container_width=True)

    # Classification Report Section
    st.header("Classification Report")
    st.write("This section shows the classification report for the model, including precision, recall, and F1-score for each class.")
    st.markdown("""
                * **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
                * **Recall**: The ratio of correctly predicted positive observations to the all observations in actual class.
                * **F1-Score**: The weighted average of Precision and Recall.
                * **macro avg**: The average of the precision, recall, and F1-score across all classes.
                * **weighted avg**: The average of the precision, recall, and F1-score across all classes, weighted by the number of true instances for each class.
                """)

    st.subheader("Class-Level Metrics (Heatmap)")
    st.image("app/images/classification_report_class_only.png", use_container_width=True)

    st.subheader("Overall Metrics (Table)")
    summary_df = pd.read_csv("app/reports/classification_report_summary.csv", index_col=0)
    st.dataframe(summary_df)

    # Confusion Matrix Section
    st.header("Confusion Matrix")
    st.write("This section shows the confusion matrix for the model, which summarizes the performance of the classification model.")
    st.image("app/images/conf_matrix.png", use_container_width=True)

    # Misclassification Examples Section
    with st.expander("Examples of Confusion Matrix values (TP, TN, FP, FN)", expanded=False):
        st.write("This section shows examples of tweets that were correctly and incorrectly classified by the model.")
        st.markdown("""
                    * **False Negatives (FN)**: Tweets incorrectly classified as non-disasters
                    * **True Negatives (TN)**: Tweets correctly classified as non-disasters
                    * **False Positives (FP)**: Tweets incorrectly classified as disasters
                    * **True Positives (TP**): Tweets correctly classified as disasters
                    """)
        
        df_eval = pd.read_csv("app/reports/eval_misclassified.csv")
        for key in df_eval['classification'].unique():
            st.subheader(f"Classification: {key}")
            examples = df_eval[df_eval['classification'] == key].sample(3, random_state=42)
            for i, row in examples.iterrows():
                st.write(f"• {row['text']}")

# App footer
st.markdown(
    """
    <hr>
    <div style='text-align: center;'>
        <strong>Built with</strong>: Python, Streamlit, TensorFlow/Keras, NLTK, Pandas, Matplotlib.<br>
        <strong>Dataset</strong>: <a href='https://www.kaggle.com/c/nlp-getting-started/overview' target='_blank'>Kaggle - NLP with Disaster Tweets</a><br>
        <strong>Author</strong>: <a href='https://www.linkedin.com/in/dido-de-boodt/' target='_blank'>Dido De Boodt</a>
    </div>
    """,
    unsafe_allow_html=True
)

