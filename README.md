# Disaster Tweet Classifier

<img width="1224" height="680" alt="image" src="https://github.com/user-attachments/assets/8b765ea0-a218-48b6-8934-7a40224c201f" />

---

This project aims to build and deploy a machine learning model that predicts whether a given tweet refers to a real disaster. The dataset contains 10,000 labeled tweets.

## Key features

- Robust **preprocessing pipeline** (cleaning, tokenization, stopword removal, lemmatization)
- Multiple **model comparisons** (Logistic Regression, Naive Bayes, FNN, CNN)
- **Hyperparameter tuning** for performance optimization
- Deployed **Streamlit app** with both **user input** and **random test tweet** predictions and a **behind-the-scenes** look at preprocessing steps and misqualified Tweets.

## Repository Structure

```
NLP_Tweet_Disaster_Classifier/
├── app/
│   ├── app.py                                # Main Streamlit app
│   ├── model/
│   │   ├── cnn_disaster_classifier.keras     # Main model used in app deployment
│   │   └── tokenizer.pickle                  # Tokenizer used for CNN model and app deployment
│   ├── data/
│   │   └── test_sample.csv                   # Tweets from the test set for demo
│   ├── images/                               # Charts & visualizations
│   ├── reports/                              # Classification reports, CSV summaries
├── data/         
│   ├── processed/                             # cleaned data split into train/test               
│   ├── raw/                                  
├── notebooks/ 
├── presentation/                             # Slides in pdf and model comparison table
├── requirements.txt                          # Project dependencies
├── runtime.txt                               # Python version for deployment
└── README.md                                 # This file
```

## Tech stack

- **Modeling:** scikit-learn (LR, NB), TensorFlow/Keras (CNN)  
- **NLP:** NLTK (stopwords, POS), custom preprocessing  
- **App:** Streamlit  
- **Experimentation:** Jupyter/Colab

## Pipeline

### Data Cleaning

   - Lowercasing, accent removal
   - Removing URLs, mentions, hashtags, punctuation, numbers, extra whitespace
   - Dropping words with only 1 character

### Tokenization & Lemmatization

   - POS tagging to choose the correct lemma
   - Stopword removal (NLTK + custom stopwords)

### Vectorization

   - CountVectorizer (n-grams)
   - TF-IDF (incl. tuned alpha for NB)

### Model Training & Evaluation

   - Baselines: Logistic Regression, Naive Bayes
   - Deep Learning: LSTM, CNN
   - CNN achieved the best weighted F1 score (~0.80)

### Deployment

   - Trained CNN + tokenizer loaded in Streamlit app
   - User can:
     - Enter custom text
     - Generate a random tweet from the test set
     - View preprocessing steps and prediction probability
     - Model summary & EDA pages

## Why the CNN?

- Learns embeddings and local word patterns; less reliant on sparse vectorizers.  
- Chosen as the **final app model** due to balanced precision/recall, useful for disaster detection where **recall matters** without over-flagging. 

## Demo (Streamlit)

👉 Check out the live demo [here](https://nlp-tweet-disaster-classifier.streamlit.app/)

### App features

* **User text input**: type a tweet and get prediction + probability
* **Random test tweet**: sample from app/data/test_sample.csv
* **Preprocessing reveal**: see how the text is transformed before prediction
* **Model info**: brief architecture & metrics page

<img width="1036" height="1384" alt="image" src="https://github.com/user-attachments/assets/6a38a640-b248-41dd-914d-18d4c782dcf0" />

## Running Locally

1. Clone the repository

```bash
git clone hhttps://github.com/Dido-D-B/NLP_Tweet_Disaster_Classifier/tree/main.git
cd NLP_Tweet_Disaster_Classifier
```

2. Create & activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Run the app

```bash
streamlit run app/app.py
```

## Limitations

- Model trained on ~10k tweets; may not generalize perfectly to all contexts
- Sarcasm, humor, or indirect disaster references can confuse the classifier
- Tokenizer is fixed; retraining required for new vocab


## References

- Dataset: [Kaggle NLP Getting Started competition](https://www.kaggle.com/competitions/nlp-getting-started) 
- Streamlit for easy deployment
- TensorFlow/Keras for deep learning framework
- [Slide deck](https://github.com/Dido-D-B/CIFAR-10_Classifier_App/blob/main/Deep%20Learning%20with%20Shallow%20Pixels.pdf)
- [App](https://nlp-tweet-disaster-classifier.streamlit.app/)
