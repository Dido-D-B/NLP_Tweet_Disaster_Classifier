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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Baseline Logistic Regression Function
def logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model

# Baseline model evaluation function
def evaluate_model(model, X_train, y_train, X_test, y_test, label="Model"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"===== {label} =====")
    plot_conf_matrix(y_test, preds, title=f"Confusion Matrix")

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

# Function to print evaluation metrics
def evaluation_metrics(y_test, preds):
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print("Evaluation Metrics:")
    print("-------------------")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

# Function to plot confusion matrix
def plot_conf_matrix(y_test, preds, title='Confusion Matrix'):
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Disaster', 'Disaster'],
                yticklabels=['Non-Disaster', 'Disaster'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

# Function to plot classification heatmap with only class metrics
def plot_classification_heatmap_only(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, target_names=["Non-Disaster (0)", "Disaster (1)"])
    df_report = pd.DataFrame(report).T

    # Only keep class rows and metrics
    class_metrics = df_report.loc[["Non-Disaster (0)", "Disaster (1)"], ["precision", "recall", "f1-score"]].round(2)

    plt.figure(figsize=(8, 4))
    sns.heatmap(class_metrics, annot=True ,cmap='coolwarm', fmt='.2f', cbar=True, linewidths=0.5, linecolor='gray')
    plt.title("CNN Class-Level Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.show()
    plt.close()