import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_visualization import plot_wordcloud, plot_top_words
from src.data_cleaning import clean_text, tokenize_and_remove_stopwords
from src.model_evaluation import evaluate_model, logistic_regression, evaluation_metrics, plot_conf_matrix