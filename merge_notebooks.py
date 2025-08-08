import nbformat as nbf
from pathlib import Path

order = [
    "notebooks/0_disaster_tweet_classifier_eda.ipynb",
    "notebooks/1_disaster_tweet_classifier_data_cleaning.ipynb",
    "notebooks/2_disaster_tweet_classifier_LogReg.ipynb",
    "notebooks/3_disaster_tweet_classifier_MNB.ipynb",
    "notebooks/4_disaster_tweet_classifier_DL.ipynb",
    "notebooks/5_disaster_tweet_classifier_model_comparison.ipynb",
]

nb_out = nbf.v4.new_notebook()
nb_out.cells = []

# Add a title cell
nb_out.cells.append(nbf.v4.new_markdown_cell("# Disaster Tweet Classifier — Final Notebook\n"
                                             "_Merged for submission (EDA → Models → Comparison)_"))

for p in order:
    nb = nbf.read(p, as_version=4)
    # Optional: add a section header for each part
    nb_out.cells.append(nbf.v4.new_markdown_cell(f"## Source: `{Path(p).name}`"))
    nb_out.cells.extend(nb.cells)

nbf.write(nb_out, "notebooks/FINAL_disaster_tweet_classifier.ipynb")
print("Merged -> notebooks/FINAL_disaster_tweet_classifier.ipynb")