from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# Function to generate and display a wordcloud from a list of tokens
def plot_wordcloud(tokens, 
                   title=None, 
                   width=800, 
                   height=500, 
                   max_font_size=110, 
                   colormap='Blues', 
                   random_state=21):
    
    wordcloud = WordCloud(width=width, height=height, random_state=random_state, 
                          max_font_size=max_font_size).generate(str(tokens))

    # Apply colormap
    wordcloud_colored = wordcloud.recolor(colormap=colormap)

    plt.figure(figsize=(14,8))
    plt.imshow(wordcloud_colored, interpolation='bilinear')
    plt.axis('off')
    if title:
        plt.title(title, fontsize=16)
    plt.show()

# Function to plot the top N most frequent words from a list of tokens
def plot_top_words(tokens, n=10, title="Top Words"):
    
    word_freq = Counter(tokens)
    top_words = dict(word_freq.most_common(n))

    plt.figure(figsize=(10, 6))
    plt.bar(top_words.keys(), top_words.values())
    plt.title(title, fontsize=14)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks()
    plt.tight_layout()
    plt.show()

