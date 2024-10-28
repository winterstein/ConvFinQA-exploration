
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

import numpy as np


def show_wordcloud(word2freq: dict, title: str = None):
	wc = WordCloud(max_words=150, background_color="white").generate_from_frequencies(word2freq)
	#Use matplotlib.pyplot to display the fitted wordcloud	
	plt.imshow(wc)
	plt.axis('off') #Turn axis off to get rid of axis numbers
	if title:
		plt.title(title)
	plt.show()

def save_wordcloud(word2freq: dict, filename: str):
	wc = WordCloud(max_words=150, background_color="white").generate_from_frequencies(word2freq)
	wc.to_file("img/"+filename)

