import json
from exploring.qaai import QAAI
from exploring.eval_rig import load_data, run_evals, sniff_schema
import os
import re
import nltk
from nltk.corpus import stopwords

from exploring.wordcloud import save_wordcloud, show_wordcloud 
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def do_wordcloud_simple():
    """Smoke test wordcloud drawing"""
    show_wordcloud({"Rome": 0.5, "Italy": 0.2, "Europe": 0.1, "Paris": 0.1})
 
def wordcloud_for_file(infile):
	input_data = load_data("data/" + infile)
	word2freq = {}
	for datum in input_data:	
		text = json.dumps(datum)
		words = text.split()
		for word in words:
			word = re.sub(r'[^a-zA-Z]+', '', word)
			word = word.lower()
			if word in stop_words:
				continue			
			word2freq[word] = word2freq.get(word, 0) + 1
	save_wordcloud(word2freq, "img/wordcloud-" + infile+".png")
	show_wordcloud(word2freq, "Wordcloud for " + infile)


def do_wordcloud_train():
	infile = "train.json"
	wordcloud_for_file(infile)

def do_wordcloud_dev():
	infile = "dev.json"
	wordcloud_for_file(infile)
 
 
def do_wordcloud_format_answers():
	# devfile = "data/dev.json"
	devfile = "data/train.json"
	data = load_data(devfile)
	word2freq = {}
	for datum in data:
		if "qa" in datum:            
			answer = datum["qa"]["answer"]
		else:
			answer = datum["qa_1"]["answer"]
		format = re.sub(r'[1-9]', 'd', answer)
		word2freq[format] = word2freq.get(format, 0) + 1
	save_wordcloud(word2freq, "img/formatcloud.png")
	show_wordcloud(word2freq, "Format-cloud for " + devfile)

 
if __name__ == "__main__":
	do_wordcloud_train()
	do_wordcloud_dev()
	do_wordcloud_format_answers()