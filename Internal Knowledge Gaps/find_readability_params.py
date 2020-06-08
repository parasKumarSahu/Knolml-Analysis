import warnings
warnings.filterwarnings("ignore")

from datetime import datetime 
import os
import pandas as pd
import numpy as np
import re
import nltk
import sys

#For Text Segmentation
import word2vec
from sklearn.feature_extraction.text import CountVectorizer
from tools import get_penalty, get_segments
from algorithm import split_optimal

#For Finding External Gaps
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

#For Readability Measures
import textstat
from scipy.stats import variation

#For Edits and Bytes Info
import requests
from bs4 import BeautifulSoup



def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


if __name__ == "__main__":

	#Setup

	if len(sys.argv) < 2:
	    print("Input Format: python3 script_name folder_name")
	    exit()
	
	startTime = datetime.now()
	folder_name = sys.argv[1]
	corpus_path = './text8'  

	wrdvec_path = 'wrdvecs-text8.bin'
	if not os.path.exists(wrdvec_path):
	    word2vec.word2vec(corpus_path, wrdvec_path, cbow=1, iter_=5, hs=1, threads=4, sample='1e-5', window=15, size=200, binary=1)

	model = word2vec.load(wrdvec_path)
	wrdvecs = pd.DataFrame(model.vectors, index=model.vocab)
	del model
	#print(wrdvecs.shape)

	nltk.download('punkt')
	sentence_analyzer = nltk.data.load('tokenizers/punkt/english.pickle')

	segment_len = 30  # segment target length in sentences

	nltk.download('wordnet')

	#Code For Text Segmentation

	f_out = open("Output/"+folder_name+".txt", "w")

	for filename in os.listdir("Input/"+folder_name):
		with open("Input/"+folder_name+"/"+filename, 'r') as f:
			text = f.read().replace('\n', '¤')  # punkt tokenizer handles newlines not so nice

		#Process only articles having more than 100 words
		#if len(text.split()) <= 100 :
		#	continue

		sentenced_text = sentence_analyzer.tokenize(text)
		vecr = CountVectorizer(vocabulary=wrdvecs.index)

		sentence_vectors = vecr.transform(sentenced_text).dot(wrdvecs)

		penalty = get_penalty([sentence_vectors], segment_len)
		#print('penalty %4.2f' % penalty)

		segments = []

		if penalty > 0:
			optimal_segmentation = split_optimal(sentence_vectors, penalty, seg_limit=250)
			segmented_text = get_segments(sentenced_text, optimal_segmentation)
			seg_count = 0

			for s in segmented_text:
			    #print(str(seg_count)+",")
			    segment = ""
			    for ss in s:
			        #print(re.sub('\W+',' ', ss))
			        segment += re.sub('\W+',' ', ss)
			    #print("\n")
			    segments.append(segment)    
			    seg_count += 1

			#print('%d sentences, %d segments, avg %4.2f sentences per segment' % (
			#    len(sentenced_text), len(segmented_text), len(sentenced_text) / len(segmented_text)))

			#Code For Finding External Gaps

			np.random.seed(2018)

			data_text = pd.DataFrame(segments, columns = ['text'])
			data_text['index'] = data_text.index
			documents = data_text

			print(WordNetLemmatizer().lemmatize('went', pos='v'))
			stemmer = SnowballStemmer('english')


			processed_docs = documents['text'].map(preprocess)
			dictionary = gensim.corpora.Dictionary(processed_docs)

			bow_corpus = 0
			if(len(dictionary) != 0):
				bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
			else:
				print("Change filter values to run!")
				exit()

			lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=len(documents), id2word=dictionary, passes=2, workers=4)

			'''
			for idx, topic in lda_model.print_topics(-1):
			    print('Topic: {} \n======\nWords: {}'.format(idx, topic))
			'''

			M = []
			count = 0
			for bow_doc in bow_corpus:
				M_r = [0]*len(documents)
				for index, score in sorted(lda_model[bow_doc], key=lambda tup: -1*tup[1]):
					M_r[index] = score
				M.append(M_r)
				count += 1

			#print("External Knowledge Gaps:-")

			no_gaps = 0
			threshold = 1.75

			for i in range(len(M)-1):
				tmp_sum = 0	
				for j in range(len(M[0])):
					tmp_sum += abs(M[i+1][j]-M[i][j])
				if tmp_sum > threshold:
					no_gaps += 1	
					#print("Gap found between segments", i+1, "and", i+2)

			f_out.write("Filename: "+str(filename)+"\n")
			f_out.write("Sentences: "+str(len(sentenced_text))+"\nSegments: "+str(len(segmented_text))+"\nKnowledge Gaps: "+str(no_gaps)+"\n")
		else:
			segments.append(text)
			f_out.write("Filename: "+str(filename)+"\n")
			f_out.write("Sentences: "+str(len(sentenced_text))+"\nSegments: 1\nKnowledge Gaps: 0\n")

		#Code for Readability Measures

		flesh_list = []
		coleman_list = []
		ari_list = []
		#print("file:", filename)

		for segment in segments:
			seg_text = segment.replace('¤', '\n')
			#print("seg len:", len(seg_text.split()))
			flesh_list.append(textstat.flesch_kincaid_grade(seg_text))
			coleman_list.append(textstat.coleman_liau_index(seg_text))
			ari_list.append(textstat.automated_readability_index(seg_text))

		f_out.write("Flesch Kincaid Grade Values: "+str(flesh_list)+"\n")
		f_out.write("Flesch Kincaid Grade CV: "+str(variation(flesh_list, axis = 0))+"\n")
		f_out.write("Coleman Liau Index Values: "+str(coleman_list)+"\n")
		f_out.write("Coleman Liau Index CV: "+str(variation(coleman_list, axis = 0))+"\n")
		f_out.write("Automated Readability Index Values: "+str(ari_list)+"\n")
		f_out.write("Automated Readability Index CV: "+str(variation(ari_list, axis = 0))+"\n")


		#Code to get edit and bytes info

		url1 = 'https://en.wikipedia.org/wiki/'
		title = filename.replace(".txt", "").replace(" ", "_")
		#For articles having view info as prefex
		if title.find("$#@") != -1:
			title = title.split("$#@")[1]
		url2 = '?action=info'

		response = requests.get(url1+title+url2)

		if response :
			soup = BeautifulSoup(response.content, 'html.parser')
			if soup.find(id='mw-pageinfo-edits') and soup.find(id='mw-pageinfo-length'):
				edits = soup.find(id='mw-pageinfo-edits').find_all("td")[1].text.replace(",", "")
				bytes_count = soup.find(id='mw-pageinfo-length').find_all("td")[1].text.replace(",", "")
				f_out.write("Edits: "+edits+"\n")
				f_out.write("Bytes: "+bytes_count+"\n")
			else:	
				f_out.write("Edits: HTML_error\n")
				f_out.write("Bytes: HTML_error\n")
		else:
			f_out.write("Edits: Network_error\n")
			f_out.write("Bytes: Network_error\n")

		f_out.write("\n")

		print(filename, "processed")

	print("Execution Time", datetime.now() - startTime)