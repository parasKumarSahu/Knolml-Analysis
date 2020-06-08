import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


if __name__ == "__main__":

	data = pd.read_csv('segmentaion_result.csv', error_bad_lines=False, encoding="utf8")
	data_text = data[['text']]
	data_text['index'] = data_text.index
	documents = data_text

	print(len(documents), "Documents found")

	np.random.seed(2018)

	nltk.download('wordnet')

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


	for idx, topic in lda_model.print_topics(-1):
	    print('Topic: {} \n======\nWords: {}'.format(idx, topic))

	M = []
	count = 0
	for bow_doc in bow_corpus:
		M_r = [0]*len(documents)
		for index, score in sorted(lda_model[bow_doc], key=lambda tup: -1*tup[1]):
			M_r[index] = score
		M.append(M_r)
		count += 1

	print("External Knowledge Gaps:-")

	f = open("external_gaps.txt", "w")
	f.write(str(pd.DataFrame(M)))
	no_gaps = 0
	threshold = 1.75

	for i in range(len(M)-1):
		tmp_sum = 0	
		for j in range(len(M[0])):
			tmp_sum += abs(M[i+1][j]-M[i][j])
		if tmp_sum > threshold:
			no_gaps += 1	
			print("Gap found between segments", i+1, "and", i+2)
			f.write("Gap found between segments "+str(i+1)+" and "+str(i+2)+"\n")

	print("Total", no_gaps, "Knowledge-gaps found!")
	f.write("Total "+str(no_gaps)+" Knowledge-gaps found!\n")