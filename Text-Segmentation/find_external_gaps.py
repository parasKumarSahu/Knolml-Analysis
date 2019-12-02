import pandas as pd

data = pd.read_csv('result.csv', error_bad_lines=False);
data_text = data[['text']]
data_text['index'] = data_text.index
documents = data_text

print(len(documents), "Documents found")

#print(documents)

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)

import nltk
nltk.download('wordnet')

print(WordNetLemmatizer().lemmatize('went', pos='v'))

stemmer = SnowballStemmer('english')

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

processed_docs = documents['text'].map(preprocess)

#print(processed_docs[:10])

dictionary = gensim.corpora.Dictionary(processed_docs)

'''
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
'''

dictionary.filter_extremes(no_below=1, no_above=0.5)
bow_corpus = 0
if(len(dictionary) != 0):
	bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
else:
	print("Change filter values to run!")
	exit()

'''
bow_doc = bow_corpus[0]


for i in range(len(bow_doc)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc[i][0], 
                                                     dictionary[bow_doc[i][0]], 
                                                     bow_doc[i][1]))
'''
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=len(documents), id2word=dictionary, passes=2, workers=4)


for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \n======\nWords: {}'.format(idx, topic))


M = []
count = 0
for bow_doc in bow_corpus:
	M_r = [0]*len(documents)
	for index, score in sorted(lda_model[bow_doc], key=lambda tup: -1*tup[1]):
		#print("Segment", count)
		#print("========")
		#print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index)))
		M_r[index] = score
	M.append(M_r)
	count += 1

print("External Knowledge Gaps:-")
print(pd.DataFrame(M))
