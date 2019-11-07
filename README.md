# Knolml-Analysis
The aim of this project is to do various types of analysis on knolml which can be used by a reseracher who is working on wikipedia data.

## Analysis1: Controversy Analysis using wiki-links
To measure the relative controversy level of various wiki-links present in a wikipedia article.
To Run:-
Python3 controvercy_analysis.py

## Analysis2: Contributions of an author over a given period of time in a wikipedia article
To find contributions of an author in terms of words, sentences, bytes etc over a given period of time (given starting and ending dates)
To Run:-
Python3 author_contribution.py

## Analysis3: Ranking all the authors based on their contribution to a given paragraph
To rank all the authors of a wikipedia article based on their contribution to a particular paragraph present in the article. The paragraph will be given as input to the program.
To Run:-
Python3 rank_authors_based_on_para_contr.py

## Analysis4: Finding knowledge gaps in a wikipedia article
A wikipedia article represents knowledge about some related topics, like a wikipedia article on IIT Ropar may be talking about placements of IIT Ropar in a particular section. But, in this section there was no information regarding a new branch say Biotechnology which was newly introduced. So, can we write a Python program that can tell that the information regarding placements of Biotechnology is missing from the IIT Ropar wikipedia page? Or in general can we tell that there is a knowledge gap in a wikipedia article?
Steps to Run:-
1. Download GloVE from http://nlp.stanford.edu/data/glove.6B.zip in the Text-Segmentation folder
2. Run converter.py, it will convert the file glove.6B.100d.txt which is in GloVe format to word2vec.6B.100d.txt which is in word2vec format so that we can use it with Gensim library of Python.
3. Finally run 'Python2 segmentart.py [number of disions] [path to input file]', the output file will be generated in the same folder named as Resultn.txt
