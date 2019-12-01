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

Steps to Run(Text Segmentation part):-
1. Supply text to be segmented in input.txt
2. Run Python3 test.py

Steps to train word2vec (Optional):-
1. You are already provided with a trained word2vec (wrdvecs-text8.bin), you have to delete it first
2. Once the trained model is deleted, supply a coprus with name text8 and simply run the code
