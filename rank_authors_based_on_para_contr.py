import xml.etree.ElementTree as ET
import xml.dom.minidom
import re
import difflib
import collections
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

author_contribution = collections.defaultdict(lambda: 0)

def max_cosine_similarity(paragraph, revision):
	s1 = s2 = ""
	if len(paragraph) < len(revision):
		s1 = paragraph
		s2 = revision
	else:
		s2 = paragraph
		s1 = revision
	max_similarity = 0

	# sw contains the list of stopwords 
	sw = stopwords.words('english')  

	# tokenization 
	X_list = word_tokenize(s1)  

	# remove stop words from string 
	X_set = {w for w in X_list if not w in sw}  

	for i in range(len(s2)-len(s1)):
		Y_list = word_tokenize(s2[i:i+len(s1)]) 
		  
		l1 =[];l2 =[] 
		  
		Y_set = {w for w in Y_list if not w in sw} 

		# form a set containing keywords of both strings  
		rvector = X_set.union(Y_set)  
		for w in rvector: 
		    if w in X_set: l1.append(1) # create a vector 
		    else: l1.append(0) 
		    if w in Y_set: l2.append(1) 
		    else: l2.append(0) 
		c = 0
		  
		# cosine formula  
		for i in range(len(rvector)): 
		        c+= l1[i]*l2[i] 
		cosine = 0        
		if sum(l1) != 0 and sum(l2) != 0:        
			cosine = c / float((sum(l1)*sum(l2))**0.5) 
		if max_similarity < cosine:
			max_similarity = cosine
	return max_similarity	



#Main Function
#file_name = input("Enter compressed KNML file path: ")
file_name = "2006_Westchester_County_torna.knolml"

tree = ET.parse(file_name)
root = tree.getroot()
last_rev = ""
count = 0
length = len(root[0].findall('Instance'))

revision_list = []
author_list = []

for each in root.iter('Instance'):
	instanceId = int(each.attrib['Id'])
	for child in each:
		if 'Contributors' in child.tag:
			author_list.append(child[0].text)		
		if 'Body' in child.tag:
			revision = child[0].text
			revision = re.sub(r'\*?\{\{[^\}]*\}\}', "", revision)
			revision = re.sub(r'\*?\[\[[^\]]*\]\]', "", revision)
			revision = re.sub(r'\*?\<[^\>]*\>', "", revision)
			revision = ' '.join(revision.split())
			revision_list.append(revision)

print(revision_list[-1])			
paragraph = input("Enter paragraph: ")

last_val = 0

for i in range(len(revision_list)):			
	curr_val = max_cosine_similarity(paragraph, revision_list[i])
	print(author_list[i], curr_val)
	if last_val == curr_val == 1.0:
		continue
	author_contribution[author_list[i]] += curr_val	
	last_val = curr_val	
	print("Progress:", i, "/", len(revision_list))

for x in sorted(author_contribution.items() ,  key=lambda x: x[1]):
	print(x[0], "\n--->", x[1])

