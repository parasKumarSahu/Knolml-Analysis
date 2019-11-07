from gensim.models import KeyedVectors

# load the Stanford GloVe model
filename = 'word2vec.6B.100d.txt'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
if 'sentance' in model:
	print(model['sentance'])
# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)