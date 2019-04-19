from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec
import os
import gensim
import os
from nltk import sent_tokenize
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import preprocess_string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk.tokenize import word_tokenize 
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords 
from scipy import spatial
import operator
import string
import functools
import sys
import math
reload(sys)
sys.setdefaultencoding('utf-8')
from query import getting_vector




def data_make(Dir_name):
	'''
	This function reads data from the folder name, which is given in the argument 'Dir_name'.
	First it makes a dictionary in which all docuemnts corresponding to an author (key is the author) is listed.
	Finally It returns a list of dictionaries, in which text of different authors are written and authors list.
	'''
	k=0
	'''
	For reading data from the directory name
	'''
	papers_list=[]
	papers_list_common = []
	auth_list =[]
	for root, dirs,files in os.walk(Dir_name):
		if(k==0):
			k = k+1
			continue

		text_list = []
		
		papers1 = {}
		path = os.path.join(root)
		abc = path.split('/')
		papers1['auth'] = abc[2]
		auth_list.append(abc[2])
		for name in files:
			path1 = os.path.join(root, name)
			fopen = open(path1,'r')
			text = fopen.read()
			text_list.append(text)
			#print(text)
			papers = {}
			papers['auth'] = path1
			papers['text'] = text
			papers_list.append(papers)
			#print(papers)
			#print(papers_list)
		papers1['text'] = text_list
		papers_list_common.append(papers1)
	#print('list', papers_list)
	return papers_list, auth_list, papers_list_common



def doc2vec_model(train_test, analyze):
	tagged_data=[]
	
	word_tokens = []
	wt = []
	for item in train_test:
		text = item['text']
		words = text.split(' ')
		#word_tokens.append(words)
		tagged_data.append(TaggedDocument(words, [item['auth']]))
	#print(tagged_data)
	
	max_epochs = 5
	vec_size = 700
	alpha = 0.025
	model = Doc2Vec(size=vec_size,
		        alpha=alpha, 
		        min_alpha=0.00025,
		        min_count=1,
		        dm =1)
	  
	model.build_vocab(tagged_data)

	for epoch in range(max_epochs):
		print('iteration {0}'.format(epoch))
		model.train(tagged_data,total_examples=model.corpus_count,epochs=model.iter)
		model.alpha -= 0.0002
		model.min_alpha = model.alpha
	model.save("d2v.model")
	print("Model Saved")
	return model



def doc2vec_char_model(train_test, analyze):
	tagged_data=[]
	wt = []
	for item in train_test:
		word_tokens = []
		text = item['text']
		words = text.split(' ')
		ngrams=[]
		for word in words:
			ng = word2ngrams(word, 4)
			for i in range(0, len(ng)):
				ngrams.append(ng[i])
		tagged_data.append(TaggedDocument(ngrams, [item['auth']]))
	print(tagged_data)
	
	max_epochs = 5
	vec_size = 300
	alpha = 0.025
	model = Doc2Vec(size=vec_size,
		        alpha=alpha, 
		        min_alpha=0.00025,
		        min_count=1,
		        dm =1)
	  
	model.build_vocab(tagged_data)

	for epoch in range(max_epochs):
		print('iteration {0}'.format(epoch))
		model.train(tagged_data,total_examples=model.corpus_count,epochs=model.iter)
		model.alpha -= 0.0002
		model.min_alpha = model.alpha
	model.save("d2v.model")
	print("Model Saved")
	return model




def find_similarity(vector_list_train, auth):
	similarity=[]
	for author in auth:
		result=[]
		papers={}
		for item in vector_list_train:
			if item['auth'] == author:
				vec = item['vector']
				#print('vector for author', auth , vec)
				for i in range(0, len(vec)):
					sum1 = 0
					for j in range(0, len(vec)):
						if (i == j):
							continue
						ab= 1 - spatial.distance.cosine(vec[i] , vec[j])
					#	print('ab ', ab)
						sum1 = sum1+ab
					#print('sum 1 is', sum1, 'len of vec', len(vec))
					result.append(sum1/(len(vec)-1))
				#	print(' avg is resul;t', result)
		sum1 = sum(result)
		#print(' total avg', sum1/len(result))
		papers['auth'] = author
		papers['result'] = sum1/len(result)
		similarity.append(papers)
	return similarity


def comp_train_test(train_vector, unk_vector):
	ab = 0
	for vec in train_vector:
		ab += 1 - spatial.distance.cosine(vec, unk_vector)		
	return ab/len(train_vector)
		



def word2ngrams(text, n, exact=True):
	'''
	this function calculates the n-grams of the word
	'''
	return ["".join(j) for j in zip(*[text[i:] for i in range(n)])]



def read_corpus_3_grams(text_list):
	'''
	This function returns the required tagged document 3-grams
	'''
	for paper in text_list:
		tokenizer = RegexpTokenizer(r'\w+') 
		text = paper['text']
		for doc in text:
			ngrams=[]
			word_tokens = tokenizer.tokenize(doc)
			for word in word_tokens:
				ng = word2ngrams(word, 3)
				for i in range(0, len(ng)):
					ngrams.append(ng[i])
			#print(ngrams)
			yield gensim.models.doc2vec.TaggedDocument(ngrams, [paper['auth']])
	



def finding_vec_list(train, auth, model1):
	#print('in func', train)
	vec_list1 = []
	for auth1 in auth:
		papers= {}
		vec1 = []
		for item in train:
			#print('in item',item)
			taggg = item['auth']
			ab = taggg.split('/')
			if ab[2] == auth1:
				tagg = item['auth']
				vec = model1[tagg]
				#print(vec)
				vec1.append(vec)
		vec_list1.append(vec1)
	#print('vec list', vec_list1)
	return vec_list1



def making_dict(vec_list, auth):
	vect_dict = []
	for i in range(0, len(vec_list)):
		papers={}
		papers['auth'] = auth[i]
		papers['vector'] = vec_list[i]
		vect_dict.append(papers)
	#print('dict', vect_dict)
	return vect_dict


def main1(train_path, test_path):
	train, auth, train_list = data_make(train_path)
	#print('train', train)
	test, auth, test_list = data_make(test_path)
	#print('test',test)
	#print(auth)
	auth_set = set(auth)
	#print(auth_set)
	auth_list= list(auth_set)
	#print(auth_list)

	train_test = []
	for item in train:
		train_test.append(item)
	for item in test:
		train_test.append(item)

	#print(train_test)
	#train_test = combining_train_test(train, test, auth)
	print('hehe$$$$$$$$$$$$$$$$$$$$$$$$$$')
	bigram_vectorizer = CountVectorizer(ngram_range=(2, 2),  token_pattern=r'\b\w+\b', min_df=1)
	analyze = bigram_vectorizer.build_analyzer()
	model1 = doc2vec_model(train_test, analyze)
	#model1 = doc2vec_char_model(train_test, analyze)

	#print('train here', train)

	vect_list_train = finding_vec_list(train, auth, model1)
	#print('here')
	vect_list_test = finding_vec_list(test, auth, model1)



	vector_list_train = making_dict(vect_list_train, auth)

	vector_list_test = making_dict(vect_list_test, auth)

	#print(vector_list_train)
	#print(vector_list_test)		



	x_train=[]
	y_train=[]
	for item in vector_list_train:
		vec= item['vector']
		auth = item['auth']
		for vecs in vec:
			abc=[]
			for ab in vecs:
				abc.append(ab)
			x_train.append(abc)
			y_train.append(auth)

	x_test=[]
	y_test=[]
	#print('x train',x_train)
	#print('Y train', y_train)
	for item in vector_list_test:
		vec= item['vector']
		auth = item['auth']
		for vecs in vec:
			abc=[]
			for ab in vecs:
				abc.append(ab)
			x_test.append(abc)
			y_test.append(auth)

	#print(auth)
	#print('x test',x_test)
	#print('Y test', y_test)
	y_train1=[]
	y_test1=[]
	for i in range (0,len(y_train)):
		id1= auth_list.index(y_train[i])
		y_train1.append(id1+1)
	for i in range (0,len(y_test)):
		id1= auth_list.index(y_test[i])
		y_test1.append(id1+1)
	return x_train, y_train1, x_test, y_test1


#A,B,C,D = main1('C501/C50train', 'C501/C50test')
'''
print(A)
print(B)	
print(C)
print(D)

print(' model made')
vector_list_train=[]
for author in auth:
	vectors=[]
	papers={}
	for item in train_list:
		ab = item['auth']
		abc = ab.split('/')
		if item['auth'] == author:
			train_data = item['text']
			#print('train in loop',train_data)
			for data in train_data:
				ngrams1=[]
				test_data = analyze(data)
				for words in test_data:
					ng = word2ngrams(words, 4)
					for i in range(0, len(ng)):
						ngrams1.append(ng[i])	
				print('tes data', ngrams1)
				vec = model1.infer_vector(ngrams1)
				#print("V1_infer", vec)
				vectors.append(vec)
	papers['auth']= author
	papers['vector'] = vectors
	vector_list_train.append(papers)





print(' model made')
vector_list_train=[]
for author in auth:
	vectors=[]
	papers={}
	for item in train:
		if item['auth'] == author:
			train_data = item['text']
			#print('train in loop',train_data)
			for data in train_data:
				ngrams1=[]
				test_data = analyze(data)	
				print('tes data', test_data)
				vec = model1.infer_vector(test_data)
				#print("V1_infer", vec)
				vectors.append(vec)
	papers['auth']= author
	papers['vector'] = vectors
	vector_list_train.append(papers)


#print('v list',vector_list_train)
	
vector_list_test=[]
for author in auth:
	vectors=[]
	papers={}
	ngrams = []
	for item in test:
		if item['auth'] == author:
			train_data = item['text']
			#print('train',train_data)
			for data in train_data:
				ngrams1=[]
				test_data = analyze(data)
				print('tes data', test_data)
				vec = model1.infer_vector(test_data)
				#print("V1_infer", vec)
				vectors.append(vec)
	papers['auth']= author
	papers['vector'] = vectors
	#print('vector', vectors)
	#print('auth', author)
	vector_list_test.append(papers)


#print('vector list test', vector_list_test)
#print('vector list train', vector_list_train)

similarity = find_similarity(vector_list_train, auth)


#print(similarity)
count_No =0
count_Yes =0
for author in auth:
	print('author is', author)
	for item in vector_list_test:
		if item['auth'] == author:
			for item1 in vector_list_train:
				if item1['auth'] == author:
					for vec in item['vector']:
						ab = comp_train_test(item1['vector'] , vec)
						#print('ab', ab)
						for item2 in similarity:
							if item2['auth'] == author:
								#print('item 21', item2['result'])
								if ab >= item2['result']:
									#print('YeS')
									count_Yes +=1
								else:
									#print('NO')
									count_No +=1

print(count_No)
print(count_Yes)
print(float(count_Yes)/ (count_No + count_Yes))
					
			
				


test_data = word_tokenize("I love chatbots".lower())
v1 = model1.infer_vector(test_data)
print("V1_infer", v1)

# to find most similar doc using tags
similar_doc = model1.docvecs.most_similar('1')
print(similar_doc)


# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
print(model1.docvecs['1'])
'''
