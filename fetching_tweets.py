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




def finding_vec_list(train, auth, model1):
	#print('in func', train)
	vec_list1 = []
	for auth1 in auth:
		papers= {}
		vec1 = []
		for item in train:
			#print('in item',item)
			taggg = item['auth']
			ab = taggg.split('.')
			if ab[0] == auth1:
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
	#print('here')
	for root, dirs,files in os.walk(Dir_name):
		'''
		print(files)
		print(root)
		print(root)
		text_list = []
		
		'''
		for name in files:
			papers1 = {}
			path1 = os.path.join(root, name)
			fopen = open(path1,'r')
			text = fopen.read()
			texts = text.split('$$$$$$$$$')
			len1 = len(texts)
			del texts[len1-1]
			texts12 = []
			for i in  range(0, len(texts)):
				#abc = texts[i].split(' ')
				#print(len(abc))
				#if len(abc) >=15:
				papers = {}
				papers['auth'] = name+ '_' + str(i)
				papers['text'] = texts[i]
				texts12.append(texts[i])
				#print('textts', texts[i])
				papers_list.append(papers)
				#else:
				#	continue
			#len1 = len(papers_list)
			#del papers_list[len1-1]
			#print(papers)
			#print(papers_list)
			
			papers1['text'] = texts
			abc = path1.split('/')
			abc1 = abc[2].split('.')
			auth_list.append(abc1[0])
			papers1['auth'] = abc1[0]
			papers_list_common.append(papers1)
	#print('list', papers_list)
	return papers_list, auth_list, papers_list_common




def making_train_test(list1):
	texts=[]
	auths =[]
	for item in list1:
		tt = item['text']
		ab = item['auth']
		for t in tt:
			texts.append(t)
			auths.append(ab)

	return texts, auths


''''
def Making_train(dir1):
	#dir1 = 'Small_tweets/Train'
	#print('here')
	A,B,C = data_make(dir1)
	

	train, auths = making_train_test(C)

	#print(train)

	#print(auths)

	set_auth = set(auths)

	list_auths = list(set_auth)

	auths_new = []
	for auth11 in auths:
		auths_new.append(list_auths.index(auth11)+1)
	return train, auths_new


'''

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
	bigram_vectorizer = CountVectorizer(ngram_range=(3, 3),  token_pattern=r'\b\w+\b', min_df=1)
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



'''
A,B = Making_train('Small_tweets/Train')

print(A)
print(B)
'''
