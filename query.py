from gensim.parsing import preprocessing
import gensim.parsing.preprocessing as preprocess
from pprint import pprint as pp
import gensim

def getting_vector(text, model_name):
	#load the model
	model =gensim.models.doc2vec.Doc2Vec.load(model_name)
	doc_vector = model.infer_vector(text)
	return doc_vector
