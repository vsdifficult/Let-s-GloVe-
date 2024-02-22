

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

import pandas as pd, numpy as np

def encode_tables_csv(tables_path: str): 
    data = pd.read_csv(tables_path, sep=";")  
    return (data['word'].tolist())
    
def get_word_embedding(word: str): 
    glove_file = './data/glove.6B.100d.txt'  
    word2vec_output_file = './glove.6B.100d.word2vec'
    data = encode_tables_csv("./data/PolSentiLex-raw.csv") 
    
    model = Word2Vec(data, vector_size=100, window=5, min_count=1, workers=4)
    model.train(data, total_examples=model.corpus_count, epochs=10)  
    glove2word2vec(glove_file, word2vec_output_file)
    model_glove = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False, no_header=True)
    vector  = model_glove.wv[word] 
    similar_words = model_glove.wv.most_similar(word) 
    print(f"Vector: {vector}\nSimilar:{similar_words}")

get_word_embedding('hello')
