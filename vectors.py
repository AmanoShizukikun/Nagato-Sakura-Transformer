from gensim.models import Word2Vec
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
corpus_file_path = os.path.join(current_directory, 'data', 'chinese.txt')

with open(corpus_file_path, 'r', encoding='utf-8') as file:
    sentences = [line.strip().split() for line in file]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)

model.wv.save_word2vec_format('NagatoSakura_vectors.bin', binary=True)
