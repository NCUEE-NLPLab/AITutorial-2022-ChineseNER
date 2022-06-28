# -*- coding: utf-8 -*-

import logging

from gensim.models import word2vec

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence("msra_train_data.txt")
    model = word2vec.Word2Vec(sentences, vector_size=300,min_count=1,epochs=10)

    #保存模型，供日後使用
    # model.save("msra_tutorial.model")
    model.wv.save_word2vec_format('msra_word2vec.txt', binary=False)
    #模型讀取方式
    # model = word2vec.Word2Vec.load("your_model_name")

if __name__ == "__main__":
    main()


