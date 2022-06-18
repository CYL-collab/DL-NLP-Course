# -*- coding: utf-8 -*-

from __future__ import print_function
import jieba
import argparse
import numpy as np
import random
import sys
import os
import pickle
import re
from tensorflow.python.keras.callbacks import TensorBoard
# from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input, Dense, Activation, LSTM, Embedding, Bidirectional

def load_vectors(fname):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    w2v = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        w2v[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return w2v

EMBEDDING_DIM = 300

def read_dataset(maxlen=40, step=3):
    print('preparing datasets...')
    path = 'datasets'
    text = open(os.path.join(path, 'seg2.txt'), 'r', encoding="utf-8").read().strip()
    text = re.sub('[\u0000-\u3001]', '', text)
    text_words = jieba.lcut(text)  
    print('total words:', len(text_words))
    sentences = []  # 句子
    next_words = []  # 句子的下一个字符
    for i in range(0, len(text_words) - maxlen, step):
        sentences.append(" ".join(text_words[i: i + maxlen]))  
        next_words.append(text_words[i + maxlen])  
    print('nb sentences:', len(sentences))

    print('tokenizing...')
    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts(text_words)
    word_index = tokenizer.word_index
    index_word = {v: k for k, v in tokenizer.word_index.items()}
    X = tokenizer.texts_to_sequences(sentences)
    y = tokenizer.texts_to_matrix(next_words)
    X = pad_sequences(X, maxlen=maxlen)
    y = np.array(y)
    print('vocab size:', len(tokenizer.word_index))
    print('X.shape:', X.shape)
    print('y.shape:', y.shape)

    if os.path.exists('embedding_matrix.dat'):
        print('building embedding_matrix...')
        embedding_matrix = pickle.load(open('embedding_matrix.dat', 'rb'))
    else:
        print('loading word vectors...')
        word_vec = load_vectors('wiki.zh.vec')
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        print('building embedding_matrix...')
        for word, i in word_index.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open('embedding_matrix.dat', 'wb'))

    return tokenizer, index_word, embedding_matrix, text_words, X, y


class Generator:
    def __init__(self):
        self.MAX_SEQUENCE_LENGTH = 20
        self.STEP = 3
        self.ITERATION = 20
        self.tokenizer, self.index_word, self.embedding_matrix, self.text_words, self.X, self.y = \
            read_dataset(maxlen=self.MAX_SEQUENCE_LENGTH, step=self.STEP)

        if os.path.exists('saved_model.h5'):
            print('loading saved model...')
            self.model = load_model('saved_model.h5')
        else:
            print('Build model...')
            inputs = Input(shape=(self.MAX_SEQUENCE_LENGTH,))
            x = Embedding(input_dim=len(self.tokenizer.word_index) + 1,
                          output_dim=EMBEDDING_DIM,
                          input_length=self.MAX_SEQUENCE_LENGTH,
                          weights=[self.embedding_matrix],
                          trainable=False)(inputs)
            x = Bidirectional(LSTM(600, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))(x)
            x = LSTM(600, dropout=0.2, recurrent_dropout=0.1)(x)
            x = Dense(len(self.tokenizer.word_index) + 1)(x)
            predictions = Activation('softmax')(x)
            model = Model(inputs, predictions)
            model.summary()
            model.compile(loss='categorical_crossentropy', optimizer='SGD')
            # plot_model(model, to_file='model.png')
            self.model = model

    @staticmethod
    def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def train(self):
        tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
        for i in range(1, self.ITERATION):
            print()
            print('-' * 50)
            print('Iteration', i)
            self.model.fit(self.X, self.y, batch_size=128, callbacks=[tbCallBack])  # 训练
            if i % 5 == 0:
                self.model.save('saved_model.h5')
                print('model saved')
            self.test()

    def test(self, seed=None):
        if seed is None:
            # 随机生成种子文本
            start_index = random.randint(0, len(self.text_words) - self.MAX_SEQUENCE_LENGTH - 1)
            seed_words = self.text_words[start_index: start_index + self.MAX_SEQUENCE_LENGTH]
            seed = "".join(seed_words)
        else:
            seed_words = jieba.lcut(seed)

        for diversity in [0.5, 1.0, 1.2]:
            print(seed)
            print('----- diversity:', diversity)
            generated = seed
            sys.stdout.write(generated)  # 打印文本

            x = self.tokenizer.texts_to_sequences([" ".join(seed_words)])
            x = pad_sequences(x, maxlen=self.MAX_SEQUENCE_LENGTH)
            for i in range(400):  # 连续生成后续words
                preds = self.model.predict(x, verbose=0)[0]  # 预测下一个结果
                next_index = self.sample(preds, diversity)  # 抽样出下一个字符的索引值
                next_word = self.index_word[next_index]  # 检出下一个字符

                generated += next_word
                x = np.delete(x, 0, -1)
                x = np.append(x, [[next_index]], axis=1)  # 输入后移一个word

                sys.stdout.write(next_word)  # 连续打印
                sys.stdout.flush()  # 刷新控制台
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=str, default=None, help='generating with seed')
    args = parser.parse_args()
    gen = Generator()
    if args.seed:
        gen.test(args.seed)
    else:
        gen.train()
