import numpy as np
import preprocess
from sklearn.feature_extraction.text import CountVectorizer
import math

def bigram(seg_list):
    """使用sklearn的CountVectorizer对文本的一元和二元词频进行统计"""
    vec1 = CountVectorizer(token_pattern = r"(?u)\b\w+\b", ngram_range=(1,1), min_df = 1)
    vec2 = CountVectorizer(token_pattern = r"(?u)\b\w+\b", ngram_range=(2,2), min_df = 1) 
    d1 = vec1.fit_transform(seg_list).toarray()
    d2 = vec2.fit_transform(seg_list).toarray()
    # 对各句的频率求和
    sum1 = np.zeros(np.size(d1,1)) 
    sum2 = np.zeros(np.size(d2,1))
    for line in d1:
        sum1 = sum1 + line
    for line in d2:
        sum2 = sum2 + line
    # 输出key为单词，value为词频的字典
    double_prob = dict(zip(vec2.get_feature_names(), sum2))
    single_prob = dict(zip(vec1.get_feature_names(), sum1))
    return single_prob, double_prob

def entropy_estimation(test_case, single_prob, double_prob):
    """对test_case中每一句话进行二元交叉熵估计"""
    sum = 0
    num_sentence = len(test_case)
    # 计算每一句话出现概率
    for sentence in test_case:
        if len(sentence) != 0:
            r = 1
            for i in range(len(sentence)-1):
                item = sentence[i] + ' ' + sentence[i+1]
                if item in double_prob:
                    temp = double_prob[item]/single_prob[sentence[i]]
                elif sentence[i] in single_prob:
                    temp = 1/single_prob[sentence[i]]
                else:
                    temp = 1/len(single_prob)
                r *= temp
            if r < 10e-100:
                num_sentence -= 1
                continue
            sum += -(math.log(r,2))/len(sentence)
    return sum/num_sentence


if __name__ == "__main__":
    preprocess.merge_text('text/lm','merged.txt')
    seg_list = preprocess.word_seg('merged.txt')
    char_list =  preprocess.char_seg('merged.txt')
    single_prob_word, double_prob_word = bigram(seg_list)
    single_prob_char, double_prob_char = bigram(char_list)
    # preprocess.merge_text('text/test','testtext.txt')
    # test_list = preprocess.word_seg_tolist('testtext.txt')
    test_list_word = preprocess.word_seg_tolist('testtext.txt')
    test_list_char = preprocess.char_seg_tolist('testtext.txt')
    entropy_single = entropy_estimation(test_list_char, single_prob_char, double_prob_char)
    entropy_double = entropy_estimation(test_list_word, single_prob_word, double_prob_word)
    print("基于字的二元熵为",entropy_single,"bit/字")
    print("基于词的二元熵为",entropy_double,"bit/词")