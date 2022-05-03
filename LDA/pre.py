import jieba
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def get_para(path,min_char):
    with open(path, "r", encoding="ANSI") as f:
        content = f.read()
        l = len(content)
        paras = []
        tmp_char = ''
        for i, char in enumerate(content):
            if char != '\u3000':
                tmp_char += char
            if char == '\n':
                tmp_char = tmp_char[:-1]
                if len(tmp_char) >= min_char:
                    paras.append(tmp_char)
                tmp_char = ''
            if (i + 1) == l:
                if len(tmp_char) >= min_char:
                    paras.append(tmp_char)
                break
    f.close()
    paras = paras[::int(len(paras)/22)][0:22]
    return paras
  
def word_seg(text):
    """对文本进行基于jieba的分词，返回以空格分隔单词的文本"""
    seg_list = [" ".join(jieba.lcut(e, use_paddle=True, cut_all=False)) for e in text]
    return seg_list            

def docs_gen(sourcepath, stpwrdpath, char_len):
    files = os.listdir(sourcepath)
    with open(stpwrdpath, 'rb') as fp:
        stopword = fp.read().decode('utf-8') 
    stpwrdlst = stopword.splitlines()
    seg_list = []
    for file in files:
        fullpath = sourcepath + '\\' + file
        filename = file[:-4]
        seg_list.extend(word_seg(get_para(fullpath, char_len)))
    vec = CountVectorizer(token_pattern = r"(?u)\b\w+\b", ngram_range=(1,1), min_df = 1, stop_words=stpwrdlst)
    cnt = vec.fit_transform(seg_list)
    # print( 'vocabulary dic :\n\n',vec.vocabulary_)
    return cnt

# def gibbs():
    
        
if __name__ == "__main__":
    cnt = docs_gen('LDA/text', 'LDA/cn_stopwords.txt', 1000)
    lda = LatentDirichletAllocation(n_components = 2, 
                                    learning_offset = 50,
                                    random_state = 0)
    res = lda.fit_transform(cnt)
    print(res)
    