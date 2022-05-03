import jieba
import os
import pickle

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
    return paras
  
def word_seg(text):
    """对文本进行基于jieba的分词，返回以空格分隔单词的文本"""
    seg_list = [" ".join(jieba.lcut(e, use_paddle=True, cut_all=False)) for e in text]
    return seg_list            

def docs_gen(sourcepath, destination, char_len):
    files = os.listdir(sourcepath)
    docs = {}
    for file in files:
        fullpath = sourcepath + '\\' + file
        filename = file[:-4]
        docs[filename] = word_seg(get_para(fullpath, char_len))
    with open(destination, "wb") as f:
        pickle.dump(docs,f,pickle.HIGHEST_PROTOCOL)
    f.close()
    return docs
        
if __name__ == "__main__":
    docs_gen('LDA/text', 'LDA/doc.pkl', 500)