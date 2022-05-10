import jieba
import re
from gensim.models import word2vec
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle as pkl

def cut_sentences(content):
	"""文本分句处理"""
	end_flag = ['?', '!', '.', '？', '！', '。', '…']	
	content_len = len(content)
	sentences = []
	tmp_char = ''
	for idx, char in enumerate(content):
		# 拼接字符
		tmp_char += char
		# 判断是否已经到了最后一位
		if (idx + 1) == content_len:
			sentences.append(tmp_char)
			break
		# 判断此字符是否为结束符号
		if char in end_flag:
			# 再判断下一个字符是否为结束符号，如果不是结束符号，则切分句子
			next_idx = idx + 1
			if not content[next_idx] in end_flag:
				sentences.append(tmp_char)
				tmp_char = ''
				
	return sentences

def word_seg(path, dest):
    """对path指向文本进行基于jieba的分词，返回以空格分隔单词的文本"""
    with open(path, "r", encoding="ANSI") as f:
        data = f.read()
        f.close()
    text = cut_sentences(data)
    with open(dest, "w+", encoding="utf-8") as f:
        for sentence in text:
            sentence = data = re.sub('[^\u4e00-\u9fa5]+', '', sentence)
            f.write(" ".join(jieba.lcut(sentence, use_paddle=True, cut_all=False)) + '\n')

def vec_gen(path):
    """根据分句分词完成的文件路径生成词向量字典"""
    train_data = word2vec.LineSentence(path)
    model = word2vec.Word2Vec(train_data, 
                              vector_size=100, 
                              window=5, 
                              workers=4)
    model.wv.vectors = model.wv.vectors / (np.linalg.norm(model.wv.vectors, axis=1).reshape(-1, 1))
    vec_dist = dict(zip(model.wv.index_to_key,model.wv.vectors))
    with open('VEC/vec_dist', 'wb') as f:     
        pkl.dump(vec_dist,f)

def cluster(keys, n_clusters):
    """根据选定键值对相应单词使用KMeans聚类"""
    with open('VEC/vec_dist', 'rb') as f:     
        vec_dist = pkl.load(f)
    vec = []
    for k in keys:
        vec.append(vec_dist[k])
    label = KMeans(n_clusters=n_clusters).fit_predict(vec)
    vec = PCA(n_components=2).fit_transform(vec)
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(vec[:,0],vec[:, 1],c=label) 
    for i, w in enumerate(keys): 
        plt.annotate(s=w, xy=(vec[:, 0][i], vec[:, 1][i]),
                    xytext=(vec[:, 0][i] + 0.01, vec[:, 1][i] + 0.01))
    plt.show()

    
if __name__ == "__main__":
    # s = word_seg('VEC/text/merged.txt','VEC/seg.txt')
    # wv = vec_gen('VEC/seg.txt')
    cluster(['郭靖','黄蓉','杨过','小龙女','郭襄','张无忌','谢逊','韦小宝','双儿','康熙','少林','武当','昆仑','北京','嘉兴','杭州','扬州','降龙十八掌','打狗棒法','一阳指'], 5)
    
    