import os
import jieba

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

def merge_text(sourcepath, merged_name):
    """合并sourcepath目录下所有文件，输出至工作目录下merged_name"""
    files = os.listdir(sourcepath)
    with open((os.getcwd() + '\\' + merged_name), 'w+', encoding= "ANSI") as p:
        for file in files:
            fullpath = sourcepath + '\\' + file
            with open(fullpath, 'r', encoding= "ANSI") as f:
                p.write(f.read())
            f.close()

def word_seg(path):
    """对path指向文本进行基于jieba的分词，返回以空格分隔单词的文本"""
    with open(path, "r", encoding="ANSI") as f:
        data = f.read().replace('\u3000','').replace('\n','').replace(' ','')
        text = cut_sentences(data)
        f.close()
        seg_list = [" ".join(jieba.lcut(e, use_paddle=True, cut_all=False)) for e in text]
    
    return seg_list

def is_chinese(uchar):
    if uchar >= '\u4e00' and uchar <= '\u9fa5':
        return True
    else:
        return False

def is_number(uchar):
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False

def word_seg_tolist(path):
    """对path指向文本进行基于jieba的分词，返回各个单词的列表"""
    with open(path, "r", encoding="ANSI") as f:
        data = f.read()
        data = cut_sentences(data)
        content_str = []
        for sentence in data:
            str = ''
            for i in sentence:
                if is_chinese(i) or is_number(i):
                    str = str + i
            content_str.append(str)
        f.close()
        seg_list = []
        for e in content_str:
            seg_list.append(jieba.lcut(e, use_paddle=True, cut_all=False))  
    return seg_list

if __name__ == "__main__":
    # merge_text('text','merged.txt')
    word_seg_tolist('越女剑.txt')
