import jieba
import re

def cut_sentences(path):
	"""文本分句处理"""
	content = open(path, "r", encoding="ANSI").read()
	content = re.sub(r'\u3000','',content)
	content = re.sub(r'\n','',content)
	end_flag = ['?', '!', '.', '？', '！', '。', '…']	
	content_len = len(content)
	sentences = []
	tmp_char = ''
	for idx, char in enumerate(content):
		# 拼接字符
		if char == '「' or char == '」':
			continue
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
  
sen = cut_sentences('datasets/白马啸西风.txt')
with open('datasets\seg2.txt', "w+", encoding="utf-8") as f:
        for sentence in sen:
            f.write(sentence + '\n')        
