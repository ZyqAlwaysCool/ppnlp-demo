import paddle
import numpy as np
import jieba
import jieba.posseg as pseg
from sklearn.metrics.pairwise import cosine_similarity

def tokenize(text):
   # 使用 jieba 进行分词和词性标注
   words = pseg.cut(text)
   # 过滤停用词和无意义的词性（如标点符号）
   filtered_words = [(word, flag) for word, flag in words if flag.startswith(('n', 'v')) and len(word) > 1]
   return [word for word, flag in filtered_words]

def build_cooccurrence_matrix(tokens, window_size=5):
   vocab = set(tokens)
   vocab = list(vocab)
   vocab_size = len(vocab)
   token_to_index = {token: idx for idx, token in enumerate(vocab)}
   
   cooccurrence_matrix = np.zeros((vocab_size, vocab_size))
   
   for i, token in enumerate(tokens):
       start = max(0, i - window_size)
       end = min(len(tokens), i + window_size + 1)
       for j in range(start, end):
           if i != j:
               cooccurrence_matrix[token_to_index[token]][token_to_index[tokens[j]]] += 1
   
   return cooccurrence_matrix, token_to_index

def textrank(cooccurrence_matrix, max_iter=100, d=0.85):
   vocab_size = cooccurrence_matrix.shape[0]
   scores = np.ones(vocab_size)
   for _ in range(max_iter):
       new_scores = (1 - d) + d * np.dot(cooccurrence_matrix, scores)
       if np.allclose(new_scores, scores):
           break
       scores = new_scores
   return scores

def extract_keywords(text, top_k=5):
   tokens = tokenize(text)
   cooccurrence_matrix, token_to_index = build_cooccurrence_matrix(tokens)
   scores = textrank(cooccurrence_matrix)
   
   index_to_token = {idx: token for token, idx in token_to_index.items()}
   sorted_indices = np.argsort(scores)[::-1]
   keywords = [index_to_token[idx] for idx in sorted_indices[:top_k]]
   
   return keywords

# 输入文本
text = "我家电表上的数字和账单不符，能帮我查一下吗？"

# 提取关键词
keywords = extract_keywords(text, top_k=5)

print(keywords)
