import paddle
from paddlenlp.transformers import BertTokenizer, BertModel
from paddle.io import Dataset, DataLoader
import paddle.nn as nn
import paddle.optimizer as optim
from paddlenlp.data import Stack, Pad, Tuple
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
#paddle.set_device('xpu')

# 标签映射类
class LabelMapper:
    def __init__(self):
        self.label_to_id = {}
        self.id_to_label = {}
        self.default_label = 0  # 默认标签用于填充空标签

    def fit(self, labels_df):
        unique_labels = set()
        for labels in labels_df.itertuples(index=False):
            if labels:  # 仅对存在的标签进行处理
                unique_labels.update([label for label in labels if label])
        
        # print(unique_labels)
        sorted_labels = sorted(unique_labels)
        for idx, label in enumerate(sorted_labels):
            self.label_to_id[label] = idx
            self.id_to_label[idx] = label

    def transform(self, labels):
        encoded_labels = []
        for label in labels:
            if label:  # 如果标签不为空，进行正常的映射
                encoded_labels.append(self.label_to_id.get(label, self.default_label))
            else:  # 如果标签为空，使用默认标签
                encoded_labels.append(self.default_label)
        
        # 确保标签数量为5个
        while len(encoded_labels) < 5:
            encoded_labels.append(self.default_label)
        
        return encoded_labels

    def inverse_transform(self, preds):
        return [self.id_to_label.get(pred, 'unknown') for pred in preds]

# 模型定义
class MultiLabelClassifier(nn.Layer):
    def __init__(self, pretrained_model, num_classes_per_label):
        super(MultiLabelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)

        #冻结
        for param in self.bert.parameters():
            param.stop_gradient = True
        
        # self.classifiers = nn.LayerList([
        #     nn.Linear(self.bert.config['hidden_size'], num_classes) for num_classes in num_classes_per_label
        #     ])
        self.classifiers = nn.LayerList([
            nn.Sequential(
                nn.Linear(self.bert.config['hidden_size'], 256), 
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256,num_classes)
            ) for num_classes in num_classes_per_label
            ])

    def forward(self, input_ids, token_type_ids):
        _, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids)
        logits = [classifier(pooled_output) for classifier in self.classifiers]
        return logits

# 文本预处理
def preprocess_text(text, tokenizer, max_len=128):
   encoded = tokenizer(text, max_seq_len=max_len, pad_to_max_seq_len=True)
   input_ids = paddle.to_tensor(encoded['input_ids'], dtype='int64')
   token_type_ids = paddle.to_tensor(encoded['token_type_ids'], dtype='int64')
   return input_ids, token_type_ids

# 加载模型
def load_model(model_path, label_mapper):
   model = MultiLabelClassifier(pretrained_model='bert-wwm-chinese', num_classes_per_label=[len(label_mapper.label_to_id) for _ in range(5)])
   model.set_state_dict(paddle.load(model_path))
   model.eval()
   return model

# 单条文本推理
def infer_single_text(text, model, tokenizer, label_mapper):
   input_ids, token_type_ids = preprocess_text(text, tokenizer)
   logits = model(input_ids.unsqueeze(0), token_type_ids.unsqueeze(0))
   preds = [paddle.argmax(logit, axis=1).numpy()[0] for logit in logits]
   decoded_preds = label_mapper.inverse_transform(preds)
   return decoded_preds

# 主函数
if __name__ == "__main__":
    # 假设已经有了label_mapper实例和tokenizer实例
    label_mapper = LabelMapper()  # 你需要先fit这个mapper
    tokenizer = BertTokenizer.from_pretrained('bert-wwm-chinese')
   
    data7 = pd.read_excel('./test.xlsx', engine='openpyxl')
    data_all = []
    data_all.append(data7)
    df = pd.concat(data_all,ignore_index=True)
    df = df[:20]
    texts_df = df['消息详情']
    label1,label1_id,label2,label2_id,label3,label3_id,label4,label4_id,label5,label5_id = df[['一级标签','一级标签编码','二级标签','二级标签编码','三级标签','三级标签编码','四级标签','四级标签编码','五级标签','五级标签编码']]
    labels = df[['一级标签编码','二级标签编码','三级标签编码','四级标签编码','五级标签编码']].values
    for i in range(len(labels)):
        for j in range(5):
            if (str(labels[i][j]) == 'nan'):
                labels[i][j] = '000'
    labels_df = pd.DataFrame(labels)
    
    label_mapper = LabelMapper()
    label_mapper.fit(labels_df)

    # 加载模型
    model_path = 'best_model.pdparams'
    model = load_model(model_path, label_mapper)

    # 单条文本
    text = "用户在2024-01-01 01:05:31发起咨询，询问电量电费查询。"
    predictions = infer_single_text(text, model, tokenizer, label_mapper)

    # 打印预测结果
    print(predictions)