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
from sklearn.metrics import f1_score, precision_score, recall_score

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

#数据增强
def augment(text,p=0.1):
    words = text.split()
    new_words = []
    for word in words:
        if random.random() < p:
            continue
        new_words.append(word)
    return ''.join(new_words)

# 数据集构建
class MyDataset(Dataset):
    def __init__(self, texts_df, labels_df, tokenizer, label_mapper, max_len=128):
        self.texts = texts_df.tolist()
        self.labels = labels_df.values.tolist()
        self.tokenizer = tokenizer
        self.label_mapper = label_mapper
        self.max_len = max_len
        # self.aug = augment() #开启数据增强

    def __getitem__(self, index):
        text = self.texts[index]
        # text = self.aug(text) #开启数据增强
        labels = self.labels[index]
        encoded = self.tokenizer(text, max_seq_len=self.max_len, padding='max_length', truncation=True)
        input_ids = encoded['input_ids']
        token_type_ids = encoded['token_type_ids']
        
        encoded_labels = self.label_mapper.transform(labels)

        return input_ids, token_type_ids, encoded_labels

    def __len__(self):
        return len(self.texts)

class EarlyStopping:
   def __init__(self, patience=5, verbose=False, delta=0):
       self.patience = patience
       self.verbose = verbose
       self.counter = 0
       self.best_score = None
       self.early_stop = False
       self.val_loss_min = np.Inf
       self.delta = delta

   def __call__(self, val_loss, model):
       score = -val_loss

       if self.best_score is None:
           self.best_score = score
           self.save_checkpoint(val_loss, model)
       elif score < self.best_score + self.delta:
           self.counter += 1
           if self.verbose:
               print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
           if self.counter >= self.patience:
               self.early_stop = True
       else:
           self.best_score = score
           self.save_checkpoint(val_loss, model)
           self.counter = 0

   def save_checkpoint(self, val_loss, model):
       '''Saves model when validation loss decrease.'''
       if self.verbose:
           print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
       paddle.save(model.state_dict(), 'best_model.pdparams')
       self.val_loss_min = val_loss

# 模型定义
class MultiLabelClassifier(nn.Layer):
    def __init__(self, pretrained_model, num_classes_per_label):
        super(MultiLabelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)

        #冻结
        for param in self.bert.parameters():
            param.stop_gradient = True
        
        self.classifiers = nn.LayerList([
            nn.Sequential(
                nn.Linear(self.bert.config['hidden_size'], 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            ) for num_classes in num_classes_per_label
            ])

    def forward(self, input_ids, token_type_ids):
        _, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids)
        logits = [classifier(pooled_output) for classifier in self.classifiers]
        return logits

class FocalLoss(nn.Layer):
    def __init__(self,a=1,g=2):
        super(FocalLoss,self).__init__()
        self.a = a
        self.g = g
    
    def forward(self,logits,labels):
        ce_loss = paddle.nn.functional.cross_entropy(logits,labels)
        pt = paddle.exp(-ce_loss)
        focal_loss = self.a * (1-pt)** self.g * ce_loss
        return focal_loss

# 训练和验证代码
def train_and_validate(texts_df, labels_df, tokenizer, model, label_mapper, epochs=100, batch_size=1024, learning_rate=2e-5):
   train_texts, val_texts, train_labels, val_labels = train_test_split(texts_df, labels_df, test_size=0.2, random_state=42)

   train_dataset = MyDataset(train_texts, train_labels, tokenizer, label_mapper)
   val_dataset = MyDataset(val_texts, val_labels, tokenizer, label_mapper)
   
   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=Tuple(Pad(axis=0, pad_val=0), Pad(axis=0, pad_val=0), Stack()))
   val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=Tuple(Pad(axis=0, pad_val=0), Pad(axis=0, pad_val=0), Stack()))

   criterion = FocalLoss(a=1, g=2)
   optimizer = optim.AdamW(parameters=model.parameters(), learning_rate=learning_rate)
   
   # 初始化 EarlyStopping
   early_stopping = EarlyStopping(patience=5, verbose=True)

   for epoch in range(epochs):
       model.train()
       for batch in train_loader:
           input_ids, token_type_ids, labels = batch
           labels = [paddle.to_tensor(label, dtype=paddle.int64) for label in zip(*labels)]

           optimizer.clear_grad()
           logits = model(input_ids, token_type_ids)
           loss = sum(criterion(logit, label) for logit, label in zip(logits, labels) if label.sum() > 0)
           loss.backward()
           optimizer.step()

       # 验证
       model.eval()
       val_loss = 0.0
       with paddle.no_grad():
           for batch in val_loader:
               input_ids, token_type_ids, labels = batch
               labels = [paddle.to_tensor(label, dtype=paddle.int64) for label in zip(*labels)]
               logits = model(input_ids, token_type_ids)
               loss = sum(criterion(logit, label) for logit, label in zip(logits, labels) if label.sum() > 0)
               val_loss += loss.numpy().item()

       val_loss /= len(val_loader)

       print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss}")

       # 调用 EarlyStopping
       early_stopping(val_loss, model)

       if early_stopping.early_stop:
           print("Early stopping")
           break

# 测试并保存结果
def load_and_test_model(model_path, texts_df, labels_df, tokenizer, label_mapper):
   model = MultiLabelClassifier(pretrained_model='bert-wwm-chinese', num_classes_per_label=[len(label_mapper.label_to_id) for _ in range(5)])
   model.set_state_dict(paddle.load(model_path))
   model.eval()

   test_dataset = MyDataset(texts_df, labels_df, tokenizer, label_mapper)
   test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=Tuple(Pad(axis=0, pad_val=0), Pad(axis=0, pad_val=0), Stack()))

   predictions = []
   true_labels = []
   with paddle.no_grad():
       for batch in test_loader:
           input_ids, token_type_ids, labels = batch
           logits = model(input_ids, token_type_ids)
           preds = [paddle.argmax(logit, axis=1).numpy() for logit in logits]
           predictions.extend(zip(*preds))
           true_labels.extend(zip(*[label.numpy() for label in labels]))

   decoded_preds = [label_mapper.inverse_transform(pred) for pred in predictions]

   cnt = 0
   for i in range(len(decoded_preds)):
       for j in range(5):
           if decoded_preds[i][j] == labels_df.values[i][j]:
               cnt += 1
   acc = 1.0 * cnt / len(texts_df[0]) * 5 * 100
   print(f'准确率为{acc}%')

   np.save('decoded_predictions.npy', np.array(decoded_preds))
   print("Predictions saved to decoded_predictions.npy.")



# Main Function
# Main Function
if __name__ == "__main__":
    data = pd.read_excel('./test.xlsx', engine='openpyxl')
    
    data_all = []
    data_all.append(data)
    df = pd.concat(data_all,ignore_index=True)
    df = df[:20]
    print(df.shape)

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

    tokenizer = BertTokenizer.from_pretrained('bert-wwm-chinese')

    model = MultiLabelClassifier(pretrained_model='bert-wwm-chinese', num_classes_per_label=[len(label_mapper.label_to_id) for _ in range(5)])

    train_and_validate(texts_df, labels_df, tokenizer, model, label_mapper, 5)
    
    load_and_test_model('best_model.pdparams', texts_df[:20], labels_df[:20], tokenizer, label_mapper)