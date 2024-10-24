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
#paddle.set_device('xpu')

class EarlyStopping:
   def __init__(self, model_idx, patience=5, verbose=False, delta=0):
       self.patience = patience
       self.verbose = verbose
       self.counter = 0
       self.best_score = None
       self.early_stop = False
       self.val_loss_min = np.Inf
       self.delta = delta
       self.model_idx = model_idx

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
       paddle.save(model.state_dict(), 'best_model_{}.pdparams'.format(self.model_idx))
       self.val_loss_min = val_loss

# 标签映射类
class LabelMapper:
   def __init__(self):
       self.label_to_id = {}
       self.id_to_label = {}
       self.default_label = 0  # 默认标签用于填充空标签
       
   def fit(self, labels):
        # 过滤掉非字符串类型的标签
        str_labels = [label for label in labels if isinstance(label, str)]
        unique_labels = set(str_labels)
        sorted_labels = sorted(unique_labels)
        for idx, label in enumerate(sorted_labels):
            self.label_to_id[label] = idx
            self.id_to_label[idx] = label

   def transform(self, labels):
       return [self.label_to_id.get(label, self.default_label) for label in labels]

   def inverse_transform(self, preds):
       return [self.id_to_label.get(pred, 'unknown') for pred in preds]

# 数据集构建
class MyDataset(Dataset):
   def __init__(self, texts_df, labels_df, tokenizer, label_mapper, max_len=128):
       self.texts = texts_df.tolist()
       self.labels = labels_df.tolist()
       self.tokenizer = tokenizer
       self.label_mapper = label_mapper
       self.max_len = max_len

   def __getitem__(self, index):
       text = self.texts[index]
       label = self.labels[index]
       encoded = self.tokenizer(text, max_seq_len=self.max_len, padding='max_length', truncation=True)
       input_ids = encoded['input_ids']
       token_type_ids = encoded['token_type_ids']
       
       encoded_label = self.label_mapper.transform([label])[0]

       return input_ids, token_type_ids, encoded_label

   def __len__(self):
       return len(self.texts)

# 模型定义
class SingleLabelClassifier(nn.Layer):
   def __init__(self, pretrained_model, num_classes):
       super(SingleLabelClassifier, self).__init__()
       self.bert = BertModel.from_pretrained(pretrained_model)

       # 冻结
       for param in self.bert.parameters():
           param.stop_gradient = True

       self.classifier = nn.Linear(self.bert.config['hidden_size'], num_classes)

   def forward(self, input_ids, token_type_ids):
       _, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids)
       logits = self.classifier(pooled_output)
       return logits

class FocalLoss(nn.Layer):
   def __init__(self, a=1, g=2):
       super(FocalLoss, self).__init__()
       self.a = a
       self.g = g

   def forward(self, logits, labels):
       ce_loss = paddle.nn.functional.cross_entropy(logits, labels)
       pt = paddle.exp(-ce_loss)
       focal_loss = self.a * (1-pt)**self.g * ce_loss
       return focal_loss

# 训练和验证代码
def train_and_validate(texts_df, labels_df, tokenizer, model, label_mapper, model_idx, epochs=100, batch_size=1024, learning_rate=2e-5):
   train_texts, val_texts, train_labels, val_labels = train_test_split(texts_df, labels_df, test_size=0.2, random_state=42)

   train_dataset = MyDataset(train_texts, train_labels, tokenizer, label_mapper)
   val_dataset = MyDataset(val_texts, val_labels, tokenizer, label_mapper)

   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=Tuple(Pad(axis=0, pad_val=0), Pad(axis=0, pad_val=0), Stack()))
   val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=Tuple(Pad(axis=0, pad_val=0), Pad(axis=0, pad_val=0), Stack()))

   criterion = FocalLoss(a=1, g=2)
   optimizer = optim.AdamW(parameters=model.parameters(), learning_rate=learning_rate)

   # 初始化 EarlyStopping
   early_stopping = EarlyStopping(model_idx, patience=5, verbose=True)

   for epoch in range(epochs):
       model.train()
       for batch in train_loader:
           input_ids, token_type_ids, labels = batch
           labels = paddle.to_tensor(labels, dtype=paddle.int64)

           optimizer.clear_grad()
           logits = model(input_ids, token_type_ids)
           loss = criterion(logits, labels)
           loss.backward()
           optimizer.step()

       # 验证
       model.eval()
       val_loss = 0.0
       with paddle.no_grad():
           for batch in val_loader:
               input_ids, token_type_ids, labels = batch
               labels = paddle.to_tensor(labels, dtype=paddle.int64)
               logits = model(input_ids, token_type_ids)
               loss = criterion(logits, labels)
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
   model = SingleLabelClassifier(pretrained_model='bert-wwm-chinese', num_classes=len(label_mapper.label_to_id))
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
           preds = paddle.argmax(logits, axis=1).numpy()
           predictions.extend(preds)
           true_labels.extend(labels)

   decoded_preds = label_mapper.inverse_transform(predictions)

   cnt = 0
   for i in range(len(decoded_preds)):
       if decoded_preds[i] == labels_df.values[i]:
           cnt += 1
   acc = 1.0 * cnt / len(texts_df) * 100
   print(f'准确率为{acc}%')

   np.save('decoded_predictions.npy', np.array(decoded_preds))
   print("Predictions saved to decoded_predictions.npy.")

# Main Function
if __name__ == "__main__":
   data7 = pd.read_excel('./test.xlsx', engine='openpyxl')

   data_all = [data7]
   df = pd.concat(data_all, ignore_index=True)
   df = df[:20]
   print(df.shape)

   texts_df = df['消息详情']
   labels_df = df[['一级标签编码', '二级标签编码', '三级标签编码', '四级标签编码', '五级标签编码']]

   # 替换NaN标签为'000'
   for col in labels_df.columns:
       labels_df[col].fillna('000', inplace=True)

   label_mappers = [LabelMapper() for _ in range(5)]
   for i, label_col in enumerate(labels_df.columns):
       label_mappers[i].fit(labels_df[label_col].tolist())

   tokenizer = BertTokenizer.from_pretrained('bert-wwm-chinese')

   models = [SingleLabelClassifier(pretrained_model='bert-wwm-chinese', num_classes=len(label_mappers[i].label_to_id)) for i in range(5)]
   
   epochs = 1
   for i, model in enumerate(models):
       print(f"Training model for {labels_df.columns[i]}")
       print('label_mappers id2label=({}) label2id=({})'.format(label_mappers[i].id_to_label, label_mappers[i].label_to_id))
       train_and_validate(texts_df, labels_df[labels_df.columns[i]], tokenizer, model, label_mappers[i], i, epochs)
       load_and_test_model(f'best_model_{i}.pdparams', texts_df[:20], labels_df[labels_df.columns[i]][:20], tokenizer, label_mappers[i])