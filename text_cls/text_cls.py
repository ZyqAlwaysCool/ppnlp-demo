import paddle
from paddle.nn import Linear, Dropout
from paddlenlp.transformers import BertTokenizer, BertModel
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
import paddle.nn.functional as F
from paddle.optimizer import AdamW
from paddle.nn import CrossEntropyLoss
from paddle.io import DataLoader
import numpy as np
from paddle.optimizer.lr import NoamDecay
from sklearn.metrics import precision_recall_fscore_support
import logging

logging.disable(logging.WARNING)
logging.disable(logging.INFO)
pretrained_model_path = '../bert-base-chinese'

# 加载数据集
def read_data(file_path):
   with open(file_path, 'r', encoding='utf-8') as f:
       for line in f:
           text, label = line.strip().split('\t')
           yield {'text': text, 'label': label}

train_ds = load_dataset(read_data, file_path='./data/train.txt', lazy=False)
test_ds = load_dataset(read_data, file_path='./data/dev.txt', lazy=False)

# 构建标签映射
def build_label_map(dataset):
   label_set = set()
   for example in dataset:
       label_set.add(example['label'])
   label_map = {label: idx for idx, label in enumerate(label_set)}
   return label_map

label_map = build_label_map(train_ds)
print(label_map)

# 定义数据预处理函数
def convert_example(example, tokenizer, label_map, max_length=128):
   encoded_input = tokenizer(
       text=example['text'],
       #max_seq_len=max_length,
       max_length=max_length,
       truncation=True,
       padding='max_length',
       return_attention_mask=True,
       return_token_type_ids=True
   )
   label = label_map[example['label']]
   return encoded_input['input_ids'], encoded_input['token_type_ids'], encoded_input['attention_mask'], label

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)

# 定义数据加载器
batch_size = 32
train_data_loader = DataLoader(
   dataset=train_ds.map(lambda example: convert_example(example, tokenizer, label_map)),
   batch_size=batch_size,
   shuffle=True,
   collate_fn=lambda x: (Pad(axis=0, pad_val=tokenizer.pad_token_id)([d[0] for d in x]),
                        Pad(axis=0, pad_val=tokenizer.pad_token_type_id)([d[1] for d in x]),
                        Pad(axis=0, pad_val=0)([d[2] for d in x]),
                        Stack()([d[3] for d in x]))
)

test_data_loader = DataLoader(
   dataset=test_ds.map(lambda example: convert_example(example, tokenizer, label_map)),
   batch_size=batch_size,
   shuffle=False,
   collate_fn=lambda x: (Pad(axis=0, pad_val=tokenizer.pad_token_id)([d[0] for d in x]),
                        Pad(axis=0, pad_val=tokenizer.pad_token_type_id)([d[1] for d in x]),
                        Pad(axis=0, pad_val=0)([d[2] for d in x]),
                        Stack()([d[3] for d in x]))
)


#earlystopping策略
class EarlyStopping:
   def __init__(self, patience=3, verbose=False, delta=0):
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
       paddle.save(model.state_dict(), 'bert_text_classification_model_best.pdparams')
       self.val_loss_min = val_loss

# 定义模型
class BertForSequenceClassification(paddle.nn.Layer):
   def __init__(self, num_classes):
       super(BertForSequenceClassification, self).__init__()
       self.bert = BertModel.from_pretrained(pretrained_model_path)
       self.dropout = Dropout(0.1)
       self.classifier = Linear(self.bert.config['hidden_size'], num_classes)

   def forward(self, input_ids, token_type_ids, attention_mask):
       outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
       pooled_output = self.dropout(outputs[1])
       logits = self.classifier(pooled_output)
       return logits

model = BertForSequenceClassification(num_classes=len(label_map))

# 定义优化器和损失函数
optimizer = AdamW(parameters=model.parameters(), learning_rate=2e-5, weight_decay=0.01)
criterion = CrossEntropyLoss()

# 学习率预热和衰减
num_training_steps = len(train_data_loader) * 10
num_warmup_steps = int(num_training_steps * 0.1)
scheduler = NoamDecay(
  d_model=768,  # BERT模型的隐藏层大小
  warmup_steps=num_warmup_steps,
  verbose=True
)

# 梯度裁剪
grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)

# 训练函数中的学习率更新和Early Stopping
def train(model, data_loader, optimizer, criterion, scheduler, grad_clip, epochs=3, early_stopping=None):
  model.train()
  for epoch in range(epochs):
      total_loss = 0
      for step, batch in enumerate(data_loader):
          input_ids, token_type_ids, attention_mask, labels = batch
          logits = model(input_ids, token_type_ids, attention_mask)
          loss = criterion(logits, labels)
          loss.backward()
          optimizer.step()
          optimizer.clear_grad()
          scheduler.step()  # 更新学习率
          total_loss += loss.item()
      
      avg_loss = total_loss / len(data_loader)
      print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")
      
      # 验证集评估
      val_loss, accuracy, precision, recall, f1 = evaluate(model, test_data_loader, label_map, return_metrics=True)
      print(f"Validation - Loss: {val_loss}, Accuracy: {accuracy * 100:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
      
      if early_stopping is not None:
          early_stopping(val_loss, model)
          if early_stopping.early_stop:
              print("Early stopping")
              break

# 评估函数，增加返回更多指标
def evaluate(model, data_loader, label_map, return_metrics=False):
  model.eval()
  total_correct = 0
  total_samples = 0
  all_labels = []
  all_predictions = []
  total_loss = 0
  with paddle.no_grad():
      for batch in data_loader:
          input_ids, token_type_ids, attention_mask, labels = batch
          logits = model(input_ids, token_type_ids, attention_mask)
          loss = criterion(logits, labels)
          total_loss += loss.item()
          predictions = paddle.argmax(logits, axis=1)
          total_correct += (predictions == labels).sum().item()
          total_samples += labels.shape[0]
          all_labels.extend(labels.numpy())
          all_predictions.extend(predictions.numpy())
  
  accuracy = total_correct / total_samples
  val_loss = total_loss / len(data_loader)
  
  if return_metrics:
      precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
      return val_loss, accuracy, precision, recall, f1
  else:
      return accuracy

# 推理函数
def predict(text, model, tokenizer, label_map, max_length=128):
   model.eval()
   encoded_input = tokenizer(
       text=text,
       #max_seq_len=max_length,
       max_length=max_length,
       truncation=True,
       padding='max_length',
       return_attention_mask=True,
       return_token_type_ids=True
   )
   input_ids = paddle.to_tensor([encoded_input['input_ids']])
   token_type_ids = paddle.to_tensor([encoded_input['token_type_ids']])
   attention_mask = paddle.to_tensor([encoded_input['attention_mask']])

   with paddle.no_grad():
       logits = model(input_ids, token_type_ids, attention_mask)
       predictions = paddle.argmax(logits, axis=1)
   
   # 将预测的整数标签转换回原始标签
   predicted_label = list(label_map.keys())[list(label_map.values()).index(predictions.item())]
   return predicted_label

# 初始化EarlyStopping
early_stopping = EarlyStopping(patience=50, verbose=True)

# 开始训练
train(model, train_data_loader, optimizer, criterion, scheduler, grad_clip, epochs=200, early_stopping=early_stopping)

# 评估模型
evaluate(model, test_data_loader, label_map)

# 保存模型
paddle.save(model.state_dict(), 'bert_text_classification_model.pdparams')

# 加载模型
model = BertForSequenceClassification(num_classes=len(label_map))
model.load_dict(paddle.load('bert_text_classification_model_best.pdparams'))
model.eval()

# 示例文本推理
text = "你好，我这里是小区业主，想咨询一下为什么我家最近的电费突然增加了很多？以前每个月大概在300元左右，这个月居然超过500元了。"
prediction = predict(text, model, tokenizer, label_map)
print(f"Predicted label: {prediction}")