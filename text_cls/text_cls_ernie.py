import paddle
from paddle.nn import Linear
from paddlenlp.transformers import ErnieTokenizer, ErnieModel
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
import paddle.nn.functional as F

pretrained_model_path = 'ernie-3.0-base-zh'

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
      max_seq_len=max_length,
      truncation=True,
      padding='max_length',
      return_attention_mask=True,
      return_token_type_ids=True
  )
  label = label_map[example['label']]
  return encoded_input['input_ids'], encoded_input['token_type_ids'], encoded_input['attention_mask'], label

# 加载tokenizer
tokenizer = ErnieTokenizer.from_pretrained(pretrained_model_path)

# 定义数据加载器
batch_size = 32
train_data_loader = paddle.io.DataLoader(
  dataset=train_ds.map(lambda example: convert_example(example, tokenizer, label_map)),
  batch_size=batch_size,
  shuffle=True,
  collate_fn=lambda x: (Pad(axis=0, pad_val=tokenizer.pad_token_id)([d[0] for d in x]),
                       Pad(axis=0, pad_val=tokenizer.pad_token_type_id)([d[1] for d in x]),
                       Pad(axis=0, pad_val=0)([d[2] for d in x]),
                       Stack()([d[3] for d in x]))
)

test_data_loader = paddle.io.DataLoader(
  dataset=test_ds.map(lambda example: convert_example(example, tokenizer, label_map)),
  batch_size=batch_size,
  shuffle=False,
  collate_fn=lambda x: (Pad(axis=0, pad_val=tokenizer.pad_token_id)([d[0] for d in x]),
                       Pad(axis=0, pad_val=tokenizer.pad_token_type_id)([d[1] for d in x]),
                       Pad(axis=0, pad_val=0)([d[2] for d in x]),
                       Stack()([d[3] for d in x]))
)

# 定义模型
class ErnieForSequenceClassification(paddle.nn.Layer):
  def __init__(self, num_classes):
      super(ErnieForSequenceClassification, self).__init__()
      self.ernie = ErnieModel.from_pretrained(pretrained_model_path)
      self.classifier = Linear(self.ernie.config['hidden_size'], num_classes)

  def forward(self, input_ids, token_type_ids, attention_mask):
      outputs = self.ernie(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
      pooled_output = outputs[1]
      logits = self.classifier(pooled_output)
      return logits

model = ErnieForSequenceClassification(num_classes=len(label_map))

# 定义优化器和损失函数
optimizer = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=2e-5)
criterion = paddle.nn.CrossEntropyLoss()

# 训练函数
def train(model, data_loader, optimizer, criterion, epochs=3):
  model.train()
  for epoch in range(epochs):
      total_loss = 0
      for batch in data_loader:
          input_ids, token_type_ids, attention_mask, labels = batch
          logits = model(input_ids, token_type_ids, attention_mask)
          loss = criterion(logits, labels)
          loss.backward()
          optimizer.step()
          optimizer.clear_grad()
          total_loss += loss.item()
      print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader)}")

# 评估函数
def evaluate(model, data_loader, label_map):
  model.eval()
  total_correct = 0
  total_samples = 0
  with paddle.no_grad():
      for batch in data_loader:
          input_ids, token_type_ids, attention_mask, labels = batch
          logits = model(input_ids, token_type_ids, attention_mask)
          predictions = paddle.argmax(logits, axis=1)
          total_correct += (predictions == labels).sum().item()
          total_samples += labels.shape[0]
  accuracy = total_correct / total_samples
  print(f"Accuracy: {accuracy * 100:.2f}%")
  return accuracy

# 推理函数
def predict(text, model, tokenizer, label_map, max_length=128):
  model.eval()
  encoded_input = tokenizer(
      text=text,
      max_seq_len=max_length,
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

# 开始训练
train(model, train_data_loader, optimizer, criterion, epochs=10)

# 评估模型
evaluate(model, test_data_loader, label_map)

# 保存模型
paddle.save(model.state_dict(), 'ernie_text_classification_model.pdparams')

# 加载模型
model = ErnieForSequenceClassification(num_classes=len(label_map))
model.load_dict(paddle.load('ernie_text_classification_model.pdparams'))
model.eval()

# 示例文本推理
text = "我最近更换了电器，电费却没有减少，这是为什么？"
prediction = predict(text, model, tokenizer, label_map)
print(f"Predicted label: {prediction}")