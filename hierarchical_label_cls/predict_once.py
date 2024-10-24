import argparse
import os

import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.utils.log import logger

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', default="gpu", help="Select which device to train model, defaults to gpu.")
#parser.add_argument("--dataset_dir", required=True, default=None, type=str, help="Local dataset directory should include data.txt and label.txt")
parser.add_argument("--params_path", default="./checkpoint/", type=str, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
args = parser.parse_args()
# yapf: enable

def preprocess_function(tokenizer, text, max_seq_length):
   encoded_inputs = tokenizer(text, max_length=max_seq_length, truncation=True, padding=True, return_tensors="pd")
   return encoded_inputs

@paddle.no_grad()
def predict_single_text(text):
   """
   Predicts the label for a single text.
   """
   paddle.set_device(args.device)
   model = AutoModelForSequenceClassification.from_pretrained(args.params_path)
   tokenizer = AutoTokenizer.from_pretrained(args.params_path)

   label_list = []
   #label_path = os.path.join(args.dataset_dir, "label.txt")
   label_path = './data/label.txt'
   with open(label_path, "r", encoding="utf-8") as f:
       for line in f:
           label_list.append(line.strip())

   # Preprocess the single text
   inputs = preprocess_function(tokenizer, text, args.max_seq_length)

   model.eval()
   logits = model(**inputs)
   probs = F.sigmoid(logits).numpy()[0]

   labels = [label_list[i] for i, p in enumerate(probs) if p > 0.5]

   hierarchical_labels = {}
   logger.info("text: {}".format(text))
   logger.info("prediction result: {}".format(",".join(labels)))
   for label in labels:
       for i, l in enumerate(label.split("##")):
           if i not in hierarchical_labels:
               hierarchical_labels[i] = []
           if l not in hierarchical_labels[i]:
               hierarchical_labels[i].append(l)
   for d in range(len(hierarchical_labels)):
       logger.info("level {} : {}".format(d + 1, ",".join(hierarchical_labels[d])))
   logger.info("--------------------")

   return labels

if __name__ == "__main__":
   # Example usage: predict a single text
   text = "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了"
   predict_single_text(text)