# -*- coding: utf-8 -*-
import os
import json

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.transformers import BertTokenizer, BertModel
import pandas as pd

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

def get_label_mapper():
    label_mapper = LabelMapper()
    label_mapper.label_to_id = {'000': 0, 'a01': 1, 'a02': 2, 'a03': 3, 'a04': 4, 'a05': 5, 'a99': 6, 'b01': 7, 'b02': 8, 'b03': 9, 'b07': 10, 'c01': 11}
    label_mapper.id_to_label = {0: '000', 1: 'a01', 2: 'a02', 3: 'a03', 4: 'a04', 5: 'a05', 6: 'a99', 7: 'b01', 8: 'b02', 9: 'b03', 10: 'b07', 11: 'c01'}

    return label_mapper
    


class Predictor:
    '''
    InitModel函数  模型初始化参数,注意不能自行增加删除函数入参
    ret            是否正常: 正常True,异常False
    err_message    错误信息: 默认normal
    return ret,err_message
    '''
    def InitModel(self): 
        ret=True
        err_message="normal"
        '''
        模型初始化,由用户自行编写
        加载出错时给ret和err_message赋值相应的错误
        *注意模型应为相对路径
        '''
        ### 请在try内编写模型初始化,便于捕获错误
        try: 
            ### 加载模型
            model_path = './best_model.pdparams' #!!!【修改】改为训练好的模型路径
            self.label_mapper = get_label_mapper()

            self.tokenizer = BertTokenizer.from_pretrained('bert-wwm-chinese')
            self.model = MultiLabelClassifier(pretrained_model='bert-wwm-chinese', num_classes_per_label=[len(self.label_mapper.label_to_id) for _ in range(5)])
            self.model.set_state_dict(paddle.load(model_path))
            self.model.eval()
            pass
        except Exception as err:
            ret=False
            err_message="[Error] model init failed,err_message:[{}]".format(ExceptionMessage(err))
            print(err_message)

        return ret,err_message


    '''
    Detect         模型推理函数,注意不能自行增加删除函数入参
    text           输入单个文本,非批量
    return         字典dict
    '''
    def Detect(self, text): 
        ## 请在try内编写推理代码,便于捕获错误
        try: 
            input_ids, token_type_ids = preprocess_text(text, self.tokenizer)
            logits = self.model(input_ids.unsqueeze(0), token_type_ids.unsqueeze(0))
            preds = [paddle.argmax(logit, axis=1).numpy()[0] for logit in logits]
            decoded_preds = self.label_mapper.inverse_transform(preds)
            
            detect_result = {}
            for i in range(1, 6):
                if decoded_preds[i-1] != '000':
                    detect_result["predict"+str(i)+"_code"] = decoded_preds[i-1]
                else:
                    detect_result["predict"+str(i)+"_code"] = ""
            if text is None:
                print("[Error] text is None.")
                return detect_result
        
            '''
            模型推理部分,由用户自行编写
            detect_result输出格式示例:
            {
                "predict1_code":"a01",
                "predict2_code":"b01",
                "predict3_code":"c01",
                "predict4_code":"",
                "predict5_code":""
            }
            数据格式说明:
            1.字典中key:predict1-5_code分别表示 一级标签到五级标签的标签编码
            2.字典中value均为由标签编码,类型为字符串,例如"a01",标签编码请参考比赛文档
            3.每一级标签仅有1种标签,默认缺省时为""
            '''
            return detect_result
        except Exception as err:
            print("[Error] predictor.Detect failed.err_message:{}".format(ExceptionMessage(err)))
            return err
        

### 获取异常文件+行号+信息
def ExceptionMessage(err):
    err_message=(
                str(err.__traceback__.tb_frame.f_globals["__file__"])
                +":"
                +str(err.__traceback__.tb_lineno)
                +"行:"
                +str(err)
            )
    return err_message
    


if __name__ == '__main__':
    ###备注说明:main函数提供给用户内测,修改后[不影响]评估
    predictor=Predictor()
    ret,err_message=predictor.InitModel()
    if ret:
        text="用户在2024-01-01 01:05:31发起咨询，询问电量电费查询。"
        detect_result=predictor.Detect(text)
        print("detect_result",detect_result)
    else:
        print("[Error] InitModel failed. ret",ret,err_message)