# -*- coding: utf-8 -*-
# version:2024.8.8 
# 2024.8月以后竞赛必须使用该版本
# 项目名称:客户标签画像分析
import os
import json

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.transformers import BertTokenizer, BertModel
import pandas as pd
#paddle.set_device('xpu')

all_label_mapper = {
    '1': {'id2label': {0: 'a01', 1: 'a02', 2: 'a03', 3: 'a04', 4: 'a05', 5: 'a99', 6: 'b01'}, 'label2id': {'a01': 0, 'a02': 1, 'a03': 2, 'a04': 3, 'a05': 4, 'a99': 5, 'b01': 6}},
    '2': {'id2label': {0: '000', 1: 'b01', 2: 'b02', 3: 'b03', 4: 'b07'}, 'label2id': {'000': 0, 'b01': 1, 'b02': 2, 'b03': 3, 'b07': 4}},
    '3': {'id2label': {0: '000', 1: 'a02', 2: 'b02', 3: 'c01'}, 'label2id': {'000': 0, 'a02': 1, 'b02': 2, 'c01': 3}},
    '4': {'id2label': {0: '000', 1: 'b03'}, 'label2id': {'000': 0, 'b03': 1}},
    '5': {'id2label': {0: '000'}, 'label2id': {'000': 0}}
}

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

def get_label_mappers():
   label_mappers = [LabelMapper() for _ in range(5)]
   label_mappers[0].id_to_label = all_label_mapper['1']['id2label']
   label_mappers[0].label_to_id = all_label_mapper['1']['label2id']
   label_mappers[1].id_to_label = all_label_mapper['2']['id2label']
   label_mappers[1].label_to_id = all_label_mapper['2']['label2id']
   label_mappers[2].id_to_label = all_label_mapper['3']['id2label']
   label_mappers[2].label_to_id = all_label_mapper['3']['label2id']
   label_mappers[3].id_to_label = all_label_mapper['4']['id2label']
   label_mappers[3].label_to_id = all_label_mapper['4']['label2id']
   label_mappers[4].id_to_label = all_label_mapper['5']['id2label']
   label_mappers[4].label_to_id = all_label_mapper['5']['label2id']
   return label_mappers


class BertSingleLabelClassifier:
    def __init__(self, pretrained_model_path, pdparams, label_mapper):
        self.label_mapper = label_mapper
        self.pretrained_model = pretrained_model_path
        self.model = SingleLabelClassifier(pretrained_model=pretrained_model_path,
                                      num_classes=len(label_mapper.label_to_id))
        self.model.set_state_dict(paddle.load(pdparams))
        self.model.eval()
        
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    
    def predict(self, texts):
        encoded = self.tokenizer(texts, max_seq_len=128, truncation=True, padding='max_length')
        input_ids = paddle.to_tensor(encoded['input_ids'], dtype='int64').unsqueeze(0)  # 增加一个批次维度
        token_type_ids = paddle.to_tensor(encoded['token_type_ids'], dtype='int64').unsqueeze(0)
        # 推理
        with paddle.no_grad():
            logits = self.model(input_ids, token_type_ids)
            pred = paddle.argmax(logits, axis=1).numpy()[0]  # 获取预测结果
                
        # 反向映射标签
        predicted_label = self.label_mapper.inverse_transform([pred])[0]        

        return predicted_label


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
            self.label_mappers = get_label_mappers()
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
            decoded_preds = []
            for i in range(5):
                cls = BertSingleLabelClassifier('bert-wwm-chinese', './best_model_{}.pdparams'.format(i), self.label_mappers[i])
                pred = cls.predict(text)
                decoded_preds.append(pred)
            
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