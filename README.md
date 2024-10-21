### 简介
本项目包含基于paddlenlp的nlp相关任务示例代码, 旨在为类似任务、项目提供解决思路。

### 快速开始
1. 准备conda环境: `conda create -n py37-ppnlp python=3.7`
2. 安装依赖: `pip install -r requirements.txt`
3. 以文本分类任务为例, 执行命令训练模型: `cd ./text_cls && python text_cls.py`

### 目录结构
* bert-base-chinese: bert官方预训练模型, 如果使用paddlenlp自动下载的预训练模型, 默认模型路径为`~/.paddlenlp`, 其中缺少了config.json文件, 可以在huggingface bert模型里补充下载, 放到目录下
* intent_analysis: 意图分析任务示例. 本质是一个多标签文本分类任务
* keyword_extract: 关键词抽取任务示例. 使用jieba+textRank算法实现
* text_cls: 文本分类任务示例. 包含了bert和ernie两个模型代码实现, 略有差异

### 注意
* 由于模型过大，因此上传到git保存的数据中不包含源模型文件, 联网环境下paddlenlp会自动下载, 离线环境下可自行下载模型放到当前目录. 需注意的是: 若使用在线环境下载模型, 则需要把代码中模型相对路径修改, 例如将`pretrained_model_path = ../bert_base_chinese`改为`pretrained_model_path = bert_base_chinese`

