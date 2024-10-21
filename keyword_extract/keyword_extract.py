from paddlenlp import Taskflow

ner = Taskflow("ner")

print(ner('我家电表上的数字和账单不符，能帮我查一下吗？'))