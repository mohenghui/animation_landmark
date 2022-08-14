import os
from config import SplitData
from writexml import WriterXml
compareA=[os.path.splitext(i)[0] for i in os.listdir(SplitData.img_path)]
compareB=[os.path.splitext(i)[0] for i in os.listdir(WriterXml().xml_annotation_path)]
# os.listdir(compareA)
# os.listdir(compareB)
# print(compareA)
# print(compareB)
# print(set(compareA).intersection(set(compareB)))
print(len(compareA),len(compareB))
print(list(set(compareA).difference(set(compareB))))