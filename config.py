


class SplitData:
    root_path = "./data"
    annotation_path = './data/annotations'
    img_path = './data/images'


class SaveImage:
    save_path = "./data/oimages"
    save_edit_path="./data/edit"

class GetGT:
    save_smilar="./data/check"
    predict_dir="./data/yolov7pred"
class DrawCricle:
    classes = ["chin", "eyebrow", "circle", "head", "nose"]

# class WriterXml:
#     detect_class=["circle"]
#     xml_annotation_path="./data/annotations_xml"