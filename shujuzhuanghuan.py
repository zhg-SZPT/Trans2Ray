import json

with open("data/Xray_2/annotations/instances_val2017.json", 'r') as load_f:
    load_dict = json.load(load_f)
    annotations = load_dict["annotations"]
    for annotation in annotations:
        img_id = annotation["image_id"]
        idx, type = img_id.split("_")
        annotation["image_id"] = int(idx)

with open("data/Xray_2/annotations/instances_val2017.json", "w") as f:
    json.dump(load_dict, f)
    print("加载入文件完成...")
