import os

data_dir = r"./data/raw/EuroSAT_RGB"

if not os.path.exists(data_dir):
    print("数据路径不存在：", data_dir)
else:
    class_names = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    class_names.sort()

    print("数据路径存在")
    print("类别数量：", len(class_names))
    print("类别名称：")
    for name in class_names:
        print("-", name)