import os

# 目标文件夹路径（请替换成你的实际路径）
folder_path = "../logs"

# 要删除的文件名
target_file = "net_epoch1.pth"

# 遍历文件夹及其子文件夹
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file == target_file:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"已删除：{os.path.abspath(file_path)}")
            except Exception as e:
                print(f"删除失败 {os.path.abspath(file_path)}: {e}")