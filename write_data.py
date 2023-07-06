# 打开文件并追加内容
file_path = "./dataset/test.txt"  # 文件路径
path_tx="C:/Users/cwf/Desktop/summary/fby/fby/dataset/endemic_fluorosis/test/mild/mild_33.png"
content = "\n{}\t0".format(path_tx)
print(content)
# 打开文件并追加内容 
with open(file_path, "a") as file:
    file.write(content)

print("追加成功！")
