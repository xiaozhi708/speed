import shutil
import os
from sklearn.model_selection import train_test_split

def find_all_files(root, suffix=None):
    res = []
    for root, _, files in os.walk(root):
        for f in files:
            if suffix is not None and not f.endswith(suffix):
                continue
            res.append(os.path.join(root, f))
    #     print(res)
    return res


def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        os.makedirs(path)

        print(path + ' 创建成功')
        return True
    else:
        print(path + ' 目录已存在')
        return False


def copyfile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        if not os.path.exists(dstfile):
            mkdir(dstfile)  # 创建路径
        shutil.copy(srcfile, dstfile)  # 复制文件


# 往文件中追加内容
def text_append_write(path, name, msg):
    mkdir(path)
    full_path = path + name + '.txt'
    with open(full_path, 'a+') as f:
        f.write(msg + '\n')  # 加\n换行显示

if __name__ == "__main__":
  path = "/home/lzc/Caltech101/101_ObjectCategories/"
  out_path = "/home/lzc/Caltech101/"
  suffix = ".jpg"

  res = find_all_files(path, suffix)

  train_list, val_list = train_test_split(res, test_size=0.5,random_state=42)
  for i in train_list:
    catalog_name = i.split("/")[-2]  # 得到图片的种类名
    text_append_write(out_path + "train/", "train_catalog", i+" "+catalog_name)
    copyfile(i, out_path + "train/" + catalog_name)
  for i in val_list:
    catalog_name = i.split("/")[-2]  # 得到图片的种类名
    text_append_write(out_path + "val/", "val_catalog", i+" "+catalog_name)
    copyfile(i, out_path + "val/" + catalog_name)