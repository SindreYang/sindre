# 数据转换函数
from sindre.lmdb import Reader
def Lmdb2Coco(db_path,out_dir):
    # lmdb数据转换成coco数据集格式

    db = Reader(db_path)
    keys=db.get_data_keys(0)
    # 提供数据库keys，让用户通过界面，选择键进行对应写入，如果选择为空，则为空(比如只做分割数据(
    for i in range(len(db)):
        data = db[i] #dict类型字典




    db.close()

def Lmdb2Voc():
    # lmdb数据转换成Voc数据集格式
    pass

def Lmdb2ShapeNet():
    # lmdb数据转换成ShapeNet数据集格式
    pass

def Lmdb2ModelNet():
    # lmdb数据转换成ModelNet数据集格式
    pass


