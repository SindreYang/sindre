# -*- coding: UTF-8 -*-
__author__ = 'sindre'


import sindre
import os
import multiprocessing as mp


def main():
    print(dir(sindre))
    dir(sindre.utils3d.tools)
    print(f"\t\t\033[0;33;40mYou cannot get success without failure! Come on!\033[0m\033[0;31;40m{__author__}!\033[0m")
    print(sindre.__path__)
    print(dir(sindre.lmdb))
    print(help(sindre.lmdb.Reader))

def win_tools_test():
    sindre.win_tools.taskbar.set_windows_alpha(255)



def lmdb_test():
    import numpy as np 
    data = {"test":[1,23,4]}
    db=sindre.lmdb.Writer("./test",1)
    db.put_samples(data)
    db.close()
    
    db_r= sindre.lmdb.Reader("./test")
    print(db_r)
    

def write_worker(queue,dirpath,map_size_limit):
    writer = sindre.lmdb.Writer(dirpath,map_size_limit=map_size_limit, multiprocessing=True)
    try:
        while True:
            data = queue.get()
            if data is None:  # 终止信号
                break
            
            writer.put_samples(data)
    except Exception as e:
        print(f"写入失败: {e}")
    writer.close()


def lmdb_test_multi(data,dirpath="./multi.db",map_size_limit=100):
    sindre.lmdb.Writer(dirpath, map_size_limit=map_size_limit).close() 
    queue = mp.Queue()
    writer_process = mp.Process(target=write_worker, args=(queue,dirpath,map_size_limit))
    writer_process.start()

    # 其他进程通过 queue.put(data) 发送数据
    for item in data:
        queue.put(item)
    queue.put(None)  # 结束信号
    writer_process.join()

    
def fun_worker(queue,dirpath,map_size_limit):
    writer =sindre.lmdb.Writer(dirpath,map_size_limit=map_size_limit, multiprocessing=True)
    try:
        while True:
            path = queue.get()
            if path is None:  # 终止信号
                break
            
            # 读入每个path，获取数据
            data={path:[1,2]}   
            # 假设做了大量处理，得到新数据
            for i in range(10**3):
                c=i*5
            data = {path:[c]}
            
            # 写入数据
            writer.put_samples(data)
    except Exception as e:
        print(f"写入失败: {e}")
    writer.close()   
    
    


if __name__ == '__main__':
    #main()
    #win_tools_test()
    #lmdb_test()
    #lmdb_test_multi()
    
    

    data = [i for  i  in range(10)]
    #lmdb_test_multi(data)
    sindre.lmdb.multiprocessing_writer([ str(path) for path  in range(12)],fun_worker)
    print(sindre.lmdb.Reader("./multi.db"))
          
    
        


