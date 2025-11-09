

# 设置LMDB数据库路径:
datasets_path= os.path.split(os.path.dirname(__file__))[0]
router = APIRouter(prefix="/dataset", tags=["数据集服务"])

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.dtype):
            return obj.name
        return json.JSONEncoder.default(self, obj)
    
    
# 设置相关路由：

@router.post("/get_all_data_name", summary="获取所有数据集名称")
async def get_all_data_name()->list:
    """
    获取当前LMDB数据集的所有名称.\n
    可用sindre-lmdb的Reader直接打开\n
    
    Returns:\n
        数据集名称:list\n

    """
    out =[]
    for  file_name in os.listdir(datasets_path):
        if "数据集" in file_name:
            out.append(file_name)
    return out


@router.post("/get_db_data_idx", summary="根据提供的数据集名及索引,返回内部data_db的结果")
async def get_db_data_idx(db_name:str,idx:int):
    """
    根据提供的数据集名及索引,返回内部data_db的结果;\n
    因为sindre-lmdb规定构建时,会产生data,meta数据库:\n
    Args:\n
        db_name: string  数据库名\n
        idx: int  索引值\n
    return:\n
        dict : 索引对应的值\n
    
    """
    dir_path = os.path.join(datasets_path, db_name)
    idx = int(idx)
    if not os.path.exists(dir_path):
        return f"找不到{dir_path}文件夹"

    try:
        with  lmdb.Reader(dirpath=dir_path) as db:
            data =db[idx]
        return json.dumps(data, cls=NumpyArrayEncoder)
    except Exception as e:
        return f"错误:{e}"
    
@router.post("/get_db_size", summary="根据提供的数据集名,返回内部data_db的大小")
async def get_db_size(db_name)->int:
    """
    根据提供的数据集名,返回内部data_db的大小;\n
    因为sindre-lmdb规定构建时,会产生data,meta数据库:\n
    Args:\n
        db_name: string  数据库名\n
    return:\n
        size : int  数据库大小\n
    
    """
    print(db_name,type(db_name),11)
    dir_path = os.path.join(datasets_path, db_name)
    if not os.path.exists(dir_path):
        return f"找不到{dir_path}文件夹"
    try:
        with  lmdb.Reader(dirpath=dir_path) as db:
            nb_sample = db.nb_samples
        return nb_sample
    except Exception as e:
        return f"错误:{e}"
    

@router.post("/get_db_meta_key", summary="根据提供的数据集名及键名,返回内部meta_db的结果")
async def get_db_meta_key(db_name:str,key:str):
    """
    根据提供的数据集名及键名,返回内部meta_db的结果;\n
    因为sindre-lmdb规定构建时,会产生data,meta数据库:\n
    Args:\n
        db_name: string  数据库名\n
        idx: str  键名\n
    return:\n
        data 对应的值\n
    
    """
    dir_path = os.path.join(datasets_path,db_name)
    if not os.path.exists(dir_path):
        return f"找不到{dir_path}文件夹"

    try:
        with  lmdb.Reader(dirpath=dir_path) as db:
            data =db.get_meta_str(key)
        return data
    except Exception as e:
        return f"错误:{e}"
        
        

@router.post("/get_db_info", summary="根据提供的数据集名,返回数据库信息")
async def get_db_data_idx(db_name:str)->str:
    """
   根据提供的数据集名,返回数据库信息; 注意：数据集大会非常慢\n
    Args:\n
        db_name: string  数据库名;\n
    return:\n
        dict : {"nb_sample":数据库大小(int),"data":索引对应的值};\n
    
    """
    dir_path = os.path.join(datasets_path,db_name)
    if not os.path.exists(dir_path):
        return f"找不到{dir_path}文件夹"

    try:
        with  lmdb.Reader(dirpath=dir_path) as db:
            info = db.__repr__()
            print(info)
        return info
    except Exception as e:
        return f"错误:{e}"



@router.post("/get_data_specification", summary="根据提供的数据集名,返回数据库get_data_specification")
async def get_data_specification(db_name:str)->dict:
    """
   根据提供的数据集名,返回数据库第一个键的类型; \n
    Args:\n
        db_name: string  数据库名;\n
    return:\n
        dict : 第一个键的类型;\n
    
    """
    dir_path = os.path.join(datasets_path,db_name)
    if not os.path.exists(dir_path):
        return f"找不到{dir_path}文件夹"

    try:
        with  lmdb.Reader(dirpath=dir_path) as db:
            info = db.get_data_specification(0)
            s= json.dumps(info,cls=NumpyArrayEncoder)
        return json.loads(s)
    except Exception as e:
        return f"错误:{e}"
    

@router.post("/get_data_keys", summary="根据提供的数据集名,返回数据库get_data_keys")
async def get_data_keys(db_name:str)->list:
    """
   根据提供的数据集名,返回数据库子数据库所有键;\n
    Args:\n
        db_name: string  数据库名;\n
    return:\n
        list : 子数据库所有键
    """
    dir_path = os.path.join(datasets_path,db_name)
    if not os.path.exists(dir_path):
        return f"找不到{dir_path}文件夹"

    try:
        with  lmdb.Reader(dirpath=dir_path) as db:
            info = db.get_data_keys()
        return info
    except Exception as e:
        return f"错误:{e}"



    
