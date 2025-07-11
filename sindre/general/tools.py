


   
    
def load(path):
    """
    基于文件扩展名自动解析不同格式的文件并加载数据。

    支持的文件类型包括但不限于：
    - JSON: 解析为字典或列表
    - TOML: 解析为字典
    - INI: 解析为ConfigParser对象
    - Numpy: .npy/.npz格式的数值数组
    - Pickle: Python对象序列化格式
    - TXT: 纯文本文件
    - LMDB: 轻量级键值数据库
    - PyTorch: .pt/.pth模型文件
    - PTS: pts的3D点云数据文件
    - constructionInfo: XML格式的牙齿模型数据

    对于未知格式，尝试使用vedo库加载，支持多种3D模型格式。

    Args:
        path (str): 文件路径或目录路径(LMDB格式)

    Returns:
        Any: 根据文件类型返回对应的数据结构，加载失败时返回None。
             - JSON/TOML: dict或list
             - INI: ConfigParser对象
             - Numpy: ndarray或NpzFile
             - Pickle: 任意Python对象
             - TXT: 字符串
             - LMDB: sindre.lmdb.Reader对象(使用后需调用close())
             - PyTorch: 模型权重或张量
             - PTS: 包含牙齿ID和边缘点的字典
             - constructionInfo: 包含项目信息和多颗牙齿数据的字典
             - vedo支持的格式: vedo.Mesh或vedo.Volume等对象

    Raises:
        Exception: 记录加载过程中的错误，但函数会捕获并返回None

    Notes:
        - LMDB数据需要手动关闭: 使用完成后调用data.close()
        - 3D模型加载依赖vedo库，确保环境已安装
        - PyTorch模型默认加载到CPU，避免CUDA设备不可用时的错误
    """
    try:
        if path.endswith(".json"):
            import json
            with open(path, 'r',encoding="utf-8") as f:
                data = json.load(f)

        elif path.endswith(".toml"):
            import tomllib
            with open(path, "rb") as f:
                data = tomllib.load(f)

                
        elif path.endswith(".ini"):
            from configparser import ConfigParser   
            data = ConfigParser() 
            data.read(path)  
    
            
        elif path.endswith((".npy","npz")):
            import numpy as np 
            data = np.load(path, allow_pickle=True)
            
        
        elif path.endswith((".pkl",".pickle")): 
            import pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)
        
        elif path.endswith(".txt"):       
            with open(path, 'r') as f:
                data = f.read()
                
        elif path.endswith((".db",".lmdb",".mdb",".yx")) or  os.path.isdir(path):
            import sindre     
            data= sindre.lmdb.Reader(path,True)
            log.info("使用完成请关闭 data.close()")
            
        elif path.endswith((".pt", ".pth")):
            import torch
            # 使用 map_location='cpu' 避免CUDA设备不可用时的错误
            data = torch.load(path, map_location='cpu')

          
        elif path.endswith(".pts"):
            # up3d线格式
            data = []
            tooth_id=None 
            with open(path, 'r') as f:
                data_pts = f.readlines()
                try:
                    tooth_id = int(data_pts[0][-3:-1])
                except:
                    pass
                lines =data_pts[1:-1]

            data = [[float(i) for i in line.split()] for line in lines]
            data = {"id":tooth_id,"margin_points":np.array(data).reshape(-1,3)}
               
                        
        elif path.endswith('.constructionInfo'):
            # exo导出格式
            import xml.etree.ElementTree as ET
            root = ET.parse(path).getroot()
            
            project_name = root.findtext("ProjectName", "")
            teeth_data = []
            
            for tooth in root.findall('Teeth/Tooth'):
                tooth_id = tooth.findtext('Number')
                if not tooth_id:
                    continue
                    
                # 解析中心点
                center_xml = tooth.find('Center')
                if center_xml is None:
                    continue
                    
                center = [
                    float(center_xml.findtext('x', '0')),
                    float(center_xml.findtext('y', '0')),
                    float(center_xml.findtext('z', '0'))
                ]
                
                # 解析旋转矩阵 (3x3)
                axis_elements = ['Axis', 'AxisMesial', 'AxisBuccal']
                matrix = []
                for element in axis_elements:
                    e = tooth.find(element)
                    if e is None:
                        matrix.extend([0.0, 0.0, 0.0])  # 默认值
                    else:
                        matrix.extend([
                            float(e.findtext('x', '0')),
                            float(e.findtext('y', '0')),
                            float(e.findtext('z', '0'))
                        ])
                
                # 解析边缘点
                margin = []
                margin_xml = tooth.find('Margin')
                if margin_xml is not None:
                    for vec in margin_xml.findall('Vec3'):
                        p = [
                            float(vec.findtext('x', '0')),
                            float(vec.findtext('y', '0')),
                            float(vec.findtext('z', '0'))
                        ]
                        margin.append(p)
                
                teeth_data.append({
                    'id': int(tooth_id),
                    'center': np.array(center).reshape(1,3),
                    'rotation_matrix': np.array(matrix).reshape(3,3),
                    'margin_points':np.array(margin).reshape(-1,3),
                })
            return {"project_name":project_name,"teeth_data":teeth_data}
        else:
            data = vedo.load(path)
        return data
    except Exception as e:
        log.error(f"Error loading .pts file: {e}")
        return None 
  