import os
from fastapi import FastAPI, HTTPException, Query, Path, Body
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Optional, Tuple
from sindre.lmdb.WebViewer import  tools  # 导入工具函数
from sindre.lmdb.WebViewer.config import DBInfo, DBContent,DBInfoUpdate
from fastapi.middleware.cors import CORSMiddleware
DATA_ROOT =r"./datasets"



app = FastAPI(title="3D/2D数据监控系统")
# 前端静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")

# 配置跨域：允许所有来源（开发环境，生产环境需指定具体域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def read_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

# API接口
@app.get("/api/databases", response_model=List[DBInfo])
def get_databases(
        db_type: Optional[str] = Query(None, description="数据库类型过滤"),
        tag: Optional[str] = Query(None, description="标签过滤"),
        search: Optional[str] = Query(None, description="搜索关键词")
):
    """获取数据库列表，支持过滤和搜索"""
    db_list = tools.scan_directories(DATA_ROOT)

    # 应用过滤
    if db_type:
        db_list = [db for db in db_list if db.type == db_type]

    if tag:
        db_list = [db for db in db_list if tag in db.tags]

    if search:
        search_lower = search.lower()
        db_list = [
            db for db in db_list
            if search_lower in db.name.lower() or
                any(search_lower in t.lower() for t in db.tags) or
                # 搜索数据键
                any(search_lower in key.lower() for key in db.data_keys)
        ]

    return db_list

@app.get("/api/databases/{db_id}", response_model=DBContent)
def get_database_content(
        db_id: str = Path(..., description="数据库ID"),
        selected_keys: Optional[str] = Query(None, description="用户选择的键，用逗号分隔")
):
    """获取数据库详细内容和预览数据，支持用户选择键"""
    db_list = tools.scan_directories(DATA_ROOT)
    db_info = next((db for db in db_list if db.id == db_id), None)

    if not db_info:
        raise HTTPException(status_code=404, detail="数据库不存在")

    # 解析用户选择的键
    selected_keys_list = selected_keys.split(',') if selected_keys else []

    # 根据类型读取数据
    content = tools.read_database_content(db_info, selected_keys_list)
    return content

@app.put("/api/databases/{db_id}/info")
def update_database_info(
        db_id: str = Path(..., description="数据库ID"),
        update_data: DBInfoUpdate = Body(...)
):
    """更新数据库信息（类型和标签）"""
    # 查找数据库
    db_list = tools.scan_directories(DATA_ROOT)
    db_info = next((db for db in db_list if db.id == db_id), None)

    if not db_info:
        raise HTTPException(status_code=404, detail="数据库不存在")

    # 调用工具函数更新信息（这里先打印作为示例）
    print(f"更新数据库信息 - ID: {db_id}")
    print(f"新类型: {update_data.type}")
    print(f"新标签: {update_data.tags}")
    tools.set_db_info(db_info, update_data.type, update_data.tags)

    return {"status": "success", "message": "数据库信息已更新"}

@app.get("/api/databases/{db_id}/download")
def download_database(db_id: str = Path(..., description="数据库ID")):
    """下载数据库文件"""
    db_list = tools.scan_directories(DATA_ROOT)
    db_info = next((db for db in db_list if db.id == db_id), None)

    if not db_info or not os.path.exists(db_info.path):
        raise HTTPException(status_code=404, detail="数据库不存在")

    return FileResponse(
        path=db_info.path,
        filename=db_info.ori_name,
        media_type='application/octet-stream'
    )

@app.get("/api/tags")
def get_all_tags() -> List[str]:
    """获取所有可用标签"""
    db_list = tools.scan_directories(DATA_ROOT)
    tags = set()
    for db in db_list:
        tags.update(db.tags)
    return sorted(tags)

@app.get("/api/preview")
def get_preview(
        db_path: str = Query(..., description="数据库路径"),
        db_type: str = Query(..., description="数据库类型"),
        preview_type: str = Query(..., description="预览类型"),
        selected_key: str = Query(..., description="渲染"),
        data_index: int = Query(0, description="索引"),
        specific_vertices: Optional[str] = Query(None, description="顶点"),
        specific_faces: Optional[str] = Query(None, description="面片"),
        specific_image: Optional[str] = Query(None, description="图像")
) -> Dict:
    """获取数据预览"""
    # 收集特定类型信息
    specific_info = {}
    if specific_vertices:
        specific_info["vertices"] = specific_vertices
    if specific_faces:
        specific_info["faces"] = specific_faces
    if specific_image:
        specific_info["image"] = specific_image

    # 调用工具函数生成预览
    preview_data = tools.get_data_preview(
        db_path=db_path,
        db_type=db_type,
        specific_info=specific_info,
        preview_type=preview_type,
        selected_key=selected_key,
        data_index=data_index
    )

    return JSONResponse(content=preview_data)


def set_data_root(path):
    global DATA_ROOT
    DATA_ROOT = os.path.abspath(path)

def main():
    data_root = input("请输入数据根目录的绝对路径: ").strip()
    if os.path.isdir(data_root):
        set_data_root(data_root)
        os.makedirs(data_root, exist_ok=True)
    else:
        set_data_root(DATA_ROOT)
        print(f"错误：路径 '{data_root}' 不存在或不是有效目录,将使用内置目录:{os.path.abspath(DATA_ROOT)}")
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    import uvicorn
    print("请打开： http://127.0.0.1:12345/")
    uvicorn.run("sindre.lmdb.WebViewer.App:app", host="0.0.0.0", port=12345, reload=False)


if __name__ == "__main__":
    main()