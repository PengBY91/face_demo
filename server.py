import os
import shutil
import tempfile
import base64
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from typing import List, Optional, Dict

# 导入新模块
from config import (
    ARCFACE_MODEL_PATH, PROVIDERS,
    GALLERY_DIR, DET_THRESH, SERVER_PORT, SERVER_HOST
)
from utils.face_engine import FaceEngine
from utils.gallery_manager import GalleryManager
from utils.history_manager import HistoryManager
from utils.cv_utils import imread_unicode

# 初始化
app = FastAPI(title="Face DB Manager")

# Mount static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# 初始化人脸引擎和库管理器
print("服务端: 正在初始化...")
engine = FaceEngine(
    rec_model_path=ARCFACE_MODEL_PATH,
    providers=PROVIDERS,
    det_thresh=DET_THRESH
)
gallery = GalleryManager(GALLERY_DIR)
history_manager = HistoryManager(GALLERY_DIR)
print("服务端: 初始化完成")


@app.get("/", response_class=HTMLResponse)
async def read_index():
    if os.path.exists("templates/index.html"):
        with open("templates/index.html", encoding='utf-8') as f:
            return f.read()
    return "templates/index.html not found"


@app.get("/history", response_class=HTMLResponse)
async def read_history():
    if os.path.exists("templates/history.html"):
        with open("templates/history.html", encoding='utf-8') as f:
            return f.read()
    return "templates/history.html not found"


@app.get("/search", response_class=HTMLResponse)
async def read_search():
    if os.path.exists("templates/nl_query.html"):
        with open("templates/nl_query.html", encoding='utf-8') as f:
            return f.read()
    return "templates/nl_query.html not found"


@app.post("/upload/")
async def upload_face(name: str = Form(...), file: UploadFile = File(...)):
    """上传人脸照片并注册到库"""
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    try:
        # 读取图片
        # 读取图片 (支持中文或特殊路径)
        img = imread_unicode(tmp_path)
        if img is None:
            raise HTTPException(400, "无法读取上传的图片")

        # 检测并提取最大人脸
        face = engine.get_largest_face(img)
        if face is None:
            raise HTTPException(400, "无法检测到人脸")

        embedding = face['embedding']
        aligned_face = face['aligned_face']

        # 重复检测 (Duplicate Detection)
        duplicate = gallery.find_duplicate(embedding, threshold=0.7)
        if duplicate:
            dup_name, sim = duplicate
            raise HTTPException(
                status_code=400, 
                detail=f"检测到重复人员：该人脸与库中已登记的 '{dup_name}' 相似度为 {sim:.2f}，请勿重复上传。"
            )

        # 添加到库
        success = gallery.add_person(name, aligned_face, embedding)
        if not success:
            raise HTTPException(500, "添加到库失败")

        return {"status": "ok", "message": f"成功添加 {name}"}

    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/sync_data")
def sync_data():
    """
    演示端调用的核心接口：获取所有特征
    返回格式: { "张三": [[0.1, ...]], "李四": [[...]] }
    """
    names, embeddings = gallery.load_embeddings()

    # 转换为旧格式兼容 (每个人一个列表)
    result = {}
    for name, embedding in zip(names, embeddings):
        result[name] = [embedding.tolist()]

    return result


@app.get("/api/faces")
def list_faces():
    """列出所有人脸 for UI，直接嵌入 Base64 图片以减少请求数"""
    all_faces = gallery.list_all()
    
    faces = []
    for name, info in all_faces.items():
        img_data = info.get("face_image")
        if img_data:
            # 将二进制图片转为 Base64 Data URI
            base64_img = base64.b64encode(img_data).decode('utf-8')
            img_url = f"data:image/jpeg;base64,{base64_img}"
        else:
            img_url = ""
            
        faces.append({
            "name": name,
            "image_url": img_url,
            "created_at": info.get("created_at", "")
        })
        
    return faces


@app.get("/api/face_image/{name}")
async def get_face_image(name: str):
    """从数据库直接返回图片内容"""
    from fastapi.responses import Response
    import cv2
    img = gallery.get_face_image(name)
    if img is None:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # 转换为 JPEG 字节流
    _, img_encoded = cv2.imencode('.jpg', img)
    return Response(content=img_encoded.tobytes(), media_type="image/jpeg")


@app.delete("/api/faces/{name}")
def delete_face(name: str):
    """删除人脸"""
    if gallery.delete_person(name):
        return {"status": "deleted", "message": "已删除"}
    return {"status": "error", "message": "未找到人员"}


@app.put("/api/faces/{old_name}/{new_name}")
def rename_face(old_name: str, new_name: str):
    """重命名人脸"""
    return {"status": "renamed"}


@app.get("/api/history")
def get_history(
    name: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    include_images: bool = False
):
    """获取识别历史记录"""
    return history_manager.get_history(name, start_time, end_time, limit, offset, include_images)


@app.get("/api/history_image/{record_id}")
async def get_history_image(record_id: int):
    """从数据库返回历史抓拍图片"""
    from fastapi.responses import Response
    import cv2
    img = history_manager.get_history_image(record_id)
    if img is None:
        raise HTTPException(status_code=404, detail="Image not found")
    
    _, img_encoded = cv2.imencode('.jpg', img)
    return Response(content=img_encoded.tobytes(), media_type="image/jpeg")


@app.post("/api/records_batch")
async def add_records_batch(records: List[Dict]):
    """批量记录识别结果 (由客户端调用, 使用 JSON 格式包含 base64 图片)"""
    try:
        processed_records = []
        for rec in records:
            name = rec.get('name')
            confidence = rec.get('confidence')
            image_b64 = rec.get('image_b64')
            
            if image_b64:
                # 去掉可能存在的 'data:image/jpeg;base64,' 前缀
                if ',' in image_b64:
                    image_b64 = image_b64.split(',')[1]
                img_bytes = base64.b64decode(image_b64)
            else:
                img_bytes = b""
                
            processed_records.append({
                'name': name,
                'confidence': confidence,
                'image': img_bytes
            })
            
        success = history_manager.add_history_records_batch(processed_records)
        if not success:
            raise HTTPException(500, "批量保存历史记录失败")
            
        return {"status": "ok", "count": len(processed_records)}
    except Exception as e:
        print(f"服务端: 批量保存记录失败: {e}")
        raise HTTPException(500, detail=str(e))


@app.post("/api/record_v2")
async def add_record_v2(name: str = Form(...), confidence: float = Form(...), file: UploadFile = File(...)):
    """记录一次识别结果 (由客户端调用)"""
    try:
        # 直接读取原始字节，避免冗余的 cv2 解码（除非需要处理图片，但这里只是保存）
        contents = await file.read()
        
        # 保存到历史记录
        success = history_manager.add_history_record(name, confidence, contents)
        if not success:
            raise HTTPException(500, "保存历史记录失败")
            
        return {"status": "ok"}
    except Exception as e:
        print(f"服务端: 保存记录失败: {e}")
        raise HTTPException(500, detail=str(e))


class NLQueryRequest(BaseModel):
    query: str


@app.post("/api/query_nl")
async def query_nl_endpoint(req: NLQueryRequest):
    """自然语言查询接口"""
    # 这里集成 LLM 逻辑
    sql = "SELECT id, person_name, confidence, timestamp, face_image FROM recognition_history"
    params = []
    
    # 模拟处理
    query_lower = req.query.lower()
    if "张三" in query_lower:
        sql += " WHERE person_name = ?"
        params.append("张三")
    elif "李四" in query_lower:
        sql += " WHERE person_name = ?"
        params.append("李四")
    
    sql += " ORDER BY timestamp DESC LIMIT 50"
    
    try:
        results = history_manager.execute_query(sql, tuple(params))
        return {"status": "ok", "results": results, "sql": sql}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
