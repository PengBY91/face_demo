import os
import shutil
import tempfile
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from typing import List

# 导入新模块
from config import (
    ARCFACE_MODEL_PATH, PROVIDERS,
    GALLERY_DIR, DET_THRESH, SERVER_PORT, SERVER_HOST
)
from utils.face_engine import FaceEngine
from utils.gallery_manager import GalleryManager

# 初始化
app = FastAPI(title="Face DB Manager")

# Mount static files to gallery directory (only for other static assets if any)
os.makedirs(GALLERY_DIR, exist_ok=True)
# app.mount("/static", StaticFiles(directory=GALLERY_DIR), name="static") # 禁用静态目录映射，改用 API 读取

# 初始化人脸引擎和库管理器
print("服务端: 正在初始化...")
engine = FaceEngine(
    rec_model_path=ARCFACE_MODEL_PATH,
    providers=PROVIDERS,
    det_thresh=DET_THRESH
)
gallery = GalleryManager(GALLERY_DIR)
print("服务端: 初始化完成")


@app.get("/", response_class=HTMLResponse)
async def read_index():
    if os.path.exists("templates/index.html"):
        with open("templates/index.html") as f:
            return f.read()
    return "templates/index.html not found"


@app.post("/upload/")
async def upload_face(name: str = Form(...), file: UploadFile = File(...)):
    """上传人脸照片并注册到库"""
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    try:
        # 读取图片
        import cv2
        img = cv2.imread(tmp_path)
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
                detail=f"检测到重复人员: 该人脸与库中 '{dup_name}' 相似度为 {sim:.2f}"
            )

        # 添加到库
        success = gallery.add_person(name, aligned_face, embedding)
        if not success:
            raise HTTPException(500, "添加到库失败")

        return {"status": "ok", "message": f"Added {name}"}

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
    """列出所有人脸 for UI"""
    faces = []
    all_faces = gallery.list_all()

    for name, info in all_faces.items():
        # 改为使用动态 API 获取数据库中的图片
        img_url = f"/api/face_image/{name}"
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
        return {"status": "deleted"}
    return {"status": "error", "message": "Not found"}


@app.put("/api/faces/{old_name}/{new_name}")
def rename_face(old_name: str, new_name: str):
    """重命名人脸"""
    if gallery.rename_person(old_name, new_name):
        return {"status": "renamed"}
    return {"status": "error", "message": "Failed"}


# Record Storage
class RecordStore:
    def __init__(self, filepath="records.json"):
        self.filepath = filepath
        self.records = []
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            import json
            try:
                with open(self.filepath, 'r') as f:
                    self.records = json.load(f)
            except:
                self.records = []

    def save(self):
        import json
        with open(self.filepath, 'w') as f:
            json.dump(self.records, f, ensure_ascii=False)

    def add_record(self, name):
        import time
        # timestamp format: YYYY-MM-DD HH:MM:SS
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        record = {"name": name, "time": timestamp}
        self.records.insert(0, record)  # Newest first
        # Limit to last 1000 records
        if len(self.records) > 1000:
            self.records = self.records[:1000]
        self.save()


record_store = RecordStore()


class RecordRequest(BaseModel):
    name: str


@app.post("/api/record")
def add_record(req: RecordRequest):
    record_store.add_record(req.name)
    return {"status": "recorded"}


@app.get("/api/records")
def get_records():
    return record_store.records


if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
