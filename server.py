import os
import shutil
import tempfile
import base64
import asyncio
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
    GALLERY_DIR, DET_THRESH, SERVER_PORT, SERVER_HOST,
    THINNING_INTERVAL, CAMERAS
)
from utils.face_engine import FaceEngine
from utils.gallery_manager import GalleryManager
from utils.history_manager import HistoryManager
from utils.cv_utils import imread_unicode
from utils.llm_service import get_llm_service, QueryCondition

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


async def periodic_thin_history():
    """定时执行历史记录稀疏化"""
    while True:
        await asyncio.sleep(THINNING_INTERVAL)
        try:
            history_manager.thin_history()
        except Exception as e:
            print(f"服务端: 稀疏化任务异常: {e}")


@app.on_event("startup")
async def start_thinning_task():
    asyncio.create_task(periodic_thin_history())


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
    include_images: bool = False,
    camera_id: Optional[str] = None
):
    """获取识别历史记录"""
    return history_manager.get_history(name, start_time, end_time, limit, offset, include_images, camera_id)


@app.get("/api/cameras")
def get_cameras():
    """获取摄像头列表"""
    cameras = []
    for cam in CAMERAS:
        if cam.get('enabled', False):
            cameras.append({
                'id': cam['id'],
                'name': cam['name']
            })
    return cameras


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
            camera_id = rec.get('camera_id', '')  # 摄像头ID (可选)
            camera_name = rec.get('camera_name', '')  # 摄像头名称 (可选)

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
                'image': img_bytes,
                'camera_id': camera_id,
                'camera_name': camera_name
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
    """
    自然语言查询接口

    使用 LLM 将自然语言查询转换为结构化查询条件
    支持的查询条件：
    - 姓名：张三、李四等
    - 时间：今天、昨天、最近3天、本周等
    - 置信度：置信度大于90%等
    - 摄像头：按摄像头名称筛选
    """
    try:
        from utils.llm_service import get_llm_service

        # 获取 LLM 服务
        llm_service = get_llm_service()

        # 检查 LLM 是否可用
        if not llm_service.enabled:
            return {
                "status": "error",
                "error_code": "LLM_NOT_CONFIGURED",
                "message": "AI 语义检索服务未配置。请在 llm_config.py 中配置 base_url 和 api_key 后重启服务。",
                "results": [],
                "count": 0
            }

        # 获取可用摄像头列表（用于解析摄像头名称）
        cameras = []
        for cam in CAMERAS:
            if cam.get('enabled', False):
                cameras.append({
                    'id': cam['id'],
                    'name': cam['name']
                })

        # 解析自然语言查询（LLM 失败时降级到规则解析）
        try:
            condition = llm_service.parse_query(req.query, cameras)
        except Exception:
            condition = llm_service._rule_based_parse(req.query, cameras)

        # 构建 SQL 查询
        sql, params = llm_service.build_sql_query(condition)

        # 执行查询
        results = history_manager.execute_query(sql, tuple(params))

        # 格式化结果（execute_query 返回字典列表）
        formatted_results = []
        for row in results:
            formatted_results.append({
                "id": row.get("id"),
                "person_name": row.get("person_name"),
                "confidence": row.get("confidence"),
                "timestamp": row.get("timestamp"),
                "camera_id": row.get("camera_id", ""),
                "camera_name": row.get("camera_name", ""),
                "image_url": row.get("image_url", f"/api/history_image/{row.get('id')}")
            })

        # 构建调试信息
        debug_info = {
            "original_query": req.query,
            "parsed_condition": {
                "person_names": condition.person_names,
                "confidence_min": condition.confidence_min,
                "confidence_max": condition.confidence_max,
                "time_periods": condition.time_periods,
                "camera_id": condition.camera_id,
                "limit": condition.limit
            },
            "sql": sql,
            "params": list(params)
        }

        return {
            "status": "ok",
            "results": formatted_results,
            "count": len(formatted_results),
            "debug": debug_info
        }

    except Exception as e:
        error_msg = str(e)
        print(f"服务端: 语义查询失败: {error_msg}")

        # 检查是否是 LLM 未配置错误
        if "LLM 服务未配置" in error_msg or "LLM_NOT_CONFIGURED" in error_msg:
            return {
                "status": "error",
                "error_code": "LLM_NOT_CONFIGURED",
                "message": "AI 语义检索服务未配置。请在 llm_config.py 中配置 base_url 和 api_key 后重启服务。",
                "results": [],
                "count": 0
            }

        # 其他错误也返回 JSON 格式
        return {
            "status": "error",
            "error_code": "QUERY_ERROR",
            "message": f"查询执行失败: {error_msg}",
            "results": [],
            "count": 0
        }


if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
