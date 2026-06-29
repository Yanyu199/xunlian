import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config import API_TITLE, CODEXDATA_DIR, MODEL_SAVE_PATH, OUTPUT_DIR, SCALER_SAVE_PATH
from core.forward_tem import forward_backend_status
from core.real_data import parse_real_tem_bytes, quality_control
from core.training_service import (
    create_training_job,
    default_training_params,
    get_training_job,
    gpu_status,
    parse_training_params,
)


app = FastAPI(title=API_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_exists": os.path.exists(MODEL_SAVE_PATH),
        "scaler_exists": os.path.exists(SCALER_SAVE_PATH),
        "active_model_path": MODEL_SAVE_PATH,
        "active_scaler_path": SCALER_SAVE_PATH,
        "output_dir": OUTPUT_DIR,
        "codexdata_dir": CODEXDATA_DIR,
        "gpu": gpu_status(),
        "forward_backend": forward_backend_status(),
    }


@app.get("/api/model/active")
def active_model():
    return {
        "model_exists": os.path.exists(MODEL_SAVE_PATH),
        "scaler_exists": os.path.exists(SCALER_SAVE_PATH),
        "model_path": MODEL_SAVE_PATH,
        "scaler_path": SCALER_SAVE_PATH,
    }


@app.post("/api/data/qc")
async def qc_file(file: UploadFile = File(...)):
    try:
        dataset = parse_real_tem_bytes(await file.read())
        return quality_control(dataset)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/inversion/predict")
async def predict_file(file: UploadFile = File(...)):
    try:
        from core.predictor import predict_dataset

        dataset = parse_real_tem_bytes(await file.read())
        return predict_dataset(dataset)
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"缺少后端模型依赖: {exc.name}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/training/defaults")
def training_defaults():
    return default_training_params()


@app.post("/api/training/start")
async def start_training(file: UploadFile = File(...), params: str = Form("{}")):
    try:
        training_params = parse_training_params(params)
        content = await file.read()
        if not content:
            raise ValueError("上传文件为空。")
        job_id = create_training_job(content, file.filename or "z_data.txt", training_params)
        return {"job_id": job_id, "status": "queued"}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/training/status/{job_id}")
def training_status(job_id: str):
    snapshot = get_training_job(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="训练任务不存在。")
    return snapshot
