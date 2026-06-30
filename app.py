import os
import tempfile
from typing import Optional

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


@app.post("/api/model/validate")
async def validate_model_file(
    model_file: UploadFile = File(...),
    data_file: UploadFile = File(...),
    scaler_file: Optional[UploadFile] = File(None),
):
    try:
        from core.model_validator import validate_model_with_dataset

        dataset = parse_real_tem_bytes(await data_file.read())
        suffix = os.path.splitext(model_file.filename or "")[1] or ".pt"
        with tempfile.TemporaryDirectory(prefix="tem_validate_", dir=OUTPUT_DIR) as tmp_dir:
            model_path = os.path.join(tmp_dir, f"uploaded_model{suffix}")
            with open(model_path, "wb") as f:
                f.write(await model_file.read())

            scaler_path = SCALER_SAVE_PATH
            used_uploaded_scaler = False
            if scaler_file is not None:
                scaler_suffix = os.path.splitext(scaler_file.filename or "")[1] or ".json"
                scaler_path = os.path.join(tmp_dir, f"uploaded_scaler{scaler_suffix}")
                with open(scaler_path, "wb") as f:
                    f.write(await scaler_file.read())
                used_uploaded_scaler = True

            result = validate_model_with_dataset(dataset, model_path=model_path, scaler_path=scaler_path)
            result["uploaded"] = {
                "model_filename": model_file.filename,
                "data_filename": data_file.filename,
                "scaler_filename": scaler_file.filename if scaler_file else None,
                "used_uploaded_scaler": used_uploaded_scaler,
            }
            if not used_uploaded_scaler:
                result.setdefault("warnings", []).insert(
                    0,
                    "未上传对应 scaler，当前验证使用系统激活的归一化文件；若它不是同一次训练生成，验证结果只能作为参考。",
                )
            return result
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"缂哄皯鍚庣妯″瀷渚濊禆: {exc.name}") from exc
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
