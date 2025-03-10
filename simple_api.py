import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
import multiprocessing
from pydantic import BaseModel
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
from celery import Celery
import time
import gc

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    print("🔹 GPU Memory Cleared on Server Shutdown")

app = FastAPI(lifespan=lifespan)

# ✅ Celery 설정 (Redis를 메시지 큐로 사용)
celery_app = Celery(
    "tasks",
    broker="redis://172.30.0.7:6379",
    backend="redis://172.30.0.7:6379"
)

# ✅ GPU 설정 (각 Worker가 개별적으로 초기화할 수 있도록 설정)
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # CUDA 디버깅 모드 (필요시 비활성화 가능)

# ✅ Worker별 모델 전역 변수 (각 Worker가 한 번만 로드)
sam2_model = None

# ✅ 요청 데이터 구조 정의
class PredictionRequest(BaseModel):
    image_path: str
    points: list
    labels: list

# ✅ Celery Worker 시작 시 한 번만 모델 로드
def load_model():
    """각 Celery Worker에서 실행될 때 모델을 로드"""
    global sam2_model
    if sam2_model is None:
        print("🔹 Loading SAM2 Model...")
        sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

        worker_name = multiprocessing.current_process().name
        worker_pid = os.getpid()
        print(f"✅ Worker {worker_name} (PID: {worker_pid}) - SAM2 Model Loaded")

    return sam2_model

# ✅ GPU 메모리 정리 함수 (동적 empty_cache 적용)
def clear_gpu_memory(threshold=1):
    """사용 중인 GPU 메모리가 threshold GiB 이상이면 empty_cache() 실행"""
    allocated_memory = torch.cuda.memory_allocated() / 1024**3  # GiB 단위 변환
    print('allocated_memory: ', allocated_memory)
    if allocated_memory > threshold:
        print(f"🔹 Allocated Memory ({allocated_memory:.2f} GiB) exceeds {threshold} GiB. Clearing cache...")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

# ✅ Celery Task 정의 (SAM2 예측을 비동기 실행)
@celery_app.task(name="tasks.predict_task", time_limit=300)
def predict_task(image_path, points, labels):
    """ SAM2 예측을 수행하는 비동기 Celery Task """
    worker_name = multiprocessing.current_process().name
    worker_pid = os.getpid()
    print(f"🔹 Task started on Worker: {worker_name} (PID: {worker_pid})")

    global sam2_model
    if sam2_model is None:
        sam2_model = load_model()

    start_time = time.time()
    print(f"🔹 {worker_name} (PID: {worker_pid}) - Celery Task Started")

    try:
        image = np.array(Image.open(image_path).convert("RGB"))

        # ✅ Gradient 계산 완전 비활성화 (메모리 최적화)
        with torch.set_grad_enabled(False):
            predictor = SAM2ImagePredictor(sam2_model)
            predictor.set_image(image)

            # 🔹 labels 차원 수정 (N, 1)로 변환
            points = np.array(points, dtype=np.float32).reshape(-1, 2)
            labels = np.array(labels, dtype=np.int32).reshape(-1)

            print(f"🔹 {worker_name} (PID: {worker_pid}) - Running predictor.predict()")
            masks, _, _ = predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=False
            )

        elapsed_time = time.time() - start_time

        print(f"✅ {worker_name} (PID: {worker_pid}) - Prediction Completed in {elapsed_time:.2f} sec")

        return {"masks": masks.tolist()}

    except torch.cuda.OutOfMemoryError:
        print(f"❌ CUDA OOM Error on Worker {worker_name} (PID: {worker_pid})")
        torch.cuda.empty_cache()
        gc.collect()
        return {"error": "CUDA Out of Memory"}

    except Exception as e:
        print(f"❌ Unexpected Error on Worker {worker_name} (PID: {worker_pid}): {str(e)}")
        return {"error": str(e)}

    finally:
        # ✅ GPU 메모리 동적 정리 적용 (2GiB 이상 사용 시 캐시 해제)
        clear_gpu_memory(threshold=1)
        print(f"🔹 {worker_name} (PID: {worker_pid}) - GPU Memory Cleared After Task")


# ✅ FastAPI 엔드포인트 (Celery Task 실행)
@app.post("/predict")
async def predict_mask(request: PredictionRequest):
    """ 클라이언트 요청을 Celery Task로 넘기고, task_id 반환 """
    task = celery_app.send_task("tasks.predict_task", args=[request.image_path, request.points, request.labels])
    return {"task_id": task.id}

# ✅ Task 결과 조회 API
@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """ Celery Task 결과 조회 """
    task_result = celery_app.AsyncResult(task_id)
    if task_result.ready():
        return {"status": "completed", "result": task_result.result}
    return {"status": "processing"}

# ✅ FastAPI 실행
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # CUDA 문제 해결
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
