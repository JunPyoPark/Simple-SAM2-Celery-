import os
import time
import subprocess
import atexit

# ✅ 실행 중인 워커 프로세스 추적 리스트
worker_processes = []

def start_worker(worker_id, gpu_id):
    """ 주어진 Worker ID와 GPU ID에서 Celery Worker 실행 """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # 각 Worker가 사용할 GPU 설정
    print(f"🚀 Starting Worker {worker_id} on GPU {gpu_id}...")

    # ✅ Celery Worker 실행 (로그를 worker.log에 저장)
    with open("worker.log", "a") as log_file:
        process = subprocess.Popen(
            f"CUDA_VISIBLE_DEVICES={gpu_id} celery -A simple_api.celery_app worker --loglevel=info "
            "--concurrency=1 --pool=solo --max-tasks-per-child=1",
            shell=True,
            env=env,
            stdout=log_file,
            stderr=log_file
        )

    worker_processes.append(process)  # 🔥 실행된 프로세스를 리스트에 추가

    return process

def tail_worker_log():
    """ worker.log 파일을 실시간으로 모니터링하여 출력 """
    try:
        with open("worker.log", "r") as log_file:
            log_file.seek(0, os.SEEK_END)  # 🔥 파일 끝으로 이동 (새 로그만 출력)
            while True:
                line = log_file.readline()
                if line:
                    print(line.strip())  # ✅ 새로운 로그를 출력
                else:
                    time.sleep(0.5)  # CPU 점유율 방지
    except KeyboardInterrupt:
        print("\n🛑 종료 요청됨. 모든 워커 정리 중...")

def cleanup_workers():
    """ 스크립트 종료 시 Celery Worker 프로세스 종료 """
    print("\n🛑 Cleaning up all workers...")
    os.system('pkill -f "python.*sam2_env"')
    print("✅ 모든 Worker가 종료되었습니다.")

# ✅ 프로그램 종료 시 실행될 cleanup 함수 등록
atexit.register(cleanup_workers)

if __name__ == "__main__":
    gpu_ids = [1,1,2,3]  # Worker들이 사용할 GPU ID (GPU 1은 worker 2개 실행)
    num_workers = len(gpu_ids)  # 실행할 Worker 개수

    processes = []

    for i in range(num_workers):
        process = start_worker(i, gpu_ids[i])
        processes.append(process)
        time.sleep(2)  # Worker 실행 간격을 둠 (CUDA 초기화 문제 방지)

    print("✅ 모든 Worker가 실행되었습니다. 로그 출력 중...")

    # ✅ worker.log 실시간 출력 시작
    tail_worker_log()
