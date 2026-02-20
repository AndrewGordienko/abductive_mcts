import os
import time
import torch
import multiprocessing as mp
import sys
import uuid

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from algorithm.model import AlphaNet
from algorithm.collector import DataCollector
from algorithm.inference_server import inference_worker
from training_dashboard.dashboard_app import run_dashboard_server

def bootstrap_model(path):
    if not os.path.exists(path):
        print(f"[*] Initializing new 'brain' at {path}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model = AlphaNet(num_res_blocks=10, channels=128)
        torch.save(model.state_dict(), path)

def worker_task(worker_id, model_path, shared_stats, task_queue, result_dict):
    """
    Worker process. Note: It no longer loads the model itself.
    It passes the task_queue to the evaluator.
    """
    # M1 Mac Optimization: The collector uses the CPU for MCTS logic, 
    # but the evaluator sends requests to the GPU process via queues.
    collector = DataCollector(model_path=model_path, device="cpu")
    
    # Inject the batching queues into the evaluator
    collector.evaluator.set_batch_mode(task_queue, result_dict)

    print(f"[Worker {worker_id}] Starting batched self-play...")
    
    while True:
        shared_stats[worker_id] = {
            "status": "In Queue",
            "move_count": 0,
            "fen": "start",
            "start_time": time.time()
        }

        start_time = time.time()
        game_data = collector.collect_game(worker_id=worker_id, stats=shared_stats)
        duration = time.time() - start_time
        
        timestamp = int(time.time() * 1000)
        filename = f"batch_{worker_id}_{timestamp}.npz"
        collector.save_batch(game_data, filename)
        
        print(f"[Worker {worker_id}] Game finished ({duration:.1f}s). Buffer: {len(game_data)}")
        
        # In batch mode, the inference server reloads the model, 
        # so workers don't need to update locally as often.
        collector.update_model(model_path)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True) # Required for MPS/GPU on Mac
    
    MODEL_PATH = "models/best_model.pth"
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    
    bootstrap_model(MODEL_PATH)
    
    with mp.Manager() as manager:
        shared_stats = manager.dict()
        result_dict = manager.dict() # Stores (probs, value) keyed by request_id
        task_queue = mp.Queue(maxsize=256)
        processes = []

        # 1. Start Inference Server (The GPU master)
        inf_p = mp.Process(
            target=inference_worker, 
            args=(MODEL_PATH, DEVICE, task_queue, result_dict)
        )
        inf_p.start()
        processes.append(inf_p)

        # 2. Start Dashboard
        dashboard_p = mp.Process(target=run_dashboard_server, args=(shared_stats,))
        dashboard_p.start()
        processes.append(dashboard_p)

        # 3. Start Workers
        num_workers = 4 
        for i in range(num_workers):
            p = mp.Process(
                target=worker_task, 
                args=(i, MODEL_PATH, shared_stats, task_queue, result_dict)
            )
            p.start()
            processes.append(p)

        try:
            for p in processes: p.join()
        except KeyboardInterrupt:
            for p in processes: p.terminate()