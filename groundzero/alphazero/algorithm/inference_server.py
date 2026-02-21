import torch
import multiprocessing as mp
import numpy as np
import time

def inference_worker(model_path, device, task_queue, result_dict):
    """
    The GPU Master process. 
    Optimized for dynamic batching to maximize throughput.
    """
    # Import inside the function to avoid CUDA/MPS initialization issues in the main process
    from .model import AlphaNet
    
    print(f"[Inference] Initializing model on {device}...")
    model = AlphaNet(num_res_blocks=10, channels=128).to(device)
    
    # Load model with weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"[Inference] Warning: Could not load model weights: {e}")
    
    model.eval()

    # Settings for high-throughput
    BATCH_SIZE = 64
    WAIT_TIMEOUT = 0.001 # 1ms window to allow queue to fill up
    
    print(f"[Inference] Server is active. Batch Size: {BATCH_SIZE}")

    while True:
        batch = []
        ids = []
        
        # 1. Block until at least ONE task is available
        # This prevents the loop from spinning at 100% CPU when idle
        try:
            task_id, state = task_queue.get(timeout=1.0)
            batch.append(state)
            ids.append(task_id)
        except:
            continue # No tasks for 1 second, just loop back

        # 2. Dynamic Batching: Try to fill the rest of the batch
        # We wait a tiny bit to see if more tasks arrive from other workers
        start_wait = time.time()
        while len(batch) < BATCH_SIZE:
            try:
                # Non-blocking attempt to grab more tasks
                task_id, state = task_queue.get_nowait()
                batch.append(state)
                ids.append(task_id)
            except:
                # If queue is empty, wait a tiny bit for threads to catch up
                if time.time() - start_wait < WAIT_TIMEOUT:
                    time.sleep(0.0001) 
                    continue
                else:
                    break # Timeout reached, process what we have

        # 3. Batch Inference
        if batch:
            with torch.no_grad():
                # Convert list of numpy arrays to a single tensor
                # Note: np.array(batch) is fast here
                tensors = torch.from_numpy(np.stack(batch)).to(device)
                
                logits, values = model(tensors)
                
                # Softmax for probabilities and move to CPU
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                vals = values.cpu().numpy()
                
                # 4. Distribute results back to workers
                # The shared result_dict is the bottleneck here, so we do it fast
                for i, tid in enumerate(ids):
                    result_dict[tid] = (probs[i], float(vals[i]))