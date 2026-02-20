import torch
import multiprocessing as mp
import numpy as np

def inference_worker(model_path, device, task_queue, result_dict):
    """The only process that touches the GPU."""
    from .model import AlphaNet
    model = AlphaNet(num_res_blocks=10, channels=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    BATCH_SIZE = 64
    
    while True:
        batch = []
        ids = []
        
        # 1. Collect tasks from the queue up to BATCH_SIZE
        while len(batch) < BATCH_SIZE:
            try:
                # Fast polling
                task_id, state = task_queue.get(timeout=0.001)
                batch.append(state)
                ids.append(task_id)
            except:
                break # Process whatever we have if the queue is empty
        
        if not batch: continue

        # 2. Batch Inference
        with torch.no_grad():
            tensors = torch.from_numpy(np.array(batch)).to(device)
            logits, values = model(tensors)
            
            # 3. Distribute results back to specific workers
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            vals = values.cpu().numpy()
            
            for i, tid in enumerate(ids):
                result_dict[tid] = (probs[i], float(vals[i]))