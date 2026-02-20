import torch
import chess
import numpy as np
import time
import uuid
from .model import AlphaNet
from .encoder import AlphaZeroEncoder
import os

class AlphaZeroEvaluator:
    def __init__(self, model_path=None, device="cpu"):
        self.device = device
        self.encoder = AlphaZeroEncoder(history_len=2) 
        
        # Batch Mode variables
        self.batch_mode = False
        self.task_queue = None
        self.result_dict = None
        
        # Local model for fallback/non-batch mode
        self.model = AlphaNet(num_res_blocks=10, channels=128).to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        # Telemetry
        self.last_inference_time = 0.0
        self.latest_value = 0.0
        self._cache = {}

    def set_batch_mode(self, task_queue, result_dict):
        """Enables worker to send tasks to the central inference server."""
        self.task_queue = task_queue
        self.result_dict = result_dict
        self.batch_mode = True

    def clear_cache(self):
        self._cache = {}

    @torch.no_grad()
    def evaluate(self, board: chess.Board):
        fen = board.fen()
        if fen in self._cache:
            priors, value = self._cache[fen]
            self.latest_value = value
            return priors, value

        if self.batch_mode:
            return self._evaluate_batched(board, fen)
        else:
            return self._evaluate_local(board, fen)

    def _evaluate_batched(self, board, fen):
        """Sends request to the Inference Server and polls for result."""
        req_id = str(uuid.uuid4())
        encoded = self.encoder.encode(board)
        
        start_time = time.time()
        # Put task in queue for the GPU process
        self.task_queue.put((req_id, encoded))
        
        # Poll for result in the shared manager dictionary
        while req_id not in self.result_dict:
            # We use a very short sleep to yield the CPU for MCTS threads
            time.sleep(0.0001)
            
        probs, value = self.result_dict.pop(req_id)
        
        priors = self._process_outputs(board, probs)
        
        self.last_inference_time = time.time() - start_time
        self.latest_value = value
        self._cache[fen] = (priors, value)
        return priors, value

    def _evaluate_local(self, board, fen):
        """Standard sequential evaluation (used if not in worker mode)."""
        start_time = time.time()
        encoded = self.encoder.encode(board)
        tensor = torch.from_numpy(encoded).unsqueeze(0).to(self.device)
        
        logits, value_tensor = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        value = float(value_tensor.item())
        
        priors = self._process_outputs(board, probs)
        
        self.last_inference_time = time.time() - start_time
        self.latest_value = value
        self._cache[fen] = (priors, value)
        return priors, value

    def _process_outputs(self, board, probs):
        """Maps raw probabilities to legal moves."""
        legal_moves = list(board.legal_moves)
        priors = {}
        for move in legal_moves:
            idx = (move.from_square << 6) | move.to_square
            priors[move] = float(probs[idx])

        total_p = sum(priors.values())
        if total_p > 1e-8:
            inv_total = 1.0 / total_p
            priors = {m: p * inv_total for m, p in priors.items()}
        else:
            priors = {m: 1.0 / len(legal_moves) for m in legal_moves}
        return priors