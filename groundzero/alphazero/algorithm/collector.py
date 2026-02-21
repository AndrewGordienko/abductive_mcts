import numpy as np
import chess
import torch
import os
import time
from algorithm.evaluator import AlphaZeroEvaluator
from mcts.search import MCTS

class DataCollector:
    def __init__(self, model_path=None, device="cpu"):
        self.evaluator = AlphaZeroEvaluator(model_path=model_path, device=device)
        self.engine = MCTS(self.evaluator)
        self.buffer_path = "data/replay_buffer/"
        os.makedirs(self.buffer_path, exist_ok=True)
        
        self.total_games = 0
        self.total_samples = 0

    def update_model(self, path):
        if os.path.exists(path):
            try:
                self.evaluator.model.load_state_dict(torch.load(path, map_location=self.evaluator.device))
                self.evaluator.model.eval()
            except Exception: pass 

    def collect_game(self, worker_id=None, stats=None):
        board = chess.Board()
        game_data = []
        move_count = 0
        current_game_fens = [board.fen()]
        phase_times = {"opening": 0, "midgame": 0, "endgame": 0}
        
        root = None 

        # --- DYNAMIC SETTINGS PER GAME ---
        MAX_MOVES = 250
        # Randomize temp threshold so some games are more exploratory
        temp_threshold = np.random.randint(15, 35) 
        
        # 10% of games will disable resignation to train endgame/checkmate finishing
        can_resign = np.random.random() > 0.10
        
        MIN_RESIGN_MOVE = 60      
        RESIGN_THRESHOLD = 0.98   
        RESIGN_COUNT = 8          
        resign_streak = 0

        while not board.is_game_over() and move_count < MAX_MOVES:
            self.evaluator.clear_cache()
            
            start_search = time.time()
            best_move, pi_dist, root = self.engine.search(board, is_training=True, root=root)
            search_duration = time.time() - start_search

            probs_arr = np.array(list(pi_dist.values()), dtype=np.float32)
            entropy = -np.sum(probs_arr * np.log2(probs_arr + 1e-9))

            # Move selection with dynamic threshold
            if move_count > temp_threshold:
                selected_move = max(pi_dist, key=pi_dist.get)
                final_pi = {m: (1.0 if m == selected_move else 0.0) for m in pi_dist}
            else:
                moves = list(pi_dist.keys())
                selected_move = np.random.choice(moves, p=list(pi_dist.values()))
                final_pi = pi_dist

            if selected_move in root.children:
                root = root.children[selected_move]
            else:
                root = None 

            if stats is not None and worker_id is not None:
                val = float(self.evaluator.latest_value)
                
                # Check resignation logic
                if can_resign and move_count > MIN_RESIGN_MOVE and abs(val) > RESIGN_THRESHOLD:
                    resign_streak += 1
                else:
                    resign_streak = 0

                stats[worker_id] = {
                    "status": "Resigning..." if resign_streak > 0 else "Thinking (Batched)",
                    "move_count": move_count,
                    "last_depth": self.engine.latest_depth,
                    "nps": int(self.engine.params['SIMULATIONS'] / max(search_duration, 0.001)),
                    "value": val,
                    "entropy": float(entropy),
                    "inference_ms": float(self.evaluator.last_inference_time * 1000),
                    "fen": board.fen(),
                    "history_fens": list(current_game_fens),
                    "phase_times": phase_times.copy(),
                    "total_games": self.total_games,
                    "total_samples": self.total_samples,
                    "heatmap": self.engine.latest_heatmap,
                    "turn": "White" if board.turn == chess.WHITE else "Black"
                }

            state = self.evaluator.encoder.encode(board)
            pi_array = np.zeros(4096, dtype=np.float32)
            for move, prob in final_pi.items():
                idx = (move.from_square * 64) + move.to_square
                pi_array[idx] = prob
            
            game_data.append({"state": state, "pi": pi_array, "turn": board.turn})
            board.push(selected_move)
            current_game_fens.append(board.fen())
            
            if move_count < 20: phase_times["opening"] += search_duration
            elif move_count < 40: phase_times["midgame"] += search_duration
            else: phase_times["endgame"] += search_duration
            
            move_count += 1
            if resign_streak >= RESIGN_COUNT: break

        # Outcome resolution
        if board.is_game_over():
            res = board.result()
            outcome = 1.0 if res == "1-0" else -1.0 if res == "0-1" else 0.0
        else:
            # If the game was ended by resignation, use the model's high-confidence value
            outcome = float(self.evaluator.latest_value)

        self.total_games += 1
        self.total_samples += len(game_data)
        
        return [
            {"state": s["state"], "pi": s["pi"], "z": float(outcome if s["turn"] == chess.WHITE else -outcome)} 
            for s in game_data
        ]

    def save_batch(self, game_data, filename):
        path = os.path.join(self.buffer_path, filename)
        np.savez_compressed(
            path, 
            states=np.array([s["state"] for s in game_data]), 
            pis=np.array([s["pi"] for s in game_data]), 
            zs=np.array([s["z"] for s in game_data])
        )