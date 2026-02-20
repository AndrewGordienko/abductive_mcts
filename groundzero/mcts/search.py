import math
import chess
import os
import numpy as np
import time
from collections import Counter
from .node import MCTSNode

class MCTS:
    def __init__(self, evaluator):
        self.params = {
            'SIMULATIONS': 400, 
            'C_PUCT': 1.5, 
            'ALPHA': 0.3, 
            'EPS': 0.25, 
            'FPU_REDUCTION': 0.2,
            'VIRTUAL_LOSS': 3.0
        }
        self.evaluator = evaluator
        self.c_puct = self.params['C_PUCT']
        self.v_loss = self.params['VIRTUAL_LOSS']
        self.latest_depth = 0
        self.latest_heatmap = {}

    def search(self, board: chess.Board, is_training=False):
        self.evaluator.clear_cache()
        
        # 1. Root Evaluation
        priors, _ = self.evaluator.evaluate(board)
        
        if is_training and len(priors) > 0:
            noise = np.random.dirichlet([self.params['ALPHA']] * len(priors))
            for i, move in enumerate(priors):
                priors[move] = (1 - self.params['EPS']) * priors[move] + self.params['EPS'] * noise[i]
        
        root = MCTSNode(priors)
        root.v_loss = {m: 0 for m in root.P}

        max_depth_reached = 0
        square_visits = Counter() # Local counter for this search
        temp_board = board.copy(stack=False)
        sim_count = int(self.params['SIMULATIONS'])
        
        for _ in range(sim_count):
            node = root
            path = []
            depth = 0

            # --- SELECTION ---
            while True:
                move = self._select_child(node)
                path.append((node, move))
                
                # Track visits for heatmap
                square_visits[move.to_square] += 1
                
                # Apply Virtual Loss
                if not hasattr(node, 'v_loss'): node.v_loss = {m: 0 for m in node.P}
                node.v_loss[move] += self.v_loss
                
                temp_board.push(move)
                depth += 1
                
                if move not in node.children:
                    break
                node = node.children[move]

            if depth > max_depth_reached:
                max_depth_reached = depth

            # --- EXPANSION ---
            if not temp_board.is_game_over():
                p_priors, value = self.evaluator.evaluate(temp_board)
                node.children[move] = MCTSNode(p_priors)
                node.children[move].v_loss = {m: 0 for m in p_priors}
            else:
                res = temp_board.result()
                value = 1.0 if res == "1-0" else -1.0 if res == "0-1" else 0.0
                if not temp_board.turn: value = -value

            # --- BACKPROPAGATION ---
            for _ in range(len(path)):
                temp_board.pop()
            
            for n, m in reversed(path):
                n.v_loss[m] -= self.v_loss
                n.N[m] += 1
                n.W[m] += value
                n.Q[m] = n.W[m] / n.N[m]
                value = -value

        # --- UPDATE TELEMETRY FOR COLLECTOR ---
        self.latest_depth = max_depth_reached
        self.latest_heatmap = {
            chess.SQUARE_NAMES[s]: round(v / sim_count, 3) 
            for s, v in square_visits.items()
        }

        total_n = sum(root.N.values())
        pi_dist = {m: n / total_n for m, n in root.N.items()}
        best_move = max(root.N, key=root.N.get)
        
        return best_move, pi_dist

    def _select_child(self, node):
        total_n = sum(node.N.values())
        total_v_loss = sum(getattr(node, 'v_loss', {}).values())
        total_n_sqrt = math.sqrt(total_n + total_v_loss + 1)
        
        if total_n > 0:
            parent_q = -sum(node.W.values()) / total_n
            fpu_val = parent_q - self.params['FPU_REDUCTION']
        else:
            fpu_val = 0.0

        best_score = -float('inf')
        best_move = None

        for move in node.N:
            v_l = node.v_loss.get(move, 0)
            n_visited = node.N[move] + v_l
            q_val = (node.W[move] - v_l) / n_visited if n_visited > 0 else fpu_val
            u = self.c_puct * node.P[move] * total_n_sqrt / (1 + n_visited)
            score = q_val + u
            if score > best_score:
                best_score, best_move = score, move
        return best_move