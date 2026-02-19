import numpy as np
import chess

class AlphaZeroEncoder:
    def __init__(self, history_len=8):
        self.history_len = history_len # Current + 7 past

    def encode(self, board: chess.Board):
        # 119 planes total: (12 pieces * 8 history) + 23 constant planes
        planes = np.zeros((119, 8, 8), dtype=np.float32)
        
        # We need the move history. board.move_stack provides this.
        # We'll simulate the boards backwards.
        temp_board = board.copy()
        for i in range(self.history_len):
            if i > 0:
                if len(temp_board.move_stack) > 0:
                    temp_board.pop()
                else:
                    break # No more history, leave planes as zeros
            
            offset = i * 12
            self._encode_pieces(temp_board, planes, offset)

        # Meta-data planes (97-119)
        # 97: Color (1 if white, 0 if black)
        if board.turn == chess.WHITE:
            planes[96, :, :] = 1.0
        
        # 98: Total move count (normalized)
        planes[97, :, :] = board.fullmove_number / 100.0
        
        # 99-102: White Castling (K, Q)
        if board.has_kingside_castling_rights(chess.WHITE): planes[98, :, :] = 1
        if board.has_queenside_castling_rights(chess.WHITE): planes[99, :, :] = 1
        # 101-102: Black Castling (k, q)
        if board.has_kingside_castling_rights(chess.BLACK): planes[100, :, :] = 1
        if board.has_queenside_castling_rights(chess.BLACK): planes[101, :, :] = 1
        
        # 103: No-progress count (50-move rule)
        planes[102, :, :] = board.halfmove_clock / 100.0

        return planes

    def _encode_pieces(self, board, planes, offset):
        for color in [chess.WHITE, chess.BLACK]:
            for piece_type in range(1, 7): # Pawn to King
                indices = board.pieces(piece_type, color)
                for sq in indices:
                    rank, file = divmod(sq, 8)
                    planes[offset, rank, file] = 1.0
                offset += 1