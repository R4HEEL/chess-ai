"""
chess_ai.py
-----------
A chess AI engine built on top of the `python-chess` library.

Components:
  1. evaluate()        – Material + positional (piece-square table) evaluation.
  2. order_moves()     – Heuristic move ordering to improve Alpha-Beta pruning.
  3. alpha_beta()      – Minimax search with Alpha-Beta pruning.
  4. get_best_move()   – Public entry point: returns the best move for the side to move.

Score convention: positive = White is winning, negative = Black is winning.
"""

import chess

# ---------------------------------------------------------------------------
# 1. PIECE VALUES
# ---------------------------------------------------------------------------

PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:     0,   # The king's safety is handled separately.
}

# ---------------------------------------------------------------------------
# 2. PIECE-SQUARE TABLES  (from White's perspective, index 0 = a1)
#
#    These encode positional bonuses/penalties.  For example, a knight is
#    stronger in the centre than on the rim; a king should hide in the corner
#    in the middlegame.  Values are in centipawns (100 cp = 1 pawn).
#
#    Convention: the tables are written with rank 8 at the top so they are
#    easy to read as a board.  We flip them for Black automatically.
# ---------------------------------------------------------------------------

# fmt: off
PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

ROOK_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

# Middlegame king table – king should castle and stay safe.
KING_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]
# fmt: on

PIECE_SQUARE_TABLES = {
    chess.PAWN:   PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK:   ROOK_TABLE,
    chess.QUEEN:  QUEEN_TABLE,
    chess.KING:   KING_TABLE,
}


def _piece_square_bonus(piece_type: int, square: chess.Square, color: chess.Color) -> int:
    """
    Return the positional bonus for a piece on a given square.

    python-chess uses square index 0=a1, 7=h1, 56=a8, 63=h8.
    Our tables are written with rank 8 at the top (index 0 = a8 visually),
    so for White we mirror the rank; for Black we use the index directly.
    """
    table = PIECE_SQUARE_TABLES[piece_type]
    if color == chess.WHITE:
        # Mirror vertically: rank 1 -> row 7, rank 8 -> row 0
        rank = chess.square_rank(square)           # 0-7 (0 = rank 1)
        file = chess.square_file(square)           # 0-7
        index = (7 - rank) * 8 + file
    else:
        # Black's pieces are already oriented correctly without mirroring
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        index = rank * 8 + file

    return table[index]


# ---------------------------------------------------------------------------
# 3. EVALUATION FUNCTION
# ---------------------------------------------------------------------------

def evaluate(board: chess.Board) -> int:
    """
    Evaluate the board position and return a score in centipawns.

    Positive  → White is better.
    Negative  → Black is better.
    ±100      ≈ one pawn of advantage.

    The score is always from White's perspective, regardless of who is
    currently to move.  The alpha_beta() function handles the perspective
    flip internally.

    Evaluation components (in order of importance):
      1. Terminal state detection (checkmate / stalemate).
      2. Material balance.
      3. Piece-square positional bonuses.
    """
    # --- Terminal states ---
    if board.is_checkmate():
        # The side that just moved delivered checkmate, so the side to move
        # is the one being mated.
        return -100_000 if board.turn == chess.WHITE else 100_000

    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0

    for piece_type in PIECE_VALUES:
        for square in board.pieces(piece_type, chess.WHITE):
            score += PIECE_VALUES[piece_type]
            score += _piece_square_bonus(piece_type, square, chess.WHITE)

        for square in board.pieces(piece_type, chess.BLACK):
            score -= PIECE_VALUES[piece_type]
            score -= _piece_square_bonus(piece_type, square, chess.BLACK)

    return score


# ---------------------------------------------------------------------------
# 4. MOVE ORDERING
# ---------------------------------------------------------------------------

def _move_score(board: chess.Board, move: chess.Move) -> int:
    """
    Assign a heuristic priority score to a move for ordering purposes.

    Higher score = searched first = more likely to trigger early cutoffs.

    Heuristics applied (you can extend these later):
      - Captures: scored by MVV-LVA (Most Valuable Victim, Least Valuable Attacker).
        Capturing a queen with a pawn is searched before capturing a pawn with a queen.
      - Promotions: always searched early.
      - All other moves: score 0.
    """
    score = 0

    if board.is_capture(move):
        if board.is_en_passant(move):
            victim_square = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        else:
            victim_square = move.to_square
        victim = board.piece_at(victim_square)
        attacker = board.piece_at(move.from_square)

        victim_value  = PIECE_VALUES.get(victim.piece_type,  0) if victim  else 0
        attacker_value = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0

        # MVV-LVA: prefer high-value victims and low-value attackers
        score += 10 * victim_value - attacker_value

    if move.promotion:
        score += PIECE_VALUES.get(move.promotion, 0)

    return score


def order_moves(board: chess.Board, moves) -> list:
    """Return legal moves sorted by heuristic score, best first."""
    return sorted(moves, key=lambda m: _move_score(board, m), reverse=True)


# ---------------------------------------------------------------------------
# 5. ALPHA-BETA PRUNING (Minimax with Alpha-Beta)
# ---------------------------------------------------------------------------

def alpha_beta(
    board: chess.Board,
    depth: int,
    alpha: int,
    beta: int,
    maximising: bool,
) -> int:
    """
    Minimax search with Alpha-Beta pruning.

    Parameters
    ----------
    board       : current board position (modified in-place, then restored).
    depth       : how many plies (half-moves) to search.
    alpha       : best score the maximising player can already guarantee.
    beta        : best score the minimising player can already guarantee.
    maximising  : True if the current node is a max node (White to move).

    Returns
    -------
    The best evaluation score reachable from this position at the given depth.

    How Alpha-Beta works
    --------------------
    Imagine two players, Max (White) and Min (Black), taking turns.
    - Max wants the highest score; Min wants the lowest.
    - alpha tracks Max's floor: "I can already do at least this well."
    - beta  tracks Min's ceiling: "I can already do at most this well."
    - If at any point alpha >= beta, the opponent will never allow this
      branch, so we prune it (stop searching here).  This is the cut-off.
    """
    # Base case: depth exhausted or game over
    if depth == 0 or board.is_game_over():
        return evaluate(board)

    legal_moves = order_moves(board, board.legal_moves)

    if maximising:
        max_eval = -float("inf")
        for move in legal_moves:
            board.push(move)
            eval_score = alpha_beta(board, depth - 1, alpha, beta, False)
            board.pop()

            max_eval = max(max_eval, eval_score)
            alpha    = max(alpha,    eval_score)

            if beta <= alpha:
                break   # Beta cut-off: Min will never allow this branch.

        return max_eval

    else:  # minimising
        min_eval = float("inf")
        for move in legal_moves:
            board.push(move)
            eval_score = alpha_beta(board, depth - 1, alpha, beta, True)
            board.pop()

            min_eval = min(min_eval, eval_score)
            beta     = min(beta,     eval_score)

            if beta <= alpha:
                break   # Alpha cut-off: Max will never allow this branch.

        return min_eval


# ---------------------------------------------------------------------------
# 6. PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

def get_best_move(board: chess.Board, depth: int = 3) -> tuple[chess.Move | None, int]:
    """
    Search for the best move for the side currently to move.

    Parameters
    ----------
    board : the current board position.
    depth : search depth in plies (half-moves).
              depth=1 → looks 1 move ahead (very weak)
              depth=3 → solid for a university project (~seconds per move)
              depth=5 → strong but slower (~seconds to minutes per move)

    Returns
    -------
    (best_move, score)
        best_move : the chess.Move object to play, or None if no legal moves.
        score     : the evaluation of the resulting position in centipawns,
                    from White's perspective.
    """
    if not any(board.legal_moves):
        return None, 0

    best_move  = None
    best_score = -float("inf") if board.turn == chess.WHITE else float("inf")

    for move in order_moves(board, board.legal_moves):
        board.push(move)
        score = alpha_beta(
            board,
            depth - 1,
            alpha=-float("inf"),
            beta=float("inf"),
            maximising=(board.turn == chess.WHITE),  # after push, turn has flipped
        )
        board.pop()

        if board.turn == chess.WHITE:
            if score > best_score:
                best_score = score
                best_move  = move
        else:
            if score < best_score:
                best_score = score
                best_move  = move

    return best_move, best_score


# ---------------------------------------------------------------------------
# 7. QUICK DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Chess AI Demo ===\n")

    board = chess.Board()
    print("Starting position:")
    print(board)
    print(f"\nStatic evaluation: {evaluate(board)} cp  (0 = perfectly balanced)\n")

    # --- Let the AI play one move as White ---
    SEARCH_DEPTH = 3
    print(f"Searching at depth {SEARCH_DEPTH}...")
    move, score = get_best_move(board, depth=SEARCH_DEPTH)

    print(f"Best move for White: {move}  (score: {score:+d} cp)")
    board.push(move)
    print("\nBoard after White's move:")
    print(board)

    # --- Let the AI reply as Black ---
    print(f"\nSearching at depth {SEARCH_DEPTH}...")
    move, score = get_best_move(board, depth=SEARCH_DEPTH)
    print(f"Best move for Black: {move}  (score: {score:+d} cp)")
    board.push(move)
    print("\nBoard after Black's reply:")
    print(board)

    # --- Custom FEN example: White is up a queen ---
    print("\n--- Custom FEN: White queen advantage ---")
    fen = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board2 = chess.Board(fen)
    print(f"Evaluation: {evaluate(board2):+d} cp")