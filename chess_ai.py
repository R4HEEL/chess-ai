import chess

PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:     0,   
}

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

PIECE_SQUARE_TABLES = {
    chess.PAWN:   PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK:   ROOK_TABLE,
    chess.QUEEN:  QUEEN_TABLE,
    chess.KING:   KING_TABLE,
}


def _piece_square_bonus(piece_type: int, square: chess.Square, color: chess.Color) -> int:

    table = PIECE_SQUARE_TABLES[piece_type]
    if color == chess.WHITE:
        rank = chess.square_rank(square)           
        file = chess.square_file(square)           
        index = (7 - rank) * 8 + file
    else:
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        index = rank * 8 + file

    return table[index]


def evaluate(board: chess.Board) -> int:

    if board.is_checkmate():
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

def _move_score(board: chess.Board, move: chess.Move) -> int:

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

        score += 10 * victim_value - attacker_value

    if move.promotion:
        score += PIECE_VALUES.get(move.promotion, 0)

    return score


def order_moves(board: chess.Board, moves) -> list:
    return sorted(moves, key=lambda m: _move_score(board, m), reverse=True)


def alpha_beta(
    board: chess.Board,
    depth: int,
    alpha: int,
    beta: int,
    maximising: bool,
) -> int:

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
                break   

        return max_eval

    else:  
        min_eval = float("inf")
        for move in legal_moves:
            board.push(move)
            eval_score = alpha_beta(board, depth - 1, alpha, beta, True)
            board.pop()

            min_eval = min(min_eval, eval_score)
            beta     = min(beta,     eval_score)

            if beta <= alpha:
                break   

        return min_eval


def get_best_move(board: chess.Board, depth: int = 3) -> tuple[chess.Move | None, int]:

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
            maximising=(board.turn == chess.WHITE),  
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


if __name__ == "__main__":
    print("=== Chess AI Demo ===\n")

    board = chess.Board()
    print("Starting position:")
    print(board)
    print(f"\nStatic evaluation: {evaluate(board)} cp  (0 = perfectly balanced)\n")

    SEARCH_DEPTH = 3
    print(f"Searching at depth {SEARCH_DEPTH}...")
    move, score = get_best_move(board, depth=SEARCH_DEPTH)

    print(f"Best move for White: {move}  (score: {score:+d} cp)")
    board.push(move)
    print("\nBoard after White's move:")
    print(board)

    print(f"\nSearching at depth {SEARCH_DEPTH}...")
    move, score = get_best_move(board, depth=SEARCH_DEPTH)
    print(f"Best move for Black: {move}  (score: {score:+d} cp)")
    board.push(move)
    print("\nBoard after Black's reply:")
    print(board)

    print("\n--- Custom FEN: White queen advantage ---")
    fen = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board2 = chess.Board(fen)
    print(f"Evaluation: {evaluate(board2):+d} cp")
