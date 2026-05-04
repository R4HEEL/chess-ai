from flask import Flask, request, jsonify, send_from_directory
import chess
import chess.polyglot
from chess_ai      import get_best_move as alphabeta_move
from chess_ai_mcts import get_best_move as mcts_move

app = Flask(__name__)
BOOK_PATH = "Perfect2023.bin"

def get_book_move(board):
    try:
        with chess.polyglot.open_reader(BOOK_PATH) as reader:
            return reader.weighted_choice(board).move
    except (IndexError, FileNotFoundError):
        return None

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/move', methods=['POST'])
def move():
    data      = request.get_json(force=True)
    fen       = data.get('fen')
    depth     = int(data.get('depth', 3))
    algorithm = data.get('algorithm', 'alphabeta')

    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({'error': 'Invalid FEN'}), 400

    if board.is_game_over():
        return jsonify({'move': None, 'score': 0})

    book_move = get_book_move(board)
    if book_move:
        return jsonify({'move': book_move.uci(), 'score': 0, 'source': 'book'})

    if algorithm == 'mcts':
        iterations = {1: 200, 2: 500, 3: 1000, 4: 2000, 5: 4000}.get(depth, 1000)
        best_move, score = mcts_move(board, iterations=iterations)
    else:
        best_move, score = alphabeta_move(board, depth=depth)

    if best_move is None:
        return jsonify({'move': None, 'score': 0})

    return jsonify({
        'move':      best_move.uci(),
        'score':     score,
        'source':    'engine',
        'algorithm': algorithm,
    })

if __name__ == '__main__':
    app.run(debug=False)
