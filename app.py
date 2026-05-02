from flask import Flask, request, jsonify, send_from_directory
import chess
import chess.polyglot
from chess_ai import get_best_move

app = Flask(__name__)

BOOK_PATH = "Perfect2023.bin"

def get_book_move(board):
    try:
        with chess.polyglot.open_reader(BOOK_PATH) as reader:
            return reader.weighted_choice(board).move
    except IndexError:
        return None
    except FileNotFoundError:
        return None

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/move', methods=['POST'])
def move():
    data  = request.get_json(force=True)
    fen   = data.get('fen')
    depth = int(data.get('depth', 3))

    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({'error': 'Invalid FEN'}), 400

    if board.is_game_over():
        return jsonify({'move': None, 'score': 0})

    book_move = get_book_move(board)
    if book_move:
        return jsonify({'move': book_move.uci(), 'score': 0, 'source': 'book'})

    best_move, score = get_best_move(board, depth=depth)

    if best_move is None:
        return jsonify({'move': None, 'score': 0})

    return jsonify({'move': best_move.uci(), 'score': score, 'source': 'engine'})

if __name__ == '__main__':
    app.run(debug=False)
