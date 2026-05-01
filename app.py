from flask import Flask, request, jsonify, send_from_directory
import chess
from chess_ai import get_best_move

app = Flask(__name__)

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

    best_move, score = get_best_move(board, depth=depth)

    if best_move is None:
        return jsonify({'move': None, 'score': 0})

    return jsonify({'move': best_move.uci(), 'score': score})

if __name__ == '__main__':
    app.run(debug=False)
