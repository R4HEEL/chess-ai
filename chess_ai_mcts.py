import chess
import math
import random

C = math.sqrt(2)
MAX_ROLLOUT_DEPTH = 60

MATERIAL = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
    chess.KING:   0,
}

class MCTSNode:

    __slots__ = ('board', 'parent', 'move', 'children', 'untried', 'wins', 'visits')

    def __init__(self, board: chess.Board, parent=None, move=None):
        self.board    = board
        self.parent   = parent
        self.move     = move
        self.children = []
        self.wins     = 0.0
        self.visits   = 0

        self.untried = list(board.legal_moves)
        random.shuffle(self.untried)


    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried) == 0

    @property
    def is_terminal(self) -> bool:
        return self.board.is_game_over()


    def ucb1(self) -> float:

        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + \
               C * math.sqrt(math.log(self.parent.visits) / self.visits)


    def best_child_by_ucb1(self) -> 'MCTSNode':
        return max(self.children, key=lambda n: n.ucb1())

    def best_child_by_visits(self) -> 'MCTSNode':
        return max(self.children, key=lambda n: n.visits)

    def expand(self) -> 'MCTSNode':

        move      = self.untried.pop()
        new_board = self.board.copy()
        new_board.push(move)
        child = MCTSNode(new_board, parent=self, move=move)
        self.children.append(child)
        return child

    def update(self, result: float) -> None:
        self.visits += 1
        self.wins   += result



def _material_score(board: chess.Board, root_color: chess.Color) -> float:

    score = 0
    for piece_type, value in MATERIAL.items():
        score += len(board.pieces(piece_type, root_color))       * value
        score -= len(board.pieces(piece_type, not root_color))   * value

    normalised = max(-15, min(15, score)) / 15
    return 0.5 + normalised * 0.5


def _game_result(board: chess.Board, root_color: chess.Color) -> float:

    if board.is_checkmate():
        winner = not board.turn
        return 1.0 if winner == root_color else 0.0

    return 0.5


def rollout(board: chess.Board, root_color: chess.Color) -> float:
    b     = board.copy()
    depth = 0

    while not b.is_game_over() and depth < MAX_ROLLOUT_DEPTH:
        moves    = list(b.legal_moves)
        captures = [m for m in moves if b.is_capture(m)]

        if captures and random.random() < 0.7:
            b.push(random.choice(captures))
        else:
            b.push(random.choice(moves))

        depth += 1

    if b.is_game_over():
        return _game_result(b, root_color)
    else:
        return _material_score(b, root_color)



def _run_mcts(board: chess.Board, iterations: int) -> MCTSNode:
    root_color = board.turn          
    root       = MCTSNode(board.copy())

    for _ in range(iterations):

        node = root
        while node.is_fully_expanded and not node.is_terminal:
            node = node.best_child_by_ucb1()

        if not node.is_terminal and not node.is_fully_expanded:
            node = node.expand()

        result = rollout(node.board, root_color)

        current = node
        while current is not None:
            current.update(result)
            result  = 1.0 - result      
            current = current.parent

    return root



def get_best_move(
    board: chess.Board,
    iterations: int = 1000,
    depth: int = None,          
                                
) -> tuple[chess.Move | None, int]:

    if not any(board.legal_moves):
        return None, 0

    root = _run_mcts(board, iterations)

    if not root.children:
        return None, 0

    best = root.best_child_by_visits()
    return best.move, best.visits



if __name__ == "__main__":
    print("=== Chess AI — Monte Carlo Tree Search Demo ===\n")

    board = chess.Board()
    print("Starting position:")
    print(board)

    ITERATIONS = 1000
    print(f"\nRunning MCTS ({ITERATIONS} iterations) for White...")
    move, visits = get_best_move(board, iterations=ITERATIONS)
    print(f"Best move for White: {move}  (visited {visits} times)")
    board.push(move)

    print("\nBoard after White's move:")
    print(board)

    print(f"\nRunning MCTS ({ITERATIONS} iterations) for Black...")
    move, visits = get_best_move(board, iterations=ITERATIONS)
    print(f"Best move for Black: {move}  (visited {visits} times)")
    board.push(move)

    print("\nBoard after Black's reply:")
    print(board)
