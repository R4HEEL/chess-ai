"""
chess_ai_mcts.py
----------------
Monte Carlo Tree Search (MCTS) chess engine built on python-chess.

Drop-in replacement for chess_ai.py — identical get_best_move() signature.

How MCTS works (4 steps repeated for every iteration):
  1. SELECTION     – Walk down the tree picking nodes by UCB1 score.
  2. EXPANSION     – Add one new child node for an unexplored move.
  3. SIMULATION    – Play random moves until game over or depth limit.
  4. BACKPROP      – Walk back up, updating wins/visits on every ancestor.

Score convention: return value is visit count of best child, not centipawns.
The higher the visit count, the more confident MCTS is in that move.

Usage:
    from chess_ai_mcts import get_best_move
    move, score = get_best_move(board, iterations=1000)
"""

import chess
import math
import random

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Exploration constant in UCB1 formula.
# Higher C  → explores more (tries untested moves more often).
# Lower  C  → exploits more (keeps playing proven good moves).
# √2 ≈ 1.41 is the theoretical optimum for win/loss games.
C = math.sqrt(2)

# Maximum moves per rollout simulation.
# Prevents infinite loops in drawn positions.
# A real game averages ~40 moves so 60 covers most games.
MAX_ROLLOUT_DEPTH = 60

# Quick material table used during rollout scoring (no piece-square tables —
# rollouts are meant to be fast, not perfectly accurate).
MATERIAL = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
    chess.KING:   0,
}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MCTS NODE
# ─────────────────────────────────────────────────────────────────────────────

class MCTSNode:
    """
    One node in the MCTS search tree.

    Each node represents a board position reached by a specific move.
    It tracks how many times it has been visited and how many of those
    visits resulted in a win for the side that created this node.

    Attributes
    ----------
    board     : the board state at this node
    parent    : parent MCTSNode (None for root)
    move      : the chess.Move that led to this node from the parent
    children  : list of expanded child MCTSNodes
    untried   : moves not yet expanded into children
    wins      : accumulated win score from backpropagation
    visits    : number of times this node has been visited
    """

    __slots__ = ('board', 'parent', 'move', 'children', 'untried', 'wins', 'visits')

    def __init__(self, board: chess.Board, parent=None, move=None):
        self.board    = board
        self.parent   = parent
        self.move     = move
        self.children = []
        self.wins     = 0.0
        self.visits   = 0

        # Shuffle untried moves so expansion order is random.
        # This prevents the engine always trying the same opening move first.
        self.untried = list(board.legal_moves)
        random.shuffle(self.untried)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def is_fully_expanded(self) -> bool:
        """True when every legal move has been expanded into a child node."""
        return len(self.untried) == 0

    @property
    def is_terminal(self) -> bool:
        """True when the game is over at this node."""
        return self.board.is_game_over()

    # ── UCB1 score ───────────────────────────────────────────────────────────

    def ucb1(self) -> float:
        """
        Upper Confidence Bound 1 score used to select which child to visit.

        UCB1 = exploitation_term + exploration_term
             = (wins / visits)  + C × √(ln(parent_visits) / visits)

        The exploitation term favours nodes with high win rates.
        The exploration term favours nodes that have been visited less
        relative to their siblings — ensuring we don't ignore any branch.

        Unvisited nodes always return +∞ so they are always tried first.
        """
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + \
               C * math.sqrt(math.log(self.parent.visits) / self.visits)

    # ── Tree operations ───────────────────────────────────────────────────────

    def best_child_by_ucb1(self) -> 'MCTSNode':
        """Return the child with the highest UCB1 score (used in selection)."""
        return max(self.children, key=lambda n: n.ucb1())

    def best_child_by_visits(self) -> 'MCTSNode':
        """
        Return the child with the most visits (used to pick the final move).

        Visit count is more robust than win rate for the final selection
        because a move visited 1000 times with 55% win rate is more reliable
        than one visited 10 times with 80% win rate.
        """
        return max(self.children, key=lambda n: n.visits)

    def expand(self) -> 'MCTSNode':
        """
        Pop one untried move, create a child node for it, and return it.
        This is the EXPANSION step of MCTS.
        """
        move      = self.untried.pop()
        new_board = self.board.copy()
        new_board.push(move)
        child = MCTSNode(new_board, parent=self, move=move)
        self.children.append(child)
        return child

    def update(self, result: float) -> None:
        """
        Update this node's statistics during BACKPROPAGATION.

        result : 1.0 = win for root colour, 0.0 = loss, 0.5 = draw
        """
        self.visits += 1
        self.wins   += result


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ROLLOUT (SIMULATION)
# ─────────────────────────────────────────────────────────────────────────────

def _material_score(board: chess.Board, root_color: chess.Color) -> float:
    """
    Estimate position quality by raw material count when rollout hits depth limit.
    Returns a value in [0, 1] from root_color's perspective.
    """
    score = 0
    for piece_type, value in MATERIAL.items():
        score += len(board.pieces(piece_type, root_color))       * value
        score -= len(board.pieces(piece_type, not root_color))   * value

    # Normalise: clamp to [-15, +15] (queen advantage range) then map to [0,1]
    normalised = max(-15, min(15, score)) / 15
    return 0.5 + normalised * 0.5


def _game_result(board: chess.Board, root_color: chess.Color) -> float:
    """
    Convert a terminal board state to a result from root_color's perspective.
      1.0 = root_color wins
      0.0 = root_color loses
      0.5 = draw
    """
    if board.is_checkmate():
        # The side TO MOVE is mated → the OTHER side won
        winner = not board.turn
        return 1.0 if winner == root_color else 0.0

    # Stalemate, insufficient material, 50-move rule, repetition
    return 0.5


def rollout(board: chess.Board, root_color: chess.Color) -> float:
    """
    SIMULATION step: play moves from this position until terminal or depth limit.

    Uses a "capture-biased" rollout rather than pure random:
    if any captures are available, pick one at random instead of a quiet move.
    This makes rollouts slightly more tactically aware without slowing them down.

    Returns a result in [0, 1] from root_color's perspective.
    """
    b     = board.copy()
    depth = 0

    while not b.is_game_over() and depth < MAX_ROLLOUT_DEPTH:
        moves    = list(b.legal_moves)
        captures = [m for m in moves if b.is_capture(m)]

        # Prefer captures 70% of the time when available (tactical bias)
        if captures and random.random() < 0.7:
            b.push(random.choice(captures))
        else:
            b.push(random.choice(moves))

        depth += 1

    if b.is_game_over():
        return _game_result(b, root_color)
    else:
        # Hit depth limit — use material to estimate result
        return _material_score(b, root_color)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CORE MCTS LOOP
# ─────────────────────────────────────────────────────────────────────────────

def _run_mcts(board: chess.Board, iterations: int) -> MCTSNode:
    """
    Run the MCTS algorithm for a fixed number of iterations.

    Each iteration = one Selection + Expansion + Simulation + Backpropagation.

    Parameters
    ----------
    board      : starting position
    iterations : number of playouts (higher = stronger, slower)

    Returns
    -------
    The root MCTSNode with all children populated and scored.
    """
    root_color = board.turn          # the side we are finding a move for
    root       = MCTSNode(board.copy())

    for _ in range(iterations):

        # ── 1. SELECTION ─────────────────────────────────────────────────────
        # Walk down the tree always choosing the highest-UCB1 child
        # until we reach a node that is either:
        #   (a) not fully expanded — has untried moves left, OR
        #   (b) terminal — game over
        node = root
        while node.is_fully_expanded and not node.is_terminal:
            node = node.best_child_by_ucb1()

        # ── 2. EXPANSION ─────────────────────────────────────────────────────
        # If the node still has untried moves, add one new child.
        # Terminal nodes cannot be expanded.
        if not node.is_terminal and not node.is_fully_expanded:
            node = node.expand()

        # ── 3. SIMULATION ────────────────────────────────────────────────────
        # From the new node, play out a random game and get a result.
        result = rollout(node.board, root_color)

        # ── 4. BACKPROPAGATION ───────────────────────────────────────────────
        # Walk back up to root updating wins/visits.
        # The result is flipped at each level because what is a win for
        # root_color is a loss for the opponent one level up.
        current = node
        while current is not None:
            current.update(result)
            result  = 1.0 - result      # flip perspective
            current = current.parent

    return root


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PUBLIC ENTRY POINT  (same signature as chess_ai.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_best_move(
    board: chess.Board,
    iterations: int = 1000,
    depth: int = None,          # accepted but ignored — kept for API compatibility
                                # with app.py which passes depth=N from the frontend
) -> tuple[chess.Move | None, int]:
    """
    Find the best move for the side currently to move using MCTS.

    Parameters
    ----------
    board      : current board position (not modified)
    iterations : number of MCTS playouts
                   200  → very fast,  weak   (~depth 1-2 feel)
                   500  → fast,       decent (~depth 2-3 feel)
                   1000 → balanced           (~depth 3 feel)    ← default
                   2000 → slow,       strong (~depth 4 feel)
                   4000 → very slow,  strong (~depth 5 feel)
    depth      : ignored (present only for drop-in compatibility with app.py)

    Returns
    -------
    (best_move, score)
        best_move : chess.Move to play, or None if no legal moves exist
        score     : visit count of the chosen move (higher = more confident)
                    NOTE: this is NOT centipawns — it cannot be shown on
                    the eval bar the same way as chess_ai.py's score.
    """
    if not any(board.legal_moves):
        return None, 0

    root = _run_mcts(board, iterations)

    if not root.children:
        return None, 0

    best = root.best_child_by_visits()
    return best.move, best.visits


# ─────────────────────────────────────────────────────────────────────────────
# 6.  QUICK DEMO
# ─────────────────────────────────────────────────────────────────────────────

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
