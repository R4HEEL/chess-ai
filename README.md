# Chess AI Web Application

A browser-based chess engine that lets you play against two classic AI algorithms — Alpha-Beta Pruning and Monte Carlo Tree Search — deployed using Flask and Railway.

Built for the Artificial Intelligence course by Raheel Shaikh (31554), Umer Bin Affan (30497), and Rayan Adil (30556).

---

## Live Demo

Deployed on Railway and accessible from any browser. No installation required.

---

## Features

- Two AI engines: Alpha-Beta and MCTS
- Opening book using Perfect2023.bin (~first 12 moves)
- Evaluation bar showing centipawn advantage
- Optional game timers: 3, 5, or 10 minutes
- Move history in SAN notation
- Undo functionality (last two half-moves)
- Review mode for stepping through completed games
- Multiple board themes and piece styles
- Automatic deployment via GitHub and Railway

---

## Project Structure

```
chess_project/
├── app.py
├── chess_ai.py
├── chess_ai_mcts.py
├── index.html
├── Perfect2023.bin
├── requirements.txt
└── Procfile
```

---

## AI Algorithms

### Alpha-Beta Pruning

Minimax search optimized with alpha-beta pruning and move ordering.

- Evaluation: material count + positional tables
- Move ordering: MVV-LVA heuristic
- Score unit: centipawns (100 = 1 pawn advantage)

| Depth | Nodes Searched | Time |
|------|---------------|------|
| 1 | ~30 | Instant |
| 2 | ~900 | < 0.1 s |
| 3 | ~27,000 | 0.5 – 1 s |
| 4 | ~810,000 | 3 – 8 s |
| 5 | ~24,000,000 | 30 – 90 s |

---

### Monte Carlo Tree Search

A statistical search method using repeated simulations.

Phases:
Selection → Expansion → Simulation → Backpropagation

UCB1 Formula:
```
UCB1 = (wins / visits) + C * sqrt(ln(parent_visits) / visits)
C = sqrt(2)
```

| Difficulty | Iterations |
|-----------|-----------|
| Beginner | 200 |
| Easy | 500 |
| Medium | 1000 |
| Hard | 2000 |
| Expert | 4000 |

---

## Running Locally

### 1. Clone Repository
```bash
git clone https://github.com/your-username/chess-ai.git
cd chess-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Server
```bash
python app.py
```

### 4. Open in Browser
```
http://127.0.0.1:5000
```

---

## Deployment (Railway)

1. Push project to GitHub
2. Create a Railway project and deploy from repository
3. Railway uses the Procfile to run:
   ```
   gunicorn app:app --bind 0.0.0.0:$PORT
   ```
4. Every push triggers automatic redeployment

---

## API

### POST /move

Request:
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
  "depth": 3,
  "algorithm": "alphabeta"
}
```

Response:
```json
{
  "move": "e7e5",
  "score": -20,
  "source": "engine",
  "algorithm": "alphabeta"
}
```

Field descriptions:

- algorithm: "alphabeta" or "mcts"
- source: "book" or "engine"
- score: centipawns (Alpha-Beta) or visit count (MCTS)
- move: UCI format (e.g., e7e5, g1f3, e7e8q)

---

## Dependencies

```
flask
python-chess>=1.9.0
gunicorn
```

Frontend uses chess.js (CDN-based, no installation required).

---

## Requirements

- Python 3.10 or higher
- Modern browser (Chrome, Firefox, Edge, Safari)
- Internet connection for chess.js CDN

---

## Known Limitations

| Area | Current State | Improvement |
|------|-------------|------------|
| Horizon effect | Fixed-depth cutoff | Add quiescence search |
| Repeated positions | Recomputed | Use transposition tables |
| MCTS strength | Weak at low iterations | Add neural network |
| Evaluation bar | Not accurate for MCTS | Add heuristic scoring |
| Player side | Always White | Add side selection |
| Promotion | Always queen | Add promotion choice |

---


