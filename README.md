# Redrob Candidate Ranking System

Production-grade ranking system for Senior AI Engineer role.

## Setup & Run

```bash
# 1. Create fresh virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run ranking
python rank.py --candidates candidates.jsonl --output team_pseudoclan.csv

# 4. View results
head -20 team_pseudoclan.csv
