# Redrob Candidate Ranking System

Production-grade ranking system for Senior AI Engineer role using multi-dimensional scoring.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run ranking (generates team_pseudoclan.csv)
python rank_fast.py --candidates candidates.jsonl --output team_pseudoclan.csv

# 3. View results
head -20 team_pseudoclan.csv
