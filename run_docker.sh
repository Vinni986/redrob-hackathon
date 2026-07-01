#!/bin/bash

echo "======================================================================"
echo "Redrob Ranking - Docker Build & Run"
echo "======================================================================"

# Create output directory
mkdir -p output

# Build image
echo " Building Docker image..."
docker build -t redrob-ranking:latest .

# Run container
echo " Running ranking..."
docker run --rm \
    --name redrob_ranking \
    --memory=16g \
    --network=none \
    -v "$(pwd)/candidates.jsonl:/app/candidates.jsonl:ro" \
    -v "$(pwd)/output:/app/output" \
    redrob-ranking:latest \
    python rank.py --candidates /app/candidates.jsonl --out /app/output/team_pseudoclan.csv

# Check output
if [ -f "output/team_pseudoclan.csv" ]; then
    echo "======================================================================"
    echo " SUCCESS! Output: output/team_pseudoclan.csv"
    echo "======================================================================"
    echo "Preview:"
    head -6 output/team_pseudoclan.csv
else
    echo " Error: Output not created"
    exit 1
fi
