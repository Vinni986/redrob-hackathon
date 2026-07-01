FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all Python files
COPY rank.py .
COPY train_model.py .
COPY feature_extractor.py .
COPY honeypot_detector.py .
COPY scorer.py .
COPY job_description.py .

# Copy pre-trained models
COPY ranking_model.pkl .
COPY scaler.pkl .

# Create output directory
RUN mkdir -p /app/output

CMD ["python", "rank.py", "--help"]
