"""Train XGBoost ranking model."""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from tqdm import tqdm

from feature_extractor import FeatureExtractor
from honeypot_detector import HoneypotDetector
from scorer import CandidateScorer

def load_candidates(file_path: str):
    """Load candidates."""
    candidates = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    return candidates

def prepare_training_data(candidates, fe, hd, scorer):
    """Create features and labels from candidates."""
    X = []
    y = []
    
    for candidate in tqdm(candidates, desc="Preparing data"):
        try:
            # Extract features
            title_score = scorer._score_title(candidate)
            career_score = scorer._score_career_experience(candidate)
            company_score = scorer._score_product_company(candidate)
            behavior_score = scorer._score_behavioral(candidate)
            location_score = scorer._score_location(candidate)
            experience_score = scorer._score_experience(candidate)
            
            # Honeypot check
            trust_mult, _ = hd.detect_honeypots(candidate)
            
            # Feature vector (7 features)
            features = [
                title_score,
                career_score,
                company_score,
                behavior_score,
                location_score,
                experience_score,
                trust_mult  # Trust multiplier as feature
            ]
            X.append(features)
            
            # Label: weighted average (0.0-1.0)
            label = (
                title_score * 0.20 +
                career_score * 0.25 +
                company_score * 0.10 +
                behavior_score * 0.20 +
                location_score * 0.05 +
                experience_score * 0.05
            ) * trust_mult
            y.append(label)
            
        except Exception as e:
            continue
    
    return np.array(X), np.array(y)

def train_model(X_train, y_train, X_test, y_test):
    """Train XGBoost model."""
    
    print("\nTraining XGBoost model...")
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\nTrain R² Score: {train_score:.4f}")
    print(f"Test R² Score: {test_score:.4f}")
    
    return model

def main():
    """Main training pipeline."""
    
    print("=" * 80)
    print("TRAINING XGBoost RANKING MODEL")
    print("=" * 80)
    
    # Initialize
    fe = FeatureExtractor()
    hd = HoneypotDetector(fe)
    scorer = CandidateScorer(fe, hd)
    
    # Load candidates
    print("\nLoading candidates...")
    candidates = load_candidates('candidates.jsonl')
    print(f"Loaded {len(candidates)} candidates")
    
    # Prepare data
    print("\nPreparing training data...")
    X, y = prepare_training_data(candidates, fe, hd, scorer)
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Label range: {y.min():.4f} - {y.max():.4f}")
    
    # Normalize features
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data: 80% train, 20% test
    print("\nSplitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42
    )
    
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model
    model = train_model(X_train, y_train, X_test, y_test)
    
    # Save model and scaler
    print("\nSaving model...")
    joblib.dump(model, 'ranking_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("✓ Model saved as ranking_model.pkl")
    print("✓ Scaler saved as scaler.pkl")
    
    # Feature importance
    print("\nFeature Importance:")
    feature_names = ['Title', 'Career', 'Company', 'Behavior', 'Location', 'Experience', 'Trust']
    for name, importance in zip(feature_names, model.feature_importances_):
        print(f"  {name}: {importance:.4f}")

if __name__ == "__main__":
    main()