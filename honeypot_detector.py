
"""Detect suspicious candidates."""

from typing import Dict, Any, Tuple
from feature_extractor import FeatureExtractor

class HoneypotDetector:
    """Detect suspicious or mismatched candidates."""

    def __init__(self, feature_extractor: FeatureExtractor):
        self.fe = feature_extractor

    def detect_honeypots(self, candidate: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Return trust multiplier (0.3-1.0) and details."""
        
        career_history = self.fe.extract_career_history(candidate)
        skills = self.fe.extract_skills(candidate)
        experience_years = self.fe.extract_total_experience_years(candidate)
        
        trust = 1.0
        
        current_title = self.fe.extract_current_title(candidate)
        if "senior" in current_title and experience_years < 4:
            trust *= 0.5
        
        ai_skills = sum(1 for s in skills if any(kw in s for kw in ["ai", "ml", "machine learning"]))
        if len(skills) > 0 and (ai_skills / len(skills)) > 0.8 and not career_history:
            trust *= 0.4
        
        if experience_years == 0:
            trust *= 0.6
        
        return max(0.3, trust), {"trust_multiplier": trust}
