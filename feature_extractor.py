"""Feature extraction from candidate profiles."""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract features from candidate profiles."""

    def __init__(self):
        pass

    def extract_current_title(self, candidate: Dict[str, Any]) -> str:
        try:
            profile = candidate.get("profile", {})
            return (profile.get("current_title") or "").lower().strip()
        except:
            return ""

    def extract_headline(self, candidate: Dict[str, Any]) -> str:
        try:
            profile = candidate.get("profile", {})
            return (profile.get("headline") or "").lower().strip()
        except:
            return ""

    def extract_summary(self, candidate: Dict[str, Any]) -> str:
        try:
            profile = candidate.get("profile", {})
            return (profile.get("summary") or "").lower().strip()
        except:
            return ""

    def extract_location(self, candidate: Dict[str, Any]) -> str:
        try:
            profile = candidate.get("profile", {})
            return (profile.get("location") or "").lower().strip()
        except:
            return ""

    def extract_career_history(self, candidate: Dict[str, Any]) -> List[Dict[str, str]]:
        try:
            career_history = candidate.get("career_history", [])
            extracted = []
            for job in career_history:
                if isinstance(job, dict):
                    extracted.append({
                        "title": (job.get("title") or "").lower().strip(),
                        "company": (job.get("company") or "").lower().strip(),
                        "description": (job.get("description") or "").lower().strip(),
                        "duration_months": int(job.get("duration_months", 0)),
                    })
            return extracted
        except:
            return []

    def extract_skills(self, candidate: Dict[str, Any]) -> List[str]:
        try:
            skills = candidate.get("skills", [])
            if isinstance(skills, list):
                # Handle new format: list of dicts with "name" key
                skill_names = []
                for s in skills:
                    if isinstance(s, dict):
                        name = s.get("name", "")
                    else:
                        name = str(s)
                    if name:
                        skill_names.append(name.lower().strip())
                return skill_names
            return []
        except:
            return []

    def extract_education(self, candidate: Dict[str, Any]) -> List[Dict[str, str]]:
        try:
            education = candidate.get("education", [])
            if not isinstance(education, list):
                return []
            
            extracted = []
            for edu in education:
                if isinstance(edu, dict):
                    extracted.append({
                        "degree": (edu.get("degree") or "").lower().strip(),
                        "field": (edu.get("field_of_study") or "").lower().strip(),
                        "school": (edu.get("institution") or "").lower().strip(),
                    })
            return extracted
        except:
            return []

    def extract_redrob_signals(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        try:
            signals = candidate.get("redrob_signals", {})
            if not isinstance(signals, dict):
                signals = {}
            
            return {
                "open_to_work": signals.get("open_to_work_flag", False),
                "recruiter_response_rate": float(signals.get("recruiter_response_rate", 0.0)),
                "interview_completion_rate": float(signals.get("interview_completion_rate", 0.0)),
                "github_activity_score": float(signals.get("github_activity_score", 0.0)),
                "saved_by_recruiters_30d": int(signals.get("saved_by_recruiters_30d", 0)),
                "search_appearance_30d": int(signals.get("search_appearance_30d", 0)),
                "willing_to_relocate": signals.get("willing_to_relocate", False),
                "last_active_days_ago": int(signals.get("last_active_days_ago", 999)),
                "notice_period_days": int(signals.get("notice_period_days", 30)),
            }
        except:
            return {}

    def extract_total_experience_years(self, candidate: Dict[str, Any]) -> float:
        try:
            # First try profile.years_of_experience
            profile = candidate.get("profile", {})
            if "years_of_experience" in profile:
                return float(profile.get("years_of_experience", 0))
            
            # Fallback: sum from career_history
            career_history = self.extract_career_history(candidate)
            total_months = sum(job.get("duration_months", 0) for job in career_history)
            years = total_months / 12.0
            return round(years, 1)
        except:
            return 0.0

    def extract_all_text(self, candidate: Dict[str, Any]) -> str:
        texts = []
        texts.append(self.extract_current_title(candidate))
        texts.append(self.extract_headline(candidate))
        texts.append(self.extract_summary(candidate))
        
        for job in self.extract_career_history(candidate):
            texts.append(job.get("title", ""))
            texts.append(job.get("company", ""))
            texts.append(job.get("description", ""))
        
        for skill in self.extract_skills(candidate):
            texts.append(skill)
        
        return " ".join([t for t in texts if t])