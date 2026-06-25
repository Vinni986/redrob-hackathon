"""Score candidates."""

from typing import Dict, Any, Tuple
import numpy as np
from feature_extractor import FeatureExtractor
from honeypot_detector import HoneypotDetector
from job_description import (
    RANKING_KEYWORDS, TITLE_REWARDS, TITLE_PENALTIES,
    CONSULTING_COMPANIES, TARGET_LOCATIONS
)

class CandidateScorer:
    """Score candidates against job requirements."""

    def __init__(self, feature_extractor: FeatureExtractor, honeypot_detector: HoneypotDetector):
        self.fe = feature_extractor
        self.hd = honeypot_detector

    def score_candidate(self, candidate: Dict[str, Any], semantic_score: float = 0.5) -> Tuple[float, Dict[str, Any]]:
        """Score candidate. Returns (final_score, breakdown)."""
        
        breakdown = {}
        
        title_score = self._score_title(candidate)
        breakdown["title_score"] = title_score
        
        career_score = self._score_career_experience(candidate)
        breakdown["career_score"] = career_score
        
        breakdown["semantic_score"] = semantic_score
        
        product_score = self._score_product_company(candidate)
        breakdown["product_score"] = product_score
        
        behavioral_score = self._score_behavioral(candidate)
        breakdown["behavioral_score"] = behavioral_score
        
        location_score = self._score_location(candidate)
        breakdown["location_score"] = location_score
        
        experience_score = self._score_experience(candidate)
        breakdown["experience_score"] = experience_score
        
        weighted_score = (
            title_score * 0.20 +
            career_score * 0.25 +
            semantic_score * 0.15 +
            product_score * 0.10 +
            behavioral_score * 0.20 +
            location_score * 0.05 +
            experience_score * 0.05
        )
        
        trust_multiplier, _ = self.hd.detect_honeypots(candidate)
        breakdown["trust_multiplier"] = trust_multiplier
        
        final_score = weighted_score * trust_multiplier
        breakdown["final_score"] = final_score
        
        return final_score, breakdown

    def _score_title(self, candidate: Dict[str, Any]) -> float:
        """Score job title. Returns 0.0-1.0."""
        current_title = self.fe.extract_current_title(candidate)
        
        if not current_title:
            return 0.5
        
        for reward_title, score in TITLE_REWARDS.items():
            if reward_title in current_title:
                return min(1.0, score / 10.0)
        
        for penalty_title, penalty in TITLE_PENALTIES.items():
            if penalty_title in current_title:
                return max(0.0, (10 + penalty) / 10.0)
        
        if any(kw in current_title for kw in ["engineer", "scientist", "developer"]):
            return 0.6
        
        return 0.4

    def _score_career_experience(self, candidate: Dict[str, Any]) -> float:
        """Score career experience. Returns 0.0-1.0."""
        
        career_history = self.fe.extract_career_history(candidate)
        current_title = self.fe.extract_current_title(candidate)
        summary = self.fe.extract_summary(candidate)
        
        if not career_history:
            return 0.0
        
        all_text = " ".join([
            job.get("title", "") + " " + job.get("description", "")
            for job in career_history
        ]) + " " + current_title + " " + summary
        all_text = all_text.lower()
        
        keyword_matches = {}
        total_matches = 0
        
        for category, keywords in RANKING_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in all_text)
            if matches > 0:
                keyword_matches[category] = matches
                total_matches += matches
        
        if total_matches == 0:
            return 0.2
        
        strong_categories = {"ranking_systems", "retrieval_systems", "recommendation_systems", "embeddings"}
        strong_matches = sum(keyword_matches.get(cat, 0) for cat in strong_categories)
        
        if strong_matches > 3:
            return 1.0
        elif strong_matches > 1:
            return 0.85
        elif strong_matches > 0:
            return 0.7
        elif total_matches > 5:
            return 0.75
        elif total_matches > 2:
            return 0.6
        else:
            return 0.4

    def _score_product_company(self, candidate: Dict[str, Any]) -> float:
        """Score product company experience. Returns 0.0-1.0."""
        
        career_history = self.fe.extract_career_history(candidate)
        if not career_history:
            return 0.5
        
        consulting_jobs = 0
        product_jobs = 0
        total_jobs = len(career_history)
        
        for job in career_history:
            company = job.get("company", "").lower()
            
            is_consulting = any(consulting_co in company for consulting_co in CONSULTING_COMPANIES)
            is_product = any(kw in company for kw in ["startup", "saas", "product", "tech"])
            
            if is_consulting and not is_product:
                consulting_jobs += 1
            elif is_product:
                product_jobs += 1
        
        if product_jobs == 0 and consulting_jobs > 0:
            consulting_ratio = consulting_jobs / total_jobs
            if consulting_ratio > 0.8:
                return 0.3
            elif consulting_ratio > 0.5:
                return 0.5
            else:
                return 0.65
        
        if product_jobs > 0:
            product_ratio = product_jobs / total_jobs
            if product_ratio > 0.7:
                return 1.0
            elif product_ratio > 0.5:
                return 0.85
            else:
                return 0.7
        
        return 0.6

    def _score_behavioral(self, candidate: Dict[str, Any]) -> float:
        """Score behavioral signals. Returns 0.0-1.0."""
        
        signals = self.fe.extract_redrob_signals(candidate)
        score = 0.5
        
        if signals.get("open_to_work"):
            score += 0.15
        if signals.get("willing_to_relocate"):
            score += 0.05
        
        recruiter_response = signals.get("recruiter_response_rate", 0.0)
        if recruiter_response > 0.7:
            score += 0.15
        elif recruiter_response > 0.5:
            score += 0.10
        
        interview_completion = signals.get("interview_completion_rate", 0.0)
        if interview_completion > 0.7:
            score += 0.10
        elif interview_completion > 0.5:
            score += 0.05
        
        github = signals.get("github_activity_score", 0.0)
        if github > 0.7:
            score += 0.10
        elif github > 0.4:
            score += 0.05
        
        saved = signals.get("saved_by_recruiters_30d", 0)
        if saved > 5:
            score += 0.10
        elif saved > 2:
            score += 0.05
        
        last_active = signals.get("last_active_days_ago", 999)
        if last_active < 7:
            score += 0.10
        elif last_active < 30:
            score += 0.05
        elif last_active > 365:
            score -= 0.15
        
        notice = signals.get("notice_period_days", 30)
        if notice > 90:
            score -= 0.10
        
        return max(0.0, min(1.0, score))

    def _score_location(self, candidate: Dict[str, Any]) -> float:
        """Score location. Returns 0.0-1.0."""
        
        location = self.fe.extract_location(candidate)
        signals = self.fe.extract_redrob_signals(candidate)
        
        if not location:
            return 0.6 if signals.get("willing_to_relocate") else 0.4
        
        if any(target in location for target in TARGET_LOCATIONS):
            return 1.0
        
        if signals.get("willing_to_relocate"):
            return 0.7
        
        if "india" in location:
            return 0.5
        
        return 0.3

    def _score_experience(self, candidate: Dict[str, Any]) -> float:
        """Score years of experience. Returns 0.0-1.0."""
        
        years = self.fe.extract_total_experience_years(candidate)
        
        if 6 <= years <= 8:
            return 1.0
        elif 5 <= years <= 9:
            return 0.9
        elif 4 <= years < 5:
            return 0.7
        elif 9 < years <= 12:
            return 0.75
        elif 12 < years <= 15:
            return 0.65
        elif years < 4:
            return 0.3
        else:
            return 0.5

    def generate_reasoning(self, candidate: Dict[str, Any], breakdown: Dict[str, Any]) -> str:
        """Generate specific, fact-based reasoning with candidate profile details."""
        
        # Extract candidate facts
        title = self.fe.extract_current_title(candidate)
        years = self.fe.extract_total_experience_years(candidate)
        location = self.fe.extract_location(candidate)
        career_history = self.fe.extract_career_history(candidate)
        signals = self.fe.extract_redrob_signals(candidate)
        
        companies = [job.get("company", "").strip() for job in career_history]
        companies = [c for c in companies if c]
        
        career_text = " ".join([
            job.get("description", "") for job in career_history
        ]).lower()
        
        # Extract scores
        career_score = breakdown.get("career_score", 0)
        title_score = breakdown.get("title_score", 0)
        product_score = breakdown.get("product_score", 0)
        behavioral_score = breakdown.get("behavioral_score", 0)
        experience_score = breakdown.get("experience_score", 0)
        final_score = breakdown.get("final_score", 0)
        
        reasons = []
        concerns = []
        
        # ========== CAREER EVIDENCE ==========
        if career_score > 0.85:
            if "ranking" in career_text or "learning to rank" in career_text:
                reasons.append("Built ranking systems at scale")
            elif "recommendation" in career_text or "recommender" in career_text:
                reasons.append("Shipped recommendation engines in production")
            elif "retrieval" in career_text or "dense retrieval" in career_text:
                reasons.append("Strong retrieval/search systems expertise")
            elif "embeddings" in career_text or "vector" in career_text:
                reasons.append("Deep embeddings and vector search experience")
            else:
                reasons.append(f"Strong ML background with {years:.1f} years experience")
        elif career_score > 0.70:
            if len(career_history) > 0:
                latest_title = career_history[0].get("title", "").strip()
                reasons.append(f"ML engineering background ({latest_title})")
            else:
                reasons.append("Relevant ML/AI experience")
        elif career_score < 0.40:
            concerns.append("Limited evidence of ranking/recommendation/retrieval work")
        
        # ========== TITLE MATCH ==========
        if title_score > 0.90:
            reasons.append(f"Strong title match: {title}")
        elif title_score > 0.75:
            reasons.append(f"Relevant title: {title}")
        elif title_score < 0.50:
            concerns.append(f"Title '{title}' doesn't align with role")
        
        # ========== COMPANY TYPE ==========
        if product_score > 0.85 and companies:
            reasons.append(f"Product company background ({companies[0]})")
        elif product_score < 0.50 and companies:
            consulting = [c for c in companies if any(
                kw in c.lower() for kw in ["tcs", "infosys", "wipro", "cognizant", "capgemini"]
            )]
            if consulting:
                concerns.append(f"Consulting-dominated career ({consulting[0]})")
        
        # ========== EXPERIENCE LEVEL ==========
        if experience_score > 0.90:
            if 6 <= years <= 8:
                reasons.append(f"Ideal experience level ({years:.1f} years)")
            else:
                reasons.append(f"{years:.1f} years experience")
        elif experience_score < 0.50:
            if years < 4:
                concerns.append(f"Junior ({years:.1f} years; role needs 6-8)")
            elif years > 15:
                concerns.append(f"May be overqualified ({years:.1f} years)")
        
        # ========== BEHAVIORAL SIGNALS ==========
        engagement = []
        if signals.get("open_to_work"):
            engagement.append("open to work")
        if signals.get("recruiter_response_rate", 0) > 0.75:
            engagement.append("high recruiter engagement")
        if signals.get("last_active_days_ago", 999) < 14:
            engagement.append("recently active")
        if signals.get("interview_completion_rate", 0) > 0.75:
            engagement.append("completes interviews")
        
        if len(engagement) >= 2:
            reasons.append(f"Strong engagement ({', '.join(engagement[:2])})")
        elif len(engagement) == 1:
            reasons.append(f"Positive signal: {engagement[0]}")
        elif behavioral_score < 0.50:
            if signals.get("last_active_days_ago", 0) > 180:
                concerns.append(f"Inactive for {signals['last_active_days_ago']} days")
            if signals.get("recruiter_response_rate", 0) < 0.30:
                concerns.append("Low recruiter engagement")
        
        # ========== LOCATION ==========
        if location:
            target_locs = ["noida", "pune", "delhi", "gurgaon", "hyderabad", "mumbai"]
            if any(t in location.lower() for t in target_locs):
                reasons.append(f"Located in {location}")
            elif signals.get("willing_to_relocate"):
                reasons.append("Willing to relocate")
        elif signals.get("willing_to_relocate"):
            reasons.append("Willing to relocate")
        
        # ========== NOTICE PERIOD ==========
        notice = signals.get("notice_period_days", 30)
        if notice > 90:
            concerns.append(f"{notice}-day notice period")
        
        # ========== BUILD FINAL REASONING ==========
        parts = []
        
        if final_score > 0.85:
            # Top tier: strongest signals
            parts = reasons[:2]
        elif final_score > 0.75:
            # Good tier: 1-2 reasons + maybe 1 concern
            parts = reasons[:1]
            if concerns:
                parts.append(f"minor: {concerns[0]}")
        elif final_score > 0.60:
            # Mid tier: balanced
            parts = reasons[:1]
            if concerns:
                parts.append(f"concern: {concerns[0]}")
        elif final_score > 0.45:
            # Lower tier: concerns dominate
            if concerns:
                parts = concerns[:1]
                if reasons:
                    parts.append(f"but {reasons[0].lower()}")
            else:
                parts = ["Below target profile"]
        else:
            # Bottom tier
            if concerns:
                parts = [concerns[0]]
            else:
                parts = ["Limited fit"]
        
        reasoning = "; ".join(parts)
        
        # Ensure max 160 characters
        if len(reasoning) > 160:
            reasoning = reasoning[:157] + "..."
        
        return reasoning