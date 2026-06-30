import json
import logging
import time
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm

from feature_extractor import FeatureExtractor
from honeypot_detector import HoneypotDetector
from scorer import CandidateScorer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastCandidateRankingPipeline:
    """Fast ranking - career evidence + behavior signals only."""

    def __init__(self):
        logger.info("Initializing fast pipeline (no semantic encoding)...")
        self.fe = FeatureExtractor()
        self.hd = HoneypotDetector(self.fe)
        self.scorer = CandidateScorer(self.fe, self.hd)

    def load_candidates(self, file_path: str) -> List[Dict[str, Any]]:
        """Load candidates from file."""
        logger.info(f"Loading candidates from {file_path}")
        candidates = []
        
        try:
            if file_path.endswith('.gz'):
                import gzip
                opener = gzip.open(file_path, 'rt', encoding='utf-8')
            else:
                opener = open(file_path, 'r', encoding='utf-8')
            
            with opener as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    try:
                        candidate = json.loads(line)
                        candidates.append(candidate)
                    except json.JSONDecodeError:
                        continue
                    
                    if (i + 1) % 10000 == 0:
                        logger.info(f"Loaded {i + 1} candidates")
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        
        logger.info(f"Total candidates: {len(candidates)}")
        return candidates

    def rank_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank all candidates."""
        logger.info(f"Ranking {len(candidates)} candidates...")
        
        ranked = []
        
        for i, candidate in enumerate(tqdm(candidates, desc="Scoring candidates")):
            candidate_id = candidate.get("candidate_id", f"candidate_{i}")
            try:
                final_score, breakdown = self.scorer.score_candidate(
                    candidate, 
                    semantic_score=0.5
                )
                ranked.append({
                    "candidate_id": candidate_id,
                    "final_score": final_score,
                    "breakdown": breakdown,
                    "candidate": candidate,
                    "education": self.fe.extract_education(candidate),
                    "skills": self.fe.extract_skills(candidate),
                    "location": self.fe.extract_location(candidate),
                    "experience_years": self.fe.extract_total_experience_years(candidate),
                })
            except Exception as e:
                logger.debug(f"Error scoring {candidate_id}: {e}")
                ranked.append({
                    "candidate_id": candidate_id,
                    "final_score": 0.0,
                    "breakdown": {},
                    "candidate": candidate,
                    "education": [],
                    "skills": [],
                    "location": "",
                    "experience_years": 0,
                })
        
        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        logger.info(f"Top score: {ranked[0]['final_score']:.3f}")
        return ranked

    def generate_submission(self, ranked: List[Dict[str, Any]], output_path: str = "submission.csv", top_k: int = 100):
        """Generate submission CSV."""
        logger.info(f"Generating submission with top {top_k} candidates...")
        
        rows = []
        for rank, result in enumerate(ranked[:top_k], 1):
            reasoning = self.scorer.generate_reasoning(
                result["candidate"],
                result["breakdown"]
            )
            
            # Format education
            education = result.get("education", [])
            education_str = "; ".join([
                f"{e.get('degree', '')} in {e.get('field', '')}"
                for e in education if e.get('degree')
            ])
            
            # Format skills (top 5)
            skills = result.get("skills", [])
            skills_str = ", ".join(skills[:5]) if skills else ""
            
            # Get location
            location = result.get("location", "")
            
            # Get experience
            experience = result.get("experience_years", 0)
            
            rows.append({
                "candidate_id": result["candidate_id"],
                "rank": rank,
                "score": round(result["final_score"], 4),
                "education": education_str,
                "skills": skills_str,
                "location": location,
                "experience_years": round(experience, 1),
                "reasoning": reasoning,
            })
        
        # CREATE DATAFRAME AND SAVE
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Submission saved to {output_path}")
        return df

    def run(self, candidates_file: str, output_path: str = "submission.csv"):
        """Run complete pipeline."""
        start = time.time()
        logger.info("=" * 80)
        logger.info("REDROB CANDIDATE RANKING (FAST MODE)")
        logger.info("=" * 80)
        
        try:
            candidates = self.load_candidates(candidates_file)
            ranked = self.rank_candidates(candidates)
            submission = self.generate_submission(ranked, output_path)
            
            elapsed = time.time() - start
            logger.info("=" * 80)
            logger.info(f"✓ Complete in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
            logger.info(f"✓ Total candidates: {len(candidates)}")
            logger.info(f"✓ Submission size: {len(submission)}")
            logger.info(f"✓ Top score: {submission['score'].iloc[0]:.4f}")
            logger.info(f"✓ Avg score: {submission['score'].mean():.4f}")
            logger.info("=" * 80)
            return submission
        except Exception as e:
            logger.error(f"Failed: {e}")
            raise

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast rank candidates")
    parser.add_argument("--candidates", type=str, default="candidates.jsonl", help="Candidate file")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output file")
    args = parser.parse_args()
    
    pipeline = FastCandidateRankingPipeline()
    submission = pipeline.run(args.candidates, args.output)
    
    print("\n" + "=" * 80)
    print("TOP 10 RANKED CANDIDATES:")
    print("=" * 80)
    print(submission.head(10).to_string(index=False))

if __name__ == "__main__":
    main()