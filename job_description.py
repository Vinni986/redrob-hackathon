"""Job description and keywords."""

JOB_DESCRIPTION = """
Senior AI Engineer

We are looking for a Senior AI Engineer to join our team. The ideal candidate will have:

Requirements:
- 6-8 years of experience in AI/ML engineering
- Strong expertise in ranking and recommendation systems
- Experience with retrieval systems and semantic search
- Proficiency with embeddings and vector search technologies
- Product engineering background with real-world deployment experience
- Proven ability to evaluate and improve ranking quality (NDCG, MRR, MAP)
- Experience with A/B testing and experimentation frameworks
- Strong Python and software engineering fundamentals
- Experience with large-scale systems and distributed computing
- Understanding of information retrieval and candidate matching

Nice to have:
- Open source contributions
- Recent AI/ML activity
- Willingness to relocate
- Located in Noida, Pune, Delhi NCR, Gurgaon, Hyderabad, or Mumbai
"""

RANKING_KEYWORDS = {
    "ranking_systems": [
        "ranking", "rank", "ranking engine", "ranking algorithm", "learning to rank",
        "listwise", "pairwise", "pointwise", "gradient boosting", "xgboost",
        "lightgbm", "ranking quality", "relevance ranking"
    ],
    "recommendation_systems": [
        "recommendation", "recommender", "collaborative filtering", "content-based",
        "matrix factorization", "neural networks recommendation", "recommendation engine",
        "personalization"
    ],
    "retrieval_systems": [
        "retrieval", "dense retrieval", "passage retrieval", "document retrieval",
        "information retrieval", "ir system", "bm25", "sparse retrieval",
        "retrieval augmented generation", "rag"
    ],
    "embeddings": [
        "embedding", "embeddings", "word embedding", "sentence embedding",
        "document embedding", "representation learning", "semantic embedding"
    ],
    "vector_search": [
        "vector search", "vector database", "semantic search", "similarity search",
        "approximate nearest neighbor", "ann", "faiss", "annoy", "milvus"
    ],
    "search_relevance": [
        "search relevance", "relevance", "relevance scoring", "search quality",
        "query understanding"
    ],
    "evaluation_metrics": [
        "ndcg", "mrr", "map", "mean reciprocal rank", "mean average precision",
        "normalized discounted cumulative gain"
    ],
    "ab_testing": [
        "a/b test", "a/b testing", "ab test", "experiment", "experimentation"
    ],
    "matching": [
        "matching", "candidate matching", "entity matching", "job matching"
    ]
}

TITLE_REWARDS = {
    "ai engineer": 10,
    "ml engineer": 9,
    "machine learning engineer": 9,
    "search engineer": 10,
    "relevance engineer": 10,
    "recommendation engineer": 10,
    "applied scientist": 8,
    "data scientist": 7,
}

TITLE_PENALTIES = {
    "marketing manager": -8,
    "hr manager": -8,
    "accountant": -8,
    "operations manager": -6,
    "customer support": -6,
}

CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "cognizant", "capgemini", "accenture",
    "deloitte", "pwc", "kpmg", "ey", "mindtree", "hcl"
}

TARGET_LOCATIONS = {
    "noida", "pune", "delhi", "delhi ncr", "gurgaon", "hyderabad", "mumbai"
}
