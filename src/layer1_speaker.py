"""Layer 1: enrollment and speaker-match interfaces. Implemented incrementally from Day 5."""

def cosine_similarity(a, b):
    import numpy as np
    a=np.asarray(a); b=np.asarray(b)
    denom=(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/denom) if denom else 0.0
