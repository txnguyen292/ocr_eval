from rapidfuzz.distance import Levenshtein

def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER).
    
    CER = (S + D + I) / N
    where S is the number of substitutions,
    D is the number of deletions,
    I is the number of insertions,
    and N is the number of characters in the reference.
    """
    if not reference:
        return 1.0 if hypothesis else 0.0
        
    distance = Levenshtein.distance(reference, hypothesis)
    return distance / len(reference)

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER).
    
    WER = (S + D + I) / N
    where S is the number of substitutions,
    D is the number of deletions,
    I is the number of insertions,
    and N is the number of words in the reference.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if not ref_words:
        return 1.0 if hyp_words else 0.0
        
    distance = Levenshtein.distance(ref_words, hyp_words)
    return distance / len(ref_words)
