import numpy as np
from .utils import bg_color

def rank_candidates_for_test(cand_pool, survivors):
    best_y, best_score = None, 1e18
    for lbl, y in cand_pool:
        score = 0.0
        if survivors and (lbl not in survivors):
            score += 100.0
        score += len(lbl) * 0.01
        score += float(np.count_nonzero(y != bg_color(y))) * 0.0001
        if score < best_score:
            best_score = score; best_y = y
    return best_y