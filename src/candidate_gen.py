import itertools, time
from collections import Counter, defaultdict
from .ops import BASIC_OPS
from .utils import hamming

def generate_candidates(x_in, H, W, train_pairs, use_compositions=True):
    cands = []
    for f in BASIC_OPS:
        try:
            y = f(x_in, H, W, train_pairs)
            if y.shape == (H, W):
                cands.append((f"op:{f.__name__}", y))
        except Exception:
            pass
    if use_compositions:
        for f,g in itertools.product(BASIC_OPS, BASIC_OPS):
            try:
                y1 = g(x_in, H, W, train_pairs)
                if y1.shape != (H, W): 
                    continue
                y2 = f(y1, H, W, train_pairs)
                if y2.shape == (H, W):
                    cands.append((f"op2:{f.__name__}+{g.__name__}", y2))
            except Exception:
                pass
    uniq = {}
    for lbl, arr in cands:
        key = arr.tobytes()
        if key not in uniq:
            uniq[key] = (lbl, arr)
    return list(uniq.values())

def infer_target_shape(train_pairs):
    Hs = [y.shape[0] for _, y in train_pairs]
    Ws = [y.shape[1] for _, y in train_pairs]
    H = Counter(Hs).most_common(1)[0][0]
    W = Counter(Ws).most_common(1)[0][0]
    return H, W

def select_survivor_labels(train_pairs, H, W, use_compositions, time_limit=1.5):
    t0 = time.time()
    x0, y0 = train_pairs[0]
    pool0 = generate_candidates(x0, H, W, train_pairs, use_compositions=use_compositions)
    labels = [lbl for lbl,_ in pool0]
    survivors = []
    for lbl in labels:
        ok = True
        for x, y_true in train_pairs:
            if time.time() - t0 > time_limit:
                break
            best = 10**9
            for lbl2, ycand in generate_candidates(x, H, W, train_pairs, use_compositions=False):
                if lbl2 != lbl: 
                    continue
                best = min(best, hamming(ycand, y_true))
                if best == 0:
                    break
            if best != 0:
                ok = False
                break
        if ok:
            survivors.append(lbl)
        if time.time() - t0 > time_limit:
            break
    if not survivors:
        scores = {}
        for lbl in labels:
            tot = 0
            for x, y_true in train_pairs:
                best = 10**9
                for lbl2, ycand in generate_candidates(x, H, W, train_pairs, use_compositions=False):
                    if lbl2 != lbl: 
                        continue
                    best = min(best, hamming(ycand, y_true))
                tot += best
            scores[lbl] = tot
        if scores:
            survivors = [l for l,_ in sorted(scores.items(), key=lambda kv: kv[1])[:3]]
    return survivors
