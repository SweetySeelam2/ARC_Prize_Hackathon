import time
from .utils import to_np, to_list, pad_to
from .candidate_gen import generate_candidates, infer_target_shape, select_survivor_labels
from .ranker import rank_candidates_for_test

MAX_TIME_PER_TASK = 2.0
USE_COMPOSITIONS = True

def set_runtime_flags(max_time=None, compositions=None):
    global MAX_TIME_PER_TASK, USE_COMPOSITIONS
    if max_time is not None:
        MAX_TIME_PER_TASK = float(max_time)
    if compositions is not None:
        USE_COMPOSITIONS = bool(compositions)

def solve_task(task, time_limit=None):
    if time_limit is None:
        time_limit = MAX_TIME_PER_TASK
    t0 = time.time()
    train_pairs = [(to_np(p["input"]), to_np(p["output"])) for p in task["train"]]
    test_inputs = [to_np(p["input"]) for p in task["test"]]

    H, W = infer_target_shape(train_pairs)
    survivors = select_survivor_labels(train_pairs, H, W, USE_COMPOSITIONS, time_limit=max(0.5, time_limit*0.5))

    preds = []
    for x in test_inputs:
        if time.time() - t0 > time_limit:
            preds.append(to_list(pad_to(x, H, W)))
            continue
        cand_pool = generate_candidates(x, H, W, train_pairs, use_compositions=USE_COMPOSITIONS)
        if not cand_pool:
            preds.append(to_list(pad_to(x, H, W)))
            continue
        y_best = rank_candidates_for_test(cand_pool, survivors)
        preds.append(to_list(y_best if y_best is not None else pad_to(x, H, W)))
    return preds

def solve_many(tasks):
    outputs = []
    for i, task in enumerate(tasks):
        if (i % 50) == 0:
            print(f"Solving task {i}/{len(tasks)}")
        preds = solve_task(task, time_limit=MAX_TIME_PER_TASK)
        outputs.append([{"output": p} for p in preds])
    return outputs