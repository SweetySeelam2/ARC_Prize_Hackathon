from .utils import to_np
from .solver import solve_task
from .utils import load_json
from .utils import hamming

def evaluate_on_training(tasks):
    solved = 0
    total = 0
    per_task = []
    for i, task in enumerate(tasks):
        fake_task = {
            "train": task["train"],
            "test": [{"input": t["input"]} for t in task["train"]],
        }
        preds = solve_task(fake_task)
        y_true_all = [to_np(t["output"]) for t in task["train"]]
        ok = all(hamming(to_np(p), y_true_all[j])==0 for j,p in enumerate(preds))
        solved += int(ok)
        total += 1
        per_task.append(ok)
    return solved/total, per_task