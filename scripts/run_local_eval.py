import argparse, json
from src.evaluate import evaluate_on_training
from src.utils import load_json
from src.solver import MAX_TIME_PER_TASK, USE_COMPOSITIONS, set_runtime_flags

parser = argparse.ArgumentParser()
parser.add_argument("--train_json", required=True)
parser.add_argument("--max_time_per_task", type=float, default=2.0)
parser.add_argument("--use_compositions", type=int, default=1)
args = parser.parse_args()

set_runtime_flags(max_time=args.max_time_per_task, compositions=bool(args.use_compositions))
tasks = load_json(args.train_json)
proxy_acc, per_task = evaluate_on_training(tasks)
print(f"Proxy accuracy: {proxy_acc:.4f}  ({sum(per_task)}/{len(per_task)} solved)")