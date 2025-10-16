import argparse, json
from src.utils import load_json
from src.solver import solve_many, set_runtime_flags

parser = argparse.ArgumentParser()
parser.add_argument("--test_json", required=True)
parser.add_argument("--out", default="submission.json")
parser.add_argument("--max_time_per_task", type=float, default=2.0)
parser.add_argument("--use_compositions", type=int, default=1)
args = parser.parse_args()

set_runtime_flags(max_time=args.max_time_per_task, compositions=bool(args.use_compositions))
tasks = load_json(args.test_json)
preds = solve_many(tasks)
with open(args.out, "w") as f:
    json.dump(preds, f)
print("Wrote:", args.out)