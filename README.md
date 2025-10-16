# ARC Prize 2025 — Heuristic + CC Solver (WIP)

> **Status:** 🟢 Actively competing in Kaggle’s ARC Prize 2025 — repo updated continuously  
> **Latest public score:** 0.00 (early iterations; focusing on correctness/format + generalization)  
> **Notebook:** `notebooks/ARC_Prize_2025_Submission.ipynb`

---

## Overview

This competition asks us to build systems that can **learn new skills efficiently** and **generalize** to novel tasks (ARC: *Abstraction & Reasoning Corpus*).  
This repo is my active work-in-progress solver that blends **symbolic/heuristic operations** with **connected-components reasoning** and a **per-input target-shape inference** step.

---

## Objectives

A practical, reproducible baseline that I’m iterating on daily:

- Validate locally (proxy accuracy) before Kaggle submissions.
- **Heuristic ensemble** of robust grid ops with **depth-2 compositions** and **candidate ranking**.
- Add **IO color mapping** and **recursive CC transformation** (object-wise).
- Guarantee at least one safe guess via **identity backstop** as Attempt-2.
- Clean repo layout with scripts to evaluate and generate `submission.json`.

---

## ✨ Why this repo

ARC is a reasoning benchmark; conventional ML tends to overfit. This baseline focuses on **symbolic/grid heuristics**:

- rotations / mirrors / transposes  
- connected components (extract, grow/shrink, replace-with-pixel), bbox crop/center  
- color remapping (learned from training pairs)  
- integer resize & tiling, symmetry completion  
- **candidate generation → filter (fits train signature) → rank (diversity + simplicity)**  
- **per-input target-shape inference** and **identity backstop** to avoid shape mismatches.

---

## 📂 Project Structure

ARC_Prize_Hackathon/                      
├─ README.md                                
├─ LICENSE                               
├─ requirements.txt                             
├─ .gitignore                          
├─ notebooks/                    
│ └─ arc-prize-notebook.ipynb                          
├─ src/                         
│ ├─ init.py                  
│ ├─ ops.py                       
│ ├─ utils.py                   
│ ├─ candidate_gen.py                      
│ ├─ ranker.py                          
│ ├─ solver.py                               
│ └─ evaluate.py                        
├─ scripts/                               
│ ├─ run_local_eval.py                               
│ └─ make_submission.py                            
├─ data/                                  
│ ├─json files                            
├─ reports/                                    
│ ├─ local_eval.md                                 
│ └─ experiments.csv                             
├─ images/                             
  └─kaggle_submissions.png

> **Note:** `data/` and submission artifacts are **ignored** from git.                                    
> ⚠️ Data Note: Due to Kaggle competition rules, ARC Prize 2025 datasets are **not included** in this repository.                                
> Please download them directly from the competition page and place them under `/data/` before running notebooks.                        

---

## 🚀 Quickstart

**1) Install**
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2) Put data locally (not committed)**
Download from the Kaggle ARC Prize 2025 dataset and place here:

data/
  arc-agi_training_challenges.json
  arc-agi_test_challenges.json

**3) Run local proxy evaluation**
python scripts/run_local_eval.py \
  --train_json data/arc-agi_training_challenges.json \
  --max_time_per_task 3.0 \
  --use_compositions 1

**4) Make a submission (only when proxy improves)**
```
python scripts/make_submission.py \
  --test_json data/arc-agi_test_challenges.json \
  --out submission.json \
  --max_time_per_task 3.0 \
  --use_compositions 1
```

Upload submission.json to Kaggle.

**Kaggle submission flow**

- Open notebooks/ARC_Prize_2025_Submission.ipynb on Kaggle.

- Run all cells (uses per-input shape inference + identity backstop).

- A submission.json is written to /kaggle/working/ and submitted.

---

## Submissions log (public)

| Date (UTC) | Notebook    | Notes                                                   | Public Score |
| ---------- | ----------- | ------------------------------------------------------- | ------------ |
| 2025-10-15 | Submission5 | Two-pronged heuristic + CC; per-input shape coming next | 0.00         |
| 2025-10-16 | Submission4 | Heuristic solver w/ size inference                      | 0.00         |
| 2025-10-03 | Submission3 | Multi-modal heuristic                                   | 0.00         |
| 2025-09-16 | Submission1 | Wide-coverage heuristic solver                          | 0.00         |

Screenshots in /images (e.g., kaggle_submissions.png)

---

## 🧠 Method

- Per-input target H×W inference from training pairs using both additive and multiplicative rules; fallback to input size when uncertain.

- Candidate generation per test input using:

  - single ops (rotate, mirror, transpose, crop/center, recolor, integer resize, symmetry completion, tiling, CC manipulations)

  - optional depth-2 compositions

  - IO color mapping abstraction

  - Recursive CC transformation (learn a transform on the largest object; apply to all)

- Signature filter: keep candidates whose grid features (colors/CCs/filled) match median train signature (tolerant).

- Survivor selection across train pairs + diversity-aware ranking for top-2 predictions.

- Identity backstop: if diversity fails or time runs out, Attempt-2 is always a safe identity at input size.

---

## 🔧 Configuration

--max_time_per_task: more time → more candidates (watch runtime).

--use_compositions (0/1): enable shallow depth-2 search for cheap gains.

---

## 🧪 Logging & Reports

Append results to reports/experiments.csv:

```
date,max_time_per_task,use_compositions,proxy_acc,solved/total,notes
```

Paste brief summaries into reports/local_eval.md for a readable trail.

---

## 🗺️ Roadmap

✅ Per-input (H,W) inference

✅ Identity fallback as Attempt-2

⏳ Cleaner survivor selection across all train pairs

⏳ Grid grammar induction & object alignment

⏳ Deterministic ablation & timing budget control

PRs welcome if they preserve determinism + clarity.

---

## 👤 Author

**Sweety Seelam** — ARC Prize 2025 participant.

Focused on symbolic + hybrid reasoning for program induction.

---

## 📜 License

MIT License

Copyright (c) 2025 Sweety Seelam