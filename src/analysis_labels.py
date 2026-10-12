"""Compare keyword task labels with manual labels in data/label_check/label_check.csv."""
import os, json, pandas as pd
OUT = 'results/label_check'
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv('data/label_check/label_check.csv')
df = df[df.human_task.notna()].copy()
df['task'] = df.task.astype(str).str.strip()
df['human_task'] = df.human_task.astype(str).str.strip()
ct = pd.crosstab(df.human_task, df.task)
ct.to_csv(f'{OUT}/confusion.csv')
res = dict(n=len(df), accuracy=float((df.task == df.human_task).mean()), per_rule_label_precision={t: float((g.task == g.human_task).mean()) for t, g in df.groupby('task')})
json.dump(res, open(f'{OUT}/summary.json', 'w'), indent=1)
print(res)
print(ct)
