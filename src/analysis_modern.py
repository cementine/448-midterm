"""Evaluate the saved seed-0 CNN and LSTM on data/modern/modern.csv."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import FAMILIES, Vocab, TextCNN, LSTMClassifier, collate
OUT = 'results/modern'
os.makedirs(OUT, exist_ok=True)
os.makedirs('report/figures', exist_ok=True)
torch.set_num_threads(1)
df = pd.read_csv('data/modern/modern.csv')
df = df[df.response.notna() & (df.response.astype(str).str.len() > 0)].copy()
df['family'] = df.family.astype(str).str.strip()
res = []
for tag, run in [('CNN', 'cnn_output_tr-all_te-all_none_s0'), ('LSTM', 'lstm_output_tr-all_te-all_none_s0')]:
    ck = torch.load(f'results/{run}/model.pt', weights_only=False)
    v = Vocab([])
    v.itos = ck['itos']
    v.stoi = {w: i for i, w in enumerate(v.itos)}
    m = (TextCNN if ck['args']['model'] == 'cnn' else LSTMClassifier)(len(v), 6)
    m.load_state_dict(ck['state_dict'])
    m.eval()
    seqs = [torch.tensor(v.encode(str(t), ck['args']['max_len']) or [1]) for t in df.response]
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs), 64):
            x, lens, _ = collate([(s, torch.tensor(0)) for s in seqs[i:i + 64]])
            preds += m(x, lens).argmax(1).tolist()
    df[f'pred_{tag}'] = [FAMILIES[i] for i in preds]
    ct = pd.crosstab(df.model, df[f'pred_{tag}']).reindex(columns=FAMILIES, fill_value=0)
    ct.to_csv(f'{OUT}/predicted_family_by_model_{tag}.csv')
    for mdl, g in df.groupby('model'):
        res.append(dict(classifier=tag, model=mdl, family=g.family.iloc[0], n=len(g), correct_family_rate=float((g[f'pred_{tag}'] == g.family).mean()), most_common_prediction=g[f'pred_{tag}'].value_counts().index[0]))
R = pd.DataFrame(res)
R.to_csv(f'{OUT}/summary.csv', index=False)
df.to_csv(f'{OUT}/predictions.csv', index=False)
print(R.to_string(index=False))
print('overall:', {t: float((df[f'pred_{t}'] == df.family).mean()) for t in ['CNN', 'LSTM']})
ct = pd.read_csv(f'{OUT}/predicted_family_by_model_CNN.csv', index_col=0)
sh = ct.div(ct.sum(1), axis=0)
fig, ax = plt.subplots(figsize=(9, 0.5 * len(sh) + 1.2))
left = np.zeros(len(sh))
for i, f in enumerate(FAMILIES):
    ax.barh(range(len(sh)), sh[f], left=left, color=plt.cm.tab10(i), label=f)
    left += sh[f].values
ax.set_yticks(range(len(sh)))
ax.set_yticklabels(sh.index, fontsize=8)
ax.set_xlabel('share of responses predicted as each family (CNN)')
ax.legend(ncol=6, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, 1.25))
fig.tight_layout()
fig.savefig('report/figures/fig_modern.png', dpi=160)

# Count the same bold markers used in the full-response feature analysis.
bold = df.response.map(lambda text: text.count('**') // 2)
old = pd.read_csv('data/processed/dataset.csv')
old_bold = old.response.map(lambda text: text.count('**') // 2)
formatting = {
    'modern_bold': bold.groupby(df.model).mean().to_dict(),
    'mean_bold_all': float(bold.mean()),
    'old_bold_by_family': old_bold.groupby(old.family).mean().to_dict(),
}
with open(f'{OUT}/formatting.json', 'w') as handle:
    json.dump(formatting, handle, indent=2)
