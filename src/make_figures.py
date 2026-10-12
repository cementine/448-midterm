"""Collect saved experiment results into tables and report figures."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import FAMILIES
FIG = 'report/figures'
os.makedirs(FIG, exist_ok=True)
SHORT = {'Gemini/PaLM': 'Gemini/\nPaLM'}
C4 = {'NB': '#9e9e9e', 'LR': '#4c72b0', 'CNN': '#dd8452', 'LSTM': '#55a868'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
runs, rows = ({}, [])
for f in sorted(glob.glob('results/*/metrics.json')):
    m = json.load(open(f))
    a = m['args']
    runs[m['run']] = m
    rows.append(dict(run=m['run'], model=a['model'], mode=a['mode'], train_tasks=a['train_tasks'].replace(',', '+'), test_tasks=a['test_tasks'].replace(',', '+'), ablation=a['ablation'], seed=a['seed'], test_acc=m['test']['accuracy'], test_macro_f1=m['test']['macro_f1'], n_test=m['test']['n'], seen_task_acc=m.get('test_seen_tasks', {}).get('accuracy', np.nan), best_epoch=m['best_epoch'], best_val_acc=m['best_val_acc'], n_params=m['n_params'], vocab=m['vocab_size'], n_train=m['n_train'], minutes=m['train_time_s'] / 60))
S = pd.DataFrame(rows)
S.to_csv('results/summary.csv', index=False)
B = json.load(open('results/bow/results.json'))
base = lambda model, seed=0, mode='output', ab='none', tr='all', te='all': f'{model}_{mode}_tr-{tr}_te-{te}_{ab}_s{seed}'
rq1 = S[(S['mode'] == 'output') & (S.train_tasks == 'all') & (S.ablation == 'none')]
t1 = rq1.groupby('model')[['test_acc', 'test_macro_f1']].agg(['mean', 'std'])
t1.to_csv('results/rq1_table.csv')
fam_f1 = pd.DataFrame({'NB': B['nb_output']['f1'], 'LR': B['lr_output']['f1'], 'CNN': pd.DataFrame([runs[base('cnn', s)]['test']['f1'] for s in range(3)]).mean(), 'LSTM': pd.DataFrame([runs[base('lstm', s)]['test']['f1'] for s in range(3)]).mean()}).reindex(FAMILIES)
fam_f1.to_csv('results/rq1_per_family_f1.csv')
fam_rec = pd.DataFrame({'CNN': runs[base('cnn')]['test']['recall'], 'LSTM': runs[base('lstm')]['test']['recall']}).reindex(FAMILIES)
fam_rec.to_csv('results/rq1_per_family_recall_seed0.csv')
pd.DataFrame({'CNN': runs[base('cnn')]['test']['per_model_accuracy'], 'LSTM': runs[base('lstm')]['test']['per_model_accuracy']}).to_csv('results/rq1_per_model_version.csv')
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, model, title in [(axes[0], 'cnn', 'CNN'), (axes[1], 'lstm', 'LSTM')]:
    m = runs[base(model)]
    C = np.array(m['test']['confusion'])
    Cn = C / C.sum(1, keepdims=True)
    ax.imshow(Cn, cmap='Blues', vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, C[i, j], ha='center', va='center', fontsize=8, color='white' if Cn[i, j] > 0.5 else 'black')
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels([SHORT.get(f, f) for f in FAMILIES], fontsize=8)
    ax.set_yticklabels([SHORT.get(f, f) for f in FAMILIES], fontsize=8)
    ax.set_xlabel('predicted family')
    ax.set_ylabel('true family')
    ax.set_title(f"{title}: accuracy {m['test']['accuracy']:.3f} (seed 0)")
fig.tight_layout()
fig.savefig(f'{FIG}/fig_rq1_confusion.png', dpi=160)
plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, model, title in [(axes[0], 'cnn', 'CNN'), (axes[1], 'lstm', 'LSTM')]:
    for s in range(3):
        h = pd.read_csv(f'results/{base(model, s)}/history.csv')
        ax.plot(h.epoch, h.train_acc, '-o', ms=3, color=plt.cm.tab10(s), label=f'train, seed {s}')
        ax.plot(h.epoch, h.val_acc, '--s', ms=3, color=plt.cm.tab10(s), label=f'validation, seed {s}')
    ax.axhline(1 / 6, ls=':', color='gray')
    ax.set_xlabel('epoch')
    ax.set_title(f'{title}: training vs validation accuracy')
axes[0].set_ylabel('accuracy')
axes[1].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(f'{FIG}/fig_rq1_curves.png', dpi=160)
plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 3.6))
w = 0.2
for i, k in enumerate(['NB', 'LR', 'CNN', 'LSTM']):
    ax.bar(np.arange(6) + (i - 1.5) * w, fam_f1[k], w, label={'NB': 'Naive Bayes', 'LR': 'Logistic regression'}.get(k, k), color=C4[k])
ax.set_xticks(range(6))
ax.set_xticklabels(FAMILIES)
ax.set_ylabel('F1 on test split')
ax.axhline(1 / 6, ls=':', color='gray')
ax.legend(ncol=4, fontsize=8)
ax.set_title('RQ1: how well is each family recognised? (response only)')
fig.tight_layout()
fig.savefig(f'{FIG}/fig_rq1_per_family.png', dpi=160)
plt.close(fig)
rq2 = pd.DataFrame({mode: {'NB': B[f'nb_{mode}']['accuracy'], 'LR': B[f'lr_{mode}']['accuracy'], 'CNN': runs[base('cnn', mode=mode)]['test']['accuracy'], 'LSTM': runs[base('lstm', mode=mode)]['test']['accuracy']} for mode in ['input', 'output', 'both']})
rq2.to_csv('results/rq2_table.csv')
fig, ax = plt.subplots(figsize=(7.5, 3.6))
w = 0.26
for i, (mode, lab, col) in enumerate([('input', 'prompt only', '#c7c7c7'), ('output', 'response only', '#4c72b0'), ('both', 'prompt + response', '#dd8452')]):
    ax.bar(np.arange(4) + (i - 1) * w, rq2.loc[['NB', 'LR', 'CNN', 'LSTM'], mode], w, label=lab, color=col)
ax.axhline(1 / 6, ls=':', color='gray', label='chance (1/6)')
ax.set_xticks(range(4))
ax.set_xticklabels(['Naive Bayes', 'Logistic reg.', 'CNN', 'LSTM'])
ax.set_ylabel('test accuracy')
ax.legend(fontsize=8)
ax.set_title('RQ2: does the prompt help?')
fig.tight_layout()
fig.savefig(f'{FIG}/fig_rq2.png', dpi=160)
plt.close(fig)
rq3 = S[S.train_tasks != 'all'].copy()
rq3['all_task_model_on_target'] = [runs[base(r.model)]['test']['per_task_accuracy'][r.test_tasks] for r in rq3.itertuples()]
rq3 = rq3[['model', 'train_tasks', 'test_tasks', 'n_test', 'seen_task_acc', 'test_acc', 'all_task_model_on_target']].sort_values(['test_tasks', 'model'])
rq3.to_csv('results/rq3_table.csv', index=False)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for ax, (trt, tet) in zip(axes, [('coding+math+qa_other', 'writing'), ('qa_other', 'coding'), ('coding+math+writing', 'qa_other')]):
    sub = rq3[(rq3.train_tasks == trt) & (rq3.test_tasks == tet)].set_index('model').reindex(['cnn', 'lstm'])
    x = np.arange(2)
    w = 0.26
    ax.bar(x - w, sub.seen_task_acc, w, label='test on the training tasks', color='#4c72b0')
    ax.bar(x, sub.test_acc, w, label=f'test on unseen task ({tet})', color='#dd8452')
    ax.bar(x + w, sub.all_task_model_on_target, w, label=f'reference: model trained on all tasks, tested on {tet}', color='#c7c7c7')
    ax.axhline(1 / 6, ls=':', color='gray')
    ax.set_xticks(x)
    ax.set_xticklabels(['CNN', 'LSTM'])
    ax.set_title(f'train on {trt}\ntest on {tet}', fontsize=9)
    ax.legend(fontsize=6.5, loc='lower center')
axes[0].set_ylabel('test accuracy')
fig.tight_layout()
fig.savefig(f'{FIG}/fig_rq3.png', dpi=160)
plt.close(fig)
means = pd.read_csv('results/rq4/feature_means.csv', index_col=0)
groups = {'length': ['n_words', 'n_sentences', 'words_per_sentence'], 'vocabulary': ['type_token_ratio', 'avg_word_len'], 'formatting': ['n_line_breaks', 'n_bullets', 'n_numbered', 'n_headers', 'n_bold', 'n_code_blocks'], 'phrases': ['starts_with_sure', 'refusal_phrases', 'first_person_rate']}
feats = [f for g in groups.values() for f in g]
gof = {f: g for g, fs in groups.items() for f in fs}
fig, axes = plt.subplots(4, 4, figsize=(13, 10))
axes = axes.ravel()
for ax, f in zip(axes, feats):
    ax.bar(range(6), means.loc[f, FAMILIES], color=[plt.cm.tab10(i) for i in range(6)])
    ax.set_xticks(range(6))
    ax.set_xticklabels(FAMILIES, rotation=40, ha='right', fontsize=7)
    ax.set_title(f'{f}  [{gof[f]}]', fontsize=9)
for ax in axes[len(feats):]:
    ax.axis('off')
fig.suptitle('RQ4: average of each feature per family')
fig.tight_layout()
fig.savefig(f'{FIG}/fig_rq4_features.png', dpi=150)
plt.close(fig)
W = pd.read_csv('results/rq4/lr_weights.csv', index_col=0).reindex(FAMILIES)[feats]
fig, ax = plt.subplots(figsize=(11, 3.6))
v = np.abs(W.values).max()
im = ax.imshow(W.values, cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto')
ax.set_xticks(range(len(feats)))
ax.set_xticklabels(feats, rotation=40, ha='right', fontsize=8)
ax.set_yticks(range(6))
ax.set_yticklabels(FAMILIES)
for i in range(6):
    for j in range(len(feats)):
        ax.text(j, i, f'{W.values[i, j]:+.1f}', ha='center', va='center', fontsize=6.5)
fig.colorbar(im, ax=ax, fraction=0.02, label='weight (red: points to this family)')
ax.set_title('RQ4: logistic-regression weights on the 14 features (features on a common scale)')
fig.tight_layout()
fig.savefig(f'{FIG}/fig_rq4_weights.png', dpi=160)
plt.close(fig)
ab = pd.DataFrame({k: {'full text': runs[base(k)]['test']['accuracy'], **{lab: runs[base(k, ab=a)]['test']['accuracy'] for a, lab in [('strip_format', 'formatting removed'), ('trunc64', 'first 64 tokens only'), ('shuffle', 'word order shuffled')]}} for k in ['cnn', 'lstm']}).T
ab.to_csv('results/rq4_ablation_table.csv')
fig, ax = plt.subplots(figsize=(8, 3.6))
w = 0.36
for i, k in enumerate(['cnn', 'lstm']):
    ax.bar(np.arange(4) + (i - 0.5) * w, ab.loc[k], w, label=k.upper(), color=C4[k.upper()])
ax.axhline(1 / 6, ls=':', color='gray')
ax.set_xticks(range(4))
ax.set_xticklabels(ab.columns, fontsize=9)
ax.set_ylabel('test accuracy')
ax.legend()
ax.set_title('RQ4: remove one kind of signal, retrain, test')
fig.tight_layout()
fig.savefig(f'{FIG}/fig_rq4_ablation.png', dpi=160)
plt.close(fig)
kt = pd.read_csv('results/rq4/kmeans_family_table.csv', index_col=0)[FAMILIES]
kp = pd.read_csv('results/rq4/kmeans_cluster_profiles.csv', index_col=0)
fig, ax = plt.subplots(figsize=(9, 3.8))
left = np.zeros(len(kt))
share = kt.div(kt.sum(1), axis=0)
for i, f in enumerate(FAMILIES):
    ax.barh(range(len(kt)), share[f], left=left, color=plt.cm.tab10(i), label=f)
    left += share[f].values
ax.set_yticks(range(len(kt)))
ax.set_yticklabels([f"cluster {c}: n={kt.loc[c].sum()}, {kp.loc[c, 'n_words']:.0f} words, {kp.loc[c, 'n_bold']:.1f} bold" for c in kt.index], fontsize=8)
ax.set_xlabel("share of the cluster's responses")
ax.legend(ncol=6, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, 1.18))
fig.tight_layout()
fig.savefig(f'{FIG}/fig_rq4_kmeans.png', dpi=160)
plt.close(fig)
lc = pd.read_csv('results/bow/learning_curve.csv').groupby('n_train').accuracy.agg(['mean', 'min', 'max']).reset_index()
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.plot(lc.n_train, lc['mean'], '-o')
ax.fill_between(lc.n_train, lc['min'], lc['max'], alpha=0.2)
ax.set_xscale('log', base=2)
ax.set_xlabel('number of training responses (log scale)')
ax.set_ylabel('test accuracy')
ax.set_title('Learning curve (logistic regression)')
ax.axhline(1 / 6, ls=':', color='gray')
fig.tight_layout()
fig.savefig(f'{FIG}/fig_learning_curve.png', dpi=160)
plt.close(fig)
print(S[['run', 'test_acc', 'seen_task_acc', 'best_epoch', 'minutes']].round(3).to_string(index=False))
print(t1.round(4))
print(rq2.round(3))
print(rq3.round(3).to_string(index=False))
print(ab.round(3))
