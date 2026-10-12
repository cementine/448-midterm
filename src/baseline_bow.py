"""Fit binary unigram/bigram baselines and a logistic-regression learning curve.
Naive Bayes smoothing is selected on the validation split."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from common import FAMILIES, tokenize, build_text
OUT = 'results/bow'
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv('data/processed/dataset.csv')
tr, va, te = (df[df.split == 'train'], df[df.split == 'val'], df[df.split == 'test'])
texts = lambda d, mode: [build_text(r, mode) for _, r in d.iterrows()]

def vectorizer():
    return CountVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False, ngram_range=(1, 2), min_df=2, max_features=100000, binary=True)

def scores(y, p):
    return dict(accuracy=accuracy_score(y, p), macro_f1=f1_score(y, p, average='macro'), precision=dict(zip(FAMILIES, precision_score(y, p, average=None, labels=FAMILIES, zero_division=0).tolist())), recall=dict(zip(FAMILIES, recall_score(y, p, average=None, labels=FAMILIES, zero_division=0).tolist())), f1=dict(zip(FAMILIES, f1_score(y, p, average=None, labels=FAMILIES, zero_division=0).tolist())), confusion=confusion_matrix(y, p, labels=FAMILIES).tolist())
res = {}
for mode in ['output', 'input', 'both']:
    vec = vectorizer()
    Xtr = vec.fit_transform(texts(tr, mode))
    Xva = vec.transform(texts(va, mode))
    Xte = vec.transform(texts(te, mode))
    val = {a: accuracy_score(va.family, BernoulliNB(alpha=a).fit(Xtr, tr.family).predict(Xva)) for a in [0.1, 0.5, 1.0]}
    alpha = max(val, key=val.get)
    nb = BernoulliNB(alpha=alpha).fit(Xtr, tr.family)
    res[f'nb_{mode}'] = dict(alpha=alpha, val_acc=val, **scores(te.family, nb.predict(Xte)))
    lr = LogisticRegression(max_iter=3000).fit(Xtr, tr.family)
    res[f'lr_{mode}'] = dict(val_acc=accuracy_score(va.family, lr.predict(Xva)), **scores(te.family, lr.predict(Xte)))
    print(mode, 'NB', round(res[f'nb_{mode}']['accuracy'], 4), 'LR', round(res[f'lr_{mode}']['accuracy'], 4), flush=True)
    if mode == 'output':
        names = vec.get_feature_names_out()
        rows = []
        for i, fam in enumerate(lr.classes_):
            for rank, j in enumerate(np.argsort(-lr.coef_[i])[:30]):
                rows.append(dict(family=fam, rank=rank + 1, phrase=names[j], weight=float(lr.coef_[i, j])))
        pd.DataFrame(rows).to_csv(f'{OUT}/top_phrases.csv', index=False)
json.dump(res, open(f'{OUT}/results.json', 'w'), indent=1)
rows = []
for frac in [0.125, 0.25, 0.5, 1.0]:
    for seed in [0, 1, 2] if frac < 1 else [0]:
        sub = tr.groupby('family').sample(frac=frac, random_state=seed)
        vec = vectorizer()
        Xs = vec.fit_transform(sub.response)
        p = LogisticRegression(max_iter=3000).fit(Xs, sub.family).predict(vec.transform(te.response))
        rows.append(dict(frac=frac, seed=seed, n_train=len(sub), accuracy=accuracy_score(te.family, p)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/learning_curve.csv', index=False)
