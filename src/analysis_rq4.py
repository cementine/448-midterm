"""Compare length, vocabulary, formatting, and phrase features across families.
Fit feature-based classifiers and compare K-means clusters with family labels."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from common import FAMILIES
OUT = 'results/rq4'
os.makedirs(OUT, exist_ok=True)
WORD = re.compile('\\b\\w+\\b')
SENT = re.compile('[.!?]+(?:\\s|$)')
REFUSAL = re.compile("\\b(i cannot|i can't|i am unable|i'm unable|i'm not able|i apologize|i'm sorry|i am sorry|as an ai|as a language model)\\b", re.I)
GROUPS = {'length': ['n_words', 'n_sentences', 'words_per_sentence'], 'vocabulary': ['type_token_ratio', 'avg_word_len'], 'formatting': ['n_line_breaks', 'n_bullets', 'n_numbered', 'n_headers', 'n_bold', 'n_code_blocks'], 'phrases': ['starts_with_sure', 'refusal_phrases', 'first_person_rate']}
FEATS = [f for g in GROUPS.values() for f in g]

def features(t):
    words = WORD.findall(t)
    lw = [w.lower() for w in words]
    n = max(len(words), 1)
    lines = t.split('\n')
    sents = [s for s in SENT.split(t) if s.strip()]
    return dict(n_words=len(words), n_sentences=len(sents), words_per_sentence=n / max(len(sents), 1), type_token_ratio=len(set(lw)) / n, avg_word_len=float(np.mean([len(w) for w in words])) if words else 0.0, n_line_breaks=t.count('\n'), n_bullets=sum((bool(re.match('\\s*[-*•]\\s+', l)) for l in lines)), n_numbered=sum((bool(re.match('\\s*\\d+[.)]\\s+', l)) for l in lines)), n_headers=sum((bool(re.match('\\s*#{1,6}\\s+', l)) for l in lines)), n_bold=t.count('**') // 2, n_code_blocks=t.count('```') // 2, starts_with_sure=int(bool(re.match('\\s*(sure|certainly|of course|absolutely)\\b', t, re.I))), refusal_phrases=len(REFUSAL.findall(t)), first_person_rate=1000 * sum((w in ('i', 'my', 'me') for w in lw)) / n)

def purity(labels, clusters):
    ct = pd.crosstab(clusters, labels)
    return float(ct.max(axis=1).sum() / len(labels))
df = pd.read_csv('data/processed/dataset.csv')
F = pd.DataFrame([features(t) for t in df.response], index=df.index)
data = pd.concat([df[['id', 'family', 'model', 'task', 'split']], F], axis=1)
data.to_csv(f'{OUT}/features.csv', index=False)
means = data.groupby('family')[FEATS].mean().reindex(FAMILIES).T
means.to_csv(f'{OUT}/feature_means.csv')
medians = data.groupby('family')[['n_words', 'n_sentences']].median().reindex(FAMILIES).T
medians.to_csv(f'{OUT}/feature_medians.csv')
# Log-scale skewed counts before standardization.
Z = np.log1p(F[FEATS].clip(lower=0))
# A constant feature becomes zero after centering.
Z = (Z - Z.mean()) / Z.std().replace(0, 1)
trm, tem = (data.split.isin(['train', 'val']).values, (data.split == 'test').values)
res = {}
for gname, cols in [('all 14 features', FEATS)] + [(f'{g} only', c) for g, c in GROUPS.items()]:
    lr = LogisticRegression(max_iter=3000).fit(Z.loc[trm, cols], data.family[trm])
    res[gname] = dict(n_features=len(cols), accuracy=accuracy_score(data.family[tem], lr.predict(Z.loc[tem, cols])))
    if gname == 'all 14 features':
        pd.DataFrame(lr.coef_, index=lr.classes_, columns=FEATS).reindex(FAMILIES).to_csv(f'{OUT}/lr_weights.csv')
json.dump(res, open(f'{OUT}/feature_classifiers.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
km = KMeans(n_clusters=6, n_init=10, random_state=0).fit(Z.values)
data['cluster'] = km.labels_
rng = np.random.RandomState(0)
kres = dict(purity_vs_family=purity(data.family, data.cluster), purity_random_clusters=float(np.mean([purity(data.family, rng.permutation(data.cluster.values)) for _ in range(20)])), cluster_sizes=data.cluster.value_counts().sort_index().tolist())
pd.crosstab(data.cluster, data.family)[FAMILIES].to_csv(f'{OUT}/kmeans_family_table.csv')
pd.crosstab(data.cluster, data.task).to_csv(f'{OUT}/kmeans_task_table.csv')
data.groupby('cluster')[['n_words', 'n_line_breaks', 'n_bullets', 'n_bold', 'n_code_blocks', 'refusal_phrases']].mean().round(2).to_csv(f'{OUT}/kmeans_cluster_profiles.csv')
json.dump(kres, open(f'{OUT}/kmeans.json', 'w'), indent=1)
print(kres)
print(pd.crosstab(data.cluster, data.family)[FAMILIES])
print(pd.read_csv(f'{OUT}/kmeans_cluster_profiles.csv'))
