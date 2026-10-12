"""Build a balanced first-turn dataset from lmarena-ai/arena-human-preference-55k.
Keep responses from six model families and assign splits by prompt."""
import argparse, hashlib, json, os, re, sys, urllib.request
import pandas as pd
RAW_URL = 'https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k/resolve/main/train.csv'
FAMILY_RULES = [(lambda m: m.startswith('gpt-'), 'GPT'), (lambda m: m.startswith('claude'), 'Claude'), (lambda m: m.startswith('gemini') or m == 'palm-2', 'Gemini/PaLM'), (lambda m: m.startswith('llama-2') or m.startswith('codellama'), 'Llama-2'), (lambda m: m.startswith('mistral') or m.startswith('mixtral'), 'Mistral'), (lambda m: m.startswith('qwen'), 'Qwen')]
CODE = re.compile('```|\\b(code|coding|python|javascript|typescript|java|c\\+\\+|c#|rust|golang|sql|regex|html|css|bash|shell|script|function|class|compile|debug|bug|program|programming|algorithm|api|json|dataframe|pandas|numpy|pytorch|tensorflow|docker|kubernetes|git|linux|ubuntu|react|node\\.?js|django|flask|array|variable|loop|syntax|snippet)\\b', re.I)
MATH = re.compile('\\b(calculate|compute|solve|equation|math|mathematic\\w*|integral|derivative|probability|how many|sum of|product of|prime|theorem|proof|prove|algebra|geometry|matrix|vector|percent|percentage|average|mean of|median|fraction|decimal|square root|logarithm|factorial|arithmetic|digits?|odd|even number|divisible|multiply|divide|subtract|plus|minus|times)\\b|\\d+\\s*[\\+\\-\\*/×÷=\\^]\\s*\\d+', re.I)
WRITE = re.compile('\\b(write|compose|draft|create|generate|make up|tell me)\\b.{0,60}\\b(story|stories|poem|poetry|essay|email|e-mail|letter|haiku|song|lyrics|blog|article|novel|speech|slogan|tagline|joke|limerick|sonnet|script|dialogue|narrative|paragraph|cover letter|bio|description|caption|tweet|post)\\b|\\b(poem|haiku|limerick|sonnet|short story|fan ?fiction|rap|lyrics)\\b', re.I)
# Match task counts across families.
QUOTA = {'coding': 340, 'math': 190, 'writing': 110, 'qa_other': 1300}

def family_of(model: str):
    m = model.lower()
    for test, fam in FAMILY_RULES:
        if test(m):
            return fam
    return None

def task_of(prompt: str) -> str:
    if CODE.search(prompt):
        return 'coding'
    if MATH.search(prompt):
        return 'math'
    if WRITE.search(prompt):
        return 'writing'
    return 'qa_other'

def split_of(prompt: str, seed: int) -> str:
    """Group split by prompt: all responses to the same prompt land in one split."""
    h = int(hashlib.md5((str(seed) + prompt).encode('utf-8')).hexdigest(), 16) % 100
    return 'train' if h < 70 else 'val' if h < 85 else 'test'

def _loads(s):
    try:
        return json.loads(s)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='data/raw/arena55k_train.csv')
    ap.add_argument('--out', default='data/processed/dataset.csv')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.raw), exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if not os.path.exists(args.raw):
        print('downloading', RAW_URL, file=sys.stderr)
        urllib.request.urlretrieve(RAW_URL, args.raw)
    df = pd.read_csv(args.raw)
    rows = []
    for r in df.itertuples():
        p = _loads(r.prompt)
        if not p or not isinstance(p, list) or (not isinstance(p[0], str)):
            continue
        for side in 'ab':
            model = getattr(r, f'model_{side}')
            fam = family_of(model)
            if fam is None:
                continue
            resp = _loads(getattr(r, f'response_{side}'))
            if not resp or not isinstance(resp, list) or (not isinstance(resp[0], str)):
                continue
            rows.append(dict(battle_id=r.id, model=model, family=fam, prompt=p[0].strip(), response=resp[0].strip()))
    d = pd.DataFrame(rows)
    d = d[(d.response.str.len() >= 20) & (d.prompt.str.len() >= 5)]
    d = d.drop_duplicates(['prompt', 'response']).copy()
    d['task'] = d.prompt.map(task_of)
    print('after filtering:', d.shape, file=sys.stderr)
    print(pd.crosstab(d.family, d.task), file=sys.stderr)
    parts = []
    for (fam, task), g in d.groupby(['family', 'task']):
        n = min(QUOTA[task], len(g))
        parts.append(g.sample(n=n, random_state=args.seed))
    out = pd.concat(parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    out['split'] = out.prompt.map(lambda p: split_of(p, args.seed))
    out.insert(0, 'id', range(len(out)))
    out = out[['id', 'battle_id', 'model', 'family', 'task', 'prompt', 'response', 'split']]
    out.to_csv(args.out, index=False)
    print('saved', args.out, out.shape, file=sys.stderr)
    print(pd.crosstab(out.family, out.split), file=sys.stderr)
    print(pd.crosstab(out.family, out.task), file=sys.stderr)
    print(out.groupby('family').model.value_counts().to_string(), file=sys.stderr)
if __name__ == '__main__':
    main()
