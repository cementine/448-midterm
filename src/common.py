"""Tokenization, vocabulary, text ablations, and CNN/LSTM classifiers."""
import random, re
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
FAMILIES = ['GPT', 'Claude', 'Gemini/PaLM', 'Llama-2', 'Mistral', 'Qwen']
FAM2ID = {f: i for i, f in enumerate(FAMILIES)}
PAD, UNK, SEP, NL = ('<pad>', '<unk>', '<sep>', '<nl>')
_MD = re.compile('(^|\\n)\\s*(#{1,6}\\s+|[-*+•]\\s+|\\d+[.)]\\s+)|\\*\\*|__|`{1,3}|~~|^>\\s?', re.M)

def strip_format(t: str) -> str:
    """Remove Markdown markers and collapse whitespace while retaining word order."""
    t = _MD.sub(lambda m: m.group(1) or '', t)
    return re.sub('\\s+', ' ', t).strip()

def apply_ablation(text: str, ablation: str, rng: random.Random) -> str:
    if ablation in (None, '', 'none'):
        return text
    if ablation == 'strip_format':
        return strip_format(text)
    if ablation == 'shuffle':
        toks = tokenize(text)
        rng.shuffle(toks)
        return ' '.join(toks)
    if ablation.startswith('trunc'):
        return ' '.join(tokenize(text)[:int(ablation[5:])])
    raise ValueError(ablation)
_TOK = re.compile('<nl>|<sep>|\\w+|[^\\w\\s]', re.U)

def tokenize(text: str):
    """Lowercase text and tokenize words, punctuation, and line breaks."""
    text = text.replace('\r\n', '\n').replace('\n', f' {NL} ')
    return _TOK.findall(text.lower())

class Vocab:
    """Build a frequency-limited vocabulary with padding and unknown tokens."""

    def __init__(self, texts, max_size=30000, min_freq=2):
        c = Counter()
        for t in texts:
            c.update(tokenize(t))
        self.itos = [PAD, UNK, SEP, NL] + [w for w, n in c.most_common(max_size) if n >= min_freq and w not in (PAD, UNK, SEP, NL)]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text, max_len):
        return [self.stoi.get(w, 1) for w in tokenize(text)][:max_len]

def build_text(row, mode):
    """Select the response, prompt, or prompt and response separated by <sep>."""
    if mode == 'output':
        return row['response']
    if mode == 'input':
        return row['prompt']
    if mode == 'both':
        return row['prompt'] + f' {SEP} ' + row['response']
    raise ValueError(mode)

def encode_both(vocab, prompt, response, max_len, prompt_max=128):
    p = vocab.encode(prompt, prompt_max)
    r = vocab.encode(response, max_len - len(p) - 1)
    return p + [2] + r

class TextDataset(torch.utils.data.Dataset):

    def __init__(self, df: pd.DataFrame, vocab: Vocab, mode: str, max_len: int):
        self.ids, self.y = ([], [])
        for _, r in df.iterrows():
            ids = encode_both(vocab, r['prompt'], r['response'], max_len) if mode == 'both' else vocab.encode(build_text(r, mode), max_len)
            self.ids.append(torch.tensor(ids or [1], dtype=torch.long))
            self.y.append(FAM2ID[r['family']])
        self.y = torch.tensor(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (self.ids[i], self.y[i])

def collate(batch):
    """Pad a batch to a common length of at least five tokens for convolution."""
    seqs, ys = zip(*batch)
    lens = torch.tensor([len(s) for s in seqs])
    T = max(5, int(lens.max()))
    x = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = s
    return (x, lens, torch.stack(ys))

class TextCNN(nn.Module):
    """Pool local convolution features across the text and classify the result."""

    def __init__(self, vocab_size, n_classes, emb_dim=128, n_filters=100, kernel_sizes=(3, 4, 5)):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, n_filters, k) for k in kernel_sizes])
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)

    def forward(self, x, lens):
        e = self.emb(x).transpose(1, 2)
        feats = []
        for conv in self.convs:
            k = conv.kernel_size[0]
            h = F.relu(conv(e))
            # Exclude padded windows from max pooling; activations are nonnegative.
            pos = torch.arange(h.size(2)).unsqueeze(0)
            valid = pos < (lens.unsqueeze(1) - k + 1).clamp(min=1)
            h = h.masked_fill(~valid.unsqueeze(1), 0.0)
            feats.append(h.max(dim=2).values)
        return self.fc(torch.cat(feats, dim=1))

class LSTMClassifier(nn.Module):
    """Classify using the hidden state at the last non-padding token."""

    def __init__(self, vocab_size, n_classes, emb_dim=128, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x, lens):
        out, _ = self.lstm(self.emb(x))
        # Read the state at the last actual token, before padding.
        h_last = out[torch.arange(x.size(0)), lens - 1]
        return self.fc(h_last)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
