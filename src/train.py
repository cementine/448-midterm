"""Train and evaluate a CNN or LSTM family classifier.
Select the checkpoint by validation accuracy and save test predictions and metrics."""
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from common import FAMILIES, Vocab, TextDataset, collate, TextCNN, LSTMClassifier, set_seed, apply_ablation, build_text

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed/dataset.csv')
    ap.add_argument('--model', choices=['cnn', 'lstm'], required=True)
    ap.add_argument('--mode', choices=['output', 'input', 'both'], default='output')
    ap.add_argument('--train_tasks', default='all', help="comma list or 'all'")
    ap.add_argument('--test_tasks', default='all')
    ap.add_argument('--ablation', default='none', help='none | strip_format | shuffle | truncK (applied to the response)')
    ap.add_argument('--max_len', type=int, default=200)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--save_model', action='store_true', help='save model.pt (weights + vocabulary) for src/predict.py')
    ap.add_argument('--name', default=None)
    ap.add_argument('--out_dir', default='results')
    return ap.parse_args()

def predict(model, loader):
    model.eval()
    preds, ys = ([], [])
    with torch.no_grad():
        for x, lens, y in loader:
            preds.append(model(x, lens).argmax(1))
            ys.append(y)
    return (torch.cat(preds).numpy(), torch.cat(ys).numpy())

def main():
    a = get_args()
    torch.set_num_threads(a.threads)
    set_seed(a.seed)
    name = a.name or f"{a.model}_{a.mode}_tr-{a.train_tasks.replace(',', '+')}_te-{a.test_tasks.replace(',', '+')}_{a.ablation}_s{a.seed}"
    out = os.path.join(a.out_dir, name)
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(a.data)
    if a.ablation != 'none':
        rng = random.Random(a.seed)
        df['response'] = [apply_ablation(t, a.ablation, rng) for t in df['response']]
    tr_tasks = None if a.train_tasks == 'all' else set(a.train_tasks.split(','))
    te_tasks = None if a.test_tasks == 'all' else set(a.test_tasks.split(','))
    tr, va, te = (df[df.split == 'train'], df[df.split == 'val'], df[df.split == 'test'])
    if tr_tasks:
        tr, va = (tr[tr.task.isin(tr_tasks)], va[va.task.isin(tr_tasks)])
    te_seen = te if tr_tasks is None else te[te.task.isin(tr_tasks)]
    te_target = te if te_tasks is None else te[te.task.isin(te_tasks)]
    vocab = Vocab([build_text(r, a.mode) for _, r in tr.iterrows()])
    mk = lambda d, shuffle: torch.utils.data.DataLoader(TextDataset(d, vocab, a.mode, a.max_len), batch_size=a.batch_size, shuffle=shuffle, collate_fn=collate)
    tr_loader, va_loader, te_loader = (mk(tr, True), mk(va, False), mk(te_target, False))
    model = (TextCNN if a.model == 'cnn' else LSTMClassifier)(len(vocab), len(FAMILIES))
    n_params = sum((p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state, hist, t0 = (-1.0, None, [], time.time())
    for ep in range(1, a.epochs + 1):
        model.train()
        tot, n, correct = (0.0, 0, 0)
        for x, lens, y in tr_loader:
            opt.zero_grad()
            scores = model(x, lens)
            loss = loss_fn(scores, y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
            correct += (scores.argmax(1) == y).sum().item()
        pv, yv = predict(model, va_loader)
        val_acc = accuracy_score(yv, pv)
        hist.append(dict(epoch=ep, train_loss=tot / n, train_acc=correct / n, val_acc=val_acc, elapsed_s=time.time() - t0))
        print(f'[{name}] ep{ep} loss={tot / n:.4f} train_acc={correct / n:.4f} val_acc={val_acc:.4f} ({time.time() - t0:.0f}s)', flush=True)
        if val_acc > best_acc:
            best_acc, best_epoch = (val_acc, ep)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    def report(frame, loader, tag):
        p, y = predict(model, loader)
        L = list(range(len(FAMILIES)))
        res = dict(n=int(len(y)), accuracy=accuracy_score(y, p), macro_f1=f1_score(y, p, average='macro'), precision=dict(zip(FAMILIES, precision_score(y, p, average=None, labels=L, zero_division=0).tolist())), recall=dict(zip(FAMILIES, recall_score(y, p, average=None, labels=L, zero_division=0).tolist())), f1=dict(zip(FAMILIES, f1_score(y, p, average=None, labels=L, zero_division=0).tolist())), confusion=confusion_matrix(y, p, labels=L).tolist())
        fr = frame[['id', 'model', 'family', 'task']].copy()
        fr['pred'] = [FAMILIES[i] for i in p]
        fr['correct'] = fr.pred == fr.family
        res['per_task_accuracy'] = fr.groupby('task').correct.mean().to_dict()
        res['per_model_accuracy'] = fr.groupby('model').correct.mean().to_dict()
        fr.to_csv(os.path.join(out, f'predictions_{tag}.csv'), index=False)
        return res
    metrics = dict(run=name, args=vars(a), n_params=n_params, vocab_size=len(vocab), n_train=len(tr), n_val=len(va), best_epoch=best_epoch, best_val_acc=best_acc, train_time_s=time.time() - t0, test=report(te_target, te_loader, 'test'))
    if tr_tasks is not None:
        metrics['test_seen_tasks'] = report(te_seen, mk(te_seen, False), 'test_seen_tasks')
    json.dump(metrics, open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
    pd.DataFrame(hist).to_csv(os.path.join(out, 'history.csv'), index=False)
    if a.save_model:
        torch.save(dict(state_dict=model.state_dict(), itos=vocab.itos, args=vars(a)), os.path.join(out, 'model.pt'))
    print(f"[{name}] TEST acc={metrics['test']['accuracy']:.4f} (best epoch {best_epoch})" + (f" | seen-task acc={metrics['test_seen_tasks']['accuracy']:.4f}" if tr_tasks else ''), flush=True)
if __name__ == '__main__':
    main()
