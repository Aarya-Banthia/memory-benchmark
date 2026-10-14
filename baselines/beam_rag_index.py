"""Paper-style RAG index for BEAM-1M: one document per user-assistant pair (deduplicated), BAAI/bge-small-en-v1.5, normalized."""
import json, time, numpy as np, os, torch
from sentence_transformers import SentenceTransformer
torch.set_num_threads(2)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = ROOT + '/third_party/BEAM/chats/1M/'
OUT = ROOT + '/baselines/index/'
os.makedirs(OUT, exist_ok=True)
m = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')
t0 = time.time(); done = 0
for c in range(1, 36):
    if os.path.exists(f'{OUT}{c}.npy'): continue
    d = json.load(open(f'{B}{c}/chat.json')); docs = []
    for bn, batch in enumerate(d):
        for tn, turn in enumerate(batch['turns']):
            for i in range(0, len(turn), 2):
                p = turn[i:i + 2]
                docs.append({'text': f"\nUSER: {p[0]['content']} \n\n\nASSISTANT: {p[1]['content'] if len(p) > 1 else 'N/A'}\n",
                             'ids': [x.get('id') for x in p], 'batch': bn + 1, 'turn': tn + 1})
    order = sorted(range(len(docs)), key=lambda i: len(docs[i]['text']))
    E = np.zeros((len(docs), 384), dtype=np.float32)
    for s in range(0, len(order), 64):
        idx = order[s:s + 64]
        E[idx] = m.encode([docs[i]['text'] for i in idx], batch_size=64, normalize_embeddings=True)
    np.save(f'{OUT}{c}.npy', E); json.dump(docs, open(f'{OUT}{c}.json', 'w'))
    done += len(docs); print(f'conv {c}: {len(docs)} docs, total {done}, {done/(time.time()-t0):.1f}/s', flush=True)
print('finished', flush=True)
