"""BEAM-1M RAG baseline, paper setup (arXiv 2510.27246 and src/answer_probing_questions/long_term_memory_methods.py):
one document per user-assistant pair, BAAI/bge-small-en-v1.5 dense retrieval, top-k, context capped at 29,000 tokens
(len//3.5, as in handling_context), official answer_generation_for_rag prompt, single call. Only the answering model
changes: GLM-5.3-flash, the same model as the read-time harness. Documents are deduplicated (the official chunker
repeats each pair once per message in a turn), which can only help RAG.
Usage: python3 baselines/beam_rag_answer.py 5 results/beam_1m/rag_same_model/rag_top5_answers.jsonl
       python3 baselines/beam_rag_answer.py 100 results/beam_1m/rag_same_model/rag_full32k_answers.jsonl   (k=100, capped at 29K tokens)"""
import json, sys, os, time, ast, threading, urllib.request, numpy as np, concurrent.futures as cf, torch
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
torch.set_num_threads(2)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = ROOT + '/baselines'
K_TOP = int(sys.argv[1]); OUT = sys.argv[2]
KEY = os.environ['LLM_API_KEY']
URL = os.environ.get('LLM_BASE_URL', 'https://router.requesty.ai/v1') + '/chat/completions'
MODEL = os.environ.get('LLM_MODEL', 'zai/glm-5.3-flash')
psrc = open(ROOT + '/third_party/BEAM/src/prompts.py').read()
ns = {}; exec(psrc[psrc.index('answer_generation_for_rag = """'):psrc.index('# ################################################## Our method')], ns)
PROMPT = ns['answer_generation_for_rag']
m = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')
convs = pq.read_table(os.environ.get('BEAM_DATA', ROOT + '/data/beam_1m.parquet')).to_pylist()
done = {json.loads(l)['key'] for l in open(OUT)} if os.path.exists(OUT) else set()
jobs = []
for c in convs:
    cid = str(c['conversation_id'])
    E = np.load(f'{H}/index/{cid}.npy'); docs = json.load(open(f'{H}/index/{cid}.json'))
    P = ast.literal_eval(c['probing_questions'])
    for ab, qs in P.items():
        for qi, q in enumerate(qs):
            key = f'{cid}|{ab}|{qi}'
            if key in done: continue
            qv = m.encode([q['question']], normalize_embeddings=True)[0]
            top = np.argsort(-(E @ qv))[:K_TOP]
            ctx = ''; used = []
            for j in top:
                if len(ctx + docs[j]['text']) // 3.5 < 29000:
                    ctx += docs[j]['text']; used.append(int(j))
            src = q.get('source_chat_ids')
            gold = sorted({int(x) for x in (sum(src.values(), []) if isinstance(src, dict) else (src or [])) if str(x).isdigit()}) if src else []
            ret = sorted({i for j in used for i in docs[j]['ids'] if i is not None})
            jobs.append((key, cid, ab, q['question'], PROMPT.replace('<context>', ctx).replace('<question>', q['question']), len(used), ret, gold))
print('jobs', len(jobs), flush=True)
LK = threading.Lock()
def call(prompt):
    body = json.dumps({'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 4096, 'temperature': 0.2}).encode()
    for a in range(6):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(URL, body,
                   {'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'}), timeout=180))
        except Exception:
            time.sleep(5 * (a + 1))
    raise RuntimeError('call failed')
def run(j):
    key, cid, ab, question, prompt, n_used, ret, gold = j
    r = call(prompt); u = r.get('usage', {})
    rec = {'key': key, 'conversation_id': cid, 'ability': ab, 'question': question,
           'answer': (r['choices'][0]['message'].get('content') or '').strip(), 'docs_in_context': n_used,
           'retrieved_ids': ret, 'gold_ids': gold, 'prompt_tokens': u.get('prompt_tokens', 0),
           'completion_tokens': u.get('completion_tokens', 0), 'cached_tokens': (u.get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0}
    with LK:
        with open(OUT, 'a') as f: f.write(json.dumps(rec) + '\n')
with cf.ThreadPoolExecutor(12) as ex:
    for f in cf.as_completed([ex.submit(run, j) for j in jobs]):
        try: f.result()
        except Exception as e: print('error', repr(e)[:200], flush=True)
print('finished', flush=True)
