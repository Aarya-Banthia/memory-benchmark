"""Official BEAM scoring: the evaluate_* functions and judge prompt from the official repo
(src/evaluation/compute_metrics.py, src/prompts.py) executed unchanged; only the LLM object is a thin
wrapper calling gpt-4.1-mini (temperature 0), matching src/llm.py. Event ordering reported as tau_norm (as report_results.py does)."""
import json, sys, re, time, ast, threading, urllib.request, collections, concurrent.futures as cf
import pyarrow.parquet as pq
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = ROOT + '/third_party/BEAM/src/'
KEY = os.environ['OPENAI_API_KEY']
USAGE = collections.Counter(); UL = threading.Lock()

class Msg:
    def __init__(self, c): self.content = c
class GPT:
    def invoke(self, x):
        msgs = [{'role': 'user', 'content': x}] if isinstance(x, str) else x
        body = json.dumps({'model': 'gpt-4.1-mini', 'messages': msgs, 'temperature': 0}).encode()
        for a in range(10):
            try:
                j = json.load(urllib.request.urlopen(urllib.request.Request('https://api.openai.com/v1/chat/completions', body,
                    {'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'}), timeout=120))
                with UL:
                    USAGE['in'] += j['usage']['prompt_tokens']; USAGE['out'] += j['usage']['completion_tokens']; USAGE['calls'] += 1
                return Msg(j['choices'][0]['message']['content'])
            except Exception as e:
                time.sleep(min(60, 3 * (a + 1)))
        raise RuntimeError('judge call failed')

# official prompt + metric code, imports stripped (embedding helpers are not used on the LLM-judge path)
ns = {}
psrc = open(R + 'prompts.py').read()
exec(psrc[psrc.index('unified_llm_judge_base_prompt = """'):psrc.index('# **************** Metric-based prompts')], ns)
msrc = open(R + 'evaluation/compute_metrics.py').read()
from scipy.stats import kendalltau
from json_repair import repair_json
from typing import List, Tuple
ns.update(kendalltau=kendalltau, repair_json=repair_json, List=List, Tuple=Tuple, json=json, re=re)
exec(msrc[msrc.index('def llm_equivalence'):msrc.index('def semantic_fact_precision_recall_f1')], ns)
exec(msrc[msrc.index('def event_ordering_score'):], ns)
model = GPT()

ANS = sys.argv[1] if len(sys.argv) > 1 else ROOT + '/results/beam_1m/answers.jsonl'
rows = {json.loads(l)['key']: json.loads(l) for l in open(ANS)}
convs = {str(c['conversation_id']): ast.literal_eval(c['probing_questions']) for c in pq.read_table(os.environ.get('BEAM_DATA', ROOT + '/data/beam_1m.parquet')).to_pylist()}
OUT = sys.argv[2] if len(sys.argv) > 2 else ROOT + '/results/beam_1m/official_verdicts_gpt-4.1-mini.jsonl'
import os
done = {json.loads(l)['key'] for l in open(OUT)} if os.path.exists(OUT) else set()
LK = threading.Lock()
def job(k):
    r = rows[k]; cid, ab, qi = k.split('|')
    q = convs[cid][ab][int(qi)]
    res = ns['evaluate_' + ab](rubric=q['rubric'], llm_response=r['answer'], probing_question=q['question'], model=model)
    rec = {'key': k, 'ability': ab, 'score': res['tau_norm'] if ab == 'event_ordering' else res['llm_judge_score'],
           'llm_judge_score': res['llm_judge_score'], **({kk: res[kk] for kk in ('precision', 'recall', 'f1', 'tau_norm', 'final_score')} if ab == 'event_ordering' else {}),
           'judge': res['llm_judge_responses']}
    with LK:
        with open(OUT, 'a') as f: f.write(json.dumps(rec) + '\n')
todo = [k for k in rows if k not in done]
print('todo', len(todo), flush=True)
with cf.ThreadPoolExecutor(16) as ex:
    for f in cf.as_completed([ex.submit(job, k) for k in todo]):
        try: f.result()
        except Exception as e: print('error', repr(e)[:200], flush=True)
print('usage', dict(USAGE), flush=True)
