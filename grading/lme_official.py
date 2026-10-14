"""Official LongMemEval answer check: prompts from the official evaluate_qa.py (imported from the repo), judge gpt-4o-2024-08-06, temperature 0, max_tokens 10."""
import json, sys, time, urllib.request, importlib.util, types, collections, concurrent.futures as cf
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# load only the prompt function from the official file, without its openai/tqdm imports
src = open(ROOT + '/third_party/LongMemEval/src/evaluation/evaluate_qa.py').read()
fn_src = src[src.index('def get_anscheck_prompt'):src.index("if __name__ == '__main__':")]
ns = {}; exec(fn_src, ns); get_anscheck_prompt = ns['get_anscheck_prompt']
KEY = os.environ['OPENAI_API_KEY']
HYP, OUT = sys.argv[1], sys.argv[2]
ref = {e['question_id']: e for e in json.load(open(os.environ.get('LME_DATA', ROOT + '/data/longmemeval_s.json')))}
hyps = [json.loads(l) for l in open(HYP)]
def judge(h):
    r = ref[h['question_id']]
    p = get_anscheck_prompt(r['question_type'], r['question'], r['answer'], h['hypothesis'], abstention='_abs' in h['question_id'])
    body = json.dumps({'model': 'gpt-4o-2024-08-06', 'messages': [{'role': 'user', 'content': p}], 'n': 1, 'temperature': 0, 'max_tokens': 10}).encode()
    for a in range(8):
        try:
            j = json.load(urllib.request.urlopen(urllib.request.Request('https://api.openai.com/v1/chat/completions', body, {'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'}), timeout=60))
            t = j['choices'][0]['message']['content'].strip(); break
        except Exception as e:
            time.sleep(3 * (a + 1)); t = None
    return {'question_id': h['question_id'], 'question_type': r['question_type'], 'raw': t, 'label': ('yes' in t.lower()) if t else None}
with cf.ThreadPoolExecutor(12) as ex: res = list(ex.map(judge, hyps))
json.dump(res, open(OUT, 'w'), indent=1)
by = collections.defaultdict(list)
for x in res: by[x['question_type']].append(1 if x['label'] else 0)
print('failed', sum(x['label'] is None for x in res))
print('Accuracy', round(sum(sum(v) for v in by.values()) / len(res), 4))
for k, v in sorted(by.items()): print(f'  {k}: {sum(v)/len(v):.4f} ({len(v)})')
abs_ = [1 if x['label'] else 0 for x in res if '_abs' in x['question_id']]; print('  abstention subset:', round(sum(abs_)/len(abs_), 4), len(abs_))
print('  task average:', round(sum(sum(v)/len(v) for v in by.values()) / len(by), 4))
