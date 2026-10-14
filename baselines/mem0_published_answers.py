"""Converts mem0's own published BEAM-1M answers into this repo's answer format, so they can be scored with
the official BEAM grader exactly like ours.

Source: github.com/mem0ai/memory-benchmarks, results/platform/beam_1m_results.json (Apache-2.0).
Those answers come from mem0's managed platform with GPT-5 as the answering model (top_200 memories);
mem0 scored them with GPT-5 and its own rubric code (reported 64.1). We change nothing in the answers.

Usage:
  git clone --depth 1 https://github.com/mem0ai/memory-benchmarks third_party/memory-benchmarks
  python3 baselines/mem0_published_answers.py results/beam_1m/mem0_platform/mem0_platform_answers.jsonl
  python3 grading/beam_official.py results/beam_1m/mem0_platform/mem0_platform_answers.jsonl \\
          results/beam_1m/mem0_platform/mem0_platform_official_verdicts_gpt-4.1-mini.jsonl"""
import json, ast, os, sys
import pyarrow.parquet as pq
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = json.load(open(ROOT + '/third_party/memory-benchmarks/results/platform/beam_1m_results.json'))
idx = {}
for c in pq.read_table(os.environ.get('BEAM_DATA', ROOT + '/data/beam_1m.parquet')).to_pylist():
    for ab, qs in ast.literal_eval(c['probing_questions']).items():
        for qi, q in enumerate(qs): idx[(ab, q['question'].strip())] = f"{c['conversation_id']}|{ab}|{qi}"
with open(sys.argv[1], 'w') as f:
    for e in src['evaluations']:
        r = e['cutoff_results']['top_200']
        f.write(json.dumps({'key': idx[(e['question_type'], e['question'].strip())], 'ability': e['question_type'], 'question': e['question'],
                            'answer': r['generated_answer'], 'mem0_reported_score': r['score']}) + '\n')
print('wrote', len(src['evaluations']))
