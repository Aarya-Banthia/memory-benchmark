"""Recompute the headline tables from the verdict files in results/. No API calls."""
import json, collections, os
R = os.path.dirname(os.path.abspath(__file__)) + '/results/'
mean = lambda v: sum(v) / len(v)

lme = json.load(open(R + 'longmemeval_s/official_verdicts_gpt-4o-2024-08-06.json'))
by = collections.defaultdict(list)
for x in lme: by[x['question_type']].append(1 if x['label'] else 0)
print(f"LongMemEval-S, official GPT-4o judge: {mean([1 if x['label'] else 0 for x in lme]):.1%} overall ({len(lme)} questions), task average {mean([mean(v) for v in by.values()]):.1%}")
for k, v in sorted(by.items()): print(f'  {k:28s} {mean(v):6.1%}  n={len(v)}')
ab = [1 if x['label'] else 0 for x in lme if '_abs' in x['question_id']]
print(f'  {"abstention (subset)":28s} {mean(ab):6.1%}  n={len(ab)}')

beam = [json.loads(l) for l in open(R + 'beam_1m/official_verdicts_gpt-4.1-mini.jsonl')]
by = collections.defaultdict(list)
for x in beam: by[x['ability']].append(x['score'])
print(f"\nBEAM-1M, official scoring (GPT-4.1-mini judge; event ordering = Kendall tau_norm): {mean([mean(v) for v in by.values()]):.3f} ({len(beam)} questions)")
for k, v in sorted(by.items(), key=lambda kv: -mean(kv[1])): print(f'  {k:28s} {mean(v):.3f}  n={len(v)}')

rows = [json.loads(l) for l in open(R + 'beam_1m/answers.jsonl')] + [json.loads(l) for l in open(R + 'longmemeval_s/answers.jsonl')]
print(f"\nAnswering model tokens (both benchmarks): {sum(r['prompt_tokens'] for r in rows):,} prompt ({sum(r['cached_tokens'] for r in rows):,} cached), {sum(r['completion_tokens'] for r in rows):,} completion")
