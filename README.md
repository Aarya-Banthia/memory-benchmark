# Read-time memory: a cheap model that searches raw history

Most AI memory layers work at **write time**. As a conversation happens, they extract "facts", embed them and store them in a vector database, and later they retrieve those facts. Anything the extractor dropped is gone.

This repo tests the opposite approach, **read-time memory**:

- Keep the raw history exactly as it was.
- Decide nothing until a question arrives.
- Then let a small, cheap model search the record itself with plain tools (regex search, read a range, count), the way you would grep a codebase.
- A final verify step re-checks the answer against everything that was retrieved.

There is no extraction, no embeddings and no vector database.

## Results

Both benchmarks were scored with their **official scoring code and official judge models**.

| Benchmark | Official judge | Score | Reference points |
| --- | --- | --- | --- |
| [BEAM](https://github.com/mohammadtavakoli78/BEAM), 1M-token conversations (699 of 700 questions) | GPT-4.1-mini, `src/evaluation/compute_metrics.py`; event ordering scored by Kendall tau as in `report_results.py` | **0.746** | BEAM paper at 1M: long-context models 0.19 to 0.26, RAG 0.27 to 0.31, LIGHT (paper's best) 0.28 to 0.34 |
| [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) (500 questions) | GPT-4o-2024-08-06, `src/evaluation/evaluate_qa.py` | **94.6%** | Paper: full-context GPT-4o 60.2%. Vendors self-report, on their own setups: Zep 71.2%, mem0 94.4% |

- **Answering model:** GLM-5.3-flash (`zai/glm-5.3-flash`), a small, cheap model, called through an OpenAI-compatible router.
- **Cost:** about $0.003 per question on LongMemEval and about $0.01 per question on BEAM-1M.

### BEAM-1M by ability (official scoring)

| Ability | Score |
| --- | --- |
| instruction_following | 0.921 |
| preference_following | 0.913 |
| information_extraction | 0.905 |
| abstention | 0.857 |
| knowledge_update | 0.764 |
| multi_session_reasoning | 0.756 |
| contradiction_resolution | 0.743 |
| summarization | 0.709 |
| temporal_reasoning | 0.552 |
| event_ordering (Kendall tau, normalised) | 0.340 |
| **Average of the 10 abilities** | **0.746** |

### LongMemEval-S by question type (official scoring)

| Type | n | Accuracy |
| --- | --- | --- |
| single-session-user | 70 | 98.6% |
| single-session-assistant | 56 | 100% |
| single-session-preference | 30 | 93.3% |
| knowledge-update | 78 | 96.2% |
| temporal-reasoning | 133 | 94.7% |
| multi-session | 133 | 89.5% |
| abstention (subset of the above) | 30 | 93.3% |
| **Overall** | 500 | **94.6%** (task average 95.4%) |

Run `python3 summarize.py` to recompute every number above from the verdict files. It makes no API calls.

## Honest caveats

- **Single run.** All results come from one run with one answering model, with no seeds and no repeats.
- **Vendor figures are self-reported** on their own models and harnesses, so this is not a controlled head-to-head with any memory product. The BEAM paper baselines do share the same scoring code and judge model as ours.
- **Event ordering is the weak spot.** Our answers score 0.75 on BEAM's rubric judge for this ability, but 0.34 on the official Kendall-tau metric. That metric splits the answer on newlines and aligns it line by line with the reference, so formatting matters. The F1 of that alignment is 0.16.
- **Temporal reasoning (0.55 on BEAM)** is the other clear gap.
- **Read-time search is slower** than a single vector lookup: about 3 model steps per question on LongMemEval and about 7 on BEAM-1M.
- **One BEAM question has no answer** because of a network stall during the run.
- **The verify step was fixed before the final run.** An earlier version truncated the retrieved evidence before re-checking, which wrongly overturned correct answers. Both benchmarks were fully re-run after the fix, and only the fixed run is reported here.
- **Cross-check verdicts.** We also graded every answer with Claude as an independent judge, before the official re-grade: 93.4% on LongMemEval and 0.73 on BEAM. Those verdicts are included as cross-checks, not headline numbers.

## What's in here

```
harness/
  lme_harness.py        read-time search harness for LongMemEval-S
  beam_harness.py       read-time search harness for BEAM-1M
grading/
  lme_official.py       official LongMemEval answer check (prompts loaded from the official repo), GPT-4o
  beam_official.py      official BEAM metric functions and judge prompt, executed unchanged, GPT-4.1-mini
results/
  longmemeval_s/answers.jsonl                              every answer, with steps, tokens and sessions searched
  longmemeval_s/official_verdicts_gpt-4o-2024-08-06.json   every official verdict
  longmemeval_s/crosscheck_verdicts_claude.json
  beam_1m/answers.jsonl                                    every answer, with retrieved and gold message ids
  beam_1m/official_verdicts_gpt-4.1-mini.jsonl             every rubric verdict with the judge's reasons
  beam_1m/crosscheck_verdicts_claude.json
summarize.py            recomputes all tables from results/
```

## How the harness works

1. **Overview.** The model gets the question and a short overview of the history: sessions or blocks, dates and opening lines. It does not get the history itself.
2. **Search.** It calls tools:
   - `search` (case-insensitive regex, optionally filtered by role or by session or block range)
   - `read` (full text of a range of turns)
   - `count` (matches grouped by session)
3. **Iterate.** It searches with several phrasings, reads around promising hits, and repeats until it is confident, then calls `answer`.
4. **Verify.** A second call sees the question, the candidate answer and all retrieved text. It can correct the answer, for example to use a later updated value, add a missed item or fix date arithmetic, but it may not discard a specific answer just because its support is incomplete.

## Reproduce

```bash
# data
mkdir -p data third_party
curl -L -o data/longmemeval_s.json https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
curl -L -o data/beam_1m.parquet https://huggingface.co/datasets/Mohammadta/BEAM/resolve/main/data/1M-00000-of-00001.parquet
git clone --depth 1 https://github.com/xiaowu0162/LongMemEval third_party/LongMemEval
git clone --depth 1 https://github.com/mohammadtavakoli78/BEAM third_party/BEAM
pip install pyarrow scipy json_repair

# answer (any OpenAI-compatible endpoint; defaults shown)
export LLM_API_KEY=...  LLM_BASE_URL=https://router.requesty.ai/v1  LLM_MODEL=zai/glm-5.3-flash
python3 harness/lme_harness.py results/longmemeval_s/answers.jsonl
python3 harness/beam_harness.py results/beam_1m/answers.jsonl

# official grading
export OPENAI_API_KEY=...
python3 grading/lme_official.py results/longmemeval_s/answers.jsonl results/longmemeval_s/official_verdicts_gpt-4o-2024-08-06.json
python3 grading/beam_official.py
python3 summarize.py
```

The harness scripts append to their output file and skip questions already answered, so an interrupted run can be resumed. Delete the output file to start fresh.

## License

MIT for the code in this repo. The benchmark data and official evaluation code belong to their authors. See the LongMemEval and BEAM repositories for their licenses.
