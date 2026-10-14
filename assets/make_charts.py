import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch
import json, collections
import os
OUT = os.path.dirname(os.path.abspath(__file__)) + '/'
for f in fm.findSystemFonts():
    if '/inter/' in f.lower(): fm.fontManager.addfont(f)
plt.rcParams['font.family'] = 'Inter'
THEMES = {
 'light': dict(bg='#ffffff', ink='#0b0b0b', ink2='#52514e', muted='#6f6e69', grid='#e8e7e3', hi='#2a78d6', base='#8f8d86', alt='#eb6834'),
 'dark':  dict(bg='#0d1117', ink='#f0efec', ink2='#c3c2b7', muted='#9a9990', grid='#262c34', hi='#3987e5', base='#75746e', alt='#d95926'),
}

def hbar_chart(fname, theme, title, subtitle, rows, xmax, fmt, note, size=(10, 4.6), hi_label=None, ann=None, title_size=17, label_bold=True):
    """rows: (label, value, highlight, sublabel); value None makes the row a group header."""
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=size, dpi=200)
    fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
    n = len(rows); ys = {}
    for i, (lab, v, hi, sub) in enumerate(rows):
        y = n - 1 - i; ys[lab] = y
        if v is None:
            ax.text(-xmax * 0.02, y - 0.15, lab, va='center', ha='right', fontsize=10.5, color=t['ink2'], fontweight='bold')
            ax.plot([0, xmax], [y - 0.42, y - 0.42], color=t['grid'], lw=1, zorder=0, clip_on=False)
            continue
        ax.barh(y, v, height=0.56, color=t['hi'] if hi else t['base'], linewidth=0)
        ax.text(v + xmax * 0.012, y, fmt(v), va='center', ha='left', fontsize=13, color=t['ink'], fontweight='bold' if hi else 'medium')
        ax.text(-xmax * 0.02, y + (0.11 if sub else 0), lab, va='center', ha='right', fontsize=12, color=t['ink'], fontweight='bold' if (hi and label_bold) else 'regular')
        if sub: ax.text(-xmax * 0.02, y - 0.2, sub, va='center', ha='right', fontsize=9.5, color=t['muted'])
    ax.set_xlim(0, xmax); ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.tick_params(axis='x', colors=t['muted'], labelsize=10, length=0)
    for x in ax.get_xticks():
        if 0 < x <= xmax: ax.axvline(x, color=t['grid'], lw=1, zorder=0)
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.axvline(0, color=t['muted'], lw=1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: ('0' if not v else (f'{int(v)}%' if xmax == 100 else f'{v:.1f}'))))
    if ann: ann(ax, t, ys)
    fig.text(0.02, 0.955, title, fontsize=title_size, fontweight='bold', color=t['ink'], va='top')
    fig.text(0.02, 0.885, subtitle, fontsize=11, color=t['ink2'], va='top')
    fig.text(0.02, 0.03, note, fontsize=9, color=t['muted'], va='bottom')
    fig.subplots_adjust(left=0.36, right=0.95, top=0.78, bottom=0.2 if '\n' in note else 0.17)
    fig.savefig(OUT + fname, facecolor=t['bg'])
    plt.close(fig)

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/results/'
def ability_means(path, keys=None):
    by = collections.defaultdict(list)
    for l in open(path):
        x = json.loads(l)
        if keys is None or x['key'] in keys: by[x['ability']].append(x['score'])
    return {a: sum(v) / len(v) for a, v in by.items()}
ours_keys = {json.loads(l)['key'] for l in open(R + 'beam_1m/official_verdicts_gpt-4.1-mini.jsonl')}
OURS = ability_means(R + 'beam_1m/official_verdicts_gpt-4.1-mini.jsonl')
RAG5 = ability_means(R + 'beam_1m/rag_same_model/rag_top5_official_verdicts_gpt-4.1-mini.jsonl', ours_keys)
RAG32 = ability_means(R + 'beam_1m/rag_same_model/rag_full32k_official_verdicts_gpt-4.1-mini.jsonl', ours_keys)
MEM0 = ability_means(R + 'beam_1m/mem0_platform/mem0_platform_official_verdicts_gpt-4.1-mini.jsonl', ours_keys)
mean = lambda d: sum(d.values()) / len(d)
o, r5, r32, m0 = round(mean(OURS), 3), round(mean(RAG5), 3), round(mean(RAG32), 3), round(mean(MEM0), 3)

f2 = lambda v: f'{v:.3f}' if v in (o, m0) else f'{v:.2f}'
def beam_ann(ax, t, ys):
    y = ys['Read-time search']
    ax.text(o + 0.012, y - 0.42, f'+{o - m0:.2f} over mem0', va='center', ha='left', fontsize=10.5, color=t['ink2'])
beam_rows = [
    ('BEAM paper, older models', None, False, None),
    ('Long-context model', 0.259, False, 'whole conversation in the prompt'),
    ('RAG', 0.307, False, 'vector retrieval, top 5'),
    ('LIGHT', 0.336, False, "the paper's best memory system"),
    ('Write-time memory, same official grader', None, False, None),
    ('mem0 (managed platform)', m0, False, 'its published answers, GPT-5 answering'),
    ('Same model: GLM-5.3-flash', None, False, None),
    ('RAG, paper setup', r5, False, 'top 5 turn pairs'),
    ('RAG, full 32K context', r32, False, 'about 25 turn pairs'),
    ('Read-time search', o, True, 'this repo: search the raw history'),
]
NOTE = ("Paper rows: best result per method across 4 backbone models, BEAM paper (arXiv 2510.27246), Table 1. mem0: its own published answers\n"
        "(github.com/mem0ai/memory-benchmarks). Same-model rows: GLM-5.3-flash. All non-paper rows scored with the official BEAM code and judge, same 699 questions.")
for th in THEMES:
    hbar_chart(f'beam_vs_paper_{th}.png', th,
               'BEAM-1M: a cheap model searching raw history beats mem0 and RAG',
               'Average score across 10 memory abilities on 1M-token conversations. Higher is better.',
               beam_rows, 1.0, f2, NOTE, size=(10, 7.0), ann=beam_ann, title_size=16)
hbar_chart('social_card.png', 'light',
           "Relevance can't be judged at write time",
           'BEAM-1M, 1M-token conversations, official grader. Read-time search vs mem0 and RAG.',
           beam_rows, 1.0, f2,
           'Official BEAM scoring code and judge. mem0: its own published answers. Paper rows: BEAM paper, Table 1.   github.com/Aarya-Banthia/memory-benchmark',
           size=(12, 8.0), ann=beam_ann, title_size=24)

# abilities: paired bars, read-time vs best same-model RAG
names = {'instruction_following': 'Following standing instructions', 'preference_following': 'Following preferences', 'information_extraction': 'Recalling facts',
         'abstention': 'Saying "not in the history"', 'knowledge_update': 'Using the latest value', 'multi_session_reasoning': 'Reasoning across sessions',
         'contradiction_resolution': 'Flagging contradictions', 'summarization': 'Summarizing', 'temporal_reasoning': 'Date and time reasoning', 'event_ordering': 'Ordering events (strict metric)'}
ab = sorted(OURS, key=lambda k: -(OURS[k] - MEM0[k]))
for th in THEMES:
    t = THEMES[th]
    fig, ax = plt.subplots(figsize=(10, 8.4), dpi=200)
    fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
    n = len(ab); h = 0.25
    series = [(OURS, t['hi'], True), (MEM0, t['alt'], False), (RAG32, t['base'], False)]
    for i, k in enumerate(ab):
        y = n - 1 - i
        for j, (S, col, bold) in enumerate(series):
            yy = y + h + 0.03 - j * (h + 0.03)
            ax.barh(yy, S[k], height=h, color=col, linewidth=0)
            ax.text(S[k] + 0.01, yy, f'{S[k]:.2f}', va='center', fontsize=9.5, color=t['ink'] if bold else t['ink2'], fontweight='bold' if bold else 'regular')
        ax.text(-0.02, y, names[k], va='center', ha='right', fontsize=11.5, color=t['ink'])
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, n - 0.3); ax.set_yticks([])
    ax.tick_params(axis='x', colors=t['muted'], labelsize=10, length=0)
    for x in ax.get_xticks():
        if 0 < x <= 1: ax.axvline(x, color=t['grid'], lw=1, zorder=0)
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.axvline(0, color=t['muted'], lw=1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: '0' if not v else f'{v:.1f}'))
    fig.text(0.02, 0.96, 'The biggest gains are where facts change or conflict', fontsize=16, fontweight='bold', color=t['ink'], va='top')
    fig.text(0.02, 0.913, 'BEAM-1M by memory ability, official grader, same 699 questions. Sorted by the gap to mem0.', fontsize=11, color=t['ink2'], va='top')
    lx = 0.30
    for col, lab in ((t['hi'], 'Read-time search (GLM-5.3-flash)'), (t['alt'], 'mem0 platform (GPT-5)'), (t['base'], 'RAG, full 32K (GLM-5.3-flash)')):
        fig.patches.append(plt.Rectangle((lx, 0.857), 0.012, 0.018, transform=fig.transFigure, color=col, figure=fig))
        fig.text(lx + 0.017, 0.866, lab, fontsize=9.5, color=t['ink'], va='center'); lx += 0.235 if 'Read' in lab else 0.195
    fig.text(0.02, 0.025, "70 questions per ability (69 for summarizing). mem0: its own published answers. Official BEAM scoring code and judge.\nOrdering is scored by Kendall tau on line-by-line alignment, so list format matters.", fontsize=9, color=t['muted'], va='bottom')
    fig.subplots_adjust(left=0.36, right=0.95, top=0.83, bottom=0.11)
    fig.savefig(OUT + f'beam_by_ability_{th}.png', facecolor=t['bg']); plt.close(fig)
print('BEAM overall', o, 'rag5', r5, 'rag32', r32)

# LongMemEval
lme_rows = [
    ('Full context, GPT-4o', 60.2, False, 'LongMemEval paper'),
    ('Zep', 71.2, False, 'self-reported, own setup'),
    ('Supermemory', 85.2, False, 'self-reported, own setup'),
    ('mem0', 94.4, False, 'self-reported, own setup'),
    ('Read-time search', 94.6, True, 'this repo: flash model + grep'),
]
for th in THEMES:
    hbar_chart(f'longmemeval_{th}.png', th,
               'LongMemEval-S: on par with the best vendor-reported results',
               'Accuracy on 500 questions over long chat histories. Higher is better.',
               lme_rows, 100, lambda v: f'{v:.1f}%' if v else '0',
               'Ours: official LongMemEval answer check with GPT-4o. Vendor figures come from their own models and harnesses and are not a controlled comparison.',
               size=(10, 5.0))
print('ok')
