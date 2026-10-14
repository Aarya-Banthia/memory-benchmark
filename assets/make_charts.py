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
 'light': dict(bg='#ffffff', ink='#0b0b0b', ink2='#52514e', muted='#6f6e69', grid='#e8e7e3', hi='#2a78d6', base='#8f8d86'),
 'dark':  dict(bg='#0d1117', ink='#f0efec', ink2='#c3c2b7', muted='#9a9990', grid='#262c34', hi='#3987e5', base='#75746e'),
}

def rbar(ax, y, w, h, color):
    # bar with 4px-ish rounded data end, square at baseline
    ax.barh(y, w, height=h, color=color, linewidth=0)

def hbar_chart(fname, theme, title, subtitle, rows, xmax, fmt, note, size=(10, 4.6), hi_label=None, ann=None, title_size=17, label_bold=True):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=size, dpi=200)
    fig.patch.set_facecolor(t['bg']); ax.set_facecolor(t['bg'])
    n = len(rows)
    for i, (lab, v, hi, sub) in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, v, height=0.56, color=t['hi'] if hi else t['base'], linewidth=0)
        ax.text(v + xmax * 0.012, y, fmt(v), va='center', ha='left', fontsize=13, color=t['ink'], fontweight='bold' if hi else 'medium')
        ax.text(-xmax * 0.02, y + (0.11 if sub else 0), lab, va='center', ha='right', fontsize=12, color=t['ink'], fontweight='bold' if (hi and label_bold) else 'regular')
        if sub: ax.text(-xmax * 0.02, y - 0.2, sub, va='center', ha='right', fontsize=9.5, color=t['muted'])
    ax.set_xlim(0, xmax); ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.tick_params(axis='x', colors=t['muted'], labelsize=10, length=0)
    for x in ax.get_xticks():
        if 0 < x <= xmax: ax.axvline(x, color=t['grid'], lw=1, zorder=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.axvline(0, color=t['muted'], lw=1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: ('0' if not v else (f'{int(v)}%' if xmax == 100 else f'{v:.1f}'))))
    if ann: ann(ax, t)
    fig.text(0.02, 0.955, title, fontsize=title_size, fontweight='bold', color=t['ink'], va='top')
    fig.text(0.02, 0.885, subtitle, fontsize=11, color=t['ink2'], va='top')
    fig.text(0.02, 0.03, note, fontsize=9, color=t['muted'], va='bottom')
    fig.subplots_adjust(left=0.36, right=0.95, top=0.76, bottom=0.2 if '\n' in note else 0.17)
    fig.savefig(OUT + fname, facecolor=t['bg'])
    plt.close(fig)

f2 = lambda v: f'{v:.2f}' if v not in (0.746,) else '0.746'
def beam_ann(ax, t):
    ax.annotate('', xy=(0.746, 0.42), xytext=(0.336, 0.42), arrowprops=dict(arrowstyle='<->', color=t['ink2'], lw=1.2))
    ax.text((0.746 + 0.336) / 2, 0.55, '2.2x', ha='center', va='bottom', fontsize=13, fontweight='bold', color=t['ink'])
beam_rows = [
    ('Long-context model', 0.259, False, 'whole conversation in the prompt'),
    ('RAG', 0.307, False, 'vector retrieval'),
    ('LIGHT', 0.336, False, 'best memory system in the paper'),
    ('Read-time search', 0.746, True, 'this repo: flash model + grep'),
]
for th in THEMES:
    hbar_chart(f'beam_vs_paper_{th}.png', th,
               'BEAM-1M: searching the raw history scores 2.2x the best published system',
               'Average score across 10 memory abilities on 1M-token conversations. Higher is better.',
               beam_rows, 1.0, f2,
               'Baselines: best result per method across 4 backbone models, BEAM paper (arXiv 2510.27246), Table 1.\nOurs: official BEAM scoring code and judge (GPT-4.1-mini), 699 of 700 questions, answered by GLM-5.3-flash.',
               ann=beam_ann, title_size=16)

# social card (light, 16:9)
hbar_chart('social_card.png', 'light',
           'AI memory without a vector database',
           'BEAM-1M (1M-token conversations): a cheap model that just searches the raw history vs the published systems',
           beam_rows, 1.0, f2,
           'Official scoring code and judge. Baselines from the BEAM paper, Table 1.   github.com/Aarya-Banthia/memory-benchmark',
           size=(12, 6.75), ann=beam_ann, title_size=22)

# abilities
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/results/'
by = collections.defaultdict(list)
for l in open(R + 'beam_1m/official_verdicts_gpt-4.1-mini.jsonl'):
    x = json.loads(l); by[x['ability']].append(x['score'])
names = {'instruction_following': 'Following standing instructions', 'preference_following': 'Following preferences', 'information_extraction': 'Recalling facts',
         'abstention': 'Saying "not in the history"', 'knowledge_update': 'Using the latest value', 'multi_session_reasoning': 'Reasoning across sessions',
         'contradiction_resolution': 'Flagging contradictions', 'summarization': 'Summarizing', 'temporal_reasoning': 'Date and time reasoning', 'event_ordering': 'Ordering events'}
subs = {}
names['event_ordering'] = 'Ordering events (strict metric)'
ab = sorted(by, key=lambda k: -sum(by[k]) / len(by[k]))
ab_rows = [(names[k], round(sum(by[k]) / len(by[k]), 3), True, subs.get(k)) for k in ab]
def ref_line(ax, t):
    ax.axvline(0.336, color=t['ink2'], lw=1.2, ls=(0, (1, 2)))
    ax.text(0.336, len(ab_rows) - 0.42, "paper's best overall (0.34)", ha='center', va='bottom', fontsize=9.5, color=t['ink2'])
for th in THEMES:
    hbar_chart(f'beam_by_ability_{th}.png', th,
               'Where read-time search is strong, and where it is not',
               'BEAM-1M score by memory ability, official scoring. Ordering long event sequences and date arithmetic are the gaps.',
               ab_rows, 1.0, lambda v: f'{v:.2f}',
               '70 questions per ability (69 for summarizing). Ordering is scored by Kendall tau on line-by-line alignment, so list format matters.\nDotted line: the best overall average in the BEAM paper, for scale.',
               size=(10, 6.4), ann=ref_line, label_bold=False)

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
