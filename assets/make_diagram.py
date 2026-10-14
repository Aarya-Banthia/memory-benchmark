"""Draws assets/why_{light,dark}.svg: write-time vs read-time memory, split at the moment a question arrives."""
import os
OUT = os.path.dirname(os.path.abspath(__file__)) + '/'
T = {
 'light': dict(bg='#ffffff', ink='#0b0b0b', ink2='#52514e', muted='#6f6e69', chip='#f4f3f0', line='#c9c7c0',
               w_acc='#eb6834', w_tint='#fdf0ea', r_acc='#2a78d6', r_tint='#eaf2fc', q='#0b0b0b'),
 'dark':  dict(bg='#0d1117', ink='#f0efec', ink2='#c3c2b7', muted='#9a9990', chip='#1b2028', line='#3a414b',
               w_acc='#d95926', w_tint='#2a1c16', r_acc='#3987e5', r_tint='#14243a', q='#f0efec'),
}
W, H = 1020, 490
CW, CH, GAP = 130, 70, 25
QX = 510
def chips_left(n):  # n chips ending just left of the question line
    end = QX - 30; start = end - (n * CW + (n - 1) * GAP)
    return [start + i * (CW + GAP) for i in range(n)]
def chips_right(n):
    start = QX + 30
    return [start + i * (CW + GAP) for i in range(n)]
def esc(s): return s.replace('&', '&amp;').replace('<', '&lt;')

def draw(th):
    t = T[th]; o = []
    o.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif">')
    o.append(f'<rect width="{W}" height="{H}" fill="{t["bg"]}"/>')
    o.append(f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{t["muted"]}"/></marker></defs>')
    # question line
    o.append(f'<line x1="{QX}" y1="52" x2="{QX}" y2="{H-20}" stroke="{t["q"]}" stroke-width="1.5" stroke-dasharray="5 5"/>')
    o.append(f'<text x="{QX}" y="30" text-anchor="middle" font-size="15" font-weight="700" fill="{t["ink"]}">a question arrives</text>')
    o.append(f'<text x="{QX-16}" y="48" text-anchor="end" font-size="12" fill="{t["muted"]}">before</text>')
    o.append(f'<text x="{QX+16}" y="48" text-anchor="start" font-size="12" fill="{t["muted"]}">after</text>')

    def row(y, label, sub, acc, tint, left, right, decide_side, decide_idx, callout):
        o.append(f'<text x="40" y="{y-14}" font-size="16" font-weight="700" fill="{t["ink"]}">{esc(label)}</text>')
        o.append(f'<text x="40" y="{y+4}" font-size="12" fill="{t["muted"]}">{esc(sub)}</text>')
        y += 22
        xs = [(x, c, 'L', i) for i, (x, c) in enumerate(zip(chips_left(len(left)), left))] + [(x, c, 'R', i) for i, (x, c) in enumerate(zip(chips_right(len(right)), right))]
        for k, (x, (title, s1, s2), side, i) in enumerate(xs):
            dec = (side == decide_side and i == decide_idx)
            o.append(f'<rect x="{x}" y="{y}" width="{CW}" height="{CH}" rx="10" fill="{tint if dec else t["chip"]}" stroke="{acc if dec else "none"}" stroke-width="2"/>')
            o.append(f'<text x="{x+12}" y="{y+25}" font-size="13.5" font-weight="700" fill="{t["ink"]}">{esc(title)}</text>')
            o.append(f'<text x="{x+12}" y="{y+43}" font-size="11" fill="{t["ink2"]}">{esc(s1)}</text>')
            if s2: o.append(f'<text x="{x+12}" y="{y+57}" font-size="11" fill="{t["ink2"]}">{esc(s2)}</text>')
            if k + 1 < len(xs):
                nx = xs[k + 1][0]
                o.append(f'<line x1="{x+CW+4}" y1="{y+CH/2}" x2="{nx-5}" y2="{y+CH/2}" stroke="{t["muted"]}" stroke-width="1.5" marker-end="url(#ah)"/>')
            if dec:
                cx = x + CW / 2
                o.append(f'<line x1="{cx}" y1="{y+CH+4}" x2="{cx}" y2="{y+CH+20}" stroke="{acc}" stroke-width="2"/>')
                cy = y + CH + 38
                anchor = 'start' if side == 'R' else 'middle'
                tx = x if side == 'R' else cx
                for j, ln in enumerate(callout):
                    o.append(f'<text x="{tx}" y="{cy + j*17}" text-anchor="{anchor}" font-size="{13 if j == 0 else 12}" font-weight="{700 if j == 0 else 400}" fill="{t["ink"] if j == 0 else t["ink2"]}">{esc(ln)}</text>')

    row(80, 'Write-time memory', 'how most memory layers work: extract facts, embed, retrieve', t['w_acc'], t['w_tint'],
        [('Conversation', 'every message', ''), ('Extract facts', 'an LLM guesses', 'what will matter'), ('Embed + store', 'only the facts', 'are kept')],
        [('Retrieve top-k', 'facts that look', 'similar'), ('Answer', 'from whatever', 'survived')],
        'L', 1, ['Relevance is judged here,', 'before any question exists.', 'Whatever it drops is gone.'])
    row(318, 'Read-time memory', 'this repo: keep everything, decide when asked', t['r_acc'], t['r_tint'],
        [('Conversation', 'every message', ''), ('Keep it raw', 'nothing is', 'thrown away')],
        [('Search, read', 'the model hunts', 'with the question'), ('Verify', 're-check against', 'the evidence'), ('Answer', 'grounded in the', 'original words')],
        'R', 0, ['Relevance is judged here,', 'with the question in hand.'])
    o.append('</svg>')
    open(OUT + f'why_{th}.svg', 'w').write('\n'.join(o))
for th in T: draw(th)
print('ok')
