# -*- coding: utf-8 -*-
"""Supplementary figures/tables from archived JSONs (2026-11-23).
Paths are relative to this file's directory (repo root = dissertation/).
Reads results/results_all34.json + results/phase_metrics.json +
results/demo_single/*.mp4 (rendered single-panel by pipeline/render_demo.py).
Writes the generated figures into BOTH paper/figs/ and paper_cn/figs/.
8-run component numbers for head-50/PF-oracle/mask are the archived values
published in Tables 1 and 3 of the paper (per-arm medians in mm).
Usage:  python _gen_figs2.py"""
import json, os, statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(BASE, 'results') + os.sep
FIGS = [os.path.join(BASE, 'paper', 'figs'), os.path.join(BASE, 'paper_cn', 'figs')]

def save_fig(fig, name):
    for d in FIGS:
        fig.savefig(os.path.join(d, name), dpi=200, bbox_inches='tight')
a34 = json.load(open(R + 'results_all34.json', encoding='utf-8'))
pm = json.load(open(R + 'phase_metrics.json', encoding='utf-8'))

UNSEEN = {'000020'} | {'%06d' % i for i in range(25, 34)}

# ---------- collect per-run medians ----------
runs = []  # (label, pack, arm, median, unseen)
for pk in sorted(a34['packs']):
    for arm in ('PSM1', 'PSM3'):
        m = a34['packs'][pk][arm]['err3d_mm']['median']
        runs.append((f'{pk} {arm}', m, pk in UNSEEN))
meds = [r[1] for r in runs]
un = [r[1] for r in runs if r[2]]; se = [r[1] for r in runs if not r[2]]
print('F8/T-A detector full corpus: n=%d median=%.2f mean=%.2f' % (len(meds), st.median(meds), st.mean(meds)))
print('  unseen n=%d median=%.2f mean=%.2f | seen n=%d median=%.2f mean=%.2f' %
      (len(un), st.median(un), st.mean(un), len(se), st.median(se), st.mean(se)))
comp = ['000000', '000001', '000020', '000021']
det8 = [a34['packs'][p][a]['err3d_mm']['median'] for p in comp for a in ('PSM1', 'PSM3')]
print('  detector subset (same 8 runs): median=%.2f mean=%.2f' % (st.median(det8), st.mean(det8)))

# ---------- F8 per-run bars ----------
fig, ax = plt.subplots(figsize=(10.5, 3.2))
x = range(len(runs))
ax.bar(x, [r[1] for r in runs], color=['#c44e52' if r[2] else '#4c72b0' for r in runs], width=0.8)
ax.axhline(10, color='k', lw=1, ls='--')
ax.text(len(runs) - 0.5, 10.35, '10 mm criterion', ha='right', fontsize=9)
ax.set_xticks(x[::4]); ax.set_xticklabels([runs[i][0].split()[0] for i in range(0, len(runs), 4)], fontsize=8)
ax.set_ylabel('median 3D error (mm)'); ax.set_ylim(0, 10.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#4c72b0', label='seen in detector training'),
                   Patch(color='#c44e52', label='unseen')], fontsize=8, loc='upper left', ncol=2)
fig.tight_layout(); save_fig(fig, 'fig5_8_perrun.png'); plt.close(fig)

# ---------- phase data ----------
arms = pm['arms']
phases = ['approach', 'contact', 'manipulation', 'retract']
tip = {p: [a['phase_stats']['tip_err_by_phase_mm'][p] for a in arms if p in a['phase_stats'].get('tip_err_by_phase_mm', {})] for p in phases}
agr = {p: [a['phase_stats']['gt_label_agreement_pct'][p] for a in arms if p in a['phase_stats'].get('gt_label_agreement_pct', {})] for p in phases}
pct = {p: [a['phase_stats']['pct_time'][p] for a in arms] for p in phases + ['other']}
print('T-B per phase: n=%d' % len(arms))
for p in phases:
    print('  %-12s tip_median=%.2f tip_iqr=[%.2f,%.2f] agree_median=%.1f time_mean=%.1f sd=%.1f' % (
        p, st.median(tip[p]), st.quantiles(tip[p], n=4)[0], st.quantiles(tip[p], n=4)[2],
        st.median(agr[p]), st.mean(pct[p]), st.stdev(pct[p])))
print('  other        time_mean=%.1f sd=%.1f' % (st.mean(pct['other']), st.stdev(pct['other'])))

# ---------- F9 stacked time share ----------
fig, ax = plt.subplots(figsize=(10.5, 3.0))
bottom = [0.0] * len(arms); colors = {'approach': '#4c72b0', 'contact': '#55a868',
                                      'manipulation': '#c44e52', 'retract': '#8172b2', 'other': '#937860'}
for p in phases + ['other']:
    ax.bar(range(len(arms)), pct[p], bottom=bottom, color=colors[p], width=0.85,
           label=p.capitalize() if p != 'other' else 'Unclassified')
    bottom = [b + v for b, v in zip(bottom, pct[p])]
ax.set_xticks(range(0, len(arms), 4))
ax.set_xticklabels(['%s %s' % (a['pack'], a['arm']) for a in arms][0::4], fontsize=7, rotation=45, ha='right')
ax.set_ylabel('share of time (%)'); ax.set_ylim(0, 100)
ax.legend(fontsize=8, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.18))
fig.tight_layout(); save_fig(fig, 'fig5_9_timeshare.png'); plt.close(fig)

# ---------- F10 phase tip error boxes ----------
fig, ax = plt.subplots(figsize=(5.6, 3.2))
data = [tip[p] for p in phases]
bp = ax.boxplot(data, tick_labels=[p.capitalize() for p in phases], showfliers=True,
                flierprops=dict(ms=2, mfc='none'))
ax.axhline(10, color='k', lw=1, ls='--')
ax.set_ylabel('tracked tip error per phase (mm)')
fig.tight_layout(); save_fig(fig, 'fig5_10_phaseerr.png'); plt.close(fig)

# ---------- F11 observation-source comparison (same 8 runs) ----------
head50 = [8.1, 16.4, 16.7, 21.4, 18.5, 19.6, 7.8, 21.3]
pf_gt = [2.0, 1.8, 1.8, 2.0, 3.9, 3.2, 2.2, 2.7]
m12 = [55.3, 78.5, 62.0, 75.9, 68.5, 82.1, 84.1, 88.7]
m60 = [35.9, 53.9, 46.0, 31.0, 30.8, 60.5, 48.2, 47.3]
srcs = [('head-50\n(no observation)', head50), ('PF, ground\ntruth keypoints', pf_gt),
        ('PF, mask\n$\\sigma_{px}{=}12$', m12), ('PF, mask\n$\\sigma_{px}{=}60$', m60),
        ('PF, in-domain\ndetector', det8)]
print('T-A means:', [(s.replace(chr(10), ' '), round(st.mean(v), 2), round(st.median(v), 2)) for s, v in srcs])
fig, ax = plt.subplots(figsize=(6.4, 3.2))
xs = range(len(srcs))
ax.bar(xs, [st.mean(v) for _, v in srcs], color=['#937860', '#4c72b0', '#c44e52', '#dd8452', '#55a868'], width=0.62)
for x, (_, v) in zip(xs, srcs):
    ax.plot([x] * len(v), v, 'k.', ms=4, alpha=0.55)
ax.axhline(10, color='k', lw=1, ls='--')
ax.set_xticks(list(xs)); ax.set_xticklabels([s for s, _ in srcs], fontsize=8)
ax.set_ylabel('median 3D error, 8 runs (mm)')
fig.tight_layout(); save_fig(fig, 'fig5_11_obssource.png'); plt.close(fig)

# ---------- T-C sensitivity grid ----------
sens = pm['sensitivity']
print('T-C grid (F, Tmin | R=1.0 / 1.5 / 2.0 -> episodes, manip median s):')
grid = {}
for s in sens:
    grid[(s['f_low'], s['t_min_s'], s['r_mult'])] = (s['episodes'], s['manip_s_median'])
for f in (0.15, 0.25, 0.40):
    for t in (0.3, 0.5, 1.0):
        row = [grid[(f, t, r)] for r in (1.0, 1.5, 2.0)]
        print('  F=%.2f Tmin=%.1f | %s' % (f, t, ' | '.join('%d, %.2f' % rv for rv in row)))
print('DONE')

# ---------- F12 runtime screenshot grid ----------
import subprocess, os
from PIL import Image, ImageDraw, ImageFont
frames_pick = [0, 250, 500, 750]
vids = [('000021', 'PSM1', 'demo_single/overlay_000021_PSM1.mp4'),
        ('000027', 'PSM1', 'demo_single/overlay_000027_PSM1.mp4')]
font = None
for fp in ('C:/Windows/Fonts/arial.ttf',
           os.path.join(os.path.dirname(matplotlib.get_data_path()),
                        'fonts', 'ttf', 'DejaVuSans.ttf'),
           os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans.ttf')):
    if os.path.exists(fp):
        font = ImageFont.truetype(fp, 30)
        break
if font is None:
    font = ImageFont.load_default()
panels = [[None, None] for _ in frames_pick]
for vi, (rec, arm, mp4) in enumerate(vids):
    for fi, f in enumerate(frames_pick):
        out = os.path.join(FIGS[0], f'_rt_{rec}_{f}.png')
        if not os.path.exists(out):
            subprocess.run(['ffmpeg', '-v', 'error', '-i', R + mp4,
                            '-vf', f"select='eq(n\,{f})'", '-vsync', '0',
                            '-frames:v', '1', out], check=True)
        im = Image.open(out)
        strip = Image.new('RGB', (im.width, 40), 'white')
        d = ImageDraw.Draw(strip)
        lbl = '%s %s  frame %d' % (rec, arm, f)
        d.text((10, 8), lbl, fill='black', font=font)
        cell = Image.new('RGB', (im.width, im.height + 40), 'white')
        cell.paste(strip, (0, 0)); cell.paste(im, (0, 40))
        panels[fi][vi] = cell
W = 2 * panels[0][0].width + 3 * 8
H = len(frames_pick) * panels[0][0].height + (len(frames_pick) + 1) * 8
grid = Image.new('RGB', (W, H), 'white')
for fi in range(len(frames_pick)):
    y = 8 + fi * (panels[0][0].height + 8)
    for vi in range(2):
        grid.paste(panels[fi][vi], (8 + vi * (panels[0][0].width + 8), y))
for d in FIGS: grid.save(os.path.join(d, 'fig5_12_runtime.png'))
for rec, arm, _ in vids:
    for f in frames_pick:
        os.remove(os.path.join(FIGS[0], f'_rt_{rec}_{f}.png'))
print('F12 runtime grid saved', grid.size)
