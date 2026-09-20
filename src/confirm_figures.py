"""Figures 0-2 from PART II DATA ONLY: last layer, fresh seeds '+','.join(map(str,SEEDS))+'."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sys; sys.path.insert(0,'/Users/alityb/projects/cubemoe/src')
from analyze import cp_test
from extract import extract_seq
from model import device_auto
ROOT='/Users/alityb/projects/cubemoe'
SURF='#fcfcfb'; INK='#0b0b0b'; INK2='#52514e'; GRID='#e3e2df'
C1,C2,C3,C4='#2a78d6','#eb6834','#1baf7a','#eda100'
import os as _os
SEEDS=[int(x) for x in _os.environ.get('FIG_SEEDS','10,11,12,13,14').split(',')]
CONF=_os.environ.get('FIG_CONF','confirm')
DATA=_os.environ.get('FIG_DATA','mixed')
def style(ax):
    ax.set_facecolor(SURF)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    for s in ('left','bottom'): ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(1)
    ax.grid(True,axis='y',color=GRID,lw=.8); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2,labelsize=9,length=0)
def L(t): return json.load(open(f'{ROOT}/out/{CONF}/{t}.json'))

# fig0 — dataset property: how much the step index alone gives away
fig,ax=plt.subplots(figsize=(7.2,4.0),facecolor=SURF); style(ax)
for k,col,lab in [('naive',C2,'stock Kociemba'),(DATA,C1,'mixed (250k scale set)' if DATA!='mixed' else 'mixed (confirmatory set)')]:
    d=np.load(f'{ROOT}/data/{k}.npz'); ps,ph=d['pos'],d['phase']
    xs,ys=[],[]
    for p in np.unique(ps):
        m=ps==p
        if m.sum()<20: continue
        xs.append(int(p)); ys.append(float((ph[m]==2).mean()))
    ax.plot(xs,ys,lw=2,color=col,label=lab,solid_capstyle='round')
ax.axhspan(0.05,0.95,color='#f2f1ee',zorder=0)
ax.set_xlabel('step index within solution',color=INK2,fontsize=10)
ax.set_ylabel('P(phase 2 | step index)',color=INK2,fontsize=10)
ax.set_title('Fig 0 — the position confound (dataset property)',color=INK,fontsize=11.5,loc='left',pad=26)
ax.text(0,1.02,'shaded = steps where BOTH phases occur, the only rows a conditional analysis can use',
        transform=ax.transAxes,fontsize=8.5,color=INK2,va='bottom')
ax.legend(frameon=False,fontsize=9,labelcolor=INK2,loc='lower right')
fig.tight_layout(); fig.savefig(f'{ROOT}/out/fig0_overlap.png',dpi=160,facecolor=SURF); plt.close(fig)

# fig1 — router vs nulls, last layer, per seed, both arms
fig,axes=plt.subplots(1,2,figsize=(11.5,4.2),facecolor=SURF)
for ax,(base,arm,hasH) in zip(axes,[('moe_mixed_s','Arm A (sequence)',True),('state_mixed_s','Arm C (position-blind)',False)]):
    style(ax); x=np.arange(len(SEEDS))
    R=[L(f'{base}{s}') for s in SEEDS]
    series=[('router',C1,[r['cond_nmi'] for r in R]),
            ('shuffled null',C3,[r['shuffled'] for r in R]),
            ('position-only router',C2,[r['position_only'] for r in R])]
    if hasH: series.append(('hash twin',C4,[r['hash_twin'] for r in R]))
    n=len(series); w=.8/n
    for j,(lab,col,v) in enumerate(series):
        b=ax.bar(x+(j-(n-1)/2)*w,v,w*.88,color=col,label=lab,zorder=3)
        for rr,vv in zip(b,v): ax.text(rr.get_x()+rr.get_width()/2,vv,f'{vv:.3f}',ha='center',va='bottom',fontsize=6.5,color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([f'seed {s}' for s in SEEDS])
    ax.set_ylabel('conditional NMI (last layer)',color=INK2,fontsize=10)
    ax.set_title(arm,color=INK,fontsize=10.5,loc='left',pad=8)
fig.suptitle(f'Fig 1 — router vs pre-registered nulls, last layer, seeds {SEEDS[0]}-{SEEDS[-1]} (6L/d256, 250k)',
             color=INK,fontsize=12,x=.008,ha='left')
h,l=axes[0].get_legend_handles_labels()
fig.legend(h,l,loc='upper right',bbox_to_anchor=(0.995,0.975),ncol=4,frameon=False,fontsize=9,labelcolor=INK2)
fig.tight_layout(rect=[0,0,1,.90]); fig.savefig(f'{ROOT}/out/fig1_conditional_nmi.png',dpi=160,facecolor=SURF); plt.close(fig)

# fig2 — change-point: scatter (arm A seed 10) + per-seed b1 both arms
dev=device_auto()
ex=extract_seq(f'moe_mixed_s{SEEDS[0]}',DATA,dev,1200); Lx=ex['NL']-1
cp=cp_test(ex['e1'][Lx],ex['solve'],ex['seam'],ex['sollen'],nperm=200)
fig,axes=plt.subplots(1,2,figsize=(11.5,4.4),facecolor=SURF)
ax=axes[0]; style(ax)
xs=np.array(cp['_xs']); ys=np.array(cp['_ys']); r=np.random.RandomState(0)
ax.scatter(xs+r.uniform(-.3,.3,len(xs)),ys+r.uniform(-.3,.3,len(ys)),s=9,color=C1,alpha=.25,linewidths=0,zorder=3)
lo,hi=xs.min()-1,xs.max()+1
a,b=np.polyfit(xs,ys,1); ax.plot([lo,hi],[a*lo+b,a*hi+b],color=INK,lw=2,zorder=4)
ax.set_xlabel('true seam index (G1 entry)',color=INK2,fontsize=10)
ax.set_ylabel('router change-point index',color=INK2,fontsize=10)
ax.set_title(f'Arm A seed {SEEDS[0]}, last layer',color=INK,fontsize=10.5,loc='left',pad=8)
ax.text(.03,.95,f"b1 (length-controlled) {cp['partial_slope']:+.3f}\np {cp['partial_p']:.3f}   n {cp['n']}",
        transform=ax.transAxes,va='top',fontsize=9,color=INK,bbox=dict(fc=SURF,ec=GRID,boxstyle='round,pad=0.4'))
ax=axes[1]; style(ax); x=np.arange(len(SEEDS)); w=.38
for j,(base,lab,col) in enumerate([('moe_mixed_s','Arm A',C1),('state_mixed_s','Arm C',C2)]):
    v=[L(f'{base}{s}')['b1'] for s in SEEDS]; p=[L(f'{base}{s}')['b1_p'] for s in SEEDS]
    bb=ax.bar(x+(j-.5)*w,v,w*.88,color=col,label=lab,zorder=3)
    for rr,vv,pp in zip(bb,v,p):
        ax.text(rr.get_x()+rr.get_width()/2,vv,('*' if pp<0.01 else '')+f'{vv:.2f}',ha='center',
                va='bottom' if vv>=0 else 'top',fontsize=7,color=INK2)
ax.axhline(0,color=GRID,lw=1)
ax.set_xticks(x); ax.set_xticklabels([f'seed {s}' for s in SEEDS])
ax.set_ylabel('b1 (seam coeff., length-controlled)',color=INK2,fontsize=10)
ax.set_title('per-seed b1  (* = p < 0.01)',color=INK,fontsize=10.5,loc='left',pad=8)
ax.legend(frameon=False,fontsize=9,labelcolor=INK2)
fig.suptitle('Fig 2 — CONFIRMATORY: does the router move its switch point when the seam moves?',
             color=INK,fontsize=12,x=.008,ha='left')
fig.tight_layout(rect=[0,0,1,.93]); fig.savefig(f'{ROOT}/out/fig2_changepoint.png',dpi=160,facecolor=SURF); plt.close(fig)
print('Part II figures written (last layer, seeds 10-14 only)')
