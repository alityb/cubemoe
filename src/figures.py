import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT='/Users/alityb/projects/cubemoe'
R=json.load(open(f'{ROOT}/out/results.json'))
SURF='#fcfcfb'; INK='#0b0b0b'; INK2='#52514e'; GRID='#e3e2df'
C={'obs':'#2a78d6','posr':'#eb6834','shuf':'#1baf7a','hash':'#eda100','dense':'#e87ba4'}
def style(ax):
    ax.set_facecolor(SURF)
    for sp in ('top','right'): ax.spines[sp].set_visible(False)
    for sp in ('left','bottom'): ax.spines[sp].set_color(GRID); ax.spines[sp].set_linewidth(1)
    ax.grid(True, axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9, length=0)

# ---------- fig 0 ----------
fig,ax=plt.subplots(figsize=(7.2,4.0),facecolor=SURF); style(ax)
for name,col,lab in [('naive',C['posr'],'naive (stock Kociemba)'),('forced',C['obs'],'forced L~U(10,20)')]:
    if name not in R.get('overlap',{}): continue
    o=R['overlap'][name]; ax.plot(o['pos'],o['p_phase2'],lw=2,color=col,label=lab,solid_capstyle='round')
    i=int(np.argmin(np.abs(np.array(o['p_phase2'])-0.5)))
    ax.annotate(lab.split(' ')[0], (o['pos'][i],o['p_phase2'][i]), textcoords='offset points',
                xytext=(6,10 if name=='forced' else -16), color=INK, fontsize=9, fontweight='bold')
ax.axhspan(0.05,0.95,color='#f2f1ee',zorder=0)
ax.set_xlabel('step index within solution',color=INK2,fontsize=10)
ax.set_ylabel('P(phase 2 | step index)',color=INK2,fontsize=10)
ax.set_title('Fig 0 — how much the step index alone gives away the phase',color=INK,fontsize=11.5,loc='left',pad=26)
ax.text(0,1.02,'shaded band = steps where BOTH phases occur (the only rows a conditional analysis can use)',
        transform=ax.transAxes,fontsize=8.5,color=INK2,va='bottom')
ax.legend(frameon=False,fontsize=9,labelcolor=INK2,loc='lower right')
fig.tight_layout(); fig.savefig(f'{ROOT}/out/fig0_overlap.png',dpi=160,facecolor=SURF); plt.close(fig)

# ---------- fig 1 ----------
M=R['models']; base=M.get('moe_mixed')
if base:
    NL=len(base['layers']); x=np.arange(NL)
    series=[('MoE (observed)',C['obs'],lambda L:L['observed']),
            ('position-only router',C['posr'],lambda L:L['null_position_router']),
            ('shuffled-router null',C['shuf'],lambda L:L['null_shuffled'])]
    extra=[]
    if 'hash_mixed' in M: extra.append(('hash router',C['hash'],M['hash_mixed']))
    if 'dense_mixed' in M: extra.append(('dense twin + k-means',C['dense'],M['dense_mixed']))
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.2),facecolor=SURF)
    for ax,key,title in [(axes[0],'nmi_raw','RAW NMI(expert; phase) — the number being debunked'),
                         (axes[1],'nmi_strat_wt','CONDITIONAL NMI(expert; phase | step index) — primary metric')]:
        style(ax); n=len(series)+len(extra); w=0.8/n
        for j,(lab,col,get) in enumerate(series):
            v=[get(L)[key] for L in base['layers']]
            b=ax.bar(x+(j-(n-1)/2)*w,v,w*0.88,color=col,label=lab,zorder=3)
            for rr,vv in zip(b,v): ax.text(rr.get_x()+rr.get_width()/2,vv,f'{vv:.3f}',ha='center',va='bottom',fontsize=6.5,color=INK2)
        for j,(lab,col,mm) in enumerate(extra):
            v=[L['observed'][key] for L in mm['layers']][:NL]; v+=[0]*(NL-len(v))
            jj=len(series)+j
            b=ax.bar(x+(jj-(n-1)/2)*w,v,w*0.88,color=col,label=lab,zorder=3)
            for rr,vv in zip(b,v): ax.text(rr.get_x()+rr.get_width()/2,vv,f'{vv:.3f}',ha='center',va='bottom',fontsize=6.5,color=INK2)
        ax.set_xticks(x); ax.set_xticklabels([f'layer {i}' for i in range(NL)])
        ax.set_title(title,color=INK,fontsize=10.5,loc='left',pad=8)
        ax.set_ylabel('NMI',color=INK2,fontsize=10)
    axes[1].legend(frameon=True,facecolor=SURF,edgecolor=GRID,fontsize=8.5,labelcolor=INK2,ncol=1,loc='upper left')
    fig.suptitle('Fig 1 — expert/phase alignment, raw vs conditioned on step index (mixed set)',
                 color=INK,fontsize=12,x=0.008,ha='left')
    fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(f'{ROOT}/out/fig1_conditional_nmi.png',dpi=160,facecolor=SURF); plt.close(fig)

# ---------- fig 2 ----------
sc=R.get('cp_scatter',{})
panels=[(t,C['obs'] if 'mixed' in t else C['posr']) for t in ['moe_mixed','moe_naive'] if t in sc]
if panels:
    fig,axes=plt.subplots(1,len(panels),figsize=(5.6*len(panels),4.4),facecolor=SURF,squeeze=False)
    for ax,(tag,col) in zip(axes[0],panels):
        style(ax)
        def _ps(k):
            v=M[tag]['layers'][int(k)]['observed']['cp'].get('partial_slope',np.nan)
            return v if np.isfinite(v) else -9
        best=max(sc[tag], key=_ps)
        xs=np.array(sc[tag][best]['xs']); ys=np.array(sc[tag][best]['ys'])
        cp=M[tag]['layers'][int(best)]['observed']['cp']
        _r=np.random.RandomState(0)
        ax.scatter(xs+_r.uniform(-.3,.3,len(xs)),ys+_r.uniform(-.3,.3,len(ys)),
                   s=9,color=col,alpha=.25,linewidths=0,zorder=3)
        lo,hi=xs.min()-1,xs.max()+1
        ax.plot([lo,hi],[lo,hi],color=GRID,lw=1.5,ls='--',zorder=2)
        if np.isfinite(cp['slope']):
            a,b=np.polyfit(xs,ys,1); ax.plot([lo,hi],[a*lo+b,a*hi+b],color=INK,lw=2,zorder=4)
        ax.set_xlabel('true seam index (G1 entry)',color=INK2,fontsize=10)
        ax.set_ylabel('router change-point index',color=INK2,fontsize=10)
        ax.set_title(f"{tag}  (layer {best})",color=INK,fontsize=10.5,loc='left',pad=8)
        ax.text(.03,.95,f"partial slope {cp.get('partial_slope',float('nan')):.3f}  (p {cp.get('partial_p',float('nan')):.3f})\n"
                        f"naive slope {cp['slope']:.3f}   r {cp['r']:.3f}\nn {cp['n']}   seam sd {cp['seam_sd']:.2f}",
                transform=ax.transAxes,va='top',fontsize=9,color=INK,
                bbox=dict(fc=SURF,ec=GRID,boxstyle='round,pad=0.4'))
    _an={}
    try:
        _cal=json.load(open(f'{ROOT}/out/calibration.json'))
        for _k,_v in _cal.items():
            _an[_k]=[r for r in _v['rows'] if r['name'].startswith('perfect')][0]['partial_slope']
    except Exception: pass
    _a=_an.get('mixed')
    fig.suptitle('Fig 2 — does the router move its switch point when the seam moves?\n'
                 'dashed = identity (slope 1) · solid = fitted partial slope · a CALIBRATED perfect phase '
                 + (f'tracker reads {_a:.2f} on this seam distribution, not 1.0' if _a else 'tracker reads <1 here'),
                 color=INK,fontsize=10.5,x=0.008,ha='left')
    fig.tight_layout(rect=[0,0,1,0.88]); fig.savefig(f'{ROOT}/out/fig2_changepoint.png',dpi=160,facecolor=SURF); plt.close(fig)
print('figures written')
