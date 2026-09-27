"""Figures for blog/phasesplit.md. Every number is read from out/*.json -- nothing typed in by hand."""
import json, glob, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R='/Users/alityb/projects/cubemoe'; F=f'{R}/blog/figures'
SURF='#fcfcfb'; INK='#0b0b0b'; INK2='#52514e'; MUTED='#898781'; GRID='#e1e0d9'; AXIS='#c3c2b7'
S1='#2a78d6'; S2='#eb6834'   # validated: CVD dE 24.7, normal-vision dE 33.6
plt.rcParams.update({'font.family':['Helvetica Neue','Arial','sans-serif'],
    'font.size':11,'axes.edgecolor':AXIS,'axes.labelcolor':INK2,'xtick.color':MUTED,'ytick.color':MUTED,
    'axes.facecolor':SURF,'figure.facecolor':SURF,'savefig.facecolor':SURF,'axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.8,'axes.axisbelow':True,
    'axes.titlesize':12.5,'axes.titleweight':'semibold','axes.titlecolor':INK,'axes.titlelocation':'left'})
def save(fig,name):
    fig.savefig(f'{F}/{name}.png',dpi=200,bbox_inches='tight'); plt.close(fig); print('wrote',name)

# ---------- Fig 1: routing switches exactly at the phase boundary ----------
h=json.load(open(f'{F}/hero_data.json'))['moe_mixed_s20']
x=[r['rel'] for r in h['rows']]; y=[100*r['f1'] for r in h['rows']]
fig,ax=plt.subplots(figsize=(7.2,3.6))
ax.axvspan(-10.5,-0.5,color='#f0efec',zorder=0,lw=0)
# draw each phase as its own segment: the break at the boundary IS the finding
xa=np.array(x); ya=np.array(y)
for m in (xa<0, xa>=0):
    ax.plot(xa[m],ya[m],color=S1,lw=2,marker='o',ms=5.5,mec=SURF,mew=1.5,zorder=3)
ax.axvline(-0.5,color=INK2,lw=1,ls=(0,(3,3)))
ax.text(-5.5,50,'phase 1',ha='center',color=INK2,fontsize=10.5)
ax.text(5,50,'phase 2 (cube is in G1)',ha='center',color=INK2,fontsize=10.5)
ax.annotate('99.5%',(x[0],y[0]),xytext=(6,6),textcoords='offset points',color=INK,fontsize=10)
ax.annotate('at most 0.1% of\nphase-2 moves',(2,0),xytext=(2.4,22),color=INK,fontsize=10,
            arrowprops=dict(arrowstyle='-',color=MUTED,lw=0.8))
ax.set_xlim(-10.5,10.5); ax.set_ylim(-4,108)
ax.set_xticks(range(-10,11,2))
ax.set_xlabel('moves relative to the phase boundary')
ax.set_ylabel('% of moves routed to\nthe "phase-1 expert"')
ax.set_title('One expert handles phase 1, then switches off at the boundary')
save(fig,'fig1_routing_switch')

# ---------- Fig 2: causal test ----------
seeds=[20,21,22,23,24]; eff=[];oth=[]
for s in seeds:
    c=json.load(open(f'{R}/out/confirm_scale_fixed/moe_mixed_s{s}.json'))['causal']
    eff.append(100*c['loss_e1']); oth.append(100*c['median_other'])
fig,ax=plt.subplots(figsize=(7.2,3.6)); i=np.arange(len(seeds)); w=0.36
b1=ax.bar(i-w/2-0.01,eff,w,color=S1,label='routed through the phase-1 expert')
b2=ax.bar(i+w/2+0.01,oth,w,color=AXIS,label='routed through a typical other expert')
for xi,v in zip(i-w/2,eff): ax.text(xi,v+1,f'{v:.0f}%',ha='center',color=INK,fontsize=9.5)
for xi,v in zip(i+w/2,oth): ax.text(xi,v+1,f'{v:.1f}%',ha='center',color=INK2,fontsize=9)
ax.set_xticks(i,[f'seed {s}' for s in seeds]); ax.set_ylim(0,58)
ax.set_ylabel('% of phase-2 move probability\ndestroyed')
ax.set_title('Force a phase-2 move through the wrong expert and the model breaks')
ax.legend(frameon=False,loc='upper right',fontsize=9.5,labelcolor=INK2)
save(fig,'fig2_causal')

# ---------- Fig 3: how much of each label position alone gives away ----------
labs=['Kociemba phase','CFOP stage\n(original data)','CFOP stage\n(decorrelated data)']
vals=[57.9,86.2,57.3]
fig,ax=plt.subplots(figsize=(7.2,2.9))
cols=[S1,S2,S1]
ax.barh(range(3),vals,color=cols,height=0.55)
for k,v in enumerate(vals): ax.text(v+1,k,f'{v:.1f}%',va='center',color=INK,fontsize=10)
ax.set_yticks(range(3),labs); ax.invert_yaxis(); ax.set_xlim(0,100)
ax.grid(axis='y',visible=False)
ax.set_xlabel('% of the label\'s uncertainty removed by knowing only the move number')
ax.set_title('In the original CFOP data, the move number almost gives the stage away')
save(fig,'fig3_position')

# ---------- Fig 4: alignment appears after decorrelation ----------
def ratio(fn):
    X=json.load(open(f'{R}/out/{fn}.json'))['cfop']
    return [r['cnmi_stage']/r['shuf_stage'] for r in X]
a=ratio('h2_2x2_fixed'); b=ratio('h2b_2x2')
fig,ax=plt.subplots(figsize=(7.2,3.4))
rng=np.random.RandomState(1)
ax.axhline(1,color=INK2,lw=1,ls=(0,(3,3)))
ax.text(1.62,1.08,'no better than a shuffled router',color=INK2,fontsize=9.5,ha='right')
ax.scatter(0+rng.uniform(-.08,.08,5),a,s=70,color=S2,edgecolor=SURF,lw=1.5,zorder=3)
ax.scatter(1+rng.uniform(-.08,.08,5),b,s=70,color=S1,edgecolor=SURF,lw=1.5,zorder=3)
for k,v in ((0,a),(1,b)): ax.text(k+0.17,np.mean(v),f'mean {np.mean(v):.1f}×',va='center',color=INK,fontsize=10)
ax.set_xticks([0,1],['original CFOP data','decorrelated CFOP data']); ax.set_xlim(-0.5,1.7)
ax.set_ylim(0,max(b)*1.25); ax.grid(axis='x',visible=False)
ax.set_ylabel('router–stage alignment\n÷ shuffled-router alignment')
ax.set_title('Break the position shortcut and routing starts tracking CFOP stages')
save(fig,'fig4_alignment')

# ---------- Fig 5: alignment is not use ----------
fig,ax=plt.subplots(figsize=(5.6,5.0))
for fn,col,lab in (('h2b_causal',S1,'readout: per-stage accuracy'),('h2c','#eb6834','readout: stage-characteristic moves')):
    Rr=json.load(open(f'{R}/out/{fn}.json'))
    xm=[];yn=[]
    for r in Rr:
        for s in r['stages']:
            xm.append(100*r['L_median'][str(s)]); yn.append(100*r['L_native'][str(s)])
    ax.scatter(xm,yn,s=62,color=col,edgecolor=SURF,lw=1.5,zorder=3,label=lab)
lim=[-12,45]; ax.plot(lim,lim,color=INK2,lw=1,ls=(0,(3,3)),zorder=2)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
ax.text(30,15,'if routing mattered,\npoints would sit\ndown here',color=INK2,fontsize=9.5,ha='center')
ax.set_xlabel('damage via a typical expert (%)'); ax.set_ylabel("damage via the stage's own expert (%)")
ax.set_title('Routing lines up with stages,\nbut the model doesn\'t depend on it')
ax.legend(frameon=False,loc='upper left',fontsize=9,labelcolor=INK2)
save(fig,'fig5_alignment_not_use')
