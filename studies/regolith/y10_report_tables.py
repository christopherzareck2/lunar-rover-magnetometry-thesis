"""Build the regolith study tables (y10_tables.md, y10_mitigation_table.csv) from y10_results.json and y10_scan.json.

Run from this folder: the JSON files are read from the current directory.
"""
import json, numpy as np, pandas as pd
def load(f):
    d=pd.DataFrame(json.load(open(f))["rows"]); fr=d[d["mode"]=="fieldrms"].copy(); d=d[d["mode"]!="fieldrms"].copy(); d["noisy"]=d["noisy"].astype(bool); return d,fr
df,fr=load("y10_results.json"); sc,_=load("y10_scan.json")
MREF_PER_CHI_100=0.04*100e-9/(4e-7*np.pi)/2.4e-3/0.04   # M_reg/Mref per unit chi at 100 nT
def md(t, floatfmt="{:.1f}"):
    cols=[str(c) for c in t.columns]; idx=t.index
    names=list(t.index.names) if isinstance(t.index,pd.MultiIndex) else [t.index.name or ""]
    lines=["| "+" | ".join(names+cols)+" |","|"+"|".join(["---"]*(len(names)+len(cols)))+"|"]
    for i,row in t.iterrows():
        lab=list(i) if isinstance(i,tuple) else [i]
        lines.append("| "+" | ".join([str(x) for x in lab]+[("" if pd.isna(v) else (floatfmt.format(v) if isinstance(v,(float,np.floating)) else str(v))) for v in row])+" |")
    return "\n".join(lines)
def panel(d, mit="none", mode="S1", noisy=False, dx=0.0, b=0.0, col="dir_mean", B=(100.0,300.0), agg="mean"):
    q=d[(d["mode"]==mode)&(d.mit==mit)&(d.noisy==noisy)&(d.dx==dx)&(d.b==b)&(d.B.isin(B))&(d.chi_reg>0)]
    t=q.groupby(["B","chi_reg","h"])[col].agg(agg).unstack("h"); t.index=t.index.set_names(["B (nT)","chi_reg"]); return t
out=[]
# T1 field rms
t=fr.groupby(["B","chi_reg","h"]).apply(lambda g: (g.rms_reg_nT/g.rms_out_nrm_nT).mean(), include_groups=False).unstack("h")
t.index=t.index.set_names(["B (nT)","chi_reg"])
mr=fr.groupby(["B","chi_reg"]).Mreg_over_Mref.mean()
out.append("### T1 regolith field rms / outcrop field rms (3-component rms over both loops, outcrop at M_ref, mean of 3 NRM directions)\n"+md(t,"{:.2f}"))
rms_out=fr.groupby("dir").rms_out_nrm_nT.mean()
out.append("outcrop NRM field rms (nT): "+", ".join(f"{k} {v:.3f}" for k,v in rms_out.items())+f"; mean {rms_out.mean():.3f}")
rr=fr.groupby(["h","B","chi_reg"]).rms_reg_nT.mean().unstack("h"); rr.index=rr.index.set_names(["B (nT)","chi_reg"])
out.append("### T1b regolith field rms (nT)\n"+md(rr,"{:.3f}"))
out.append("M_reg/M_ref: "+"; ".join(f"B={B:.0f} nT, chi={c}: {v:.2f}" for (B,c),v in mr.items()))
ind=fr.groupby(["b","B"]).rms_out_induced_nT.mean()
out.append("basalt induced field rms (nT, chi_b=2e-3, outcrop+buried b): "+"; ".join(f"b={b}, B={B:.0f}: {v:.4f}" for (b,B),v in ind.items()))
# T2 noise-free
for (dx,b) in ((0.0,0.0),(0.0,0.5),(0.5,0.5),(0.5,0.0),(0.0,1.0)):
    out.append(f"### T2 noise-free direction bias (deg), none, loops {'centred' if dx==0 else '0.5 m off centre'}, b={b}, S1\n"+md(panel(df,dx=dx,b=b)))
for (dx,b) in ((0.0,0.0),(0.0,0.5),(0.5,0.5)):
    out.append(f"### T2c control chi_reg=0, none, noise-free / noisy mean (deg), dx={dx}, b={b}")
    q=df[(df["mode"]=="S1")&(df.mit=="none")&(df.chi_reg==0)&(df.dx==dx)&(df.b==b)&(df.h==0.5)].groupby(["B","noisy"]).dir_mean.mean().unstack("noisy")
    out.append(md(q,"{:.1f}"))
# intensity bias
for (dx,b) in ((0.0,0.0),(0.0,0.5)):
    out.append(f"### T2d noise-free signed intensity error (%), none, dx={dx}, b={b}\n"+md(panel(df,dx=dx,b=b,col="dM_mean"),"{:+.0f}"))
# worst direction
for (dx,b) in ((0.0,0.5),):
    q=df[(df["mode"]=="S1")&(df.mit=="none")&(df.noisy==False)&(df.dx==dx)&(df.b==b)&(df.chi_reg>0)]
    t=q.groupby(["B","chi_reg","h"]).dir_mean.agg(["min","max"]).unstack("h")
    out.append(f"### T2e noise-free dir bias min..max over 3 NRM directions, dx={dx}, b={b}")
    tt=pd.DataFrame({h:[f"{t[('min',h)].iloc[i]:.0f}-{t[('max',h)].iloc[i]:.0f}" for i in range(len(t))] for h in HS} if (HS:=[0.5,1.0,3.0,5.0]) else {}, index=t.index); tt.index=tt.index.set_names(["B (nT)","chi_reg"])
    out.append(md(tt))
# noisy
for (dx,b) in ((0.0,0.0),(0.0,0.5),(0.5,0.5)):
    out.append(f"### T3 noisy (100 draws) mean direction error (deg), none, dx={dx}, b={b}\n"+md(panel(df,noisy=True,dx=dx,b=b)))
    out.append(f"p95:\n"+md(panel(df,noisy=True,dx=dx,b=b,col="dir_p95")))
# mitigations
rows=[]
for (dx,b) in ((0.0,0.5),(0.5,0.5),(0.0,0.0)):
  for B in (100.0,300.0):
    for chi in (0.01,0.02,0.04):
      for h in (1.0,3.0):
        r={"dx":dx,"b":b,"B":B,"chi":chi,"h":h}
        for mit in ("none","a","b","b_u20","b_hwrong","c1.4","c2.0"):
          for noisy in (False,True):
            q=df[(df["mode"]=="S1")&(df.mit==mit)&(df.noisy==noisy)&(df.dx==dx)&(df.b==b)&(df.B==B)&(df.chi_reg==chi)&(df.h==h)]
            r[f"{mit}|{'N' if noisy else 'F'}"]=q.dir_mean.mean()
        rows.append(r)
mit=pd.DataFrame(rows)
mit.to_csv("y10_mitigation_table.csv",index=False)
for (dx,b) in ((0.0,0.5),(0.5,0.5),(0.0,0.0)):
    q=mit[(mit.dx==dx)&(mit.b==b)].set_index(["B","chi","h"]).drop(columns=["dx","b"])
    out.append(f"### T4 mitigations, dx={dx}, b={b}: mean dir err (deg), F=noise-free, N=noisy\n"+md(q))
# control per mitigation
rows=[]
for (dx,b) in ((0.0,0.0),(0.0,0.5),(0.5,0.5)):
  for h in (1.0,3.0):
    r={"dx":dx,"b":b,"h":h}
    for m_ in ("none","a","b","b_u20","b_hwrong"):
      for noisy in (False,True):
        q=df[(df["mode"]=="S1")&(df.mit==m_)&(df.noisy==noisy)&(df.dx==dx)&(df.b==b)&(df.chi_reg==0)&(df.h==h)]
        r[f"{m_}|{'N' if noisy else 'F'}"]=q.dir_mean.mean()
    rows.append(r)
out.append("### T4c controls (chi_reg=0) per mitigation\n"+md(pd.DataFrame(rows).set_index(["dx","b","h"]))) 
rows=[]
for (dx,b) in ((0.0,0.0),(0.0,0.5)):
  for B in (100.0,300.0):
    for h in (0.5,1.0,3.0):
      r={"dx":dx,"b":b,"B":B,"h":h}
      for m_ in ("none","c2.0","c1.4","c1.2","c1.1"):
        q=df[(df["mode"]=="S1")&(df.mit==m_)&(df.noisy==False)&(df.dx==dx)&(df.b==b)&(df.B==B)&(df.chi_reg==0.04)&(df.h==h)]
        r[m_]=q.dir_mean.mean()
      rows.append(r)
out.append("### T4d chi_reg=0.04, noise-free bias vs how well chi_reg is known (lognormal factor)\n"+md(pd.DataFrame(rows).set_index(["dx","b","B","h"])))
# tolerance from scan + main
al=pd.concat([sc[sc.B==100.0], df[(df.B==100.0)]])
def cross(g, thr=5.0):
    g=g.groupby("chi_reg").dir_mean.mean().sort_index()
    x=g.index.values*MREF_PER_CHI_100; y=g.values
    for i in range(len(y)-1):
        if y[i]<thr<=y[i+1]:
            return x[i]+(thr-y[i])*(x[i+1]-x[i])/(y[i+1]-y[i])
    return np.nan if y.max()<thr else 0.0
tol=[]
for (dx,b) in ((0.0,0.0),(0.0,0.5),(0.5,0.0)):
  for noisy in (False,True):
    for mit_ in ("none","c1.4","c1.2","b_u20"):
      r={"dx":dx,"b":b,"noisy":noisy,"mit":mit_}
      for h in (0.5,1.0,3.0,5.0):
        q=al[(al["mode"]=="S1")&(al.mit==mit_)&(al.noisy==noisy)&(al.dx==dx)&(al.b==b)&(al.h==h)&(al.chi_reg>0)]
        r[h]=cross(q)
      tol.append(r)
tol=pd.DataFrame(tol)
out.append("### T5 M_reg/M_ref at which the mean direction error crosses 5 deg (B=100 nT grid; chi_reg*B scales it: M_reg/Mref = 33.15 * chi_reg * (B/100 nT))\n"+md(tol.set_index(["dx","b","noisy","mit"]),"{:.2f}"))
# scan table none noise-free dx0 b0
for (dx,b) in ((0.0,0.0),):
    q=al[(al["mode"]=="S1")&(al.mit=="none")&(al.noisy==False)&(al.dx==dx)&(al.b==b)&(al.chi_reg>0)]
    t=q.groupby(["chi_reg","h"]).dir_mean.mean().unstack("h"); t.index=[f"{c:g} ({c*MREF_PER_CHI_100:.2f})" for c in t.index]; t.index.name="chi_reg (M_reg/Mref @100nT)"
    out.append("### T5b scan noise-free, none, dx=0, b=0\n"+md(t))
# S2, S1fill comparison
rows=[]
for mode_ in ("S1","S2","S1fill"):
  for b in (0.5,1.0):
    for h in (0.5,1.0,3.0,5.0):
      q=df[(df["mode"]==mode_)&(df.mit=="none")&(df.noisy==False)&(df.dx==0.0)&(df.b==b)&(df.B==100.0)&(df.chi_reg==0.02)&(df.h==h)]
      rows.append({"mode":mode_,"b":b,"h":h,"dir":q.dir_mean.mean()})
t=pd.DataFrame(rows).pivot_table(index=["mode","b"],columns="h",values="dir")
out.append("### T6 hole modes, noise-free, centred, B=100, chi_reg=0.02\n"+md(t))
rows=[]
for mode_ in ("S1","S1fill"):
  for h in (0.5,1.0,3.0,5.0):
    q=df[(df["mode"]==mode_)&(df.mit=="none")&(df.noisy==False)&(df.dx==0.0)&(df.b==0.0)&(df.B==100.0)&(df.chi_reg==0.01)&(df.h==h)]
    rows.append({"mode":mode_,"b":0.0,"h":h,"dir":q.dir_mean.mean()})
out.append(md(pd.DataFrame(rows).pivot_table(index=["mode","b"],columns="h",values="dir")))
# plug scalar recovery
q=df[(df.mit=="b")&(df.chi_reg>0)&(df["mode"]=="S1")&(df.noisy==True)]
out.append("### T7 plug scalar recovered / true (mitigation b, noisy): mean "+f"{q.plug_scalar_over_true.mean():.3f}; sd of scalar (A/m) mean {q.plug_scalar_sd_A_per_m.mean():.2e}")
q=q.groupby(["dx","b","h"]).agg(ratio=("plug_scalar_over_true","mean"),sd=("plug_scalar_sd_A_per_m","mean"))
out.append(md(q,"{:.3g}"))
open("y10_tables.md","w").write("\n\n".join(out)); print("\n\n".join(out))
