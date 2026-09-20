#!/usr/bin/env python3
"""
Regenerate all three paper figures from the canonical result files.

Usage (from the repo root, or anywhere):
    python3 code/make_figures.py

Reads:
    partitions/provisioned/dist_query_type.csv
    partitions/provisioned/instance_signatures.csv
    results/provisioned_results.json
    results/serverless_results.json
    results/null_and_fixedci.json
Writes:
    results/figures/fig2_calibration.png
    results/figures/alpha_forest.png

Requires: numpy, matplotlib   (pip install numpy matplotlib)
"""
import os, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PART = os.path.join(ROOT, "partitions")
RES  = os.path.join(ROOT, "results")
FIGS = os.path.join(RES, "figures")
SEED = 7

NAVY="#1f2a44"; INK="#222222"; BLUE="#4269d0"; ORANGE="#e6902e"
PURPLE="#8250c4"; GREEN="#3aa06b"; GREY="#8a93a3"; LGREY="#eef1f6"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10.5})


def _mean_pairwise_js(D):
    eps=1e-12; D=np.clip(D,eps,None); D=D/D.sum(1,keepdims=True); logD=np.log2(D)
    tot=0.0; cnt=0
    for i in range(D.shape[0]-1):
        p=D[i]; q=D[i+1:]; mmix=0.5*(p[None,:]+q); lm=np.log2(mmix)
        kl_p=(p[None,:]*(logD[i][None,:]-lm)).sum(1)
        kl_q=(q*(np.log2(q)-lm)).sum(1)
        js=0.5*kl_p+0.5*kl_q; tot+=js.sum(); cnt+=js.size
    return tot/cnt


# ---------------------------------------------------------------- Figure 2
def figure2():
    rng=np.random.default_rng(SEED)
    ids=[]; rows=[]
    with open(os.path.join(PART,"provisioned","dist_query_type.csv")) as f:
        r=csv.reader(f); next(r)
        for ln in r: ids.append(int(ln[0])); rows.append([float(x) for x in ln[1:]])
    P=np.array(rows); n=P.shape[0]
    cnt={}
    with open(os.path.join(PART,"provisioned","instance_signatures.csv")) as f:
        r=csv.reader(f); h=next(r); ci=h.index("n_queries"); ii=h.index("instance_id")
        for ln in r: cnt[int(ln[ii])]=int(float(ln[ci]))
    m=np.array([cnt[i] for i in ids])
    canon=json.load(open(os.path.join(RES,"provisioned_results.json")))["proxies"]["query_type"]
    obs=canon["observed_mean_js"]; rec=canon["matched_alpha"]; rec_ci=canon["alpha_ci95"]
    pbar=P.mean(0); pbar/=pbar.sum()
    def synth(alpha,reps=6):
        vals=[]
        for _ in range(reps):
            ptil=rng.dirichlet(alpha*pbar,size=n); Ph=np.empty_like(ptil)
            for c in range(n):
                x=rng.multinomial(min(m[c],2_000_000),ptil[c]); Ph[c]=x/x.sum()
            vals.append(_mean_pairwise_js(Ph))
        return np.mean(vals)
    alphas=np.geomspace(0.1,30,26); curve=np.array([synth(a) for a in alphas])
    fig,ax=plt.subplots(figsize=(7.4,4.9),dpi=200)
    ax.axvspan(0.1,1.0,color=GREY,alpha=0.14,lw=0)
    ax.text(0.32,0.05,"literature's\ntested band",ha="center",va="bottom",fontsize=8.6,color="#5b6472")
    ax.plot(alphas,curve,"-o",color=ORANGE,ms=4,lw=2,label="Synthetic spread vs a  (Dirichlet)")
    ax.axhline(obs,color=BLUE,ls="--",lw=1.8,label=f"Observed real spread = {obs:.2f}")
    ax.axvspan(rec_ci[0],rec_ci[1],color=GREEN,alpha=0.13,lw=0)
    ax.axvline(rec,color=GREEN,ls=":",lw=1.8); ax.plot([rec],[obs],"o",color=GREEN,ms=9,zorder=5)
    ax.annotate(f"recovered a = {rec:.2f}\n[{rec_ci[0]:.2f}, {rec_ci[1]:.2f}]",xy=(rec,obs),
                xytext=(5.2,0.60),fontsize=9.6,color=NAVY,
                arrowprops=dict(arrowstyle="-|>",color=GREEN,lw=1.4))
    ax.set_xscale("log"); ax.set_xlabel("Dirichlet concentration  a  (log scale)")
    ax.set_ylabel("Mean pairwise Jensen-Shannon spread")
    ax.set_title("Calibration: provisioned fleet, query-type mix",fontsize=11.5,color=NAVY,pad=8)
    ax.set_xticks([0.1,0.3,1,3,10,30]); ax.set_xticklabels(["0.1","0.3","1","3","10","30"])
    ax.set_ylim(0,0.9); ax.grid(True,which="both",ls=":",lw=0.6,color="#dfe4ec")
    ax.legend(loc="upper right",fontsize=9,frameon=True,facecolor="white",edgecolor="#d0d6e0")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS,"fig2_calibration.png"),bbox_inches="tight",facecolor="white",dpi=200)
    plt.close(); print(f"wrote fig2_calibration.png  (curve crosses {obs:.2f} near a={rec:.2f})")


# ---------------------------------------------------------------- Figure 3
def figure3():
    prov=json.load(open(os.path.join(RES,"provisioned_results.json")))["proxies"]
    srv =json.load(open(os.path.join(RES,"serverless_results.json")))["proxies"]
    nullf=json.load(open(os.path.join(RES,"null_and_fixedci.json")))
    proxies=["query_type","scan_volume","joins","hour_of_day"]
    labels={"query_type":"Query type","scan_volume":"Scanned volume",
            "joins":"Join complexity","hour_of_day":"Hour of day"}
    fig,ax=plt.subplots(figsize=(7.6,5.0),dpi=200)
    ax.axvspan(0.1,1.0,color=GREY,alpha=0.14,lw=0)
    ax.text(0.32,7.5,"literature's tested band",fontsize=8.6,color="#5b6472",ha="center")
    y=0; yticks=[]; ylabels=[]
    for pr in reversed(proxies):
        for fleet,src,col,mk in [("Serverless",srv,ORANGE,"s"),("Provisioned",prov,BLUE,"o")]:
            a=src[pr]["matched_alpha"]; ci=src[pr]["alpha_ci95"]
            p=nullf[fleet.lower()][pr]["null_test"]["p_value"]; sig=p<0.05
            ax.errorbar([a],[y],xerr=[[a-ci[0]],[ci[1]-a]],fmt=mk,color=col,ms=7,capsize=3.5,
                        lw=1.8,mec="white",mew=0.8,label=fleet if pr=="hour_of_day" else None)
            ax.text(ci[1]+0.5,y,"*" if sig else "n.s.",va="center",
                    fontsize=11 if sig else 8.5,color=NAVY if sig else "#9aa2b1",
                    fontweight="bold" if sig else "normal")
            yticks.append(y); ylabels.append(fleet); y+=1
        ax.text(-0.02,y-1.5,labels[pr],transform=ax.get_yaxis_transform(),ha="right",va="center",
                fontsize=10.5,fontweight="bold",color=NAVY)
        y+=0.6
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels,fontsize=9)
    ax.set_xscale("log"); ax.set_xticks([0.1,0.3,1,3,10,30])
    ax.set_xticklabels(["0.1","0.3","1","3","10","30"])
    ax.set_xlabel("Recovered Dirichlet a  (log scale)   <- more heterogeneous      more homogeneous ->")
    ax.set_title("Calibrated heterogeneity by dimension and fleet",fontsize=11.5,color=NAVY,pad=8)
    ax.grid(True,axis="x",which="both",ls=":",lw=0.6,color="#dfe4ec")
    ax.set_ylim(-0.8,y-0.4)
    for s in ["top","right","left"]: ax.spines[s].set_visible(False)
    ax.tick_params(axis="y",length=0)
    h,l=ax.get_legend_handles_labels()
    ax.legend(h,l,loc="upper right",fontsize=9,frameon=True,facecolor="white",
              edgecolor="#d0d6e0",title="Fleet")
    fig.text(0.5,-0.02,"*  cluster structure beyond the Dirichlet null (p<0.05);   n.s. = not significant",
             ha="center",va="top",fontsize=8.6,color="#5b6472")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS,"alpha_forest.png"),bbox_inches="tight",facecolor="white",dpi=200)
    plt.close(); print("wrote alpha_forest.png")


if __name__ == "__main__":
    figure2(); figure3()
    print("All figures regenerated into", RES)
