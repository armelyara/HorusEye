# ============================================================
# HorusEye Q1 (niveau 1) — Classification d'urgence sous voile
# Voiles plausibles uniquement + calibration ECE/AUROC
# Environnement : Kaggle, GPU T4. Qwen2.5-VL seul.
# ============================================================
# Settings : GPU T4 ON, Internet ON, dataset AIDER en input.
#
# CHANGEMENTS vs ancien F0 :
#  - PLUS de "thermal" (le feu n'est pas un voile ; abandonné)
#  - Voiles appliqués UNIQUEMENT là où physiquement plausibles :
#      fumée  -> fire, traffic_accident, normal
#      brouillard -> flood, collapsed_building, normal
#  - Calibration ECE + AUROC ajoutée (self-reported confidence)
# ============================================================

import subprocess, sys
def pip(*a): subprocess.run([sys.executable,"-m","pip","install","-q",*a], check=False)
pip("-U","transformers","accelerate"); pip("qwen-vl-utils")

import os, re, json, random, gc
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
from tqdm import tqdm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
AIDER_ROOT = "/kaggle/input/aider-dataset/AIDER"   # ⚠️ ajuste
OUTPUT_DIR = "/kaggle/working/q1_results"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
NUM_PER_CLASS = 50
SEED = 42
SEVERITY = 0.5
LABELS = ["fire","flood","collapsed_building","traffic_accident","normal"]
CLASS_MAP = {"fire":"fire","flooded_areas":"flood","collapsed_building":"collapsed_building",
             "traffic_incident":"traffic_accident","normal":"normal"}
IMG_EXT = {".jpg",".jpeg",".png",".bmp"}

# --- MATRICE VOILES PLAUSIBLES (le cœur de la correction) ---
# pour chaque classe, la liste des voiles physiquement plausibles
PLAUSIBLE_VEILS = {
    "fire":               ["clean", "smoke"],
    "traffic_accident":   ["clean", "smoke"],
    "flood":              ["clean", "fog"],
    "collapsed_building": ["clean", "fog"],
    "normal":             ["clean", "smoke", "fog"],
}
ALL_CONDITIONS = ["clean", "fog", "smoke"]   # plus de thermal

os.makedirs(OUTPUT_DIR, exist_ok=True); random.seed(SEED)

# ---------- TA PIPELINE (voiles seulement : fog + smoke) ----------
class DegradationPipeline:
    def __init__(self, severity=0.5): self.severity=np.clip(severity,0.,1.)
    def add_fog(self,image,depth_map=None):
        f=image.astype(np.float32)/255.; h,w=image.shape[:2]
        if depth_map is None:
            depth_map=np.tile(np.linspace(1.,0.,h).reshape(h,1),(1,w))
            n=cv2.GaussianBlur(np.random.rand(h,w).astype(np.float32)*.15,(51,51),0)
            depth_map=np.clip(depth_map+n,0,1)
        A=.95; beta=self.severity*2.; t=np.exp(-beta*depth_map); t=np.stack([t]*3,-1)
        return np.clip((f*t+A*(1-t))*255,0,255).astype(np.uint8)
    def add_smoke(self,image):
        h,w=image.shape[:2]; f=image.astype(np.float32)/255.
        sm=self._smoke_tex(h,w); col=np.array([.9,.9,.92])
        a=np.stack([sm*self.severity*1.2]*3,-1); layer=np.ones_like(f)*col
        return np.clip((f*(1-a)+layer*a)*255,0,255).astype(np.uint8)
    def _smoke_tex(self,h,w):
        s=np.zeros((h,w),np.float32)
        for sc in [4,8,16,32,64]:
            n=np.random.rand(max(h//sc+1,2),max(w//sc+1,2)).astype(np.float32)
            s+=cv2.resize(n,(w,h),interpolation=cv2.INTER_CUBIC)*(sc/64.)
        s=(s-s.min())/(s.max()-s.min()+1e-8); s=np.clip(s*2.,0,1)
        return cv2.GaussianBlur(np.power(s,2.5),(41,41),0)
    def apply(self,image,cond):
        if cond=="fog":   return self.add_fog(image)
        if cond=="smoke": return self.add_smoke(image)
        return image.copy()   # clean

degrader = DegradationPipeline(SEVERITY)

# ---------- PROMPT / PARSING ----------
PROMPT=('You are analyzing a drone/aerial image of a potential emergency scene. '
        'Classify the emergency type as EXACTLY one of: '
        '[fire, flood, collapsed_building, traffic_accident, normal]. '
        'Give your confidence between 0.0 and 1.0 (how certain you are). '
        'Respond ONLY with valid JSON, no markdown:\n'
        '{"emergency_type": "<one of the 5>", "confidence": <float 0-1>}')

def parse_response(text):
    if not text: return None
    text=text.replace("```json","").replace("```","").strip()
    m=re.search(r"\{.*\}",text,re.DOTALL)
    if not m: return None
    try:
        o=json.loads(m.group(0)); e=str(o.get("emergency_type","")).lower().strip()
        if e not in LABELS:
            for l in LABELS:
                if l.split("_")[0] in e: e=l; break
        conf=float(o.get("confidence",0.0))
        conf=min(max(conf,0.0),1.0)
        return {"label": e if e in LABELS else "unknown", "confidence": conf}
    except: return None

# ---------- MODÈLE ----------
print("Chargement Qwen2.5-VL...")
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_NAME,torch_dtype=torch.float16,device_map="auto")
processor=AutoProcessor.from_pretrained(MODEL_NAME)
print("✓ chargé")

def qwen_predict(pil, retries=1):
    msgs=[{"role":"user","content":[{"type":"image","image":pil},{"type":"text","text":PROMPT}]}]
    text=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    ii,vi=process_vision_info(msgs)
    inp=processor(text=[text],images=ii,videos=vi,padding=True,return_tensors="pt").to(model.device)
    try:
        with torch.no_grad(): gen=model.generate(**inp,max_new_tokens=64,do_sample=False)
        tr=[o[len(i):] for i,o in zip(inp.input_ids,gen)]
        out=processor.batch_decode(tr,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        del inp,gen,tr; return out
    except torch.cuda.OutOfMemoryError:
        del inp; gc.collect(); torch.cuda.empty_cache()
        return qwen_predict(pil,retries-1) if retries>0 else "ERROR: OOM"
    finally:
        gc.collect(); torch.cuda.empty_cache()

# ---------- SAMPLES ----------
def load_samples():
    root=Path(AIDER_ROOT)
    if not root.exists():
        for p in Path("/kaggle/input").rglob("*"):
            if p.is_dir(): print("  ",p)
        raise SystemExit("Ajuste AIDER_ROOT")
    s=[]
    for folder,label in CLASS_MAP.items():
        d=root/folder
        if not d.exists(): continue
        imgs=[f for f in d.iterdir() if f.suffix.lower() in IMG_EXT]
        random.shuffle(imgs)
        if NUM_PER_CLASS: imgs=imgs[:NUM_PER_CLASS]
        for img in imgs: s.append({"path":str(img),"true_label":label})
    return s

# ---------- MÉTRIQUES : F1 macro + confusion ----------
def compute_f1(results):
    idx={l:i for i,l in enumerate(LABELS)}; cm=np.zeros((5,5),int); c=v=0
    for r in results:
        ti=idx[r["true_label"]]; p=r["pred_label"]
        if p in idx:
            cm[ti][idx[p]]+=1; v+=1
            if idx[p]==ti: c+=1
    acc=c/v if v else 0.; f1s={}
    for i,l in enumerate(LABELS):
        tp=cm[i][i]; fp=cm[:,i].sum()-tp; fn=cm[i,:].sum()-tp
        pr=tp/(tp+fp) if tp+fp else 0.; rc=tp/(tp+fn) if tp+fn else 0.
        f1=2*pr*rc/(pr+rc) if pr+rc else 0.
        f1s[l]={"precision":pr,"recall":rc,"f1":f1,"support":int(cm[i].sum())}
    present=[l for l in LABELS if f1s[l]["support"]>0]
    f1_macro=float(np.mean([f1s[l]["f1"] for l in present])) if present else 0.
    return acc,f1_macro,f1s,cm

# ---------- CALIBRATION : ECE + AUROC ----------
def compute_ece(confidences, correctness, n_bins=10):
    """Expected Calibration Error."""
    confidences=np.array(confidences); correctness=np.array(correctness)
    if len(confidences)==0: return 0.0
    bins=np.linspace(0,1,n_bins+1); ece=0.0; N=len(confidences)
    for i in range(n_bins):
        lo,hi=bins[i],bins[i+1]
        mask=(confidences>lo)&(confidences<=hi) if i>0 else (confidences>=lo)&(confidences<=hi)
        if mask.sum()==0: continue
        acc_bin=correctness[mask].mean()
        conf_bin=confidences[mask].mean()
        ece+=(mask.sum()/N)*abs(acc_bin-conf_bin)
    return float(ece)

def compute_auroc(confidences, correctness):
    """AUROC : la confiance discrimine-t-elle correct vs incorrect ?"""
    confidences=np.array(confidences); correctness=np.array(correctness)
    pos=confidences[correctness==1]; neg=confidences[correctness==0]
    if len(pos)==0 or len(neg)==0: return float("nan")
    # AUROC = P(conf_correct > conf_incorrect) via comptage de paires
    count=0; total=len(pos)*len(neg)
    for p in pos:
        count+=np.sum(p>neg)+0.5*np.sum(p==neg)
    return float(count/total)

def reliability_diagram(confidences, correctness, path, n_bins=10):
    confidences=np.array(confidences); correctness=np.array(correctness)
    bins=np.linspace(0,1,n_bins+1); xs=[]; ys=[]
    for i in range(n_bins):
        lo,hi=bins[i],bins[i+1]
        mask=(confidences>lo)&(confidences<=hi) if i>0 else (confidences>=lo)&(confidences<=hi)
        if mask.sum()==0: continue
        xs.append(confidences[mask].mean()); ys.append(correctness[mask].mean())
    fig,ax=plt.subplots(figsize=(6,6))
    ax.plot([0,1],[0,1],"--",color="gray",label="Calibration parfaite")
    ax.plot(xs,ys,"o-",color="#2196F3",label="Qwen2.5-VL")
    ax.set_xlabel("Confiance annoncée"); ax.set_ylabel("Exactitude réelle")
    ax.set_title("Q1 — Diagramme de fiabilité"); ax.legend(); ax.grid(alpha=.3)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()

def plot_cm(cm,cond,path):
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(cm,cmap="Blues")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(LABELS,rotation=45,ha="right"); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Prédiction"); ax.set_ylabel("Vérité")
    ax.set_title(f"Q1 — Confusion ({cond})")
    for i in range(5):
        for j in range(5):
            ax.text(j,i,cm[i][j],ha="center",va="center",color="white" if cm[i][j]>cm.max()/2 else "black")
    plt.colorbar(im); plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()

# ============================================================
# RUN — voiles plausibles uniquement
# ============================================================
samples=load_samples()
print(f"{len(samples)} images | severity {SEVERITY}")
print("Voiles plausibles :", PLAUSIBLE_VEILS)

# on stocke les prédictions par condition
by_condition = {c: [] for c in ALL_CONDITIONS}
all_records = []   # pour calibration globale

for s in tqdm(samples, desc="Q1"):
    img=cv2.imread(s["path"])
    if img is None: continue
    true=s["true_label"]
    # appliquer SEULEMENT les voiles plausibles pour cette classe
    for cond in PLAUSIBLE_VEILS[true]:
        deg=degrader.apply(img,cond)
        pil=Image.fromarray(cv2.cvtColor(deg,cv2.COLOR_BGR2RGB))
        raw=qwen_predict(pil); p=parse_response(raw)
        if p is None: pred,conf="unknown",0.0
        else: pred,conf=p["label"],p["confidence"]
        rec={"path":s["path"],"true_label":true,"condition":cond,
             "pred_label":pred,"confidence":conf,
             "correct":int(pred==true)}
        by_condition[cond].append(rec)
        all_records.append(rec)

# ---------- MÉTRIQUES PAR CONDITION ----------
summary={}
for cond in ALL_CONDITIONS:
    recs=by_condition[cond]
    if not recs: continue
    acc,f1m,f1s,cm=compute_f1(recs)
    confs=[r["confidence"] for r in recs]; corr=[r["correct"] for r in recs]
    ece=compute_ece(confs,corr); auroc=compute_auroc(confs,corr)
    plot_cm(cm,cond,f"{OUTPUT_DIR}/q1_{cond}_confusion.png")
    summary[cond]={"accuracy":acc,"f1_macro":f1m,"ece":ece,"auroc":auroc,
                   "n":len(recs),"per_class":f1s,"confusion_matrix":cm.tolist(),
                   "classes_present":[l for l in LABELS if f1s[l]["support"]>0]}
    print(f"\n{cond}: n={len(recs)} acc={acc:.3f} F1={f1m:.3f} ECE={ece:.3f} AUROC={auroc:.3f}")
    print("  classes:", summary[cond]["classes_present"])

# ---------- CALIBRATION GLOBALE ----------
confs_all=[r["confidence"] for r in all_records]; corr_all=[r["correct"] for r in all_records]
ece_all=compute_ece(confs_all,corr_all); auroc_all=compute_auroc(confs_all,corr_all)
reliability_diagram(confs_all,corr_all,f"{OUTPUT_DIR}/q1_reliability_diagram.png")

# ---------- SAUVEGARDE ----------
with open(f"{OUTPUT_DIR}/q1_all_records.json","w") as f: json.dump(all_records,f,indent=2)
with open(f"{OUTPUT_DIR}/q1_summary.json","w") as f:
    json.dump({"by_condition":summary,
               "global":{"ece":ece_all,"auroc":auroc_all,"n":len(all_records)},
               "config":{"severity":SEVERITY,"num_per_class":NUM_PER_CLASS,
                         "plausible_veils":PLAUSIBLE_VEILS}}, f, indent=2)

import shutil
shutil.make_archive("/kaggle/working/q1_results","zip","/kaggle/working/q1_results")

print("\n"+"="*60)
print("RÉCAP Q1 (voiles plausibles + calibration)")
print("="*60)
print(f"{'Condition':<10}{'n':<6}{'Acc':<8}{'F1':<8}{'ECE':<8}{'AUROC':<8}")
for cond in ALL_CONDITIONS:
    if cond in summary:
        s=summary[cond]
        print(f"{cond:<10}{s['n']:<6}{s['accuracy']:<8.3f}{s['f1_macro']:<8.3f}{s['ece']:<8.3f}{s['auroc']:<8.3f}")
print(f"\nGLOBAL   ECE={ece_all:.3f}  AUROC={auroc_all:.3f}  (n={len(all_records)})")
print("\nInterprétation :")
print("  ECE bas = bien calibré (confiance ≈ exactitude)")
print("  AUROC haut = la confiance sépare bien correct/incorrect")
print("  ⚠️ Si ECE monte sous voile = le modèle devient sur-confiant quand la scène est obscurcie")
print("\n✓ Télécharge /kaggle/working/q1_results.zip")
