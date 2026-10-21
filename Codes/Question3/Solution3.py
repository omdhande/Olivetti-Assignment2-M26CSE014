#!/usr/bin/env python3
"""Q3(a)-(d): measured bias/variance on Olivetti Faces.
Run from Codes/Question3 with: python Solution3.py
Expected files: Question1/q1_split.npz, Question1/best_tree.joblib,
Question1/final_bagged_models.joblib, Question2/q2_selected_samme.joblib.
"""
from pathlib import Path
import json, time, csv, warnings
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

HERE = Path(__file__).resolve().parent
Q1 = HERE.parent / 'Question1'
Q2 = HERE.parent / 'Question2'
SEED = 26014
R = 30
B = 100
SHALLOW_DEPTH = 2


def get_selected_boost():
    candidates = [Q2/'q2_selected_samme.joblib', Q2/'selected_samme.joblib',
                  Q2/'q2_selected_boosting.joblib']
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError('Cannot find selected Q2 SAMME model. Expected q2_selected_samme.joblib in '+str(Q2))
    try:
        model = joblib.load(path)
    except Exception as e:
        raise RuntimeError(f'Could not load {path}. The SAMME class may need to be imported from Solution2.py.') from e
    learners = getattr(model, 'models', getattr(model, 'estimators_', None))
    if learners is None or len(learners) == 0:
        raise AttributeError('Selected SAMME model must expose .models or .estimators_.')
    depth = learners[0].get_params().get('max_depth', 1)
    alphas = getattr(model, 'alphas', None)
    rounds = len(alphas) if alphas is not None else len(learners)
    return model, depth, rounds, path


class SAMME:
    """SAMME implementation using the Q2 alpha and sample-weight update."""
    def __init__(self, depth, rounds, seed):
        self.depth, self.rounds, self.seed = depth, rounds, seed
        self.models, self.alphas, self.classes_ = [], [], None
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        K, n = len(self.classes_), len(y)
        w = np.full(n, 1.0/n)
        self.models, self.alphas = [], []
        for t in range(self.rounds):
            clf = DecisionTreeClassifier(max_depth=self.depth, random_state=self.seed+t)
            clf.fit(X, y, sample_weight=w)
            pred = clf.predict(X)
            err = float(np.sum(w * (pred != y)))
            if err >= 1.0 - 1.0/K:
                if not self.models:
                    self.models.append(clf); self.alphas.append(1.0)
                break
            err = max(err, 1e-12)
            alpha = np.log((1-err)/err) + np.log(K-1)
            self.models.append(clf); self.alphas.append(float(alpha))
            w *= np.exp(alpha * (pred != y))
            w /= w.sum()
            if err <= 1e-12: break
        return self
    def predict_proba(self, X):
        scores = np.zeros((len(X), len(self.classes_)))
        cols = {int(c): i for i,c in enumerate(self.classes_)}
        alphas = np.asarray(self.alphas, float)
        total = alphas.sum()
        if total <= 0: alphas = np.ones(len(self.models)); total = len(self.models)
        for clf, a in zip(self.models, alphas):
            for i, c in enumerate(clf.predict(X)): scores[i, cols[int(c)]] += a
        return scores / total
    def predict(self, X): return self.classes_[self.predict_proba(X).argmax(axis=1)]


def onehot(y, classes):
    m = {int(c):i for i,c in enumerate(classes)}
    out = np.zeros((len(y),len(classes)))
    for i,c in enumerate(y): out[i,m[int(c)]] = 1
    return out


def tree_probs(model, X, classes):
    raw = model.predict_proba(X)
    out = np.zeros((len(X),len(classes)))
    cols = {int(c):i for i,c in enumerate(classes)}
    for j,c in enumerate(model.classes_): out[:,cols[int(c)]] = raw[:,j]
    return out


def bag_probs(models, X, classes):
    return np.mean([tree_probs(m,X,classes) for m in models], axis=0)


def boost_probs(model, X, classes):
    # SAMME confidence is the normalized weighted vote q_k = S_k / sum(alpha).
    raw = np.asarray(model.predict_proba(X), float)
    model_classes = getattr(model, 'classes_', classes)
    if raw.shape[1] == len(classes) and np.array_equal(model_classes, classes): return raw
    out = np.zeros((len(X),len(classes)))
    cols = {int(c):i for i,c in enumerate(classes)}
    for j,c in enumerate(model_classes): out[:,cols[int(c)]] = raw[:,j]
    sums = out.sum(axis=1,keepdims=True)
    return np.divide(out,sums,out=np.zeros_like(out),where=sums>0)


def metrics(prob_reps, y_test, classes):
    p = np.asarray(prob_reps)
    meanp = p.mean(axis=0)
    target = onehot(y_test, classes)
    bias2 = float(np.mean(np.sum((meanp-target)**2,axis=1)))
    variance = float(np.mean(np.sum((p-meanp[None,:,:])**2,axis=2)))
    pred = classes[p.argmax(axis=2)]
    acc = np.array([accuracy_score(y_test, pred[r]) for r in range(len(p))])
    return {'bias_squared':bias2, 'variance':variance,
            'test_accuracy_mean':float(acc.mean()),
            'test_accuracy_std':float(acc.std(ddof=1)),
            'test_accuracy_repetitions':acc.tolist()}


def resample_by_identity(X, y, rng):
    # Six draws with replacement from each identity's six original training faces.
    xs, ys = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        take = rng.choice(idx, size=6, replace=True)
        xs.append(X[take]); ys.append(y[take])
    return np.concatenate(xs), np.concatenate(ys)


def fit_bag(X, y, depth, min_leaf, ntrees, seed):
    rng = np.random.default_rng(seed)
    models=[]; n=len(y)
    for b in range(ntrees):
        idx=rng.choice(n,size=n,replace=True)
        clf=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=min_leaf,random_state=seed+b+1)
        clf.fit(X[idx],y[idx]); models.append(clf)
    return models


def fit_tree_like(template, X, y, seed):
    params=template.get_params().copy(); params['random_state']=seed
    clf=DecisionTreeClassifier(**params); clf.fit(X,y); return clf


def main():
    start=time.time()
    data=fetch_olivetti_faces(); X,y=data.data,data.target; classes=np.unique(y)
    split_path=Q1/'q1_split.npz'; tree_path=Q1/'best_tree.joblib'
    if not split_path.exists() or not tree_path.exists():
        raise FileNotFoundError('Run Question1/Solution1.py first; need q1_split.npz and best_tree.joblib.')
    split=np.load(split_path); train_idx=split['train_idx']; test_idx=split['test_idx']
    Xtr,ytr=X[train_idx],y[train_idx]; Xte,yte=X[test_idx],y[test_idx]
    selected_tree=joblib.load(tree_path)
    selected_depth=selected_tree.get_params().get('max_depth')
    min_leaf=selected_tree.get_params().get('min_samples_leaf',1)
    selected_boost, boost_depth, boost_rounds, boost_path=get_selected_boost()
    print('Dataset:',X.shape,' train:',Xtr.shape,' fixed test:',Xte.shape)
    print('Selected tree depth/min leaf:',selected_depth,min_leaf)
    print('Selected boosting depth/rounds:',boost_depth,boost_rounds,' from ',boost_path)
    print(f'Running {R} resamples; Q3(b) uses {B} trees per bagged ensemble.')

    masked_img=Xte.reshape(-1,64,64).copy(); masked_img[:,24:40,24:40]=0
    Xmasked=masked_img.reshape(len(Xte),-1)
    q3a={k:[] for k in ('single_tree','bagging','boosting')}
    q3b_single={}; q3b_bag={}; q3b_acc={}
    depth_cfg=[('shallow_depth_2',SHALLOW_DEPTH),('selected_depth',selected_depth),('deep_depth_unrestricted',None)]
    for name,_ in depth_cfg:
        q3b_single[name]=[]; q3b_bag[name]=[]
    rng=np.random.default_rng(SEED+3000)

    for r in range(R):
        Xr,yr=resample_by_identity(Xtr,ytr,rng); rs=SEED+10000+r*1000
        # Q3(a), selected single tree, 100-tree bagging, selected SAMME settings.
        st=fit_tree_like(selected_tree,Xr,yr,rs)
        q3a['single_tree'].append(tree_probs(st,Xte,classes))
        bag=fit_bag(Xr,yr,selected_depth,min_leaf,B,rs+100)
        q3a['bagging'].append(bag_probs(bag,Xte,classes))
        boost=SAMME(boost_depth,boost_rounds,rs+500).fit(Xr,yr)
        q3a['boosting'].append(boost_probs(boost,Xte,classes))
        # Q3(b): single tree and 100-tree bagging at shallow, selected, deep depths.
        for name,depth in depth_cfg:
            if name=='selected_depth':
                st_d, bag_d=st,bag
            else:
                st_d=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=min_leaf,random_state=rs+(600 if name.startswith('shallow') else 700))
                st_d.fit(Xr,yr)
                bag_d=fit_bag(Xr,yr,depth,min_leaf,B,rs+(800 if name.startswith('shallow') else 900))
            q3b_single[name].append(tree_probs(st_d,Xte,classes))
            q3b_bag[name].append(bag_probs(bag_d,Xte,classes))
        print(f'Resample {r+1:02d}/{R} complete; elapsed {(time.time()-start)/60:.1f} min')

    q3a_results={k:metrics(v,yte,classes) for k,v in q3a.items()}
    print('\n========== Q3(a): SELECTED MODELS ==========')
    for name,m in q3a_results.items():
        print(f"{name:12s} bias^2={m['bias_squared']:.6f} variance={m['variance']:.6f} "
              f"test accuracy mean={m['test_accuracy_mean']:.4f} std={m['test_accuracy_std']:.4f}")

    q3b_results={}
    print('\n========== Q3(b): DEPTH EFFECT ==========')
    for name,depth in depth_cfg:
        ms=metrics(q3b_single[name],yte,classes); mb=metrics(q3b_bag[name],yte,classes)
        q3b_results[name]={'max_depth':depth,'single_tree':ms,'bagging_100':mb,
          'variance_reduction':ms['variance']-mb['variance'],
          'accuracy_improvement':mb['test_accuracy_mean']-ms['test_accuracy_mean']}
        print(f"{name} (depth={depth}): single bias²={ms['bias_squared']:.6f}, var={ms['variance']:.6f}, "
              f"acc={ms['test_accuracy_mean']:.4f}±{ms['test_accuracy_std']:.4f}")
        print(f"  bagging-100: bias²={mb['bias_squared']:.6f}, var={mb['variance']:.6f}, "
              f"acc={mb['test_accuracy_mean']:.4f}±{mb['test_accuracy_std']:.4f}, "
              f"variance reduction={ms['variance']-mb['variance']:.6f}")

    print('\n========== Q3(c): MEASURED CONCLUSION ==========')
    best_improve=max(q3b_results,key=lambda k:q3b_results[k]['accuracy_improvement'])
    for name,v in q3b_results.items():
        print(f"{name}: bagging accuracy improvement={v['accuracy_improvement']:+.4f}; "
              f"variance reduction={v['variance_reduction']:+.6f}")
    print('Largest measured accuracy improvement from bagging:',best_improve)
    print('Use these measured results (and the leaf occupancy from Q1(a)) to discuss whether deepest trees benefit most; do not assume they do.')

    # Q3(d): evaluate frozen original selected models on clean and masked test sets.
    bag_path=Q1/'final_bagged_models.joblib'
    if bag_path.exists():
        try: original_bag=joblib.load(bag_path)
        except Exception as e:
            warnings.warn(f'Cannot load {bag_path}: {e}; refitting with frozen Q1 settings.')
            original_bag=fit_bag(Xtr,ytr,selected_depth,min_leaf,B,SEED+77)
    else:
        original_bag=fit_bag(Xtr,ytr,selected_depth,min_leaf,B,SEED+77)
    q3d={}
    print('\n========== Q3(d): CLEAN VS MASKED TEST ==========')
    for split_name,Xeval in [('clean',Xte),('masked',Xmasked)]:
        q3d[split_name]={}
        for name,kind in [('single_tree','tree'),('bagging','bag'),('boosting','boost')]:
            if kind=='tree': p=tree_probs(selected_tree,Xeval,classes)
            elif kind=='bag': p=bag_probs(original_bag,Xeval,classes)
            else: p=boost_probs(selected_boost,Xeval,classes)
            pred=classes[p.argmax(axis=1)]
            q3d[split_name][name]={'accuracy':float(accuracy_score(yte,pred)),
                                   'mean_winning_probability':float(np.mean(p.max(axis=1)))}
            print(f"{split_name:6s} | {name:12s} accuracy={q3d[split_name][name]['accuracy']:.4f} "
                  f"mean winning probability={q3d[split_name][name]['mean_winning_probability']:.4f}")
    print('\nModel          Clean acc.  Masked acc.  Clean mean p  Masked mean p')
    for name in ('single_tree','bagging','boosting'):
        a=q3d['clean'][name]; b=q3d['masked'][name]
        print(f"{name:14s}{a['accuracy']:10.4f}{b['accuracy']:13.4f}{a['mean_winning_probability']:14.4f}{b['mean_winning_probability']:15.4f}")

    # Plot Q3(a) measured bias/variance and accuracy mean +/- std.
    names=['Single tree','Bagging','Boosting']; keys=['single_tree','bagging','boosting']
    fig,ax=plt.subplots(1,3,figsize=(14,4.5))
    ax[0].bar(names,[q3a_results[k]['bias_squared'] for k in keys]); ax[0].set_title('Measured bias squared'); ax[0].grid(axis='y',alpha=.25)
    ax[1].bar(names,[q3a_results[k]['variance'] for k in keys]); ax[1].set_title('Measured variance'); ax[1].grid(axis='y',alpha=.25)
    ax[2].bar(names,[q3a_results[k]['test_accuracy_mean'] for k in keys],yerr=[q3a_results[k]['test_accuracy_std'] for k in keys],capsize=4); ax[2].set_title('Test accuracy over 30 resamples'); ax[2].grid(axis='y',alpha=.25)
    for a in ax: a.tick_params(axis='x',rotation=15)
    fig.tight_layout(); fig.savefig(HERE/'q3_bias_variance.png',dpi=250,bbox_inches='tight'); plt.close(fig)

    # JSON summary without large arrays.
    summary={'seed':SEED,'repetitions':R,'bagging_trees':B,
      'selected_tree':{'max_depth':selected_depth,'min_samples_leaf':min_leaf},
      'selected_boosting':{'depth':boost_depth,'rounds':boost_rounds,'file':str(boost_path)},
      'q3a':q3a_results,'q3b':q3b_results,'q3d':q3d}
    with (HERE/'q3_results.json').open('w') as f: json.dump(summary,f,indent=2)
    with (HERE/'q3_summary.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['section','model_or_depth','bias_squared','variance','test_accuracy_mean','test_accuracy_std'])
        for name,m in q3a_results.items(): writer.writerow(['Q3(a)',name,m['bias_squared'],m['variance'],m['test_accuracy_mean'],m['test_accuracy_std']])
        for name,v in q3b_results.items():
            for sub in ('single_tree','bagging_100'):
                m=v[sub]; writer.writerow(['Q3(b)',name+':'+sub,m['bias_squared'],m['variance'],m['test_accuracy_mean'],m['test_accuracy_std']])
    print('\nSaved q3_results.json, q3_summary.csv, q3_bias_variance.png in',HERE)
    print(f'All Q3 sections completed in {(time.time()-start)/60:.2f} minutes.')

if __name__=='__main__': main()
