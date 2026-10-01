"""Controlled active mechanism identification. No LLM judge or hidden-law policy input.

Statistical claim applies ONLY to the declared finite catalog and specified Gaussian
observation model. This is a research pilot, not unrestricted equation discovery.
"""
from __future__ import annotations
import argparse, dataclasses, hashlib, itertools, json, math, os, platform, time
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import torch

VERSION = '1.0.0'
METHODS = ('random_safe', 'variance_safe', 'eig_safe', 'chernoff_safe',
           'deficit_safe', 'eig_posterior', 'deficit_posterior', 'eig_fixed')

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()

def seed_for(*parts):
    return int(digest(parts)[:15], 16) % (2**32-1)

def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, path)

@dataclass(frozen=True)
class Config:
    preset: str = 'pilot'
    seed: int = 20261001
    budget: int = 60
    alpha: float = .05
    batch: int = 32
    eig_samples: int = 12
    game_iterations: int = 32
    device: str = 'auto'
    trace_per_group: int = 2
    max_minutes: float = 90
    methods: tuple = METHODS

def settings(preset):
    return {
        'smoke': dict(repeats=1, per_family=1, noises=(.15,), stress=False),
        'pilot': dict(repeats=8, per_family=2, noises=(.08,.20), stress=True),
        'confirmatory': dict(repeats=100, per_family=2, noises=(.08,.20), stress=True),
    }[preset]

def value(domain, family, a, b, x, z):
    """Works with exact rational numbers as well as float inputs."""
    if domain == 'mechanics':
        return {
            'linear': lambda: a*x,
            'additive': lambda: a*x+b*z,
            'cubic': lambda: a*x+b*x**3,
            'interaction': lambda: a*x+b*x*z,
            'squared': lambda: a*x*x+b*z*z,
            'mixed': lambda: a*x*x+b*z,
        }[family]()
    return {
        'michaelis_menten': lambda: a*x/(b+x),
        'competitive': lambda: a*x/(b*(1+z)+x),
        'uncompetitive': lambda: a*x/(b+x*(1+z)),
        'noncompetitive': lambda: a*x/((b+x)*(1+z)),
        'substrate_inhibition': lambda: a*x/(b+x+x*x),
        'hill2': lambda: a*x*x/(b*b+x*x),
    }[family]()

def expression(domain, family, a, b):
    forms = {
        'linear': '{a}*x', 'additive': '{a}*x+{b}*z',
        'cubic': '{a}*x+{b}*x**3', 'interaction': '{a}*x+{b}*x*z',
        'squared': '{a}*x**2+{b}*z**2', 'mixed': '{a}*x**2+{b}*z',
        'michaelis_menten': '{a}*x/({b}+x)',
        'competitive': '{a}*x/({b}*(1+z)+x)',
        'uncompetitive': '{a}*x/({b}+x*(1+z))',
        'noncompetitive': '{a}*x/(({b}+x)*(1+z))',
        'substrate_inhibition': '{a}*x/({b}+x+x**2)',
        'hill2': '{a}*x**2/({b}**2+x**2)',
    }
    return forms[family].format(a=f'({a})',b=f'({b})')

def make_catalog(domain, regime='full', remove=None):
    families = (('linear','additive','cubic','interaction','squared','mixed')
                if domain == 'mechanics' else ('michaelis_menten','competitive',
                'uncompetitive','noncompetitive','substrate_inhibition','hill2'))
    axis = [F(-2),F(-1),F(-1,2),F(1,2),F(1),F(2)] if domain=='mechanics' else [F(1,4),F(1,2),F(1),F(2),F(4)]
    zs = axis if domain=='mechanics' else [F(0),F(1,4),F(1,2),F(1),F(2)]
    full_actions = list(itertools.product(axis,zs))
    actions = [p for p in full_actions if regime=='full' or
               (p[1]==p[0] if domain=='mechanics' else p[1]==0)]
    models=[]; exact=[]
    for family in families:
        if family == remove: continue
        for a,b in itertools.product((F(1,2),F(1),F(2)), repeat=2):
            if family=='linear' and b!=1: continue  # remove unused-parameter duplicates
            models.append(dict(family=family,a=str(a),b=str(b),formula=expression(domain,family,a,b)))
            exact.append(tuple(value(domain,family,a,b,x,z) for x,z in actions))
    # Exact equivalence on the COMPLETE, publicly specified finite intervention menu.
    # No tolerance-based numerical clustering, and no assertion beyond this menu.
    signatures={}; groups=[]
    for sig in exact:
        if sig not in signatures: signatures[sig]=len(signatures)
        groups.append(signatures[sig])
    fnames=sorted({m['family'] for m in models})
    family=np.array([fnames.index(m['family']) for m in models])
    prior=np.array([1/(len(fnames)*sum(family==f)) for f in family])
    costs=np.array([1 if (z==x if domain=='mechanics' else z==0) else 3 for x,z in actions])
    # Public, fixed units; never normalize by hidden ground-truth coefficients.
    scale=4.0 if domain=='mechanics' else 1.0
    mu=np.array([[float(v)/scale for v in row] for row in exact])
    return dict(domain=domain,regime=regime,models=models,actions=actions,
                exact=exact,mu=mu,family=family,fnames=fnames,groups=np.array(groups),
                prior=prior,costs=costs,scale=scale)

def build_cases(cfg):
    s=settings(cfg.preset); cases=[]
    for domain in ('mechanics','kinetics'):
        full=make_catalog(domain)
        selected=[]
        for fam in full['fnames']:
            ids=[i for i,m in enumerate(full['models']) if m['family']==fam]
            rng=np.random.default_rng(seed_for(cfg.seed,domain,fam,'selection'))
            selected.extend(rng.choice(ids,size=min(s['per_family'],len(ids)),replace=False).tolist())
        contexts=[(r,'closed') for r in ('full','restricted')]
        if s['stress']:
            contexts += [('full','off_grid'),('full','noise_mismatch'),('full','missing_family')]
        for regime,scenario in contexts:
            for noise in s['noises']:
                for truth_index in selected:
                    model=full['models'][truth_index]
                    for rep in range(s['repeats']):
                        base=dict(domain=domain,regime=regime,scenario=scenario,
                                  noise=noise,truth_index=truth_index,rep=rep,seed=cfg.seed)
                        base['id']=digest(base)[:24]
                        base['cluster']=f'{domain}:{truth_index}'
                        base['truth_family']=model['family']
                        cases.append(base)
    return cases

def case_group(case):
    return (case['domain'],case['regime'],case['scenario'],case['noise'],
            case['truth_family'] if case['scenario']=='missing_family' else '')

def prepare_group(cases):
    c=cases[0]; remove=c['truth_family'] if c['scenario']=='missing_family' else None
    bank=make_catalog(c['domain'],c['regime'],remove)
    full=make_catalog(c['domain'])
    truth=[]; signatures=[]
    for case in cases:
        m=full['models'][case['truth_index']]; a,b=F(m['a']),F(m['b'])
        if case['scenario']=='off_grid': a*=F(113,100); b*=F(107,100)
        sig=tuple(value(c['domain'],m['family'],a,b,x,z) for x,z in bank['actions'])
        signatures.append(sig); truth.append([float(v)/bank['scale'] for v in sig])
    bank['noise']=float(c['noise'])
    return bank,np.array(truth),signatures

def get_device(name='auto'):
    return torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if name=='auto' else name)

def tensors(bank,device):
    f=lambda x: torch.as_tensor(x,device=device,dtype=torch.float64)
    return dict(mu=f(bank['mu']),prior=f(bank['prior']),cost=f(bank['costs']),
                family=torch.as_tensor(bank['family'],device=device),
                groups=torch.as_tensor(bank['groups'],device=device))

def update_evidence(ll, max_e, increment, prior):
    ll=ll+increment
    log_q=torch.logsumexp(ll+prior.log(),dim=-1)
    log_e=log_q[:,None]-ll
    return ll,torch.maximum(max_e,log_e)

def partition_mass(p,labels):
    out=torch.zeros((len(p),int(labels.max())+1),dtype=p.dtype,device=p.device)
    out.scatter_add_(1,labels.expand(len(p),-1),p)
    return out

def choose_action(policy,ll,max_e,t,noise,allowed,uniform,mc_u,mc_z,cfg):
    """Only public candidates + past observations enter this function."""
    mu,cost,prior=t['mu'],t['cost'],t['prior']
    p=torch.softmax(ll+prior.log(),dim=-1)
    b,m=ll.shape; a=mu.shape[1]
    if policy=='random':
        score=torch.zeros_like(allowed,dtype=mu.dtype)
    elif policy=='variance':
        score=((p @ mu.square())-(p @ mu).square()).clamp_min(0)/cost
    elif policy=='eig':
        # Monte Carlo mutual information about catalog atom, not claimed to be
        # a reproduction of an external paper's Bayesian design implementation.
        idx=torch.searchsorted(p.cumsum(-1).contiguous(),mc_u.contiguous()).clamp_max(m-1)
        ys=mu[idx]+noise*mc_z[:,:,None]
        logp=-.5*((ys[:,:,:,None]-mu.T[None,None,:,:])/noise).square()
        mix=torch.logsumexp(logp+p.clamp_min(1e-300).log()[:,None,None,:],dim=-1)
        score=(-.5*mc_z[:,:,None].square()-mix).mean(1)/cost
    elif policy in ('chernoff','deficit'):
        leader=(ll+prior.log()).argmax(-1)
        d=.5*((mu[leader,None,:]-mu[None,:,:])/noise).square()
        # Do not try to separate indistinguishable atoms on this intervention menu.
        rivals=t['groups'][None,:]!=t['groups'][leader,None]
        if policy=='deficit':
            rivals &= max_e < math.log(1/cfg.alpha)
            gap=(math.log(1/cfg.alpha)-(torch.logsumexp(ll+prior.log(),-1)[:,None]-ll)).clamp_min(.25)
            d=d/gap[:,:,None]
        # Multiplicative-weights zero-sum game: maximize worst-rival separation
        # per unit experimental cost. Approximate mixed design, no optimality claim.
        d=d/cost[None,None,:]
        scale=d.amax((1,2),keepdim=True).clamp_min(1e-12)
        game=d/scale
        rival_weight=rivals.to(mu.dtype)
        allocation=torch.zeros((b,a),dtype=mu.dtype,device=mu.device)
        for _ in range(cfg.game_iterations):
            q=rival_weight/rival_weight.sum(-1,keepdim=True).clamp_min(1e-30)
            scores=torch.einsum('bm,bma->ba',q,game).masked_fill(~allowed,-torch.inf)
            best=scores.argmax(-1)
            allocation.scatter_add_(1,best[:,None],torch.ones((b,1),dtype=mu.dtype,device=mu.device))
            gain=game.gather(2,best[:,None,None].expand(-1,m,1)).squeeze(-1)
            rival_weight *= torch.exp(-2*gain)
        # allocation describes spending fractions; convert to action frequencies.
        probs=(allocation/cost)*allowed
        probs=probs/probs.sum(-1,keepdim=True).clamp_min(1e-30)
        fallback=allowed.to(mu.dtype)/allowed.sum(-1,keepdim=True).clamp_min(1)
        probs=torch.where(rivals.any(-1)[:,None],probs,fallback)
        probs=.98*probs+.02*fallback  # declared exploration, shared for both policies
        return torch.searchsorted(probs.cumsum(-1).contiguous(),uniform[:,None].contiguous()).squeeze(-1).clamp_max(a-1)
    else: raise ValueError(policy)
    score=score.masked_fill(~allowed,-torch.inf)
    # Random tie-breaking; separate policy randomness from observation noise.
    ties=score==score.max(-1,keepdim=True).values
    probs=ties.to(mu.dtype)/ties.sum(-1,keepdim=True)
    return torch.searchsorted(probs.cumsum(-1).contiguous(),uniform[:,None].contiguous()).squeeze(-1).clamp_max(a-1)

def declaration(ll,max_e,t,cfg,rule):
    b=len(ll); device=ll.device
    kind=torch.zeros(b,dtype=torch.long,device=device) # 0 unresolved,1 family,2 class,3 empty
    answer=torch.full((b,),-1,dtype=torch.long,device=device)
    if rule=='safe':
        alive=max_e < math.log(1/cfg.alpha)
        families=partition_mass(alive.to(ll.dtype),t['family'])>0
        groups=partition_mass(alive.to(ll.dtype),t['groups'])>0
        single_family=families.sum(-1)==1
        single_group=(groups.sum(-1)==1)&~single_family
        kind[single_family]=1; answer[single_family]=families.long().argmax(-1)[single_family]
        kind[single_group]=2; answer[single_group]=groups.long().argmax(-1)[single_group]
        kind[~alive.any(-1)]=3
    else:
        p=torch.softmax(ll+t['prior'].log(),dim=-1)
        fm=partition_mass(p,t['family']); gm=partition_mass(p,t['groups'])
        sf=fm.max(-1).values >= 1-cfg.alpha
        sg=(gm.max(-1).values >= 1-cfg.alpha)&~sf
        kind[sf]=1; answer[sf]=fm.argmax(-1)[sf]
        kind[sg]=2; answer[sg]=gm.argmax(-1)[sg]
    return kind,answer

def run_batch(cases,method,cfg,advisor=None):
    bank,truth,signatures=prepare_group(cases)
    device=get_device(cfg.device); t=tensors(bank,device); b,m=len(cases),len(bank['models'])
    policy,rule=method.rsplit('_',1)
    ll=torch.zeros((b,m),device=device,dtype=torch.float64); max_e=torch.zeros_like(ll)
    cost=torch.zeros(b,device=device,dtype=torch.float64); steps=torch.zeros(b,device=device,dtype=torch.long)
    kind=torch.zeros_like(steps); answer=torch.full_like(steps,-1)
    stopped=torch.zeros(b,device=device,dtype=torch.bool)
    noise=bank['noise']; truth_t=torch.tensor(truth,device=device,dtype=torch.float64)
    rows_idx=torch.arange(b,device=device)
    # Random tapes are keyed by episode, so chunk size / resume cannot change draws.
    eps=np.stack([np.random.default_rng(seed_for(c['id'],'noise')).normal(size=cfg.budget) for c in cases])
    u=np.stack([np.random.default_rng(seed_for(c['id'],policy,'policy')).random(size=cfg.budget) for c in cases])
    mc_u=np.stack([np.random.default_rng(seed_for(c['id'],'eig_u')).random((cfg.budget,cfg.eig_samples)) for c in cases])
    mc_z=np.stack([np.random.default_rng(seed_for(c['id'],'eig_z')).normal(size=(cfg.budget,cfg.eig_samples)) for c in cases])
    eps,u,mc_u,mc_z=[torch.tensor(z,device=device,dtype=torch.float64) for z in (eps,u,mc_u,mc_z)]
    elapsed0=time.perf_counter(); traces=[]; history=[]; invalid_llm=0
    true_matches=torch.tensor([[sig==s for s in bank['exact']] for sig in signatures],device=device)
    lost_truth=torch.zeros(b,dtype=torch.bool,device=device)
    noise_factor=2 if cases[0]['scenario']=='noise_mismatch' else 1
    for step in range(cfg.budget):
        allowed=t['cost'][None,:] <= (cfg.budget-cost[:,None])
        active=~stopped & allowed.any(-1)
        if not active.any(): break
        allowed[~active,0]=True
        if policy=='llm':
            if b!=1 or advisor is None: raise ValueError('LLM panel requires single episode and advisor')
            fallback=int(choose_action('eig',ll,max_e,t,noise,allowed,u[:,step],mc_u[:,step],mc_z[:,step],cfg)[0])
            act,valid=advisor.choose(bank,ll[0].detach().cpu().numpy(),history,
                                      allowed[0].cpu().numpy(),fallback)
            invalid_llm+=int(not valid); action=torch.tensor([act],device=device)
        else:
            action=choose_action(policy,ll,max_e,t,noise,allowed,u[:,step],mc_u[:,step],mc_z[:,step],cfg)
        means=t['mu'][:,action].T
        y=truth_t[rows_idx,action]+noise*noise_factor*eps[:,step]
        inc=-.5*((y[:,None]-means)/noise).square() # identical normalizing constants cancel
        ll_new,emax_new=update_evidence(ll,max_e,inc,t['prior'])
        ll=torch.where(active[:,None],ll_new,ll); max_e=torch.where(active[:,None],emax_new,max_e)
        cost+=active*t['cost'][action]; steps+=active
        # Used ONLY for evaluation, never supplied to choose_action / advisor.
        lost_truth |= active & true_matches.any(-1) & ~((max_e < math.log(1/cfg.alpha)) & true_matches).any(-1)
        if rule!='fixed':
            new_kind,new_answer=declaration(ll,max_e,t,cfg,rule)
            done=active&(new_kind>0)
            kind[done]=new_kind[done]; answer[done]=new_answer[done]; stopped|=done
        if policy=='llm': history.append(dict(action=int(action[0]),y=float(y[0])))
        for i in range(min(cfg.trace_per_group,b)):
            if bool(active[i]): traces.append(dict(id=cases[i]['id'],method=method,step=step+1,
                action=int(action[i]),y=float(y[i]),cost=float(cost[i]),
                survivors=int((max_e[i]<math.log(1/cfg.alpha)).sum()),decision=int(kind[i])))
    if rule=='fixed':
        p=torch.softmax(ll+t['prior'].log(),-1)
        kind[:]=1; answer=partition_mass(p,t['family']).argmax(-1)
    if device.type=='cuda': torch.cuda.synchronize()
    wall=time.perf_counter()-elapsed0
    out=[]
    for i,c in enumerate(cases):
        k=int(kind[i]); ans=int(answer[i]); correct=False; description=None
        if k==1:
            description=bank['fnames'][ans]; correct=description==c['truth_family']
        elif k==2:
            ids=np.flatnonzero(bank['groups']==ans)
            description=sorted({bank['models'][j]['family'] for j in ids})
            correct=signatures[i]==bank['exact'][ids[0]]
        resolved=k in (1,2)
        out.append(dict(**c,method=method,decision={0:'unresolved',1:'family',2:'equivalence',3:'empty_catalog'}[k],
            answer=description,resolved=resolved,correct=bool(correct),false_declaration=bool(resolved and not correct),
            cost=float(cost[i]),queries=int(steps[i]),guarantee_applicable=c['scenario']=='closed',
            true_mean_in_catalog=bool(true_matches[i].any()),true_mean_eliminated=bool(lost_truth[i]),
            surviving_atoms=int((max_e[i]<math.log(1/cfg.alpha)).sum()),
            batch_seconds=wall,batch_size=b,amortized_seconds=wall/b,
            llm_invalid_responses=invalid_llm if policy=='llm' else 0))
    return out,traces

def environment():
    return dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                vram_gib=round(torch.cuda.get_device_properties(0).total_memory/2**30,2) if torch.cuda.is_available() else None)

def load_rows(outdir):
    rows=[]
    for p in sorted((Path(outdir)/'episodes').glob('*.json')):
        data=json.loads(p.read_text())
        rows.extend(data if isinstance(data,list) else [data])
    keys=[(r['id'],r['method']) for r in rows]
    if len(keys)!=len(set(keys)): raise ValueError('Episódios duplicados nos checkpoints.')
    return rows

def run_suite(cfg,outdir,checkpoint=None):
    if not 0<cfg.alpha<1 or cfg.budget<1 or cfg.batch<1:
        raise ValueError('Alpha, orçamento ou batch inválido.')
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    spec=dataclasses.asdict(cfg)
    # Operational settings may change across resumptions without changing experiment.
    for k in ('batch','device','max_minutes','trace_per_group'): spec.pop(k)
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    run_hash=digest(dict(spec=spec,source=source_sha))
    manifest_path=outdir/'manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old['run_hash']!=run_hash: raise ValueError('Configuração/código diferente. Escolha outra pasta de execução.')
    else:
        atomic_json(manifest_path,dict(run_hash=run_hash,config=dataclasses.asdict(cfg),
            source_sha256=source_sha,environment=environment(),version=VERSION,created_unix=time.time()))
    groups={}
    for c in build_cases(cfg): groups.setdefault(case_group(c),[]).append(c)
    total=sum(map(len,groups.values()))*len(cfg.methods)
    existing={(r['id'],r['method']) for r in load_rows(outdir)}
    start=time.monotonic(); done=0; completed=True
    for key,cases in groups.items():
        for method in cfg.methods:
            # Atomic shards avoid thousands of tiny writes to mounted storage.
            pending=[c for c in cases if (c['id'],method) not in existing]
            done+=len(cases)-len(pending)
            for pos in range(0,len(pending),cfg.batch):
                if cfg.max_minutes and time.monotonic()-start>cfg.max_minutes*60:
                    completed=False; break
                chunk=pending[pos:pos+cfg.batch]
                result,traces=run_batch(chunk,method,cfg)
                shard=digest([method,[c['id'] for c in chunk]])[:24]
                atomic_json(outdir/'episodes'/f'{shard}.json',result)
                existing.update((r['id'],r['method']) for r in result)
                if traces:
                    atomic_json(outdir/'traces'/f"{chunk[0]['id']}_{method}.json",traces)
                done+=len(chunk)
                print(f'{done}/{total} | {key[0]} {key[1]} {key[2]} | {method}',flush=True)
                if checkpoint: checkpoint(outdir)
            if not completed: break
        if not completed: break
    atomic_json(outdir/'status.json',dict(completed=completed,expected_episodes=total,
        completed_episodes=len(existing),environment=environment()))
    if checkpoint: checkpoint(outdir,force=True)
    print('Concluído.' if completed else 'Pausa com checkpoint. Execute esta célula novamente para continuar.')
    return outdir

def summarize(outdir):
    import pandas as pd
    from scipy.stats import beta
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outdir=Path(outdir); rows=load_rows(outdir)
    if not rows: raise ValueError('Nenhum episódio salvo.')
    df=pd.DataFrame(rows)
    df.to_csv(outdir/'episodes.csv',index=False)
    keys=['domain','regime','scenario','noise','method']
    summary=df.groupby(keys).agg(n=('id','size'),resolution=('resolved','mean'),
        correct_resolution=('correct','mean'),false_declarations=('false_declaration','mean'),
        mean_cost=('cost','mean'),mean_queries=('queries','mean'),
        truth_elimination=('true_mean_eliminated','mean')).reset_index()
    summary.to_csv(outdir/'summary.csv',index=False)
    # Exact binomial intervals per fixed truth and context, independent noise seeds.
    risk=[]
    for key,g in df.groupby(keys+['truth_index']):
        n=len(g); k=int(g.false_declaration.sum())
        risk.append(dict(zip(keys+['truth_index'],key),n=n,errors=k,
            rate=k/n,lower_95=0.0 if k==0 else float(beta.ppf(.025,k,n-k+1)),
            upper_95=1.0 if k==n else float(beta.ppf(.975,k+1,n-k))))
    pd.DataFrame(risk).to_csv(outdir/'risk_by_instance.csv',index=False)
    # Paired, law-cluster bootstrap: pairing by episode id, clustering by law.
    paired=[]; baseline='eig_safe'
    for scenario in sorted(df.scenario.unique()):
        sub=df[df.scenario==scenario]
        for metric in ['false_declaration','correct','cost']:
            wide=sub.pivot(index=['id','cluster'],columns='method',values=metric)
            if baseline not in wide: continue
            for method in wide.columns:
                if method==baseline: continue
                x=wide[[baseline,method]].dropna().astype(float)
                if x.empty: continue
                delta=(x[method]-x[baseline]).groupby(level='cluster').mean().to_numpy()
                rng=np.random.default_rng(seed_for(scenario,metric,method,'bootstrap'))
                samples=rng.choice(delta,size=(2000,len(delta)),replace=True).mean(-1)
                paired.append(dict(scenario=scenario,metric=metric,method=method,baseline=baseline,
                    paired_episodes=len(x),law_clusters=len(delta),difference=float(delta.mean()),
                    lower_95=float(np.quantile(samples,.025)),upper_95=float(np.quantile(samples,.975))))
    pd.DataFrame(paired).to_csv(outdir/'paired_comparisons.csv',index=False)
    closed=df[df.scenario=='closed']
    fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    agg=closed.groupby('method')[['false_declaration','correct','cost']].mean().sort_index()
    if len(agg):
        ax[0].barh(agg.index,agg.false_declaration)
        alpha=json.loads((outdir/'manifest.json').read_text())['config']['alpha']
        ax[0].axvline(alpha,color='red',ls='--',label=f'nominal alpha={alpha}')
        ax[0].set_xlabel('False declarations / all episodes'); ax[0].legend()
        for method,g in closed.groupby('method'):
            xs=np.arange(0,int(g.cost.max())+1)
            ax[1].plot(xs,[((g.correct)&(g.cost<=x)).mean() for x in xs],label=method)
        ax[1].set(xlabel='Experimental cost',ylabel='Correct resolution / all episodes',ylim=(0,1))
        ax[1].legend(fontsize=7)
    fig.suptitle('Finite catalog, specified noise — exploratory aggregates')
    fig.savefig(outdir/'main_results.png',dpi=180); fig.savefig(outdir/'main_results.pdf'); plt.close(fig)
    status=json.loads((outdir/'status.json').read_text()) if (outdir/'status.json').exists() else {}
    text=(f'# Relatório da execução\n\nEpisódios disponíveis: {len(df)}. '
          f'Execução completa: {status.get("completed",False)}.\n\n'
          'Os resultados medem identificação em um catálogo finito. Não demonstram descoberta irrestrita de equações. '
          'A garantia é aplicável somente ao cenário closed, com o ruído especificado. '
          'Os cenários off_grid, missing_family e noise_mismatch são testes de estresse sem essa garantia.\n\n'
          'summary.csv contém resultados por contexto. risk_by_instance.csv fornece intervalos binomiais por lei. '
          'paired_comparisons.csv usa bootstrap pareado por lei; valores negativos de cost/error favorecem o método. '
          'Custo menor deve ser interpretado junto com resolução correta, pois uma recusa precoce também custa pouco.\n\n'
          'O gráfico é descritivo e não substitui os intervalos por instância. Uma amostra pequena sem erros '
          'não confirma que o risco é inferior a 5%. Nenhuma vitória estatística é declarada automaticamente.\n')
    (outdir/'REPORT.md').write_text(text)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--preset',default='smoke',choices=['smoke','pilot','confirmatory'])
    p.add_argument('--out',default='results'); p.add_argument('--device',default='auto')
    p.add_argument('--budget',type=int,default=60); p.add_argument('--minutes',type=float,default=90)
    args=p.parse_args(); cfg=Config(preset=args.preset,device=args.device,budget=args.budget,max_minutes=args.minutes)
    run_suite(cfg,args.out); print(summarize(args.out).to_string(index=False))
