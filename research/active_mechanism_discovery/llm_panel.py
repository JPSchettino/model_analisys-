"""Optional open-weight LLM acquisition panel. The LLM never grades discoveries."""
import gc, hashlib, json, re, time
from pathlib import Path
import numpy as np
import torch
from discovery import (Config, atomic_json, build_cases, case_group, digest,
                       environment, run_batch, seed_for)

class Advisor:
    def __init__(self,logdir,model_id='Qwen/Qwen2.5-Coder-7B-Instruct',revision=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from huggingface_hub import model_info
        if not torch.cuda.is_available():
            raise RuntimeError('O painel LLM requer GPU. O experimento principal funciona em CPU.')
        self.model_id=model_id
        self.revision=revision or model_info(model_id).sha
        self.logdir=Path(logdir); self.logdir.mkdir(parents=True,exist_ok=True)
        self.tokenizer=AutoTokenizer.from_pretrained(model_id,revision=self.revision)
        kwargs=dict(revision=self.revision,device_map='auto',trust_remote_code=False)
        free,total=torch.cuda.mem_get_info()
        if free/2**30 >= 23:
            kwargs['torch_dtype']=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.precision=str(kwargs['torch_dtype'])
        else:
            kwargs['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True)
            self.precision='4-bit NF4'
        self.model=AutoModelForCausalLM.from_pretrained(model_id,**kwargs).eval()
        self.calls=0; self.tokens=0; self.seconds=0.0

    def choose(self,bank,ll,history,allowed,fallback):
        # Public information only. No truth index, simulator seed or evaluation score.
        order=np.argsort(ll+np.log(bank['prior']))[-8:][::-1]
        hypotheses=[dict(expression=bank['models'][int(i)]['formula'],
                         relative_log_likelihood=round(float(ll[i]-max(ll)),3)) for i in order]
        actions=[dict(id=i,x=float(x),z=float(z),cost=int(bank['costs'][i]))
                 for i,(x,z) in enumerate(bank['actions']) if allowed[i]]
        prompt=("Choose the next experiment to distinguish the candidate mechanisms efficiently. "
                "Measurement noise is independent Gaussian with known standard deviation. "
                "An observation equals the formula divided by the supplied scale plus noise. "
                "The hypothesis list is a likelihood-ranked shortlist, not guaranteed complete. "
                "Use previous observations and differences between hypotheses. Return ONLY "
                "a JSON object with the integer action id: {\"action\": 0}.\n"+
                json.dumps(dict(hypotheses=hypotheses,actions=actions,
                                observations=history,scale=bank['scale'],sigma=bank['noise'])))
        key=digest(dict(prompt=prompt,revision=self.revision,precision=self.precision))
        path=self.logdir/f'{key}.json'
        if path.exists():
            saved=json.loads(path.read_text()); return saved['action'],saved['valid']
        messages=[dict(role='system',content='You design informative scientific experiments.'),
                  dict(role='user',content=prompt)]
        text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.tokenizer(text,return_tensors='pt').to(self.model.device)
        if inputs['input_ids'].shape[1]>12000:
            raise RuntimeError('Prompt excessivo; reduza o orçamento do painel.')
        started=time.perf_counter()
        with torch.inference_mode():
            output=self.model.generate(**inputs,max_new_tokens=96,do_sample=False,
                                       pad_token_id=self.tokenizer.eos_token_id)
        torch.cuda.synchronize()
        seconds=time.perf_counter()-started
        new=output[0,inputs['input_ids'].shape[1]:]
        response=self.tokenizer.decode(new,skip_special_tokens=True)
        match=re.search(r'"action"\s*:\s*(\d+)',response)
        action=int(match.group(1)) if match else -1
        valid=0<=action<len(allowed) and bool(allowed[action])
        if not valid: action=fallback  # recorded explicitly, never silently counted as pure LLM
        atomic_json(path,dict(action=action,valid=valid,response=response,prompt=prompt,
            model=self.model_id,revision=self.revision,precision=self.precision,
            input_tokens=int(inputs['input_ids'].numel()),output_tokens=len(new),seconds=seconds))
        self.calls+=1; self.tokens+=len(new); self.seconds+=seconds
        return action,valid

    def close(self):
        del self.model; gc.collect(); torch.cuda.empty_cache()

def run_panel(outdir,max_cases=12,budget=24,seed=20261001,checkpoint=None):
    """Balanced small panel; exploratory only. Includes paired non-LLM controls."""
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    cfg=Config(preset='smoke',seed=seed,budget=budget,trace_per_group=1,max_minutes=0)
    # Deterministic stratified interleave avoids selecting only the first domain.
    all_cases=build_cases(cfg); strata={}
    for c in all_cases: strata.setdefault((c['domain'],c['regime']),[]).append(c)
    cases=[]
    for i in range(max(map(len,strata.values()))):
        for key in sorted(strata):
            if i<len(strata[key]): cases.append(strata[key][i])
    cases=cases[:max_cases]
    manifest=outdir/'manifest.json'
    previous=json.loads(manifest.read_text()) if manifest.exists() else None
    revision=previous['specification']['revision'] if previous else None
    advisor=Advisor(outdir/'llm_calls',revision=revision)
    specification=dict(model=advisor.model_id,revision=advisor.revision,precision=advisor.precision,
        seed=seed,budget=budget,max_cases=max_cases,ids=[c['id'] for c in cases],
        core_sha=hashlib.sha256(Path(__file__).with_name('discovery.py').read_bytes()).hexdigest(),
        panel_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    fp=digest(specification)
    if manifest.exists() and json.loads(manifest.read_text())['fingerprint']!=fp:
        advisor.close(); raise ValueError('Painel diferente: use outra pasta para preservar a reprodução.')
    atomic_json(manifest,dict(fingerprint=fp,specification=specification,environment=environment()))
    try:
        for i,c in enumerate(cases):
            for method in ('llm_safe','eig_safe','deficit_safe'):
                path=outdir/'episodes'/f"{c['id']}_{method}.json"
                if path.exists(): continue
                rows,trace=run_batch([c],method,cfg,advisor)
                atomic_json(path,rows[0]); atomic_json(outdir/'traces'/path.name,trace)
                if checkpoint: checkpoint(outdir.parent)
            print(f'Painel LLM: {i+1}/{len(cases)}',flush=True)
    finally: advisor.close()
    import pandas as pd
    df=pd.DataFrame([json.loads(p.read_text()) for p in sorted((outdir/'episodes').glob('*.json'))])
    df.to_csv(outdir/'llm_panel.csv',index=False)
    atomic_json(outdir/'status.json',dict(completed=True,cases=len(cases),
        note='Painel exploratório pequeno; não juntar à inferência confirmatória. Fallbacks estão registrados.'))
    return df.groupby('method').agg(n=('id','size'),resolved=('resolved','mean'),
        correct=('correct','mean'),false_declarations=('false_declaration','mean'),cost=('cost','mean'),
        invalid_llm=('llm_invalid_responses','sum'))
