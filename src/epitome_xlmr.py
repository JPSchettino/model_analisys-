"""Adaptação EPITOME com XLM-R. Código incorporado no notebook entregue."""
import copy
import gc
import hashlib
import itertools
import json
import math
import random
import re
import shutil
import subprocess
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import transformers
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, XLMRobertaModel, get_linear_schedule_with_warmup

MODEL_ID = 'FacebookAI/xlm-roberta-base'
TASKS = ('ER', 'IP', 'EX')
MODELS = ('GEMMA', 'LLAMA', 'GPT4')
STEMS = dict(ER='emotional-reactions', IP='interpretations', EX='explorations')
REPOS = {
    'en': ('behavioral-data/Empathy-Mental-Health', '0e11f98901527550885dbf253bd2af92dbcec43b'),
    'pt': ('JPSchettino/model_analisys-', 'ff0379ea9eaab237be50cfff2826a5cff43315e1'),
}
BLOBS = {
    'en': dict(ER='6af98f52f26daef4dd76fe9404e165419e5193bb', IP='dc806b3aab4f45b36f73bc39f043d6cc430ef17b', EX='a85bd077317d7413985632d0970188bf6a32f11f'),
    'pt': dict(ER='d53a13f9bf46b109a91ee11c652c652a7567759e', IP='f1951a93abb1fa82ab5f9532328df60d5209c853', EX='dab8fea5589075704bd74eaf677844862b86033b'),
}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def normalize(s):
    return ' '.join(str(s).strip().strip('"').strip().split())


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_sources(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frames, audit = {}, {}
    for language, (repo, revision) in REPOS.items():
        frames[language] = {}
        for task in TASKS:
            path = directory / f'{language}_{task}.csv'
            url = f'https://raw.githubusercontent.com/{repo}/{revision}/dataset/{STEMS[task]}-reddit.csv'
            if not path.exists():
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                path.write_bytes(response.content)
            raw = path.read_bytes()
            git_sha = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
            if git_sha != BLOBS[language][task]:
                raise ValueError(f'Arquivo {language}/{task} difere da fonte fixada. Remova {path} e execute de novo.')
            frame = pd.read_csv(path, dtype={'sp_id': str, 'rp_id': str}).fillna('')
            required = {'sp_id', 'rp_id', 'seeker_post', 'response_post', 'level', 'rationales'}
            if not required <= set(frame):
                raise ValueError(f'{language}/{task}: faltam colunas {required - set(frame)}')
            frame['level'] = pd.to_numeric(frame.level, errors='raise').astype(int)
            if set(frame.level.unique()) != {0, 1, 2} or len(frame) != 3084:
                raise ValueError(f'{language}/{task}: conteúdo inesperado')
            if (frame[['seeker_post', 'response_post']].apply(lambda c: c.str.strip()).eq('').any().any()):
                raise ValueError(f'{language}/{task}: há textos vazios')
            frame['row_id'] = np.arange(len(frame))
            frames[language][task] = frame
            audit[f'{language}_{task}'] = {'url': url, 'git_blob': git_sha, 'sha256': hashlib.sha256(raw).hexdigest(), 'rows': len(frame)}
    # As traduções preservam a ordem/níveis, mas alguns IDs foram traduzidos ou
    # convertidos por planilha. A fonte é fixada e os desvios são registrados.
    reference = frames['en']['ER']
    for task in TASKS:
        en, pt = frames['en'][task], frames['pt'][task]
        if not en[['sp_id', 'rp_id']].equals(reference[['sp_id', 'rp_id']]):
            raise ValueError('A ordem dos três arquivos originais não coincide.')
        matching = (en.sp_id == pt.sp_id) & (en.rp_id == pt.rp_id)
        if not (en.level == pt.level).all() or matching.mean() < .97:
            raise ValueError(f'{task}: não é seguro alinhar as versões por posição.')
        audit[f'alinhamento_{task}'] = {'method': 'row_position_in_pinned_files', 'labels_identical': True,
                                      'id_matches': int(matching.sum()), 'id_mismatch_row_ids': en.loc[~matching, 'row_id'].tolist()}
    return frames, audit


def post_groups(frame):
    """Mesmo ID OU mesmo texto de seeker fica no mesmo grupo."""
    parent = list(range(len(frame)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen_id, seen_text = {}, {}
    for i, row in enumerate(frame.itertuples()):
        text = normalize(row.seeker_post).casefold()
        for seen, value in [(seen_id, row.sp_id), (seen_text, text)]:
            if value in seen:
                a, b = root(i), root(seen[value])
                parent[max(a, b)] = min(a, b)
            else:
                seen[value] = i
    return np.array([root(i) for i in range(len(frame))])


def grouped_split(frames, seed):
    reference = frames['en']['ER']
    groups = post_groups(reference)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fold = np.full(len(reference), -1, dtype=int)
    for i, (_, test) in enumerate(splitter.split(reference, reference.level, groups)):
        fold[test] = i
    indices = {'train': np.flatnonzero(fold >= 2), 'validation': np.flatnonzero(fold == 1), 'test': np.flatnonzero(fold == 0)}
    for a, b in itertools.combinations(indices, 2):
        if set(groups[indices[a]]) & set(groups[indices[b]]):
            raise AssertionError('Vazamento de enunciados entre divisões.')
    for task in TASKS:
        for name, ids in indices.items():
            if set(frames['en'][task].iloc[ids].level.unique()) != {0, 1, 2}:
                raise ValueError(f'A divisão {task}/{name} não contém as três classes.')
    return indices, groups


def align_rationales(text, annotation, offsets, level):
    """Alinha substrings originais por caracteres; não inventa spans ausentes."""
    labels = [0 if end > start else -100 for start, end in offsets]
    parts = [p.strip() for p in str(annotation).split('|') if p.strip()]
    if not parts:
        return (labels, 'no_rationale_expected') if level == 0 else ([-100] * len(offsets), 'missing_positive_rationale')
    spans = []
    for part in parts:
        # Escapa pontuação; espaço variável é permitido sem mudar o conteúdo.
        pattern = r'\s+'.join(re.escape(word) for word in part.split())
        matches = list(re.finditer(pattern, text, flags=re.I))
        if not matches:
            return [-100] * len(offsets), 'unmatched_rationale'
        spans.extend((m.start(), m.end()) for m in matches)
    for i, (start, end) in enumerate(offsets):
        if end > start and any(start < right and end > left for left, right in spans):
            labels[i] = 1
    if not any(label == 1 for label in labels):
        return labels, 'rationale_outside_truncation'
    return labels, 'aligned'


class EncodedPairs(Dataset):
    def __init__(self, frame, tokenizer, max_length, supervise_rationales=False):
        self.frame = frame.reset_index(drop=True).copy()
        sp = tokenizer(self.frame.seeker_post.tolist(), padding='max_length', truncation=True, max_length=max_length)
        rp = tokenizer(self.frame.response_post.tolist(), padding='max_length', truncation=True, max_length=max_length,
                       return_offsets_mapping=True)
        self.offsets = rp.pop('offset_mapping')
        self.tensors = {
            'sp_ids': torch.tensor(sp['input_ids']), 'sp_mask': torch.tensor(sp['attention_mask']),
            'rp_ids': torch.tensor(rp['input_ids']), 'rp_mask': torch.tensor(rp['attention_mask']),
        }
        self.audit = Counter()
        if 'level' in self.frame:
            self.tensors['labels'] = torch.tensor(self.frame.level.tolist())
            rationales = []
            for row, offsets in zip(self.frame.itertuples(), self.offsets):
                if supervise_rationales:
                    labels, status = align_rationales(row.response_post, row.rationales, offsets, row.level)
                else:
                    labels, status = [-100] * len(offsets), 'not_supervised_for_this_language'
                self.audit[status] += 1
                rationales.append(labels)
            self.tensors['rationale_labels'] = torch.tensor(rationales)
        # Conta truncamento por texto; preserva os offsets para a atribuição.
        for kind, text in [('seeker', self.frame.seeker_post), ('response', self.frame.response_post)]:
            self.audit[f'{kind}_truncated'] = sum(len(tokenizer.encode(t, add_special_tokens=True)) > max_length for t in text)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        return {key: values[i] for key, values in self.tensors.items()}


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, dropout=.1):
        super().__init__()
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(hidden_size)

    def forward(self, q, k, v, key_mask):
        scores = self.q_linear(q) @ self.k_linear(k).transpose(1, 2) / self.scale
        scores = scores.masked_fill(~key_mask[:, None, :].bool(), torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        return self.out(weights @ self.v_linear(v))


class EpitomeXLMR(nn.Module):
    def __init__(self, revision=None, config=None, dropout=.1):
        super().__init__()
        if config is not None:  # usado nos testes com modelo pequeno e sem rede
            self.seeker = XLMRobertaModel(config, add_pooling_layer=False)
        else:
            self.seeker = AutoModel.from_pretrained(MODEL_ID, revision=revision, add_pooling_layer=False)
        self.responder = copy.deepcopy(self.seeker)
        hidden = self.seeker.config.hidden_size
        self.attn = CrossAttention(hidden)
        self.dropout = nn.Dropout(dropout)
        self.empathy_classifier = nn.Sequential(nn.Dropout(.1), nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(.1), nn.Linear(hidden, 3))
        self.rationale_classifier = nn.Linear(hidden, 2)
        for module in [self.attn, self.empathy_classifier, self.rationale_classifier]:
            for sub in module.modules():
                if isinstance(sub, nn.Linear):
                    nn.init.normal_(sub.weight, mean=0, std=.02)
                    nn.init.zeros_(sub.bias)
        self.seeker.requires_grad_(False)
        self.seeker.eval()

    def train(self, mode=True):
        super().train(mode)
        self.seeker.eval()
        return self

    def forward(self, sp_ids, sp_mask, rp_ids, rp_mask, labels=None, rationale_labels=None, lambda_ei=.5, lambda_re=.5):
        with torch.no_grad():
            seeker = self.seeker(input_ids=sp_ids, attention_mask=sp_mask).last_hidden_state
        responder = self.responder(input_ids=rp_ids, attention_mask=rp_mask).last_hidden_state
        sequence = responder + self.dropout(self.attn(responder, seeker, seeker, sp_mask))
        logits = self.empathy_classifier(sequence[:, 0])
        rationale_logits = self.rationale_classifier(self.dropout(sequence))
        result = {'logits': logits, 'rationale_logits': rationale_logits}
        if labels is not None:
            loss_ei = nn.functional.cross_entropy(logits, labels)
            active = rationale_labels != -100 if rationale_labels is not None else None
            loss_re = (nn.functional.cross_entropy(rationale_logits[active], rationale_labels[active])
                       if active is not None and active.any() else rationale_logits.sum() * 0.)
            result.update(loss=lambda_ei * loss_ei + lambda_re * loss_re, loss_ei=loss_ei, loss_re=loss_re)
        return result


def evaluate(model, loader, device):
    model.eval()
    logits, labels, rationale_true, rationale_pred = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            result = model(**batch)
            logits.extend(result['logits'].float().cpu().tolist())
            if 'labels' in batch:
                labels.extend(batch['labels'].cpu().tolist())
                active = batch['rationale_labels'] != -100
                if active.any():
                    rationale_true.extend(batch['rationale_labels'][active].cpu().tolist())
                    rationale_pred.extend(result['rationale_logits'].argmax(-1)[active].cpu().tolist())
    arr = np.asarray(logits)
    if not labels:
        return {}, arr
    predicted = arr.argmax(1)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1, 2])
    metrics = {
        'n': len(labels), 'accuracy': float(accuracy_score(labels, predicted)),
        'macro_f1': float(f1_score(labels, predicted, labels=[0, 1, 2], average='macro', zero_division=0)),
        'confusion_matrix_0_1_2': matrix.tolist(),
        'classification_report': classification_report(labels, predicted, labels=[0, 1, 2], output_dict=True, zero_division=0),
        'classes_not_predicted': [i for i in range(3) if i not in set(predicted)],
    }
    if rationale_true:
        metrics['rationale_token_f1_positive'] = float(f1_score(rationale_true, rationale_pred, zero_division=0))
        metrics['n_rationale_tokens_evaluated'] = len(rationale_true)
    return metrics, arr


def load_checkpoint(model, path, device):
    state = torch.load(path, map_location='cpu', weights_only=True)
    info = model.load_state_dict(state, strict=False)
    if info.unexpected_keys or any(not key.startswith('seeker.') for key in info.missing_keys):
        raise ValueError(f'Checkpoint incompatível: {info}')
    model.to(device)


def train_task(task, train_dataset, val_dataset, cfg, output, revision, device):
    seed_everything(cfg['seed'])
    model = EpitomeXLMR(revision=revision, dropout=cfg['dropout']).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    # Reduz memória de ativações. O encoder seeker já está congelado.
    model.responder.gradient_checkpointing_enable()
    generator = torch.Generator().manual_seed(cfg['seed'])
    loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=cfg['eval_batch_size'])
    optimizer = torch.optim.AdamW(parameters, lr=cfg['lr'], eps=1e-8, weight_decay=0.)
    n_steps = math.ceil(len(loader) / cfg['gradient_accumulation']) * cfg['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=n_steps)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    best_f1, history = -1., []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / 'best_model.pt'
    for epoch in range(cfg['epochs']):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        bar = tqdm(loader, desc=f'{task}: época {epoch+1}/{cfg["epochs"]}')
        for step, batch in enumerate(bar):
            batch = {key: value.to(device) for key, value in batch.items()}
            # Ajusta o último grupo incompleto para não subponderar o gradiente.
            group_start = (step // cfg['gradient_accumulation']) * cfg['gradient_accumulation']
            divisor = min(cfg['gradient_accumulation'], len(loader) - group_start)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                result = model(**batch, lambda_ei=cfg['lambda_ei'], lambda_re=cfg['lambda_re'])
                loss = result['loss'] / divisor
            if not torch.isfinite(loss):
                raise RuntimeError(f'{task}: perda não finita. Reduza batch_size ou desative AMP.')
            scaler.scale(loss).backward()
            losses.append(float(result['loss'].detach().cpu()))
            if (step + 1) % cfg['gradient_accumulation'] == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(parameters, 1.)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            bar.set_postfix(loss=f'{np.mean(losses):.4f}')
        val_metrics, _ = evaluate(model, val_loader, device)
        history.append({'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), 'validation_pt': val_metrics})
        write_json(output / 'training_history.json', history)
        print(f'{task}: validação PT Macro-F1={val_metrics["macro_f1"]:.4f}')
        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            # Seeker é reconstruído a partir da revisão base fixada no manifesto.
            state = {key: value.detach().cpu() for key, value in model.state_dict().items() if not key.startswith('seeker.')}
            temporary = output / 'best_model.tmp'
            torch.save(state, temporary)
            temporary.replace(best_path)
            del state
    del optimizer, scheduler, scaler
    gc.collect()
    load_checkpoint(model, best_path, device)
    return model, history


def rationale_text(text, offsets, predicted):
    spans = []
    for (start, end), positive in zip(offsets, predicted):
        if not positive or end <= start:
            continue
        if spans and start <= spans[-1][1] + 1:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    return ' | '.join(text[start:end] for start, end in spans)


def predict_pairs(model, dataset, device, batch_size):
    model.eval()
    probabilities, logits, texts = [], [], []
    position = 0
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size):
            result = model(**{key: value.to(device) for key, value in batch.items()})
            logits.extend(result['logits'].float().cpu().tolist())
            probabilities.extend(result['logits'].float().softmax(-1).cpu().tolist())
            rationale = result['rationale_logits'].argmax(-1).cpu().tolist()
            for labels in rationale:
                row = dataset.frame.iloc[position]
                texts.append(rationale_text(row.response_post, dataset.offsets[position], labels))
                position += 1
    return np.asarray(logits), np.asarray(probabilities), texts


def exact_sign_test(higher, lower):
    n = higher + lower
    if not n:
        return 1.
    return min(1., 2 * sum(math.comb(n, i) for i in range(min(higher, lower) + 1)) / 2**n)


def holm(pvalues):
    result, previous = [0.] * len(pvalues), 0.
    for rank, i in enumerate(sorted(range(len(pvalues)), key=lambda k: pvalues[k])):
        previous = max(previous, min(1., (len(pvalues) - rank) * pvalues[i]))
        result[i] = previous
    return result


def regression_table(pairs):
    """Análise adicional comparável à tabela ordinal, com SE por enunciado."""
    import statsmodels.api as sm
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    rows, status = [], {}
    x = pd.DataFrame({model: (pairs.id == model).astype(float) for model in ['LLAMA', 'GPT4']})
    for task in TASKS:
        y = pairs[task + '_label'].astype(int)
        if y.nunique() < 2:
            status[task] = {'available': False, 'reason': 'Only one observed score level'}
            continue
        # With intercept + model indicators, any group with a constant binary
        # outcome puts its group probability at a boundary. MLE is not finite,
        # even when the GLM optimizer reports convergence without warnings.
        if y.nunique() == 2:
            constant_groups = [model for model in MODELS if y[pairs.id == model].nunique() < 2]
            if constant_groups:
                status[task] = {'available': False,
                                'reason': 'Complete or quasi-complete separation: constant outcome within model groups; ordinary logistic MLE is not finite',
                                'constant_outcome_models': constant_groups,
                                'n_clusters': int(pairs.prompt_id.nunique())}
                continue
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                if y.nunique() == 2:
                    result = sm.GLM((y == y.max()).astype(int), sm.add_constant(x), family=sm.families.Binomial()).fit(
                        cov_type='cluster', cov_kwds={'groups': pairs.prompt_id, 'use_correction': True})
                    model_type = 'binary_logistic_two_observed_levels'
                    converged = bool(result.converged)
                else:
                    result = OrderedModel(y, x, distr='logit').fit(method='bfgs', disp=False, cov_type='cluster',
                        cov_kwds={'groups': pairs.prompt_id, 'use_correction': True})
                    model_type = 'proportional_odds_ordinal_logistic'
                    converged = bool(result.mle_retvals.get('converged', False))
            messages = [str(w.message) for w in caught]
            fatal = any(re.search(r'separation|singular|hessian|overflow|converg', message, re.I) for message in messages)
            if not converged or fatal:
                status[task] = {'available': False, 'reason': 'Fit did not produce a reliable estimate', 'warnings': messages}
                continue
            ci = result.conf_int()
            for model in ['LLAMA', 'GPT4']:
                coef, low, high = float(result.params[model]), float(ci.loc[model, 0]), float(ci.loc[model, 1])
                pvalue = float(result.pvalues[model])
                if not np.isfinite([coef, low, high, pvalue]).all() or max(abs(low), abs(high)) > 50:
                    raise ValueError('Unstable coefficient or confidence interval')
                rows.append({'metric': task, 'model': model, 'reference': 'GEMMA', 'fit_type': model_type,
                             'log_odds': coef, 'log_odds_ci_low': low, 'log_odds_ci_high': high,
                             'odds_ratio': math.exp(coef), 'odds_ratio_ci_low': math.exp(low),
                             'odds_ratio_ci_high': math.exp(high), 'p_cluster_wald': pvalue})
            status[task] = {'available': True, 'fit_type': model_type, 'n_clusters': int(pairs.prompt_id.nunique()),
                            'warnings': messages, 'note': 'Small sample: use the exact paired tests as the primary analysis.'}
        except Exception as exc:
            rows = [r for r in rows if r['metric'] != task]
            status[task] = {'available': False, 'reason': str(exc)}
    # Família pré-definida: duas comparações com Gemma por dimensão (6).
    positions = [(task, model) for task in TASKS for model in ['LLAMA', 'GPT4']]
    lookup = {(r['metric'], r['model']): r for r in rows}
    adjusted = holm([lookup[p]['p_cluster_wald'] if p in lookup else 1. for p in positions])
    for position, pvalue in zip(positions, adjusted):
        if position in lookup:
            lookup[position]['p_cluster_wald_holm_6'] = pvalue
    columns = ['metric', 'model', 'reference', 'fit_type', 'log_odds', 'log_odds_ci_low', 'log_odds_ci_high',
               'odds_ratio', 'odds_ratio_ci_low', 'odds_ratio_ci_high', 'p_cluster_wald', 'p_cluster_wald_holm_6']
    return pd.DataFrame(rows, columns=columns), status


def analysis_tables(pairs):
    if len(pairs) != 60 or pairs.id.value_counts().to_dict() != {model: 20 for model in MODELS}:
        raise ValueError('São necessários exatamente 20 pares de cada modelo.')
    grouped = pairs.groupby('prompt_id')
    if grouped.ngroups != 20 or not all(set(g.id) == set(MODELS) for _, g in grouped):
        raise ValueError('Os modelos não compartilham os mesmos 20 enunciados.')
    summary, tests = [], []
    for task in TASKS:
        for model in MODELS:
            labels = pairs.loc[pairs.id == model, task + '_label'].astype(int)
            if not labels.isin([0, 1, 2]).all():
                raise ValueError('Rótulos fora de 0/1/2.')
            summary.append({'metric': task, 'model': model, 'n': len(labels),
                            **{f'count_{i}': int((labels == i).sum()) for i in range(3)},
                            'mean_ordinal_score_descriptive': float(labels.mean())})
        wide = pairs.pivot(index='prompt_id', columns='id', values=task + '_label')
        for a, b in itertools.combinations(MODELS, 2):
            differences = wide[a].astype(int) - wide[b].astype(int)
            higher, lower, ties = int((differences > 0).sum()), int((differences < 0).sum()), int((differences == 0).sum())
            tests.append({'metric': task, 'model_A': a, 'model_B': b,
                          'A_score_higher': higher, 'B_score_higher': lower, 'ties': ties,
                          'p_exact_paired_sign': exact_sign_test(higher, lower)})
    for row, adjusted in zip(tests, holm([r['p_exact_paired_sign'] for r in tests])):
        row.update(p_holm_9=adjusted, significant_at_0_05=bool(adjusted < .05))
    return pd.DataFrame(summary), pd.DataFrame(tests)


def compare_baseline(pairs, baseline_path, name):
    baseline = pd.read_csv(baseline_path).fillna('')
    def key(row):
        return tuple(normalize(row[c]) for c in ['id', 'seeker_post', 'response_post'])
    old = {key(row): row for _, row in baseline.iterrows()}
    current = [key(row) for _, row in pairs.iterrows()]
    if len(old) != 60 or len(set(current)) != 60 or set(old) != set(current):
        raise ValueError(f'{name}: os 60 pares não coincidem.')
    result = pairs[['prompt_id', 'id', 'seeker_post', 'response_post']].copy()
    for task in TASKS:
        result[task + '_baseline'] = [int(old[k][task + '_label']) for k in current]
        result[task + '_epitome_xlmr'] = pairs[task + '_label'].astype(int).to_numpy()
        result[task + '_changed'] = result[task + '_baseline'] != result[task + '_epitome_xlmr']
    return result


def write_report(output, cfg, summary, tests, metrics, regression_status):
    lines = ['# EPITOME com XLM-R: resultados da execução', '',
             'Modelo base: `FacebookAI/xlm-roberta-base`. Adaptação bi-encoder multitarefa, com encoder seeker congelado, cross-attention, cabeça de níveis 0/1/2 e cabeça de rationales.', '',
             'As traduções PT recebem supervisão de nível. Rationales são supervisionadas apenas nas versões EN originais, porque há trechos incompatíveis nas traduções. A versão EN e PT de cada enunciado fica na mesma divisão. Seleção do checkpoint: Macro-F1 na validação PT. Teste EN/PT independente da seleção.', '',
             'Esta é uma adaptação com processamento atualizado, treino bilíngue e divisão por grupos; não é uma reprodução numérica exata do treinamento anterior.', '',
             '## Distribuição nos 60 pares', '', '| Dimensão | Modelo | Nível 0 | Nível 1 | Nível 2 | Média descritiva |', '|---|---|---:|---:|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'| {row.metric} | {row.model} | {row.count_0} | {row.count_1} | {row.count_2} | {row.mean_ordinal_score_descriptive:.3f} |')
    lines += ['', '## Testes pareados primários', '',
              'Teste exato dos sinais, bilateral: compara quantas vezes um modelo recebeu nível maior ou menor no mesmo enunciado. Empates são excluídos do número informativo. Holm em nove comparações predefinidas. Esse teste usa a ordem das classes sem supor distâncias iguais. Com apenas dois níveis, coincide com McNemar exato.', '',
              '| Dimensão | Comparação | A maior | B maior | Empates | p exato | p Holm |', '|---|---|---:|---:|---:|---:|---:|']
    for row in tests.itertuples():
        lines.append(f'| {row.metric} | {row.model_A} × {row.model_B} | {row.A_score_higher} | {row.B_score_higher} | {row.ties} | {row.p_exact_paired_sign:.5f} | {row.p_holm_9:.5f} |')
    lines += ['', '## Teste em dados anotados e traduzidos', '', '| Dimensão | Idioma | N | Acurácia | Macro-F1 | Classes não previstas |', '|---|---|---:|---:|---:|---|']
    quality_flags = []
    for task in TASKS:
        for lang in ['en', 'pt']:
            m = metrics[task]['test_' + lang]
            lines.append(f'| {task} | {lang} | {m["n"]} | {m["accuracy"]:.4f} | {m["macro_f1"]:.4f} | {m["classes_not_predicted"]} |')
            if m['classes_not_predicted']:
                quality_flags.append(f'{task}/{lang}: classes não previstas no teste: {m["classes_not_predicted"]}.')
    lines += ['', '## Limites para o artigo', '',
              'Os níveis das traduções PT são transferidos das anotações EN, e não foram reanotados independentemente por falantes de português. As predições nos 60 pares não demonstram eficácia clínica. A validação humana específica dos níveis ER/IP/EX é distinta de avaliações humanas de outros comportamentos de entrevista motivacional. O template de revisão humana oculta as predições para reduzir viés de anotação.', '',
              'A regressão ordinal/binária é uma análise adicional com erro-padrão agrupado por enunciado (20 grupos). A amostra é pequena. Falhas de convergência/separação são informadas em `regression_status.json`, sem criar estimativas artificiais. Log-odds, odds ratio, IC e p-valor têm colunas próprias. Resultados de significância devem priorizar os testes pareados exatos.', '']
    lines += ['- ' + flag for flag in quality_flags] or ['Nenhuma classe ausente nas predições de teste.']
    lines += ['', 'Fonte da arquitetura e escala: Sharma et al. (2020), https://aclanthology.org/2020.emnlp-main.425/.',
              'Detalhes de fontes, revisões e ambiente: `manifest.json`, `pip_freeze.txt` e `fontes_e_alinhamento.json`.', '']
    (output / 'relatorio_resultados.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(output / 'quality_flags.json', {'classes_not_predicted': quality_flags, 'independent_human_pt_validation': False})
    train_n = metrics['ER']['n_training_examples']
    val_n = metrics['ER']['validation_pt']['n']
    test_n = metrics['ER']['test_pt']['n']
    manuscript = [
        '# Draft methods and results for the manuscript', '',
        f'We implemented a multilingual adaptation of the EPITOME multi-task bi-encoder. Both encoders were initialized from `FacebookAI/xlm-roberta-base`. The seeker encoder was frozen and kept in evaluation mode. The responder representation was augmented by cross-attention over the seeker representation and used by a three-level classification head and a binary token-level rationale head. Padding tokens were masked in cross-attention. Special and padding tokens were excluded from rationale loss.', '',
        f'For each mechanism (ER, IP, and EX), training used {train_n} examples comprising English annotated pairs and their Portuguese translations. Approximately 60% of the source rows were used for training, 20% for validation, and 20% for testing. Rows sharing a seeker identifier or normalized seeker text were grouped; English and Portuguese versions stayed in the same split. Rationale supervision used only original English spans that could be aligned to response characters. Missing or unmatched spans were excluded from that auxiliary loss. Portuguese examples received level supervision from transferred English labels. These translations were not independently reannotated by Portuguese-speaking raters.', '',
        f'Training used {cfg["epochs"]} epochs, learning rate {cfg["lr"]}, effective batch size {cfg["batch_size"] * cfg["gradient_accumulation"]}, seed {cfg["seed"]}, maximum sequence length {cfg["max_length"]} per encoder, and objective L = {cfg["lambda_ei"]} L_level + {cfg["lambda_re"]} L_rationale. The checkpoint was selected by macro-F1 on {val_n} Portuguese validation examples. The corresponding {test_n}-example English and Portuguese test splits were held out from checkpoint selection. Dataset revisions, model revision, software versions, split assignments, truncation counts, and alignment exclusions were recorded in the run manifest and audit files.', '',
        'The same 20 prompts were evaluated for each generation model. Primary between-model comparisons used two-sided exact paired sign tests on ordinal score differences, with Holm adjustment across the nine predefined comparisons (three model pairs for each of ER, IP, and EX). Ties were excluded from the informative pair count. Additional proportional-odds models, or binary logistic models when only two levels were observed, used Gemma as reference and standard errors clustered by prompt. These asymptotic estimates were treated as supplementary because there were only 20 prompt clusters.', '',
        '## Held-out performance', '',
        '| Mechanism | English test macro-F1 | Portuguese translated test macro-F1 |',
        '|---|---:|---:|',
    ]
    for task in TASKS:
        manuscript.append(f'| {task} | {metrics[task]["test_en"]["macro_f1"]:.4f} | {metrics[task]["test_pt"]["macro_f1"]:.4f} |')
    manuscript += ['', '## Score distribution', '', '| Mechanism | Model | Score 0 | Score 1 | Score 2 |', '|---|---|---:|---:|---:|']
    for row in summary.itertuples():
        manuscript.append(f'| {row.metric} | {row.model} | {row.count_0} | {row.count_1} | {row.count_2} |')
    significant = tests[tests.significant_at_0_05]
    manuscript += ['', '## Primary paired comparisons', '']
    if significant.empty:
        manuscript.append('None of the nine paired comparisons met the adjusted 0.05 significance threshold. This does not establish equivalence between models.')
    else:
        for row in significant.itertuples():
            manuscript.append(f'{row.metric}: {row.model_A} versus {row.model_B}, exact p = {row.p_exact_paired_sign:.5f}, Holm-adjusted p = {row.p_holm_9:.5f}; {row.A_score_higher} prompts favored A and {row.B_score_higher} favored B, with {row.ties} ties.')
    manuscript += ['', '## Limitations to retain in the paper', '',
        'This is an adaptation with bilingual supervision and updated preprocessing, rather than an exact numerical reproduction of the earlier training run. Evaluation on translated annotations does not directly validate Portuguese ER/IP/EX scores; human assessment of other motivational interviewing behaviors is complementary. Predicted empathy mechanisms and model-to-model differences do not establish clinical effectiveness. Softmax probabilities were not calibrated. Missing predicted classes and alignment/truncation exclusions should be reported alongside overall performance.', '',
        'Framework source: Sharma et al. (2020), https://aclanthology.org/2020.emnlp-main.425/.', '']
    (output / 'draft_methods_results_EN.md').write_text('\n'.join(manuscript), encoding='utf-8')


def run_experiment(cfg, base_output, original_pairs_path, additional_baseline_path, source_dir='/content/epitome_sources'):
    if not torch.cuda.is_available():
        raise RuntimeError('Ative GPU no Colab: Ambiente de execução > Alterar tipo de ambiente de execução > GPU.')
    if cfg['train_languages'] != ['en', 'pt']:
        raise ValueError('Esta versão foi preparada para treino bilíngue EN/PT.')
    seed_everything(cfg['seed'])
    device = torch.device('cuda')
    hf_config = AutoConfig.from_pretrained(MODEL_ID)
    revision = hf_config._commit_hash
    if hf_config.model_type != 'xlm-roberta' or not revision:
        raise ValueError('Backbone ou revisão não resolvido com segurança.')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError('É necessário tokenizer rápido com offsets de caracteres.')
    frames, source_audit = download_sources(source_dir)
    split, groups = grouped_split(frames, cfg['seed'])
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    pairs = pd.read_csv(original_pairs_path).fillna('')
    baseline_hash = hashlib.sha256(Path(original_pairs_path).read_bytes()).hexdigest()
    identity = {'config': cfg, 'model_id': MODEL_ID, 'model_revision': revision,
                'sources': BLOBS, 'code_sha256': code_hash, 'baseline_sha256': baseline_hash,
                'additional_baseline_sha256': hashlib.sha256(Path(additional_baseline_path).read_bytes()).hexdigest(),
                'runtime_versions': {'torch': torch.__version__, 'transformers': transformers.__version__,
                                     'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__}}
    run_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    output = Path(base_output) / ('run_' + run_hash)
    output.mkdir(parents=True, exist_ok=True)
    pairs = pairs[['id', 'seeker_post', 'response_post']].copy()
    prompt_ids = {text: i + 1 for i, text in enumerate(dict.fromkeys(pairs.seeker_post.map(normalize)))}
    pairs.insert(0, 'prompt_id', pairs.seeker_post.map(lambda text: prompt_ids[normalize(text)]))
    # Valida os pares antes de gastar tempo de GPU.
    dummy = pairs.assign(ER_label=0, IP_label=0, EX_label=0)
    analysis_tables(dummy)
    compare_baseline(dummy, original_pairs_path, 'original')
    compare_baseline(dummy, additional_baseline_path, 'additional')
    manifest = dict(identity, run_id=run_hash, status='running',
                    selection='best validation PT macro-F1; untouched held-out EN/PT test',
                    train_count_per_language=len(split['train']), validation_count_per_language=len(split['validation']),
                    test_count_per_language=len(split['test']), frozen_seeker=True, saved_checkpoint_excludes_frozen_seeker=True,
                    architecture_adaptations=['Modern XLM-R encoders', 'Character-offset rationale alignment',
                        'Ignore missing/unmatched rationale spans', 'No supervision on special/padding tokens',
                        'Mask padding in cross-attention', 'Frozen seeker in eval mode',
                        'Bilingual level supervision; rationale supervision only on EN', 'Group-disjoint split'],
                    torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
                    pandas=pd.__version__, sklearn=sklearn.__version__, gpu=torch.cuda.get_device_name(0))
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'fontes_e_alinhamento.json', source_audit)
    split_frame = pd.DataFrame({'row_id': np.arange(len(groups)), 'group_id': groups, 'split': ''})
    for name, ids in split.items():
        split_frame.loc[ids, 'split'] = name
    split_frame.to_csv(output / 'split_assignments.csv', index=False)
    freeze = subprocess.run(['python', '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (output / 'pip_freeze.txt').write_text(freeze.stdout, encoding='utf-8')
    shutil.copyfile(__file__, output / 'epitome_xlmr.py')
    shutil.copyfile(original_pairs_path, output / 'baseline_original.csv')
    shutil.copyfile(additional_baseline_path, output / 'baseline_classificador_pares_pt.csv')
    tokenizer.save_pretrained(output / 'tokenizer')
    metrics, preprocessing = {}, {}
    for task in TASKS:
        print(f'\n===== {task}; resultados em {output} =====')
        task_output = output / task
        task_output.mkdir(exist_ok=True)
        datasets = {}
        for lang in ['en', 'pt']:
            for name, ids in split.items():
                datasets[lang + '_' + name] = EncodedPairs(frames[lang][task].iloc[ids], tokenizer, cfg['max_length'], supervise_rationales=(lang == 'en'))
        preprocessing[task] = {name: dict(ds.audit) for name, ds in datasets.items()}
        write_json(output / 'preprocessing_audit.json', preprocessing)
        completed = task_output / 'completed.json'
        if completed.exists() and (task_output / 'best_model.pt').exists():
            print(f'{task}: checkpoint e resultados concluídos recuperados do Drive.')
            model = EpitomeXLMR(revision=revision, dropout=cfg['dropout'])
            load_checkpoint(model, task_output / 'best_model.pt', device)
            history = json.loads((task_output / 'training_history.json').read_text())
        else:
            training = torch.utils.data.ConcatDataset([datasets['en_train'], datasets['pt_train']])
            model, history = train_task(task, training, datasets['pt_validation'], cfg, task_output, revision, device)
        task_metrics = {'best_epoch': max(history, key=lambda entry: entry['validation_pt']['macro_f1'])['epoch'],
                        'n_training_examples': len(datasets['en_train']) + len(datasets['pt_train'])}
        for lang in ['en', 'pt']:
            for name in ['validation', 'test']:
                ds = datasets[lang + '_' + name]
                m, logits = evaluate(model, DataLoader(ds, batch_size=cfg['eval_batch_size']), device)
                task_metrics[name + '_' + lang] = m
                pred = ds.frame[['row_id', 'level']].copy()
                pred['predicted_level'] = logits.argmax(1)
                for i in range(3):
                    pred[f'logit_{i}'] = logits[:, i]
                pred.to_csv(task_output / f'{name}_{lang}_predictions.csv', index=False)
        metrics[task] = task_metrics
        write_json(output / 'validation_and_test.json', metrics)
        pair_dataset = EncodedPairs(pairs[['id', 'seeker_post', 'response_post']], tokenizer, cfg['max_length'])
        logits, probabilities, rationales = predict_pairs(model, pair_dataset, device, cfg['eval_batch_size'])
        pairs[task + '_label'] = logits.argmax(1)
        for i in range(3):
            pairs[f'{task}_logit_{i}'] = logits[:, i]
            pairs[f'{task}_probability_{i}'] = probabilities[:, i]
        pairs[task + '_rationale_text'] = rationales
        pairs.to_csv(output / 'rotulos_parciais.csv', index=False)
        write_json(task_output / 'completed.json', {'task': task, 'run_id': run_hash, 'best_epoch': task_metrics['best_epoch']})
        del model, datasets, pair_dataset
        gc.collect()
        torch.cuda.empty_cache()
    pairs.to_csv(output / 'rotulos_epitome_xlmr_60_pares.csv', index=False)
    summary, tests = analysis_tables(pairs)
    summary.to_csv(output / 'resumo_por_modelo.csv', index=False)
    tests.to_csv(output / 'testes_pareados.csv', index=False)
    regression, regression_status = regression_table(pairs)
    regression.to_csv(output / 'regressao_log_odds_e_OR.csv', index=False)
    write_json(output / 'regression_status.json', regression_status)
    for path, name in [(original_pairs_path, 'original'), (additional_baseline_path, 'classificador_pares_pt')]:
        comparison = compare_baseline(pairs, path, name)
        comparison.to_csv(output / f'comparacao_{name}.csv', index=False)
    review = pairs[['prompt_id', 'id', 'seeker_post', 'response_post']].copy()
    for task in TASKS:
        for rater in ['rater_1', 'rater_2', 'adjudicated']:
            review[f'{task}_{rater}'] = ''
    review.to_csv(output / 'template_validacao_humana_pt.csv', index=False)
    write_report(output, cfg, summary, tests, metrics, regression_status)
    manifest['status'] = 'completed'
    write_json(output / 'manifest.json', manifest)
    # Tabelas em LaTeX para o manuscrito (método/limites descritos no relatório).
    summary.to_latex(output / 'tabela_distribuicao.tex', index=False, escape=True, float_format='%.4f')
    tests.to_latex(output / 'tabela_testes_pareados.tex', index=False, escape=True, float_format='%.5f')
    if not regression.empty:
        regression[['metric', 'model', 'odds_ratio', 'odds_ratio_ci_low', 'odds_ratio_ci_high', 'p_cluster_wald', 'p_cluster_wald_holm_6']].to_latex(
            output / 'tabela_regressao_OR.tex', index=False, escape=True, float_format='%.5f')
    import zipfile
    zip_path = output / 'resultados_epitome_xlmr.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in output.rglob('*'):
            if path.is_file() and path.suffix not in {'.pt', '.tmp', '.zip'}:
                archive.write(path, arcname=str(path.relative_to(output)))
    print(f'\nConcluído. ZIP de resultados: {zip_path}')
    print('Os três checkpoints permanecem no Drive, fora do ZIP de tabelas.')
    return output, zip_path, summary, tests
