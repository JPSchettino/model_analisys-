"""Confere hashes e métricas publicadas, sem carregar pesos."""
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def verify():
    root = Path(__file__).resolve().parents[1]
    results = root / 'results/2026-10-01'
    hashes = json.loads((results / 'SHA256SUMS.json').read_text())
    for name, expected in hashes.items():
        actual = hashlib.sha256((results / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Hash divergente: ' + name)
    manifest = json.loads((root / 'experiments/2026-10-01/manifest.json').read_text())
    source = root / 'experiments/2026-10-01/epitome_xlmr_used.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['code_sha256']
    pairs = pd.read_csv(results / 'rotulos_epitome_xlmr_60_pares.csv')
    assert len(pairs) == 60 and pairs.groupby('id').size().eq(20).all()
    assert pairs.groupby('prompt_id').size().eq(3).all()
    metrics = json.loads((results / 'validation_and_test.json').read_text())
    count = 0
    for task in ['ER', 'IP', 'EX']:
        for split in ['validation', 'test']:
            for language in ['en', 'pt']:
                preds = pd.read_csv(results / task / f'{split}_{language}_predictions.csv')
                recorded = metrics[task][f'{split}_{language}']
                assert len(preds) == recorded['n']
                assert abs(accuracy_score(preds.level, preds.predicted_level) - recorded['accuracy']) < 1e-12
                assert abs(f1_score(preds.level, preds.predicted_level, labels=[0, 1, 2], average='macro', zero_division=0) - recorded['macro_f1']) < 1e-12
                assert confusion_matrix(preds.level, preds.predicted_level, labels=[0, 1, 2]).tolist() == recorded['confusion_matrix_0_1_2']
                count += 1
    print(f'Conferidos {len(hashes)} hashes, 60 pares e {count} conjuntos de métricas.')


if __name__ == '__main__':
    verify()
