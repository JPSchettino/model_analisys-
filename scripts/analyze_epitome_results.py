"""Gera tabelas corrigidas de uma execução EPITOME já concluída, sem GPU."""
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import epitome_xlmr as e


def analyze(input_path, output):
    input_path, output = Path(input_path), Path(output)
    if input_path.is_dir():
        def read(name):
            return (input_path / name).read_bytes()
        members = {p.relative_to(input_path).as_posix() for p in input_path.rglob('*') if p.is_file()}
        archive = None
    else:
        archive = zipfile.ZipFile(input_path)
        read = archive.read
        members = set(archive.namelist())
    try:
        pairs = pd.read_csv(io.BytesIO(read('rotulos_epitome_xlmr_60_pares.csv'))).fillna('')
        summary, tests = e.analysis_tables(pairs)
        regression, status = e.regression_table(pairs)
        metrics = json.loads(read('validation_and_test.json')) if 'validation_and_test.json' in members else {}
        manifest = json.loads(read('manifest.json')) if 'manifest.json' in members else {}
    finally:
        if archive is not None:
            archive.close()
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / 'resumo_por_modelo.csv', index=False)
    tests.to_csv(output / 'testes_pareados.csv', index=False)
    regression.to_csv(output / 'regressao_log_odds_e_OR_corrigida.csv', index=False)
    e.write_json(output / 'regression_status_corrigido.json', status)
    for filename, frame in [('tabela_distribuicao.tex', summary), ('tabela_testes_pareados.tex', tests),
                            ('tabela_regressao_OR_corrigida.tex', regression)]:
        frame.to_latex(output / filename, index=False, escape=True, float_format=lambda x: f'{x:.5g}')
    performance = []
    for task, task_metrics in metrics.items():
        for lang in ['en', 'pt']:
            m = task_metrics['test_' + lang]
            performance.append({'mechanism': task, 'language': lang, 'n': m['n'], 'accuracy': m['accuracy'],
                                'macro_f1': m['macro_f1'], 'classes_not_predicted': str(m['classes_not_predicted'])})
    if performance:
        pd.DataFrame(performance).to_csv(output / 'desempenho_teste.csv', index=False)
    notes = ['# Análise das predições EPITOME', '',
             f'Execução: `{manifest.get("run_id", "não informado")}`.', '',
             'Nove comparações pareadas exatas com Holm; regressão adicional com checagem de separação. '
             'Esta análise usa os rótulos salvos: não modifica predições, checkpoints nem treino.', '']
    significant = tests[tests.significant_at_0_05]
    for row in significant.itertuples():
        notes.append(f'- {row.metric}: {row.model_A} × {row.model_B}, p Holm={row.p_holm_9:.5f}.')
    if significant.empty:
        notes.append('Nenhuma comparação passou pelo critério ajustado de 0,05. Isso não demonstra equivalência.')
    notes += ['', '## Regressão adicional', '']
    for task, item in status.items():
        notes.append(f'- {task}: ' + ('estimável; manter como análise suplementar.' if item['available'] else item['reason']))
    notes += ['', '## Limites', '',
              'A ausência de classes previstas deve ser reportada. Níveis das traduções PT são transferidos de EN. '
              'Avaliações humanas de outros comportamentos MI complementam esses escores, mas não fornecem automaticamente '
              'rótulos humanos ER/IP/EX. Predições não medem eficácia clínica.', '']
    (output / 'resumo_analise.md').write_text('\n'.join(notes), encoding='utf-8')
    print(tests.to_string(index=False))
    print('Tabelas e regressão corrigida:', output)
    return summary, tests, regression, status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='ZIP de resultados ou pasta extraída')
    parser.add_argument('--output', default='resultados_corrigidos')
    args = parser.parse_args()
    analyze(args.input, args.output)
