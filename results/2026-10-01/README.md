# Resultados conferidos — run ed1cec923ce3

Execução concluída em 01/10/2026 em A100 40 GB. O manifesto e o código exato estão em `../../experiments/2026-10-01/`.

## Leitura

- `rotulos_epitome_xlmr_60_pares.csv`: 20 respostas por modelo, níveis 0/1/2, logits, probabilidades e rationales previstos.
- `resumo_por_modelo.csv` e `tabela_distribuicao.tex`: contagens finais.
- `testes_pareados.csv`: nove testes exatos pareados com Holm.
- `regressao_log_odds_e_OR_corrigida.csv`: análise suplementar; ER não estimável por separação, IP constante, EX estimável com 20 grupos. Use a versão corrigida.
- `validation_and_test.json`, `desempenho_por_classe.csv`, `ER/`, `IP/`, `EX/`: métricas, predições de validação/teste, histórico e época selecionada.
- `split_assignments.csv`, auditorias, fontes e ambiente: rastreabilidade.
- `baseline_*.csv`: entradas das duas rodadas anteriores usadas no notebook. São predições automáticas, não anotações humanas.
- `SHA256SUMS.json`: integridade dos arquivos publicados.

## Interpretação

EX: Gemma × GPT4 p Holm = 0,02734375; Llama × GPT4 = 0,0087890625. Gemma × Llama não passou pelo critério ajustado. ER/IP não demonstram diferenças nem equivalência. O teste PT tem Macro-F1 ER 0,49736, IP 0,56398, EX 0,56819. Classes raras não são previstas, conforme métricas por classe. Estes escores não medem eficácia clínica.

A avaliação humana do artigo permanece independente destas predições. Checkpoints treinados permanecem no Drive da execução; nenhum peso está neste diretório. O código reconstrói o encoder seeker congelado pela revisão fixa registrada no manifesto.

## Reproduzir as tabelas sem GPU

Na raiz do repositório:

```bash
python scripts/analyze_epitome_results.py --input results/2026-10-01 --output resultados_corrigidos
python scripts/verify_published_results.py
```
