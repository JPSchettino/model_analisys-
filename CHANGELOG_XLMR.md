# Atualização XLM-R — 01/10/2026

## Adaptação executada

Registrados o código exato e o manifesto da execução `ed1cec923ce3`. Backbone `FacebookAI/xlm-roberta-base`, bi-encoder, cross-attention, níveis e rationales, treino EN/PT por grupos e testes independentes EN/PT. O notebook atual usa as mesmas configurações, com envio dos arquivos de avaliação pelo usuário e versões de bibliotecas registradas.

## Correção posterior da análise

A regressão automática de ER da execução original devolveu uma aproximação finita apesar de haver quase separação (Gemma/GPT4 sem casos no nível superior). A convergência declarada pelo otimizador e ausência de warnings não garantiam um estimador finito. O código atual verifica grupos com desfecho binário constante e não fornece OR/p de Wald nesses casos. IP constante também não permite regressão. EX mantém a análise adicional com Holm na família planejada de seis contrastes, incluindo contrastes não estimáveis conservadoramente na correção. Os nove testes pareados primários, scores e pesos permanecem inalterados.

O arquivo histórico contém o comportamento original e não deve ser usado para gerar a tabela de regressão atualizada. `scripts/analyze_epitome_results.py` aplica a correção aos resultados existentes, sem gastar GPU. Mensagens de relatório distinguem anotação ER/IP/EX de avaliações humanas de outros comportamentos MI.

## Verificação

Testes com dados sintéticos verificam o caso de quase separação, o teste exato e Holm por implementações independentes, e a coerência dos resultados da regressão EX. A preparação original também verificou treino/checkpoint com XLM-R pequeno e a execução completa foi realizada no Colab. O registro histórico mantém seu hash original; o código atualizado tem outro hash.
