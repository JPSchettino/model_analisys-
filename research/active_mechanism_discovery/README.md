# Experimento de descoberta ativa — Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JPSchettino/model_analisys-/blob/active-mechanism-discovery/research/active_mechanism_discovery/Discovery_Colab.ipynb)

O notebook principal busca os arquivos de um commit fixo deste repositório e verifica SHA-256 antes de executá-los. Abra pelo botão acima. A alternativa `Discovery_Colab_standalone.ipynb` contém as fontes e pode ser enviada pelo menu Arquivo → Fazer upload de notebook.

1. Selecione um runtime com GPU. Se houver A100, use-a; ela não é garantida no plano gratuito.
2. Execute as células em ordem. O padrão `pilot` faz o experimento principal. `smoke` é uma verificação rápida.
3. Autorize a montagem do seu Drive se quiser backup entre sessões. Sem isso, os resultados ficam no runtime até o download.
4. Ao final, o notebook gera gráficos, tabelas e um ZIP. Envie esse ZIP para analisarmos o resultado.
5. Se a execução parar pelo limite configurado ou desconectar, abra o mesmo notebook, mantenha a configuração e execute novamente. O backup do Drive restaura o último snapshot.

O padrão contém oito métodos, dois domínios, ruído e intervenções restritas, além de estresses em que a garantia não se aplica. O painel opcional de 7B usa pesos abertos e nenhuma API paga. Ele pode ser desligado com `RUN_LLM_PANEL=False`.

**Esta versão estuda identificação de mecanismos em um catálogo finito.** As equivalências são exatas dentro do menu de intervenções definido. O LLM não avalia resultados. Consulte PROTOCOL.md para hipóteses, regra estatística, comparações e limites.

Arquivos gerados:

- `summary.csv`: resultados por domínio, contexto, ruído e método
- `risk_by_instance.csv`: intervalos para taxas de erro por lei
- `paired_comparisons.csv`: comparações pareadas com EIG, bootstrap por lei
- `main_results.png` e `.pdf`: gráficos descritivos
- `episodes.csv`: dados completos para reanálise
- `manifest.json`, `status.json`, `runtime_packages.txt`: configuração, progresso e ambiente
- `llm_panel/`: painel separado, prompts, respostas, revisão dos pesos e ocorrências de fallback
- `PROTOCOL.md` e fontes do código: cópia junto com os resultados

Sem Colab, com dependências instaladas: `python discovery.py --preset smoke --out results --device cpu`.

Os resultados locais de `validation_cpu` validam o funcionamento do software. Não constituem a amostra principal do artigo. A execução CUDA e o carregamento real dos pesos precisam ser confirmados no Colab.
