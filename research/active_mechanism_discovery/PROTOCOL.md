# Protocolo experimental — identificação ativa sob restrições

Versão 1.0, 01/10/2026. Este é um piloto de pesquisa executável. Novidade e competitividade editorial ainda dependem de resultados e comparação externa.

## Pergunta e contribuição candidata

Quanto custa controlar falsas declarações de descoberta quando o agente seleciona as medições, os experimentos têm custos diferentes e algumas intervenções não estão disponíveis?

O estudo separa três fatores: aquisição de dados, regra de declaração e capacidade do catálogo de representar a verdade. A proposta exploratória de aquisição dá prioridade às alternativas que ainda precisam de evidência para serem eliminadas. Sua eficiência será comparada com métodos simples e desenho por informação mútua. Não se pressupõe que ela vencerá.

A unidade de inferência é a **família de equações em um catálogo finito**, com parâmetros discretizados conhecidos como possibilidades. Não há geração livre de novas equações no núcleo desta versão. Isso torna verificáveis as afirmações estatísticas e as equivalências. O painel LLM escolhe experimentos; ele não gera a verdade nem julga o resultado.

Este recorte é mais estreito que descoberta científica geral. Resultados positivos aqui justificam a etapa seguinte, não uma alegação de resolver descoberta irrestrita.

## O que mudou após a pesquisa adicional

- [Universal inference, 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7382245/) já fornece a base para testes e conjuntos de confiança sequenciais baseados em razões de verossimilhança. A validade estatística empregada aqui é uma aplicação desse princípio, não um teorema novo reivindicado.
- [Deep Adaptive Design, ICML 2021](https://proceedings.mlr.press/v139/foster21a.html) já estuda políticas de desenho amortizadas. Treinar uma rede para escolher experimentos também não bastaria como novidade.
- [AutoSciLab](https://arxiv.org/abs/2605.24043) e [MDA](https://arxiv.org/abs/2608.09696) são concorrentes externos necessários em uma versão ampliada. As políticas implementadas aqui não são reproduções desses sistemas.
- [Chernoff Sampling, AISTATS 2022](https://proceedings.mlr.press/v151/mukherjee22a.html) é antecedente do desenho por discriminação. Nossa implementação é uma aproximação simples de alocação, identificada como tal.

## Hipóteses a testar

H1: o critério sequencial mantém o risco de declaração falsa no nível especificado quando verdade, catálogo e ruído satisfazem as condições.

H2: aquisição sensível à evidência que falta pode reduzir custo de resolução correta em relação a EIG e alocação discriminativa convencional. H2 pode ser refutada pelos dados.

H3: restrições nas intervenções tornam necessária uma resposta de equivalência em parte das instâncias; forçar uma única família pode produzir excesso de conclusões.

H4: retirar a verdade do catálogo ou especificar o ruído incorretamente rompe a aplicabilidade da garantia. O experimento mede a gravidade dessa falha; não promete corrigi-la nesta versão.

## Dados e ambientes abertos

Todos os dados são gerados pelo código distribuído, sem arquivos privados, avaliação humana, serviço de API pago ou coleta externa. As leis são exemplos científicos simplificados, não dados experimentais reais. O catálogo completo é público para todos os métodos; o índice da lei verdadeira e as respostas futuras ficam no simulador/avaliador.

**Mecânica simplificada:** seis formas com termos lineares, aditivos, cúbicos, de interação e quadráticos; 48 modelos parametrizados. Valores dos parâmetros: 1/2, 1 e 2. Menu completo com 36 intervenções racionais. Menu restrito: z=x, seis intervenções.

**Cinética simplificada:** Michaelis–Menten, inibição competitiva, não competitiva, incompetitiva, inibição por substrato e Hill com expoente 2; 54 modelos parametrizados. Menu completo com 25 intervenções; restrito a inibidor z=0, cinco intervenções. As formas são controladas e não pretendem cobrir toda a bioquímica.

Medições no eixo de restrição custam 1; intervenções fora dele custam 3. São custos de benchmark declarados, não preços de laboratório. A escala de medição é pública e fixa por domínio. O ruído padrão é gaussiano independente, com desvio conhecido na unidade normalizada.

As equivalências são calculadas com aritmética racional exata em **todo o menu finito de ações permitido**. Nenhuma tolerância numérica define as classes. A conclusão vale para esse menu; não constitui prova de equivalência em todo um domínio contínuo.

## Regra sequencial e seu limite

Seja H o catálogo finito e p_h(y|a) a densidade gaussiana da hipótese h na intervenção a. O agente escolhe a_t usando apenas informações anteriores. O prior público pi tem massa positiva em todas as hipóteses.

Definimos L_t(h)=produto_s p_h(y_s|a_s), Q_t=soma_h pi(h)L_t(h) e E_t(h)=Q_t/L_t(h). A computação usa logaritmos e float64. Constantes gaussianas comuns cancelam exatamente na expressão matemática.

Sob uma hipótese verdadeira h*, Q_t é a densidade sequencial da mistura: a seleção adaptativa de ações é previsível e usa a mesma política sob as hipóteses. Portanto E_t(h*) é um martingale não negativo iniciado em 1. Pela desigualdade de Ville, P(sup_t E_t(h*) >= 1/alpha) <= alpha.

O conjunto C_t mantém hipóteses cujo E_s nunca cruzou 1/alpha até t. O sistema declara:

1. **family:** todas as hipóteses sobreviventes têm a mesma família;
2. **equivalence:** há mais de uma família sobrevivente, mas todas pertencem à mesma classe de respostas no menu;
3. **unresolved:** o orçamento terminou sem as condições anteriores;
4. **empty_catalog:** nenhuma hipótese sobreviveu. Isso sinaliza incompatibilidade com o procedimento, não prova um mecanismo externo específico.

Se h* sobrevive, uma declaração family ou equivalence não pode excluir a verdade no sentido definido acima. Logo a probabilidade de uma declaração errada durante a execução é limitada por alpha nas condições especificadas. Não é necessário dividir alpha pelo tamanho do catálogo para a cobertura da única hipótese verdadeira; isso não deve ser confundido com controle de outras famílias de descobertas simultâneas.

A garantia é por execução, não a probabilidade de zero erros em milhares de execuções. Também não implica que a fração de erros condicionada às declarações seja <= alpha. Ambas as métricas devem ser distinguidas.

**Condições:** verdade exatamente representada, ruído corretamente especificado, ausência de acesso ao futuro, catálogo/prior fixados e cálculo fiel. A discretização não aproxima uma garantia contínua automaticamente. Hiperparâmetros escolhidos após olhar o teste também não devem ser vendidos como protocolo confirmatório independente.

## Políticas e controles

| Método | Aquisição | Declaração |
|---|---|---|
| random_safe | Ação acessível aleatória | Critério sequencial |
| variance_safe | Variância posterior por custo | Critério sequencial |
| eig_safe | Informação mútua Monte Carlo por custo, sobre o átomo do catálogo | Critério sequencial |
| chernoff_safe | Jogo aproximado entre alocação e rival mais difícil; divergência gaussiana por custo | Critério sequencial |
| deficit_safe | Mesmo jogo, ponderando pela evidência que falta para eliminar rivais sobreviventes | Critério sequencial |
| eig_posterior | EIG | Probabilidade posterior de família/classe >= 1-alpha |
| deficit_posterior | Aquisição deficit | Mesmo limiar posterior |
| eig_fixed | EIG até o fim | Família MAP forçada no orçamento final |

O controle eig_fixed é deliberadamente uma regra forçada, útil para diagnóstico; não deve representar todos os métodos científicos existentes. O limiar posterior não possui automaticamente a garantia frequentista uniforme acima.

Deficit divide a separação esperada gaussiana para cada rival pelo déficit atual de log-evidência. A otimização usa atualizações multiplicativas aproximadas e exploração uniforme de 2%, também presente no controle chernoff. O código não reivindica solução ótima do jogo. A ablação crucial é deficit_safe versus chernoff_safe, além de eig_safe.

O painel opcional usa [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) em inferência. Ele recebe histórico, ações/custos, desvio do ruído e oito candidatos com maior pontuação. Responde com um índice de ação. Respostas inválidas usam EIG, com registro explícito da ocorrência. Os mesmos episódios têm controles EIG e deficit. A amostra é pequena e exploratória.

## Cenários e amostragem

- closed: verdade no catálogo, ruído correto, intervenções completas ou restritas.
- off_grid: coeficientes verdadeiros multiplicados por fatores racionais fora da grade, apenas menu completo.
- missing_family: família verdadeira retirada do catálogo, apenas menu completo.
- noise_mismatch: ruído real duas vezes maior que o informado, apenas menu completo.

Os cenários de estresse são reportados separadamente; não há promessa de controle de erro neles. Para off_grid, correct significa recuperação da família estrutural, não dos coeficientes verdadeiros.

Presets:

- smoke: um parâmetro selecionado por família, uma repetição, um ruído, dois domínios e dois menus; 24 episódios por método.
- pilot: dois parâmetros por família, oito repetições, dois ruídos, dois menus closed e três estresses; 1.920 episódios por método, 15.360 execuções no total.
- confirmatory: mesma estrutura com 100 repetições; 24.000 episódios por método, 192.000 execuções. Requer múltiplas sessões conforme tempo medido; não há promessa de completar numa sessão gratuita.

As seleções de coeficientes, ruído e aleatoriedade das políticas têm sementes rastreáveis. Regras de declaração com a mesma política compartilham a fita aleatória, permitindo comparações pareadas. Alterar o batch não altera os episódios. Não treinamos políticas no conjunto de teste.

O preset confirmatory é uma configuração de amostra maior. Antes de usá-lo como estudo confirmatório real, congelar código, decisões e hipóteses após o piloto, escolher uma nova semente e registrar o plano. O nome do preset não realiza pré-registro.

## Análises e leitura dos resultados

Medidas: declaração falsa por execução, resolução correta, tipo de resposta, custo, número de consultas, eliminação da verdade e tempo amortizado por batch. O tempo inclui aquisição e o bookkeeping do batch; não é latência isolada de uma chamada em GPU.

Cada contexto e lei recebe intervalo binomial exato para erros entre repetições independentes. Comparações entre métodos usam episódios pareados e bootstrap por lei. O agregado por lei dá peso igual às leis disponíveis; não é exatamente a mesma estimativa que agrupar todos os episódios indiscriminadamente. O gráfico agregado é descritivo.

Os controles têm os mesmos custos de medição e número máximo de unidades. Custos de computação podem diferir; reportar tempos e configurações. Comparar custo junto com resolução correta: recusar tudo custa pouco.

Nenhum texto de conclusão científica é gerado automaticamente. Não selecionar somente as figuras favoráveis. O painel LLM e os testes de software devem permanecer separados do estudo principal.

## Reprodução e limitações de Colab

O núcleo usa tensores float64 e simulações em batch. Não requer treinamento de grande modelo. A GPU acelera os cálculos de informação e os jogos de alocação. O código detecta CUDA; em CPU funciona com o preset smoke. A memória da GPU depende principalmente de batch × amostras EIG × ações × hipóteses. Batch 32 é conservador para o catálogo atual.

O modelo de 7B usa BF16/FP16 quando há pelo menos 23 GiB livres; caso contrário, tenta NF4 de quatro bits. A instalação de dependências não substitui o PyTorch do Colab. A revisão do modelo, hardware e versões são registradas.

O [Colab não garante GPUs específicas no plano gratuito](https://research.google.com/colaboratory/faq.html). Uma A100 disponível é suficiente como alvo de engenharia para esse painel, mas a execução em A100 ainda precisa ser medida. Há checkpoints atômicos, retomada por episódio e backup periódico opcional no Drive. Nenhuma chave de API é necessária.

## O que falta para uma submissão forte

1. Executar o piloto e testar se H2 sobrevive às comparações; abandonar a política se não agregar.
2. Ampliar famílias, custos e parâmetros, incluindo sensibilidade à discretização.
3. Integrar pelo menos um simulador externo e reproduzir concorrentes recentes com protocolo compatível.
4. Estudar hipóteses compostas com parâmetros contínuos; o teorema discreto não se estende por conveniência.
5. Definir uma contribuição científica original verificável além da aplicação de inferência sequencial já conhecida.

O pacote entrega a infraestrutura, um desenho falsificável e verificações do núcleo. Não entrega evidência de superioridade nem garante aceitação top tier.
