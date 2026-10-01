# EPITOME com XLM-R — avaliação em português

Adaptação multilíngue do EPITOME usada na reavaliação de 01/10/2026.
O fluxo atual está em [`src/epitome_xlmr.py`](src/epitome_xlmr.py) e no
[`notebook do Colab`](notebooks/EPITOME_XLMR_JP.ipynb).

## Executar no Colab

[Abrir no Colab](https://colab.research.google.com/github/JPSchettino/model_analisys-/blob/main/notebooks/EPITOME_XLMR_JP.ipynb)

1. Selecione GPU e execute as células.
2. Envie `baseline_original.csv` e `baseline_classificador_pares_pt.csv`, presentes no ZIP `resultados_epitome_xlmr.zip` da execução anterior. São os mesmos 60 pares com os rótulos das duas rodadas anteriores; o código confere a correspondência antes do treino.
3. Autorize o Drive. Resultados e checkpoints ficam em `Meu Drive/EPITOME_XLMR_JP/run_...`.
4. O notebook gera `resultados_epitome_xlmr.zip`. Os pesos permanecem no Drive e ficam fora do ZIP (aproximadamente 3,5 GB no total).

O notebook público solicita os arquivos de avaliação, em vez de incorporar seus rótulos ao código. As fontes de treino EN/PT são públicas e fixadas por commit e hash. O conjunto de enunciados/respostas usado no projeto também está em `dataset/avalia_modelos.csv`, mas esse arquivo não contém os rótulos das avaliações anteriores.

## Método

- Dois encoders `FacebookAI/xlm-roberta-base`, seeker congelado, atenção cruzada, classificação dos níveis 0/1/2 e extração de rationales.
- Três avaliadores: ER (reações emocionais), IP (interpretações) e EX (explorações).
- Treino bilíngue para níveis, rationales supervisionadas somente em inglês com trechos alinháveis.
- Separação por grupos de enunciado, mantendo duplicatas e traduções juntas.
- Seleção por Macro-F1 de validação PT e teste separado EN/PT.
- Quatro épocas, seed12, comprimento64, learning rate2e-5, batch efetivo32; configuração em `config/epitome_xlmr.json`.
- Comparações primárias: teste exato pareado dos sinais e Holm sobre nove contrastes.
- Regressão suplementar com erro agrupado por enunciado, detectando ausência de variação e separação antes de estimar OR.

## Reanalisar uma execução concluída, sem treino

Use um ambiente com PyTorch >=2.3, adequado à plataforma, e depois:

```bash
python -m pip install -r requirements-xlmr.txt
python scripts/analyze_epitome_results.py --input resultados_epitome_xlmr.zip --output resultados_corrigidos
python -m unittest discover -s tests -v
```

Também é aceito o caminho de uma pasta extraída em `--input`. O script gera distribuições, testes exatos/Holm, regressão corrigida, tabela de teste quando há métricas e um resumo da análise. Ele não carrega checkpoints nem baixa pesos.

## Registro da execução usada no artigo

Run `ed1cec923ce3`: A10040GB, Torch2.11.0+cu128, Transformers4.48.3, NumPy2.1.3, pandas2.2.3 e scikit-learn1.6.1. Divisão efetiva por idioma: 1.850 treino, 621 validação e 613 teste. A versão do scikit-learn influencia o agrupamento dos folds; as versões da execução estão fixadas em `requirements-xlmr.txt`.

`experiments/2026-10-01/epitome_xlmr_used.py` preserva exatamente o código executado e seu SHA-256 corresponde ao manifesto. É um registro histórico. A versão recomendada em `src/epitome_xlmr.py` acrescenta a correção da regressão ER e esclarece o texto sobre validação humana. Isso não altera os pesos treinados, os rótulos nem os testes pareados. Consulte [CHANGELOG_XLMR.md](CHANGELOG_XLMR.md).

Alterar o código cria outra identidade de execução/pasta no notebook. Para corrigir somente as tabelas de um treino existente, use o script de análise acima, sem executar novamente o treino.

Os avaliadores têm limites: classes pouco frequentes podem não ser previstas, níveis PT vêm de traduções anotadas em EN, e os escores não medem eficácia clínica. A avaliação humana de comportamentos MI e a anotação específica ER/IP/EX são avaliações complementares. Este repositório contém o avaliador de empatia, não os pesos do modelo de suporte a álcool e saúde.

## Código de origem e atribuição

Baseado em Sharma et al. (2020), [A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support](https://aclanthology.org/2020.emnlp-main.425/). O código original e suas instruções foram preservados abaixo; `requirements.txt`, `src/train.py` e `src/test.py` correspondem ao fluxo legado. Para XLM-R use as entradas indicadas acima.

---

## Documentação original (fluxo legado)
# Empathy in Text-based Mental Health Support
This repository contains codes and dataset access instructions for the [EMNLP 2020 publication](https://arxiv.org/pdf/2009.08441) on understanding empathy expressed in text-based mental health support.

If this code or dataset helps you in your research, please cite the following publication:
```bash
@inproceedings{sharma2020empathy,
    title={A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support},
    author={Sharma, Ashish and Miner, Adam S and Atkins, David C and Althoff, Tim},
    year={2020},
    booktitle={EMNLP}
}
```

## Introduction

We present a computational approach to understanding how empathy is expressed in online mental health platforms. We develop a novel unifying theoretically-grounded framework for characterizing the communication of empathy in text-based conversations. We collect and share a corpus of 10k (post, response) pairs annotated using this empathy framework with supporting evidence for annotations (rationales). We develop a multi-task RoBERTa-based bi-encoder model for identifying empathy in conversations and extracting rationales underlying its predictions. Experiments demonstrate that our approach can effectively
identify empathic conversations. We further apply this model to analyze 235k mental health interactions and show that users do not self-learn empathy over time, revealing opportunities for empathy training and feedback.

For a quick overview, check out [bdata.uw.edu/empathy](http://bdata.uw.edu/empathy/). For a detailed description of our work, please read our [EMNLP 2020 publication](https://arxiv.org/pdf/2009.08441).

## Quickstart

### 1. Prerequisites

Our framework can be compiled on Python 3 environments. The modules used in our code can be installed using:
```
$ pip install -r requirements.txt
```


### 2. Prepare dataset
A sample raw input data file is available in [dataset/sample_input_ER.csv](dataset/sample_input_ER.csv). This file (and other raw input files in the [dataset](dataset) folder) can be converted into a format that is recognized by the model using with following command:
```
$ python3 src/process_data.py --input_path dataset/sample_input_ER.csv --output_path dataset/sample_input_model_ER.csv
```

### 3. Training the model
For training our model on the sample input data, run the following command:
```
$ python3 src/train.py \
	--train_path=dataset/sample_input_model_ER.csv \
	--lr=2e-5 \
	--batch_size=32 \
	--lambda_EI=1.0 \
	--lambda_RE=0.5 \
	--save_model \
	--save_model_path=output/sample_ER.pth
```

**Note:** You may need to create an `output` folder in the main directory before running this command.

For training the models on the full Reddit dataset, these are the three commands you can run for Emotional Reactions, Interpretations, and Explorations respectively:

**1. Emotional Reactions**
```
python3 src/train.py \
--train_path=dataset/emotional-reactions-reddit.csv \
--lr=2e-5 \
--batch_size=32 \
--lambda_EI=1.0 \
--lambda_RE=0.5 \
--save_model \
--save_model_path=output/reddit_ER.pth
```

 **2. Interpretations**
```
python3 src/train.py \
--train_path=dataset/interpretations-reddit.csv \
--lr=2e-5 \
--batch_size=32 \
--lambda_EI=1.0 \
--lambda_RE=0.5 \
--save_model \
--save_model_path=output/reddit_IP.pth
```

**3. Explorations**
```
python3 src/train.py \
--train_path=dataset/explorations-reddit.csv \
--lr=2e-5 \
--batch_size=32 \
--lambda_EI=1.0 \
--lambda_RE=0.5 \
--save_model \
--save_model_path=output/reddit_EX.pth
```


### 4. Testing the model
For testing our model on the sample test input, run the following command:
```
$ python3 src/test.py \
	--input_path dataset/sample_test_input.csv \
	--output_path dataset/sample_test_output.csv \
	--ER_model_path output/sample_ER.pth \
	--IP_model_path output/sample_IP.pth \
	--EX_model_path output/sample_EX.pth
```

## Training Arguments

The training script accepts the following arguments: 

Argument | Type | Default value | Description
---------|------|---------------|------------
lr | `float` | `2e-5` | learning rate
lambda_EI | `float` | `0.5` | weight of empathy identification loss 
lambda_RE |  `float` | `0.5` | weight of rationale extraction loss
dropout |  `float` | `0.1` | dropout
max_len | `int` | `64` | maximum sequence length
batch_size | `int` | `32` | batch size
epochs | `int` | `4` | number of epochs
seed_val | `int` | `12` | seed value
train_path | `str` | `""` | path to input training data
dev_path | `str` | `""` | path to input validation data
test_path | `str` | `""` | path to input test data
do_validation | `boolean` | `False` | If set True, compute results on the validation data
do_test | `boolean` | `False` | If set True, compute results on the test data
save_model | `boolean` | `False` | If set True, save the trained model  
save_model_path | `str` | `""` | path to save model 


## Dataset Access Instructions

The Reddit portion of our collected dataset is available inside the [dataset](dataset) folder. The csv files with annotations on the three empathy communication mechanisms are `emotional-reactions-reddit.csv`, `interpretations-reddit.csv`, and `explorations-reddit.csv`. Each csv file contains six columns:
```
sp_id: Seeker post identifier
rp_id: Response post identifier
seeker_post: A support seeking post from an online user
response_post: A response/reply posted in response to the seeker_post
level: Empathy level of the response_post in the context of the seeker_post
rationales: Portions of the response_post that are supporting evidences or rationales for the identified empathy level. Multiple portions are delimited by '|'
```

For accessing the TalkLife portion of our dataset for non-commercial use, please contact the TalkLife team [here](mailto:research@talklife.co). 


