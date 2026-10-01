"""Build a standalone notebook containing the exact source and protocol."""
import argparse,hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--repo',default='')
parser.add_argument('--ref',default='')
parser.add_argument('--subdir',default='')
args=parser.parse_args()
if bool(args.repo)!=bool(args.ref): raise ValueError('--repo e --ref devem ser usados juntos')
cells=[]
def md(s): cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
def code(s): cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[]))

md('''# Quando uma descoberta científica é justificável?

Experimento reproduzível de identificação ativa de mecanismos, com ruído, intervenções restritas, custos e regras de declaração. **Versão de pesquisa 1.0 — 01/10/2026.**

O notebook é autossuficiente: ele cria o código, gera os dados, compara oito métodos, salva checkpoints e exporta as tabelas. Não há API paga nem avaliação humana. A100 é um alvo de execução, mas o Colab gratuito não garante esse hardware.

**Escopo:** identificação em um catálogo finito, não descoberta livre de toda equação possível. A garantia vale para o catálogo e ruído especificados; os testes de estresse são reportados separadamente. O objetivo do piloto é descobrir se a contribuição candidata merece ser ampliada. Não existe promessa de resultado ou aceitação top tier.

Use **Ambiente de execução → Alterar tipo de ambiente → GPU**, e execute as células em ordem. O padrão `pilot` é a primeira execução científica exploratória. O Drive é opcional, mas necessário para recuperação após perda do runtime. O painel LLM só começa depois que o núcleo termina.
''')

code('''import os, sys, subprocess, importlib.util
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ.setdefault('MPLCONFIGDIR','/tmp/discovery_matplotlib')
# Keep Colab's working PyTorch/CUDA installation. Install only missing core packages.
packages={'numpy':'numpy>=1.26', 'scipy':'scipy>=1.12', 'pandas':'pandas>=2.2',
          'matplotlib':'matplotlib>=3.8', 'torch':'torch>=2.4'}
missing=[requirement for module,requirement in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable,'-m','pip','install',*missing],check=True)
import torch
print('Python:',sys.version.split()[0], '| PyTorch:',torch.__version__)
if torch.cuda.is_available():
    props=torch.cuda.get_device_properties(0)
    print('GPU:',props.name,'| memória:',round(props.total_memory/2**30,1),'GiB')
    torch.backends.cuda.matmul.allow_tf32=False
else:
    print('Sem CUDA. Use PRESET="smoke" para a verificação em CPU.')
''')

files={name:(HERE/name).read_text() for name in ['discovery.py','llm_panel.py','test_discovery.py','PROTOCOL.md','README.md']}
checks={k:hashlib.sha256(v.encode()).hexdigest() for k,v in files.items()}
bootstrap=("# Embedded, inspectable source.\nEMBEDDED_FILES = "+repr(files)+"\n"+
           "EXPECTED_SHA256 = "+repr(checks)+"\n")
if args.repo:
    bootstrap=("# Fetch exact reviewed sources from a fixed Git commit; verify every file.\n"+
        "REPOSITORY = "+repr(args.repo)+"\nSOURCE_COMMIT = "+repr(args.ref)+"\nSOURCE_SUBDIR = "+repr(args.subdir.strip('/'))+"\n"+
        "EXPECTED_SHA256 = "+repr(checks)+"\n"+'''
import urllib.request
EMBEDDED_FILES={}
for name in EXPECTED_SHA256:
    relative='/'.join(part for part in (SOURCE_SUBDIR,name) if part)
    url=f'https://raw.githubusercontent.com/{REPOSITORY}/{SOURCE_COMMIT}/{relative}'
    with urllib.request.urlopen(url,timeout=60) as response:
        EMBEDDED_FILES[name]=response.read().decode('utf-8')
print('Código:',REPOSITORY,'commit',SOURCE_COMMIT)
''')
    cells[0]['source']=[x.replace('O notebook é autossuficiente: ele cria o código',
        'O notebook carrega código de um commit fixo do GitHub, verifica os arquivos') for x in cells[0]['source']]
code(bootstrap+'''
import hashlib
ROOT=Path(os.environ.get('DISCOVERY_ROOT','/content/mechanism_discovery'))
ROOT.mkdir(parents=True,exist_ok=True)
for name,content in EMBEDDED_FILES.items():
    assert hashlib.sha256(content.encode()).hexdigest()==EXPECTED_SHA256[name]
    (ROOT/name).write_text(content)
sys.path.insert(0,str(ROOT))
import discovery
print('Código materializado e verificado em',ROOT)
''')

md('''## Configuração

`smoke`: 192 execuções pequenas. `pilot`: 15.360 execuções, em blocos com retomada. `confirmatory`: 192.000 execuções; requer protocolo congelado, nova semente e possivelmente várias sessões.

`MAX_MINUTES` limita cada chamada do núcleo. Execute novamente a célula principal para continuar. O núcleo usa float64 para a evidência. O painel opcional usa um modelo aberto de 7B, com precisão ajustada à memória livre.
''')
code('''PRESET = 'pilot'  # 'smoke', 'pilot', 'confirmatory'
SEED = 20261001 if PRESET != 'confirmatory' else 20261107
BUDGET = 60
ALPHA = 0.05
BATCH = 32
MAX_MINUTES = 45
PERSIST_DRIVE = True
RUN_LLM_PANEL = True
LLM_CASES = 12
LLM_BUDGET = 24

from discovery import Config, build_cases, digest
cfg=Config(preset=PRESET,seed=SEED,budget=BUDGET,alpha=ALPHA,batch=BATCH,max_minutes=MAX_MINUTES)
import dataclasses
scientific_config=dataclasses.asdict(cfg)
for key in ('batch','device','max_minutes','trace_per_group'): scientific_config.pop(key)
run_id=PRESET+'_'+digest({'config':scientific_config,'source':EXPECTED_SHA256['discovery.py']})[:12]
RUN_DIR=ROOT/'runs'/run_id
RUN_DIR.mkdir(parents=True,exist_ok=True)
print('Execução:',run_id,'| episódios × métodos:',len(build_cases(cfg))*len(cfg.methods))
''')

code('''import json, time, zipfile, shutil
DRIVE_BACKUP = None
if PERSIST_DRIVE:
    try:
        from google.colab import drive
    except ImportError:
        print('Drive indisponível fora do Colab; usando armazenamento local.')
    else:
        drive.mount('/content/drive')
        backup_root=Path('/content/drive/MyDrive/Mechanism_Discovery_runs')
        backup_root.mkdir(parents=True,exist_ok=True)
        DRIVE_BACKUP=backup_root/(run_id+'.zip')
        if DRIVE_BACKUP.exists() and not (RUN_DIR/'manifest.json').exists():
            with zipfile.ZipFile(DRIVE_BACKUP) as z:
                for member in z.infolist():
                    if not (RUN_DIR/member.filename).resolve().is_relative_to(RUN_DIR.resolve()):
                        raise ValueError('Caminho inválido no backup.')
                z.extractall(RUN_DIR)
            print('Backup restaurado:',DRIVE_BACKUP.name)

last_backup=[0.0]
def checkpoint(folder,force=False):
    if DRIVE_BACKUP is None: return
    if not force and time.monotonic()-last_backup[0]<120: return
    local_zip=ROOT/'checkpoint_staging.zip'
    with zipfile.ZipFile(local_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in Path(folder).rglob('*'):
            if p.is_file() and not p.name.endswith('.tmp'):
                z.write(p,p.relative_to(folder))
    pending=DRIVE_BACKUP.with_suffix('.pending')
    shutil.copyfile(local_zip,pending)
    os.replace(pending,DRIVE_BACKUP)
    last_backup[0]=time.monotonic()

for name in EMBEDDED_FILES:
    (RUN_DIR/name).write_text(EMBEDDED_FILES[name])
print('Backup periódico:',str(DRIVE_BACKUP) if DRIVE_BACKUP else 'desligado — baixe os resultados antes de sair')
''')

md('''## Verificação antes de gastar a sessão

Testes do cálculo da evidência, equivalência exata, respeito ao orçamento, retomada, independência do tamanho do batch e um teste estatístico de regressão. Estes testes verificam o software; não substituem a amostra científica.
''')
code('''import unittest, test_discovery
suite=unittest.defaultTestLoader.loadTestsFromModule(test_discovery)
result=unittest.TextTestRunner(verbosity=1).run(suite)
assert result.wasSuccessful(), 'Corrija o erro antes do estudo principal.'
''')

md('''## Experimento principal

Pode ser reexecutado com a mesma configuração. Episódios concluídos são reutilizados. Alterações científicas ou de código criam outra pasta. Um limite de tempo salva o progresso e permite continuar depois.
''')
code('''from discovery import run_suite, summarize
started=time.monotonic()
run_suite(cfg,RUN_DIR,checkpoint=checkpoint)
print('Minutos nesta chamada:',round((time.monotonic()-started)/60,2))
''')
code('''summary=summarize(RUN_DIR)
display(summary)
from IPython.display import Image, display
display(Image(filename=str(RUN_DIR/'main_results.png')))
status=json.loads((RUN_DIR/'status.json').read_text())
print('Núcleo completo:',status['completed'])
if not status['completed']:
    print('Resultados parciais. Reexecute a célula principal para continuar.')
checkpoint(RUN_DIR,force=True)
''')

md('''## Painel exploratório com LLM aberto

O modelo sugere a próxima ação, recebe apenas informações disponíveis ao agente e não atribui notas. Os resultados ficam separados. Baixa cerca de 15 GB de pesos no modo sem quantização; tempo e memória dependem do runtime. Respostas inválidas usam uma política de fallback explicitamente registrada.

O painel requer o núcleo completo e GPU. A revisão dos pesos é fixada no manifesto. Se esta parte falhar por download ou memória, os resultados do núcleo continuam salvos; o erro é registrado e os arquivos podem ser exportados na célula seguinte.
''')
code('''if RUN_LLM_PANEL and status['completed'] and torch.cuda.is_available():
    try:
        subprocess.run([sys.executable,'-m','pip','install',
            'transformers>=4.57,<6','accelerate>=1,<2','bitsandbytes>=0.48,<1',
            'huggingface_hub>=0.34'],check=True)
        from llm_panel import run_panel
        display(run_panel(RUN_DIR/'llm_panel',max_cases=LLM_CASES,budget=LLM_BUDGET,seed=SEED,checkpoint=checkpoint))
    except Exception as exc:
        discovery.atomic_json(RUN_DIR/'llm_panel_error.json',
            {'type':type(exc).__name__,'message':str(exc),'completed':False})
        print('Painel LLM incompleto:',type(exc).__name__,str(exc))
        print('O núcleo está salvo. Você pode exportar os arquivos e depois retomar o painel.')
    finally:
        checkpoint(RUN_DIR,force=True)
else:
    print('Painel não executado: confira RUN_LLM_PANEL, conclusão do núcleo e GPU.')
''')

md('''## Exportar tudo

O ZIP contém tabelas, gráficos, episódios, fontes, protocolo, ambiente e painel LLM quando executado. Não inclui pesos de modelo. Envie esse ZIP para análise; gráficos favoráveis isolados não bastam para concluir superioridade.
''')
code('''freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
(RUN_DIR/'runtime_packages.txt').write_text(freeze)
discovery.atomic_json(RUN_DIR/'final_environment.json',discovery.environment())
checkpoint(RUN_DIR,force=True)
export_path=ROOT/(run_id+'_resultados.zip')
with zipfile.ZipFile(export_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in RUN_DIR.rglob('*'):
        if p.is_file() and not p.name.endswith('.tmp'):
            z.write(p,p.relative_to(RUN_DIR))
print('Pronto:',export_path)
try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download(str(export_path))
''')

md('''## Antes de interpretar

- `closed` testa o procedimento com verdade no catálogo e ruído correto. Os outros cenários não recebem a mesma garantia.
- `family` identifica uma forma, sem certificar seus coeficientes contínuos. `equivalence` vale para o menu finito de intervenções.
- Intervalos de erro estão em `risk_by_instance.csv`. O gráfico agregado é descritivo.
- Uma taxa baixa de erro com pouca resolução não estabelece utilidade. Compare as duas e o custo.
- A validade estatística usa resultados conhecidos de inferência sequencial. A novidade candidata está no problema experimental e na aquisição orientada à evidência; precisa ser demonstrada.
- As variantes simples deste notebook não substituem a comparação externa com AutoSciLab/MDA. Consulte `PROTOCOL.md` para o plano de ampliação.
''')

for i,c in enumerate(cells): c['id']=f'discovery-{i:02d}'
nb=dict(nbformat=4,nbformat_minor=5,metadata={
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.12'},
    'colab':{'name':'Discovery_Colab.ipynb','provenance':[]},'accelerator':'GPU'},cells=cells)
path=HERE/'Discovery_Colab.ipynb'
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1))
print(path)
