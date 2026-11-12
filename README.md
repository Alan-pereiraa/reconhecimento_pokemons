# Reconhecimento de Pokémon em Vídeo com Deep Learning

Atividade de Machine Learning e Deep Learning (6º período de Engenharia de Software). O exemplo da professora reconhecia 3 objetos de mesa. Aqui ele foi adaptado para reconhecer **3 Pokémon em vídeo, vistos de vários ângulos**, e ganhou as três melhorias pedidas na Parte 4.

| Classe | Pokémon | Observação |
| --- | --- | --- |
| `pikachu` | Pikachu (nº 25) | amarelo |
| `psyduck` | Psyduck (nº 54) | também amarelo: o par parecido na cor sugerido na atividade |
| `charmander` | Charmander (nº 4) | laranja |
| `fundo` | nenhum | cenário vazio (parede, mesa, objetos) |

## De onde vêm os vídeos

Os vídeos são gerados a partir dos modelos 3D da [Pokémon 3D API](https://pokemon-3d-api.github.io/Showcase/), indicada pela professora. O `gerar_videos.py` abre cada modelo `.glb` no `<model-viewer>` (o visualizador do Showcase) num Chrome sem janela e grava a câmera girando em volta dele:

- **Ângulos:** cada vídeo mistura frente, costas, lados, de cima (~45°), de baixo, perto (Pokémon ocupando quase a tela) e longe (um terço da tela).
- **Cenários:** cada classe tem 2 vídeos em cenário claro e 2 em cenário escuro, com parede, mesa e objetos de cores sorteadas. A câmera balança como se estivesse na mão, e cada quadro tem um pouco de ruído de sensor.
- **Fundo:** 8 vídeos sem Pokémon. O cenário muda a cada segundo, e foram 8 vídeos porque cada vídeo de fundo gera menos fotos e o `extrair_frames.py` avisava que as classes estavam desbalanceadas.
- **Vídeo de teste:** `videos_teste/teste.mp4`, com 48 s. Os 3 Pokémon aparecem um de cada vez, em ângulos variados, sobre uma toalha xadrez que não aparece no treino.

Vídeos gravados com câmera (pelúcia, carta, ou a tela do Showcase filmada) podem ir para as mesmas pastas `videos/<classe>/`. O `extrair_frames.py` usa todos.

## Como rodar

Python 3.12 ou 3.13. Na raiz do projeto:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install tensorflow numpy matplotlib opencv-python pillow playwright
```

O `gerar_videos.py` usa o Google Chrome instalado. Sem ele, rode `playwright install chromium`.

```bash
python gerar_videos.py        # Passo 0: videos/<classe>/ e videos_teste/teste.mp4 (~8 min, precisa de internet)
python extrair_frames.py      # Passo 1: fotos em data/<classe>/
python train.py               # Passo 2: modelo.keras, class_names.json, curvas.png, dataset_imagens.pdf
python detectar.py videos_teste/teste.mp4   # Passo 3: saida_teste.mp4 + linha do tempo
python detectar.py 0          # webcam (q para sair)
```

Para processar um vídeo sem abrir janela: `MOSTRAR=0 python detectar.py ...`.

## Melhorias implementadas (Parte 4)

Os trechos do grupo estão marcados com `[GRUPO]` nos comentários do código.

1. **Suavização e limiar de confiança** (`detectar.py`). Um `deque(maxlen=JANELA)` guarda as probabilidades dos últimos 8 frames, e o rótulo usa a média delas. Se a maior probabilidade média ficar abaixo de 60%, a tela mostra `? (talvez <classe>)`. Para testar outros valores: `JANELA=1`, `JANELA=15`, `LIMIAR=0.8`.
2. **Linha do tempo** (`detectar.py`). Cada frame tem seu rótulo guardado. No final, frames seguidos com o mesmo rótulo viram trechos. Trechos com menos de 0,5 s são descartados, vizinhos iguais são juntados de novo, e `?` e `fundo` ficam de fora.
3. **Fine-tuning** (`train.py`). Depois da fase 1 (base congelada, Adam 1e-3), as últimas 30 camadas da MobileNetV2 são descongeladas. O modelo é recompilado com Adam 1e-5 e treina até mais 8 épocas. O `curvas.png` junta as duas fases e marca o início do fine-tuning com uma linha tracejada. O terminal mostra a matriz de confusão ao final de cada fase.

## Resultados

Dados: 1.075 fotos (charmander 273, fundo 277, pikachu 282, psyduck 243). A validação usa vídeos inteiros: 1 vídeo de cada Pokémon e 2 de fundo.

| Modelo | Acurácia de validação |
| --- | --- |
| Fase 1 (só a cabeça) | 99,33% (1 charmander → psyduck, 1 fundo → charmander) |
| Fase 2 (após fine-tuning) | 100,00% |

![Curvas](curvas.png)

Linha do tempo no vídeo de teste (`saida_teste.mp4`). No roteiro, o Pikachu aparece de 2 a 15 s, o Psyduck de 17,5 a 30,5 s e o Charmander de 33 a 46 s:

```
Linha do tempo (Pokemon identificados):
     2.6s a   15.4s  ->  pikachu
    18.1s a   23.5s  ->  psyduck
    24.2s a   30.8s  ->  psyduck
    33.5s a   46.4s  ->  charmander
Pokemon encontrados no video: ['charmander', 'pikachu', 'psyduck']
```

O Psyduck foi partido em dois trechos porque, por volta de 24 s, ele fica pequeno e longe e a confiança média cai abaixo de 60% (`?`).

A validação alta não quer dizer que o modelo funciona no mundo real. Todos os vídeos usam o **mesmo modelo 3D** de cada Pokémon, então um vídeo de validação novo ainda se parece muito com o treino. Uma pelúcia ou carta filmada na webcam tende a ir pior. Esse é um bom ponto para discutir no relatório.

## Experimentos da Parte 5

| Pergunta | Como rodar |
| --- | --- |
| 1. Split por vídeo × por foto | `SPLIT=foto python train.py` e compare com `python train.py` |
| 2. Fine-tuning com taxa alta | `LR_FINE=1e-3 python train.py` |
| 3. Pokémon de costas fora do treino | `python gerar_videos.py sem-costas pikachu`, apague `data/`, rode `extrair_frames.py` e `train.py`, depois `python gerar_videos.py costas pikachu` e `python detectar.py videos_teste/costas_pikachu.mp4` |
| 4. Pokémon desconhecido | `python gerar_videos.py desconhecido 1` (1 = Bulbasaur) e `python detectar.py videos_teste/desconhecido_1.mp4` |
| 5. Dois Pokémon no mesmo quadro | `python gerar_videos.py dupla` e `python detectar.py videos_teste/dupla.mp4` |

Cada `train.py` sobrescreve `modelo.keras`. Ao terminar um experimento, rode o treino normal de novo.

## Entrega

| Item | Arquivo |
| --- | --- |
| 1. Imagens por classe | `dataset_imagens.pdf` |
| 2. Vídeo de teste anotado e linha do tempo | `saida_teste.mp4` + print do terminal |
| 3. Códigos comentados | `train.py`, `detectar.py` (e `gerar_videos.py`) |
| 4. Curvas e matriz de confusão | `curvas.png` + print do terminal do `train.py` |
| 5. Relatório da Parte 5 | (a fazer) |

## Arquivos

| Arquivo | O que faz |
| --- | --- |
| `gerar_videos.py` | Gera os vídeos de treino e de teste a partir dos modelos 3D |
| `utils.py` | `letterbox`, igual no treino e na detecção (código da professora) |
| `extrair_frames.py` | Extrai as fotos dos vídeos para `data/` (código da professora) |
| `train.py` | Treina o modelo: fase 1 + fine-tuning |
| `detectar.py` | Reconhece Pokémon em vídeo ou webcam, com suavização, limiar e linha do tempo |

`modelos_3d/` (cache dos `.glb`) e `data/` são gerados pelos scripts e ficam fora do git.
