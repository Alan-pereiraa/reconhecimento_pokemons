# Provas da Parte 5

Resultados dos experimentos usados no `relatorio_parte5.pdf`. Cada pasta tem os logs do treino, as curvas, a linha do tempo do `detectar.py` e prints dos vídeos anotados.

## Como os vídeos de teste foram avaliados

Os vídeos de teste são gerados pelo `gerar_videos.py` com um roteiro fixo. Por isso sabemos em que segundo cada Pokémon aparece. O `scripts/medir.py` passa o vídeo pelo modelo do mesmo jeito que o `detectar.py` (média de 8 quadros, limiar de 60%) e compara cada quadro com o roteiro. Os arquivos `medicao_*.json` são a saída desse script: o acerto de cada trecho e o que o modelo respondeu.

## Pastas

| Pasta | Pergunta | Arquivos principais |
| --- | --- | --- |
| `0_modelo_normal/` | Referência: treino normal (split por vídeo, `LR_FINE=1e-5`) | `treino.log`, `curvas.png`, `medicao_teste.json`, `linha_do_tempo_teste.txt` |
| `1_split_por_foto/` | 1. Split por foto × por vídeo | `treino_split_foto.log`, `curvas_split_foto.png`, `medicao_teste.json` |
| `2_taxa_alta/` | 2. Fine-tuning com taxa alta | `treino_lr_1e-3.log`, `curvas_lr_1e-3.png`, `medicao_teste.json`, `quadro_toalha_vazia_16s.jpg` |
| `3_costas/` | 3. Pokémon de costas fora do treino | `treino_sem_costas.log`, `medicao_costas_modelo_sem_costas.json`, `medicao_costas_modelo_normal.json`, `quadro_pikachu_de_costas.jpg` |
| `4_desconhecido/` | 4. Pokémon desconhecido | `linha_do_tempo_<pokemon>.txt`, `quadro_<pokemon>.jpg`, `medicoes/<pokemon>_limiar_<L>.json` |
| `5_dupla/` | 5. Dois Pokémon no mesmo quadro | `linha_do_tempo_dupla.txt`, `medicao_dupla.json`, `abordagens_limiar_janela_<L>.json`, `quadro_dupla_*.jpg` |
| `scripts/` | Scripts de medição | `medir.py`, `dupla_abordagens.py` |

O print `2_taxa_alta/quadro_toalha_vazia_16s.jpg` foi montado com as probabilidades que cada modelo deu para aquele quadro (salvas pelo `medir.py`). Os outros prints foram tirados dos vídeos `saida_*.mp4` gerados pelo `detectar.py`.

## Como reproduzir

Na raiz do projeto, com os vídeos de `videos/`:

```bash
python extrair_frames.py
python train.py                                   # 0: modelo normal
python docs/experimentos_parte5/scripts/medir.py videos_teste/teste.mp4 teste

SPLIT=foto python train.py                        # 1
LR_FINE=1e-3 python train.py                      # 2

python gerar_videos.py sem-costas pikachu         # 3 (depois apague data/, rode extrair_frames.py e train.py)
python gerar_videos.py costas pikachu
python docs/experimentos_parte5/scripts/medir.py videos_teste/costas_pikachu.mp4 simples:pikachu

python gerar_videos.py desconhecido 1             # 4 (também 7, 26, 172 e 133)
LIMIAR=0.8 python docs/experimentos_parte5/scripts/medir.py videos_teste/desconhecido_1.mp4 simples:desconhecido

python gerar_videos.py dupla                      # 5
python detectar.py videos_teste/dupla.mp4
LJ=0.6 python docs/experimentos_parte5/scripts/dupla_abordagens.py videos_teste/dupla.mp4 abordagens.json
```

Cada `train.py` sobrescreve o `modelo.keras`. Os experimentos 1, 2 e 3 foram treinados em cópias separadas do projeto, ao mesmo tempo. Os experimentos 4 e 5 usam o modelo normal.
