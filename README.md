# Reconhecimento de Pokémons com CNN

Classifica, frame a frame, o Pokémon que aparece em um vídeo ou na webcam. O modelo usa transfer learning com a MobileNetV2 pré-treinada na ImageNet: a base fica congelada e só a camada final é treinada.

As classes são definidas pelas subpastas de `videos/`. Hoje são `bulbasaur`, `charmander`, `pikachu`, `squirtle` e `fundo` (nenhum Pokémon em cena).

## Instalação e execução

### Pré-requisitos

- Python 3 com `pip` (o Python 3.13 é compatível com o TensorFlow 2.21)
- Webcam, se quiser rodar o reconhecimento ao vivo

No Debian/Ubuntu, se o comando `python3 -m venv` der erro, instale o pacote do venv antes:

```bash
sudo apt install python3-venv
```

### Instalar as dependências

Rode os comandos na raiz do projeto. Eles criam um ambiente virtual em `.venv/`, para não misturar as bibliotecas com as do sistema.

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow opencv-python numpy matplotlib pillow
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install tensorflow opencv-python numpy matplotlib pillow
```

O TensorFlow tem algumas centenas de MB, então a instalação pode demorar. Para conferir se tudo foi instalado, rode:

```bash
python -c "import tensorflow, cv2, matplotlib, PIL; print('TensorFlow', tensorflow.__version__)"
```

Se o `pip` não encontrar uma versão do TensorFlow, a sua versão do Python ainda não é compatível. Nesse caso, crie o ambiente com uma versão anterior do Python.

### Rodar o projeto

O ambiente precisa estar ativado em todo terminal novo: rode `source .venv/bin/activate` (ou `.venv\Scripts\Activate.ps1` no Windows). Rode os scripts a partir da raiz do projeto, pois eles usam caminhos relativos.

Depois de colocar os vídeos em `videos/` (passo 1 abaixo), a sequência é:

```bash
python extrair_frames.py         # vídeos -> imagens em data/
python train.py                  # imagens -> modelo.keras
python detectar.py meu_video.mp4 # reconhece os Pokémons em um vídeo
python detectar.py 0             # ou usa a webcam (q para sair)
```

Para sair do ambiente virtual, use `deactivate`. Cada etapa está detalhada a seguir.

## 1. Gravar os vídeos

Filme o Pokémon (pelúcia, carta, boneco...) e coloque os vídeos na subpasta correspondente:

```
videos/
├── bulbasaur/   v01_frente_claro.mp4  v02_costas_escuro.mp4  v03_lado_claro.mp4
├── charmander/  ...
├── pikachu/     ...
├── squirtle/    ...
└── fundo/       vídeos SEM nenhum Pokémon (só a mesa, a parede, a mão...)
```

- Grave pelo menos 3 vídeos por classe. Com menos de 2, o treino é interrompido com erro.
- Varie o ângulo, a iluminação e a distância entre os vídeos, porque cada vídeo inteiro vai só para o treino ou só para a validação.
- Formatos aceitos: `.mp4`, `.mov`, `.avi`, `.mkv` e `.webm`.
- Mantenha o nome `fundo` nessa pasta: o `detectar.py` usa esse nome para mostrar "nenhum Pokemon".

## 2. Extrair os frames

```bash
python extrair_frames.py
```

Os frames são salvos em `data/<classe>/<nome_do_video>__<numero_do_frame>.jpg`. O script aproveita até 4 frames por segundo e 120 por vídeo, e descarta frames quase iguais ao anterior. Ele avisa no final se as classes ficaram desbalanceadas.

Se você remover ou trocar vídeos, apague a pasta `data/` antes de extrair de novo. Caso contrário, os frames antigos continuam sendo usados no treino.

## 3. Treinar o modelo

```bash
python train.py
```

Na primeira execução é preciso ter internet, pois os pesos da MobileNetV2 são baixados automaticamente. O treino roda por até 8 épocas e para antes se a loss de validação não melhorar por 3 épocas. Ele funciona sem GPU, só que mais devagar.

No terminal aparecem a matriz de confusão, a precisão e o recall por classe e a acurácia de validação. Os arquivos gerados são:

| Arquivo | Conteúdo |
| --- | --- |
| `modelo.keras` | Modelo treinado |
| `class_names.json` | Nomes das classes, na ordem usada pelo modelo |
| `curvas.png` | Curvas de acurácia e loss (treino e validação) |
| `dataset_imagens.pdf` | Imagens do dataset, com até 300 por classe |

## 4. Rodar o modelo

O `detectar.py` precisa de `modelo.keras` e `class_names.json` na raiz do projeto.

**Em um vídeo:**

```bash
python detectar.py meu_video.mp4
```

O vídeo anotado é salvo como `saida_meu_video.mp4`. Durante o processamento, uma janela mostra o resultado. Para não abrir a janela (por exemplo, em um servidor sem interface gráfica), use:

```bash
MOSTRAR=0 python detectar.py meu_video.mp4
```

**Na webcam:**

```bash
python detectar.py 0
```

Rodar `python detectar.py` sem argumentos também abre a webcam. Use outro número se tiver mais de uma câmera. Pressione `q` com a janela selecionada para sair. O resultado da webcam não é salvo.

Cada frame mostra a classe mais provável com a confiança, uma barra de probabilidade por classe e o tempo do vídeo.

## Ajustes

Os parâmetros ficam em constantes no início de cada script:

| Script | Constante | Padrão | O que controla |
| --- | --- | --- | --- |
| `extrair_frames.py` | `FRAMES_POR_SEGUNDO` | 4 | Frames aproveitados por segundo de vídeo |
| `extrair_frames.py` | `MAX_POR_VIDEO` | 120 | Limite de frames por vídeo |
| `extrair_frames.py` | `DIFF_MINIMA` | 6.0 | Diferença mínima para o frame anterior (0 desliga o filtro) |
| `train.py` | `EPOCHS` | 8 | Número máximo de épocas |
| `train.py` | `BATCH` | 16 | Tamanho do lote |
| `train.py` | `VAL_FRACTION` | 0.2 | Fração dos vídeos usada na validação |

Para adicionar ou remover um Pokémon, altere as subpastas de `videos/`, apague `data/` e repita os passos 2 e 3.

## Problemas comuns

- **`'<classe>' tem 0 video(s)`**: a pasta da classe está sem vídeos, ou o `extrair_frames.py` não foi rodado depois de adicioná-los.
- **`FileNotFoundError` em `data`**: rode o `extrair_frames.py` antes do `train.py`.
- **Erro ao carregar `modelo.keras`**: rode o `train.py` antes do `detectar.py`.
- **`Nao consegui abrir: ...`**: confira o caminho do vídeo ou o número da webcam.
- **Erro em `cv2.imshow`**: não há interface gráfica disponível, ou o pacote instalado é o `opencv-python-headless`. Para vídeos, use `MOSTRAR=0`. A webcam precisa de janela.
