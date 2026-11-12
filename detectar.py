"""
Passo 3 - Reconhece Pokemon em um video (ou na webcam).
Uso:
  python detectar.py meu_video.mp4     -> gera saida_meu_video.mp4
  python detectar.py 0                 -> usa a webcam (q = sair)
  python detectar.py "rtsp://..."      -> camera IP (ex.: Intelbras), ao vivo (q = sair)

Melhorias implementadas pelo grupo:
  1. Suavizacao e limiar de confianca: o rotulo usa a MEDIA das
     probabilidades dos ultimos JANELA frames. Se a maior probabilidade
     media ficar abaixo de LIMIAR, mostra "?" em vez de um nome.
  2. Linha do tempo: no final do video, imprime em que trechos cada
     Pokemon apareceu e a lista dos Pokemon encontrados.

Para os testes da atividade (janelas de 1, 8 e 15 frames, limiares):
  JANELA=15 LIMIAR=0.7 python detectar.py meu_video.mp4
A preparacao da imagem e a mesma do treino (letterbox do utils.py).
"""
import sys
import os
import json
from collections import deque
import cv2
import numpy as np
import tensorflow as tf
from utils import abrir_video, letterbox

CLASSE_VAZIA = "fundo"  # classe que significa "nenhum Pokemon"
# ---- [GRUPO] melhoria 1 ----
JANELA = int(os.environ.get("JANELA", "8"))         # media dos ultimos N frames (1 = sem suavizacao)
LIMIAR = float(os.environ.get("LIMIAR", "0.6"))     # abaixo disso o rotulo vira "?"
DUVIDA = "?"
# ---- [GRUPO] melhoria 2 ----
MIN_TRECHO = 0.5  # segundos: trechos mais curtos sao ignorados na linha do tempo
# ---- [GRUPO] interface: cor de cada classe (BGR) ----
# Cor do TIPO de cada Pokemon (Pikachu e Psyduck sao amarelos, entao a cor do
# corpo nao serviria para diferenciar na tela).
CORES = {"pikachu": (0, 215, 255),     # eletrico: amarelo
         "psyduck": (235, 150, 40),    # agua: azul
         "charmander": (20, 110, 255), # fogo: laranja
         CLASSE_VAZIA: (200, 200, 200)}
NOME_NA_TELA = {CLASSE_VAZIA: "nenhum"}  # "fundo" nao diz nada para quem assiste
CINZA = (150, 150, 150)
FONTE = cv2.FONT_HERSHEY_DUPLEX

model = tf.keras.models.load_model("modelo.keras")
class_names = json.load(open("class_names.json"))

fonte = sys.argv[1] if len(sys.argv) > 1 else "0"
webcam = fonte.isdigit()
# [GRUPO] camera IP (rtsp:// ou http://) e ao vivo como a webcam: nao grava
# arquivo de saida e mostra a janela.
camera_ip = fonte.lower().startswith(("rtsp://", "http://", "https://"))
ao_vivo = webcam or camera_ip
cap = cv2.VideoCapture(int(fonte) if webcam else fonte)
if not cap.isOpened():
    raise SystemExit(f"Nao consegui abrir: {fonte}")
if camera_ip:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # menos atraso: nao acumula frames velhos
fps = cap.get(cv2.CAP_PROP_FPS)
if not 0 < fps <= 120:  # algumas cameras IP informam um FPS invalido
    fps = 30

writer = None
if not ao_vivo:
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    saida = f"saida_{os.path.splitext(os.path.basename(fonte))[0]}.mp4"
    writer = abrir_video(saida, fps, (w, h))


# ---- [GRUPO] interface ---------------------------------------------------
def escrever(img, texto, org, escala, cor, esp=1):
    """Texto com contorno preto: fica legivel em fundo claro ou escuro."""
    cv2.putText(img, texto, org, FONTE, escala, (0, 0, 0), esp + 3, cv2.LINE_AA)
    cv2.putText(img, texto, org, FONTE, escala, cor, esp, cv2.LINE_AA)


def escurecer(img, x0, y0, x1, y1, forca=0.6):
    """Faixa semitransparente atras do texto."""
    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * (1 - forca)).astype(np.uint8)


def desenhar(frame, probs, rotulo, i):
    """Desenha o resultado no frame. Tudo escala com a largura (480p a 4K).
    - faixa no topo: nome na cor do Pokemon, % e barra de confianca com a
      marca do LIMIAR (abaixo dela o rotulo vira "?");
    - moldura na cor do Pokemon enquanto ele esta identificado;
    - painel: uma barra por classe, na cor dela, com a classe atual em destaque."""
    h, w = frame.shape[:2]
    s = w / 640
    grosso = max(1, round(2 * s))
    if rotulo == DUVIDA:
        cor, titulo = CINZA, f"? talvez {class_names[i]}"
    elif rotulo == CLASSE_VAZIA:
        cor, titulo = CORES[CLASSE_VAZIA], "nenhum Pokemon"
    else:
        cor, titulo = CORES.get(rotulo, (255, 255, 255)), rotulo.upper()

    # faixa do topo: nome + porcentagem + barra de confianca
    topo = int(74 * s)
    escurecer(frame, 0, 0, w, topo)
    cv2.rectangle(frame, (0, 0), (int(8 * s), topo), cor, -1)
    escrever(frame, titulo, (int(20 * s), int(36 * s)), 1.0 * s, cor, grosso)
    pct = f"{probs[i]:.0%}"
    (tw, _), _ = cv2.getTextSize(pct, FONTE, 1.0 * s, grosso)
    escrever(frame, pct, (w - tw - int(16 * s), int(36 * s)), 1.0 * s, (255, 255, 255), grosso)
    x0, x1, y0, y1 = int(20 * s), w - int(16 * s), int(48 * s), int(62 * s)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (70, 70, 70), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + int((x1 - x0) * probs[i]), y1), cor, -1)
    xl = x0 + int((x1 - x0) * LIMIAR)
    cv2.line(frame, (xl, y0 - int(4 * s)), (xl, y1 + int(4 * s)), (255, 255, 255), grosso)

    # painel de baixo: barra de cada classe na cor dela
    lin, larg, alt, fs = int(17 * s), int(90 * s), int(10 * s), 0.38 * s
    nomes = [NOME_NA_TELA.get(c, c) for c in class_names]
    rot_larg = max(cv2.getTextSize(c, FONTE, fs, 1)[0][0] for c in nomes)
    px, py = int(10 * s), h - lin * len(class_names) - int(6 * s)
    escurecer(frame, 0, py - int(6 * s), px + rot_larg + larg + int(50 * s), h)
    for j, c in enumerate(class_names):
        y = py + j * lin
        cj = CORES.get(c, (255, 255, 255))
        escrever(frame, nomes[j], (px, y + alt - int(1 * s)), fs, cj)
        bx = px + rot_larg + int(8 * s)
        cv2.rectangle(frame, (bx, y), (bx + larg, y + alt), (70, 70, 70), -1)
        cv2.rectangle(frame, (bx, y), (bx + int(larg * probs[j]), y + alt), cj, -1)
        if j == i:  # classe mais provavel: contorno branco
            cv2.rectangle(frame, (bx, y), (bx + larg, y + alt), (255, 255, 255), max(1, round(s)))
        escrever(frame, f"{probs[j]:.0%}", (bx + larg + int(6 * s), y + alt - int(1 * s)),
                 fs, (255, 255, 255))

    # moldura na cor do Pokemon identificado (desenhada por cima de tudo)
    if rotulo not in (DUVIDA, CLASSE_VAZIA):
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), cor, max(4, int(8 * s)))


# [GRUPO] deque(maxlen=JANELA) guarda so os ultimos JANELA vetores de
# probabilidade; os mais antigos saem sozinhos.
ultimas = deque(maxlen=JANELA)
rotulos = []  # [GRUPO] rotulo final de cada frame, para a linha do tempo

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if webcam:
        frame = cv2.flip(frame, 1)
    img = cv2.cvtColor(letterbox(frame), cv2.COLOR_BGR2RGB).astype("float32")
    probs_frame = model.predict(img[None], verbose=0)[0]

    # ---- [GRUPO] melhoria 1: suavizacao + limiar ----
    ultimas.append(probs_frame)
    probs = np.mean(ultimas, axis=0)  # media dos ultimos JANELA frames
    i = int(np.argmax(probs))
    # So da nome se a confianca MEDIA passar do limiar; senao, "?".
    rotulo = class_names[i] if probs[i] >= LIMIAR else DUVIDA
    rotulos.append(rotulo)

    desenhar(frame, probs, rotulo, i)

    if writer is not None:
        writer.write(frame)
    if ao_vivo or os.environ.get("MOSTRAR", "1") == "1":
        cv2.imshow("Reconhecimento com CNN", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
if writer is not None:
    writer.release()
    print(f"Video anotado salvo em: {saida}")
cv2.destroyAllWindows()


# ---- [GRUPO] melhoria 2: linha do tempo -----------------------------------
def linha_do_tempo(rotulos, fps, min_trecho=MIN_TRECHO):
    """Agrupa frames seguidos com o mesmo rotulo em trechos (ini, fim, rotulo).
    1) trechos com menos de min_trecho segundos sao descartados (piscadas);
    2) trechos vizinhos que ficaram com o mesmo rotulo sao juntados de novo;
    3) trechos de "?" e de fundo nao entram na lista final."""
    trechos = []
    for idx, r in enumerate(rotulos):
        if trechos and trechos[-1][2] == r:
            trechos[-1][1] = idx
        else:
            trechos.append([idx, idx, r])
    # tempo de um frame = indice / fps; o trecho vai do 1o frame ate o fim do ultimo
    trechos = [t for t in trechos if (t[1] + 1 - t[0]) / fps >= min_trecho]
    juntos = []
    for t in trechos:
        if juntos and juntos[-1][2] == t[2]:
            juntos[-1][1] = t[1]
        else:
            juntos.append(t)
    return [(a / fps, (b + 1) / fps, r) for a, b, r in juntos
            if r not in (DUVIDA, CLASSE_VAZIA)]


trechos = linha_do_tempo(rotulos, fps)
print("\nLinha do tempo (Pokemon identificados):")
for ini, fim, r in trechos:
    print(f"  {ini:6.1f}s a {fim:6.1f}s  ->  {r}")
if not trechos:
    print("  (nenhum Pokemon identificado)")
print("Pokemon encontrados no video:", sorted({r for _, _, r in trechos}))
