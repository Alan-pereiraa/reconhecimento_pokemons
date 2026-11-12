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
from utils import letterbox

CLASSE_VAZIA = "fundo"  # classe que significa "nenhum Pokemon"
# ---- [GRUPO] melhoria 1 ----
JANELA = int(os.environ.get("JANELA", "8"))         # media dos ultimos N frames (1 = sem suavizacao)
LIMIAR = float(os.environ.get("LIMIAR", "0.6"))     # abaixo disso o rotulo vira "?"
DUVIDA = "?"
# ---- [GRUPO] melhoria 2 ----
MIN_TRECHO = 0.5  # segundos: trechos mais curtos sao ignorados na linha do tempo

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
    writer = cv2.VideoWriter(saida, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# [GRUPO] deque(maxlen=JANELA) guarda so os ultimos JANELA vetores de
# probabilidade; os mais antigos saem sozinhos.
ultimas = deque(maxlen=JANELA)
rotulos = []  # [GRUPO] rotulo final de cada frame, para a linha do tempo

n = 0
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

    # --- desenho ---
    if rotulo == DUVIDA:
        texto, cor = f"? (talvez {class_names[i]})", (180, 180, 180)
    elif rotulo == CLASSE_VAZIA:
        texto, cor = "nenhum Pokemon", (0, 200, 255)
    else:
        texto, cor = rotulo, (0, 200, 0)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, f"{texto} ({probs[i]:.0%})", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)
    for j, c in enumerate(class_names):
        y = 55 + j * 22
        cv2.rectangle(frame, (10, y), (10 + int(probs[j] * 150), y + 14), (0, 200, 255), -1)
        cv2.putText(frame, c, (170, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
    cv2.putText(frame, f"t = {n / fps:5.1f}s  janela={JANELA} limiar={LIMIAR:.0%}",
                (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if writer is not None:
        writer.write(frame)
    if ao_vivo or os.environ.get("MOSTRAR", "1") == "1":
        cv2.imshow("Reconhecimento com CNN", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    n += 1

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
