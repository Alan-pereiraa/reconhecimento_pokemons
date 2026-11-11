"""
Passo 3 - Reconhece o Pokemon em um video (ou na webcam).
Uso:
    python detectar.py meu_video.mp4   -> gera saida_meu_video.mp4
    python detectar.py 0               -> usa a webcam (q = sair)

Cada frame e classificado pelo modelo. Para o nome nao ficar piscando, a
tela mostra a media das ultimas probabilidades (SUAVIZACAO).
A preparacao da imagem e a mesma do treino (letterbox do utils.py).
"""
import sys
import os
import json

import cv2
import tensorflow as tf

from utils import desenhar, escrever, letterbox

JANELA = "Reconhecimento de Pokemon"
LARGURA_JANELA = 960  # a janela abre com essa largura (pode redimensionar)
SUAVIZACAO = 0.6  # peso da media anterior (0 = cada frame sozinho)

model = tf.keras.models.load_model("modelo.keras")
class_names = json.load(open("class_names.json"))

fonte = sys.argv[1] if len(sys.argv) > 1 else "0"
webcam = fonte.isdigit()
cap = cv2.VideoCapture(int(fonte) if webcam else fonte)
if not cap.isOpened():
    raise SystemExit(f"Nao consegui abrir: {fonte}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = None
if not webcam:
    saida = f"saida_{os.path.splitext(os.path.basename(fonte))[0]}.mp4"
    writer = cv2.VideoWriter(saida, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

mostrar = webcam or os.environ.get("MOSTRAR", "1") == "1"
if mostrar:
    # WINDOW_NORMAL: a janela cabe na tela mesmo com camera 1080p/4K
    cv2.namedWindow(JANELA, cv2.WINDOW_NORMAL)
    if w and h:
        cv2.resizeWindow(JANELA, LARGURA_JANELA, int(LARGURA_JANELA * h / w))

n, media = 0, None
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if webcam:
        frame = cv2.flip(frame, 1)

    img = cv2.cvtColor(letterbox(frame), cv2.COLOR_BGR2RGB).astype("float32")
    probs = model.predict(img[None], verbose=0)[0]
    media = probs if media is None else SUAVIZACAO * media + (1 - SUAVIZACAO) * probs

    desenhar(frame, media, class_names)
    if not webcam:
        s = frame.shape[1] / 640
        texto = f"{n / fps:.1f}s"
        (tw, _), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, 0.6 * s, 1)
        escrever(frame, texto, (frame.shape[1] - tw - int(16 * s), frame.shape[0] - int(14 * s)),
                 0.6 * s, (255, 255, 255))

    if writer is not None:
        writer.write(frame)
    if mostrar:
        cv2.imshow(JANELA, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    n += 1

cap.release()
if writer is not None:
    writer.release()
    print(f"Video anotado salvo em: {saida}")
cv2.destroyAllWindows()
