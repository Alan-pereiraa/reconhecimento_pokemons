"""
Passo 3 SEM WEBCAM - reconhece o Pokemon em FOTOS.
Uso:
    python detectar_imagem.py foto.png        -> uma foto
    python detectar_imagem.py teste/          -> todas as fotos da pasta

Abre uma janela por foto (qualquer tecla = proxima, q = sair) e salva cada
uma com o resultado em resultados/. Sem janela (so imprime e salva):
    MOSTRAR=0 python detectar_imagem.py teste/
Para VIDEO continue usando: python detectar.py meu_video.mp4

A preparacao da imagem e a mesma do treino (ler_imagem + letterbox do utils.py).
"""
import json
import os
import sys

import cv2
import numpy as np
import tensorflow as tf

from utils import letterbox, ler_imagem, IMG_EXT

CLASSE_VAZIA = "fundo"  # classe que significa "nenhum dos 3"
SAIDA_DIR = "resultados"

model = tf.keras.models.load_model("modelo.keras")
class_names = json.load(open("class_names.json"))

alvo = sys.argv[1] if len(sys.argv) > 1 else "teste"
if os.path.isdir(alvo):
    fotos = sorted(os.path.join(alvo, f) for f in os.listdir(alvo)
                   if f.lower().endswith(IMG_EXT))
elif os.path.isfile(alvo):
    fotos = [alvo]
else:
    raise SystemExit(f"Nao achei: {alvo}")
if not fotos:
    raise SystemExit(f"Nenhuma foto {IMG_EXT} em {alvo}")

os.makedirs(SAIDA_DIR, exist_ok=True)
mostrar = os.environ.get("MOSTRAR", "1") == "1"

for caminho in fotos:
    try:
        frame = ler_imagem(caminho)[0]
    except Exception as e:
        print(f"Pulei {caminho}: {e}")
        continue

    img = cv2.cvtColor(letterbox(frame), cv2.COLOR_BGR2RGB).astype("float32")
    probs = model.predict(img[None], verbose=0)[0]
    i = int(np.argmax(probs))
    rotulo = class_names[i]
    texto = "nenhum dos 3" if rotulo == CLASSE_VAZIA else rotulo
    print(f"{os.path.basename(caminho)}: {texto} ({probs[i]:.0%})")

    # --- desenho: painel com o resultado em cima da foto ---
    foto = letterbox(frame, 480)
    painel = np.zeros((50 + 22 * len(class_names), foto.shape[1], 3), np.uint8)
    cor = (0, 200, 0) if rotulo != CLASSE_VAZIA else (0, 200, 255)
    cv2.putText(painel, f"{texto} ({probs[i]:.0%})", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)
    for j, c in enumerate(class_names):
        y = 45 + j * 22
        cv2.rectangle(painel, (10, y), (10 + int(probs[j] * 150), y + 14), (0, 200, 255), -1)
        cv2.putText(painel, c, (170, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    tela = np.vstack([painel, foto])

    # imencode + tofile funciona mesmo com acento no nome do arquivo (Windows)
    nome = os.path.splitext(os.path.basename(caminho))[0] + ".jpg"
    cv2.imencode(".jpg", tela)[1].tofile(os.path.join(SAIDA_DIR, nome))

    if mostrar:
        cv2.imshow("Reconhecimento de Pokemon", tela)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

if mostrar:
    cv2.destroyAllWindows()
print(f"Fotos com o resultado salvas em {SAIDA_DIR}/")
