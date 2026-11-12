"""Pergunta 5: compara abordagens para 2 Pokemon no mesmo quadro, sem treinar de novo.
Uso (na raiz do projeto, com modelo.keras treinado):
    LJ=0.6 python docs/experimentos_parte5/scripts/dupla_abordagens.py videos_teste/dupla.mp4 saida.json
Roteiro do video: 0-1.5 s vazio, 1.5-16.5 s pikachu (esquerda) + psyduck (direita), 16.5-18 s vazio."""
import json
import os
import sys

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.getcwd())  # utils.py da raiz do projeto
from utils import letterbox

ALVO = {"pikachu", "psyduck"}
LIMIAR_TOP2 = 0.25
LIMIAR_JANELA = float(os.environ.get("LJ", "0.6"))  # limiar de cada janela (abordagem C)

model = tf.keras.models.load_model("modelo.keras")
cn = json.load(open("class_names.json"))
FUNDO = cn.index("fundo")
cap = cv2.VideoCapture(sys.argv[1])
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
a, b = int(1.5 * fps) + 4, int(16.5 * fps)  # trecho com os dois, sem a transicao
vazio = list(range(0, int(1.5 * fps))) + list(range(int(16.5 * fps) + 4, len(frames)))


def prep(img):
    return cv2.cvtColor(letterbox(img), cv2.COLOR_BGR2RGB).astype("float32")


def janelas(f):
    """Recortes sobrepostos em 2 escalas (independe de onde os Pokemon estao)."""
    h, w = f.shape[:2]
    out = []
    for frac in (0.6, 0.8):
        ch, cw = int(h * frac), int(w * frac * 0.75)
        for y in np.linspace(0, h - ch, 2).astype(int):
            for x in np.linspace(0, w - cw, 3).astype(int):
                out.append(f[y:y + ch, x:x + cw])
    return out


def avaliar(idx):
    inteiro = model.predict(np.stack([prep(frames[i]) for i in idx]), batch_size=32, verbose=0)
    n_j = len(janelas(frames[0]))
    jan = model.predict(np.stack([prep(c) for i in idx for c in janelas(frames[i])]),
                        batch_size=64, verbose=0).reshape(len(idx), n_j, len(cn))
    r = {"A_top1": [], "B_top2": [], "C_janelas": []}
    for p, pj in zip(inteiro, jan):
        r["A_top1"].append({cn[int(p.argmax())]})
        r["B_top2"].append({cn[j] for j in np.argsort(p)[::-1][:2] if p[j] >= LIMIAR_TOP2})
        r["C_janelas"].append({cn[j] for j in range(len(cn)) if pj[:, j].max() >= LIMIAR_JANELA})
    return r, inteiro


com, p_com = avaliar(range(a, b))
sem, _ = avaliar(vazio)
res = {"frames_com_os_dois": b - a, "frames_vazios": len(vazio),
       "prob_media_inteiro": {c: round(float(p_com[:, j].mean()), 3) for j, c in enumerate(cn)}}
for k in com:
    poks = [s - {"fundo"} for s in com[k]]
    res[k] = {
        "acha_os_dois": round(float(np.mean([ALVO <= s for s in poks])), 3),
        "acha_so_um": round(float(np.mean([len(s & ALVO) == 1 for s in poks])), 3),
        "acha_errado": round(float(np.mean([bool(s - ALVO) for s in poks])), 3),
        "vazio_com_pokemon_falso": round(float(np.mean([bool(s - {"fundo"}) for s in sem[k]])), 3),
    }
print(json.dumps(res, indent=1))
json.dump(res, open(sys.argv[2], "w"), indent=1)
