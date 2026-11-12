"""Mede um modelo num video de teste com roteiro conhecido (gerar_videos.py).
Uso (na raiz do projeto, com modelo.keras treinado):
    python docs/experimentos_parte5/scripts/medir.py <video> <roteiro> [saida.json]
Ex.: python docs/experimentos_parte5/scripts/medir.py videos_teste/teste.mp4 teste
roteiro: teste | simples:<rotulo>   (rotulo = classe, "desconhecido" ou "a+b")
Mesmo preprocessamento do detectar.py (letterbox -> RGB -> float32),
mesma suavizacao (media dos ultimos JANELA) e LIMIAR."""
import json
import os
import sys
from collections import Counter, deque

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.getcwd())  # utils.py da raiz do projeto
from utils import letterbox

JANELA = int(os.environ.get("JANELA", "8"))
LIMIAR = float(os.environ.get("LIMIAR", "0.6"))


def roteiro(nome):
    if nome == "teste":
        seg = [(2, "fundo"), (13, "pikachu"), (2.5, "fundo"), (13, "psyduck"), (2.5, "fundo"),
               (13, "charmander"), (2.5, "fundo")]
    else:
        seg = [(1.5, "fundo"), (15, nome.split(":", 1)[1]), (1.5, "fundo")]
    t, out = 0.0, []
    for d, r in seg:
        out.append((t, t + d, r))
        t += d
    return out


video, rot = sys.argv[1], roteiro(sys.argv[2])
model = tf.keras.models.load_model("modelo.keras")
cn = json.load(open("class_names.json"))
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(cv2.cvtColor(letterbox(f), cv2.COLOR_BGR2RGB).astype("float32"))
raw = model.predict(np.stack(frames), batch_size=32, verbose=0)

ult, labels = deque(maxlen=JANELA), []
for p in raw:
    ult.append(p)
    m = np.mean(ult, axis=0)
    i = int(m.argmax())
    labels.append(cn[i] if m[i] >= LIMIAR else "?")

res = {"video": video, "fps": fps, "frames": len(raw), "janela": JANELA, "limiar": LIMIAR, "trechos": []}
for ini, fim, verdade in rot:
    a, b = int(round(ini * fps)), int(round(fim * fps))
    a2 = min(b, a + JANELA)  # ignora a transicao (janela ainda com frames do trecho anterior)
    lab, pr = labels[a2:b], raw[a2:b]
    alvo = verdade.split("+") if "+" in verdade else [verdade]
    acerto = None if verdade == "desconhecido" else float(np.mean([l in alvo for l in lab]))
    res["trechos"].append({
        "ini": ini, "fim": fim, "verdade": verdade, "n": len(lab), "acerto": acerto,
        "rotulos": dict(Counter(lab).most_common()),
        "prob_media": {c: round(float(pr[:, j].mean()), 3) for j, c in enumerate(cn)},
        "conf_max_media": round(float(pr.max(axis=1).mean()), 3),
    })
pok = [t for t in res["trechos"] if t["acerto"] is not None]
res["acerto_frames"] = round(sum(t["acerto"] * t["n"] for t in pok) / max(1, sum(t["n"] for t in pok)), 4)
print(json.dumps(res, indent=1, ensure_ascii=False))
if len(sys.argv) > 3:
    json.dump(res, open(sys.argv[3], "w"), indent=1, ensure_ascii=False)
    np.save(sys.argv[3].replace(".json", ".npy"), raw)
