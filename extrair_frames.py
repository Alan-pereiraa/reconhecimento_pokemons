import os

import cv2
import numpy as np

from utils import letterbox, VIDEO_EXT

VIDEOS_DIR = "videos"
DATA_DIR = "data"
FRAMES_POR_SEGUNDO = 4
MAX_POR_VIDEO = 120
DIFF_MINIMA = 6.0

if not os.path.isdir(VIDEOS_DIR):
    raise SystemExit(f"Crie a pasta '{VIDEOS_DIR}/' com uma subpasta por classe.")

classes = sorted(d for d in os.listdir(VIDEOS_DIR)
                 if os.path.isdir(os.path.join(VIDEOS_DIR, d)))
print("Classes encontradas:", classes)

resumo = {}
for cls in classes:
    os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)
    videos = sorted(f for f in os.listdir(os.path.join(VIDEOS_DIR, cls))
                    if f.lower().endswith(VIDEO_EXT))
    total_cls = 0
    for vid in videos:
        nome = os.path.splitext(vid)[0].replace("__", "_")
        cap = cv2.VideoCapture(os.path.join(VIDEOS_DIR, cls, vid))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        passo = max(1, int(round(fps / FRAMES_POR_SEGUNDO)))
        i, salvos, ultimo = 0, 0, None
        while salvos < MAX_POR_VIDEO:
            ok, frame = cap.read()
            if not ok:
                break
            if i % passo == 0:
                img = letterbox(frame)
                pequeno = cv2.resize(img, (32, 32)).astype(np.float32)
                if ultimo is None or np.abs(pequeno - ultimo).mean() >= DIFF_MINIMA:
                    cv2.imwrite(os.path.join(DATA_DIR, cls, f"{nome}__{i:06d}.jpg"), img)
                    salvos += 1
                    ultimo = pequeno
            i += 1
        cap.release()
        total_cls += salvos
        print(f"  {cls}/{vid}: {salvos} frames")
    resumo[cls] = total_cls

print("\nTotal por classe:", resumo)
menor, maior = min(resumo.values()), max(resumo.values())
if maior > 1.5 * max(menor, 1):
    print("ATENCAO: classes desbalanceadas. Grave mais videos da classe menor "
          "ou reduza MAX_POR_VIDEO.")
print("Pronto! Agora rode: python train.py")
