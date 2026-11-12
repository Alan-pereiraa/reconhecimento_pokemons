"""
Passo 2 - Treino com Transfer Learning (MobileNetV2) + Fine-tuning.
Uso: python train.py

Fase 1: a base (MobileNetV2, pre-treinada na ImageNet) fica congelada e so a
"cabeca" nova aprende, igual a atividade anterior.
Fase 2 (MELHORIA 3, implementada pelo grupo): fine-tuning. As ultimas
camadas da MobileNetV2 sao descongeladas e o treino continua com uma taxa
de aprendizado bem menor.
A validacao e separada POR VIDEO.

Experimentos da Parte 5 (sem mudar o codigo):
    SPLIT=foto python train.py      -> split aleatorio por FOTO (pergunta 1)
    LR_FINE=1e-3 python train.py    -> fine-tuning com taxa alta (pergunta 2)
"""
import os
import json
import random
from collections import defaultdict
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "data"
IMG_SIZE = 224
BATCH = 16
EPOCHS = 8
VAL_FRACTION = 0.2          # fracao de VIDEOS (nao de fotos) para validacao
SEED = 42
# ---- [GRUPO] parametros do fine-tuning (melhoria 3) ----
EPOCHS_FINE = 8             # epocas extras da fase 2
CAMADAS_FINE = 30           # quantas camadas do FIM da MobileNetV2 descongelar
LR_FINE = float(os.environ.get("LR_FINE", "1e-5"))  # bem menor que o 1e-3 da fase 1
# ---- [GRUPO] experimento da pergunta 1: "video" (correto) ou "foto" ----
SPLIT = os.environ.get("SPLIT", "video")

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

class_names = sorted(d for d in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, d)))
print("Classes:", class_names)

# ---- Split POR VIDEO ------------------------------------------------------
# Frames do mesmo video sao quase iguais. Se um video tiver frames no treino e
# na validacao, a acuracia fica enganosamente alta. Por isso cada video inteiro
# vai OU para treino OU para validacao. O nome do video vem antes de "__".
train_paths, train_y, val_paths, val_y = [], [], [], []
for idx, cls in enumerate(class_names):
    por_video = defaultdict(list)
    for f in sorted(os.listdir(os.path.join(DATA_DIR, cls))):
        if f.lower().endswith(".jpg"):
            por_video[f.split("__")[0]].append(os.path.join(DATA_DIR, cls, f))
    videos = sorted(por_video)
    if len(videos) < 2:
        raise SystemExit(f"'{cls}' tem {len(videos)} video(s). Grave pelo menos 3 por classe.")
    if SPLIT == "foto":
        # [GRUPO] pergunta 1: split ERRADO de proposito. Sorteia 20% das fotos,
        # misturando todos os videos -> frames vizinhos caem dos dois lados.
        fotos = [p for v in videos for p in por_video[v]]
        random.Random(SEED).shuffle(fotos)
        n_val = max(1, round(len(fotos) * VAL_FRACTION))
        val_paths.extend(fotos[:n_val]); val_y.extend([idx] * n_val)
        train_paths.extend(fotos[n_val:]); train_y.extend([idx] * (len(fotos) - n_val))
        print(f"  {cls}: {len(fotos) - n_val} treino / {n_val} validacao | split por FOTO")
        continue
    random.Random(SEED).shuffle(videos)
    n_val = max(1, round(len(videos) * VAL_FRACTION))
    val_videos = set(videos[:n_val])
    for v in videos:
        destino = (val_paths, val_y) if v in val_videos else (train_paths, train_y)
        destino[0].extend(por_video[v]); destino[1].extend([idx] * len(por_video[v]))
    n_tr = sum(len(por_video[v]) for v in videos if v not in val_videos)
    n_va = sum(len(por_video[v]) for v in val_videos)
    print(f"  {cls}: {n_tr} treino / {n_va} validacao | videos de validacao: {sorted(val_videos)}")


def load(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img, label


def make_ds(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH)
    return ds.prefetch(tf.data.AUTOTUNE)


train_ds = make_ds(train_paths, train_y, True)
val_ds = make_ds(val_paths, val_y, False)

# Data augmentation (so atua no treino). O flip horizontal simula o objeto
# virado para o outro lado; a rotacao e o zoom simulam a camera se mexendo.
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomZoom(0.20),
    tf.keras.layers.RandomTranslation(0.10, 0.10),
    tf.keras.layers.RandomBrightness(0.25),
    tf.keras.layers.RandomContrast(0.25),
])

base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False,
    weights="imagenet")
base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = augment(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

parar = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                         restore_best_weights=True)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ---- Fase 1: so a cabeca (base congelada) --------------------------------
print("\n===== FASE 1: base congelada, so a cabeca aprende =====")
hist1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[parar])


# ---- Matriz de confusao + precisao/recall por classe ----------------------
# [GRUPO] virou funcao para comparar o modelo ao final da fase 1 e da fase 2
# (pergunta 2).
def avaliar(titulo):
    y_true = np.array(val_y)
    y_pred = np.argmax(model.predict(val_ds, verbose=0), axis=1)
    k = len(class_names)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    print(f"\n----- {titulo} -----")
    print("Matriz de confusao (linhas = real, colunas = previsto):")
    print(" " * 11 + " ".join(f"{c[:8]:>8}" for c in class_names))
    for i, c in enumerate(class_names):
        print(f"{c[:10]:>10} " + " ".join(f"{v:>8}" for v in cm[i]))

    print("\nClasse        precisao   recall")
    for i, c in enumerate(class_names):
        prec = cm[i, i] / max(cm[:, i].sum(), 1)
        rec = cm[i, i] / max(cm[i, :].sum(), 1)
        print(f"{c[:12]:<12} {prec:>9.2%} {rec:>8.2%}")
    print(f"\nAcuracia de validacao: {np.trace(cm) / cm.sum():.2%}")


avaliar("Ao final da FASE 1 (so a cabeca)")

# ---- Fase 2: fine-tuning (MELHORIA 3, implementada pelo grupo) -----------
# A cabeca ja foi treinada na fase 1. So agora descongelamos a base: se ela
# fosse descongelada com a cabeca aleatoria, os gradientes grandes do inicio
# destruiriam os filtros pre-treinados.
base.trainable = True
for camada in base.layers[:-CAMADAS_FINE]:
    camada.trainable = False  # so as ultimas CAMADAS_FINE camadas aprendem
# A base continua sendo chamada com training=False (ver acima): as camadas de
# BatchNormalization seguem usando as estatisticas da ImageNet.

# Mudou o que e treinavel -> e OBRIGATORIO compilar de novo. Taxa bem menor
# (1e-5) para os pesos pre-treinados mudarem aos poucos.
model.compile(optimizer=tf.keras.optimizers.Adam(LR_FINE),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
n_treinaveis = sum(1 for c in base.layers if c.trainable)
print(f"\n===== FASE 2: fine-tuning das ultimas {n_treinaveis} camadas, taxa {LR_FINE:g} =====")

epocas_fase1 = len(hist1.history["loss"])
parar2 = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                          restore_best_weights=True)
hist2 = model.fit(train_ds, validation_data=val_ds,
                  epochs=epocas_fase1 + EPOCHS_FINE, initial_epoch=epocas_fase1,
                  callbacks=[parar2])

model.save("modelo.keras")
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# ---- Curvas de treino vs validacao (as duas fases) -----------------------
# [GRUPO] junta os historicos da fase 1 e da fase 2 e marca com uma linha
# vertical a epoca em que o fine-tuning comeca.
hist = {m: hist1.history[m] + hist2.history[m] for m in hist1.history}
epocas = range(1, len(hist["loss"]) + 1)
plt.figure(figsize=(10, 4))
for i, (m, t) in enumerate([("accuracy", "Acuracia"), ("loss", "Loss")]):
    plt.subplot(1, 2, i + 1)
    plt.plot(epocas, hist[m], marker="o", label="treino")
    plt.plot(epocas, hist["val_" + m], marker="o", label="validacao")
    plt.axvline(epocas_fase1 + 0.5, color="gray", linestyle="--",
                label="inicio do fine-tuning")
    plt.title(t); plt.xlabel("epoca"); plt.legend()
plt.tight_layout(); plt.savefig("curvas.png")

avaliar("Ao final da FASE 2 (apos o fine-tuning) - modelo salvo")

# ---- PDF do dataset (entrega obrigatoria) ---------------------------------
from PIL import Image, ImageDraw, ImageFont


def gerar_pdf_dataset(data_dir="data", saida="dataset_imagens.pdf", cols=10,
                      qualidade=70, max_por_classe=300):
    """Uma secao por classe. Se houver muitas fotos, pega uma amostra espacada
    (max_por_classe) para o PDF nao ficar gigante. A legenda mostra quantos
    videos geraram as fotos daquela classe."""
    pag_w, pag_h, margem, topo = 1240, 1754, 40, 110
    cell = (pag_w - 2 * margem) // cols
    rows = (pag_h - topo - margem) // cell
    por_pagina = cols * rows
    try:
        fonte = ImageFont.load_default(size=28)
    except TypeError:
        fonte = ImageFont.load_default()
    paginas = []
    classes = sorted(d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)))
    for classe in classes:
        pasta = os.path.join(data_dir, classe)
        fotos = sorted(f for f in os.listdir(pasta) if f.lower().endswith(".jpg"))
        n_videos = len({f.split("__")[0] for f in fotos})
        total = len(fotos)
        if total > max_por_classe:
            idx = np.linspace(0, total - 1, max_por_classe).astype(int)
            fotos = [fotos[i] for i in idx]
        n_pag = max(1, -(-len(fotos) // por_pagina))
        for p in range(n_pag):
            pagina = Image.new("RGB", (pag_w, pag_h), "white")
            draw = ImageDraw.Draw(pagina)
            draw.text((margem, 40),
                      f"Classe: {classe} | {total} imagens de {n_videos} videos | pagina {p + 1}/{n_pag}",
                      fill="black", font=fonte)
            for i, nome in enumerate(fotos[p * por_pagina:(p + 1) * por_pagina]):
                img = Image.open(os.path.join(pasta, nome)).convert("RGB")
                img.thumbnail((cell - 6, cell - 6))
                pagina.paste(img, (margem + (i % cols) * cell + 3, topo + (i // cols) * cell + 3))
            paginas.append(pagina)
    paginas[0].save(saida, save_all=True, append_images=paginas[1:],
                    quality=qualidade, resolution=150)
    print(f"PDF salvo: {saida} ({len(paginas)} paginas)")


gerar_pdf_dataset(DATA_DIR)
print("\nPronto! Modelo salvo em modelo.keras. Agora rode: python detectar.py meu_video.mp4")
