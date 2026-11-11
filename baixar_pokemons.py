"""
Passo 1 SEM WEBCAM - substitui "gravar os videos" + extrair_frames.py.
Uso: python baixar_pokemons.py      (precisa de internet so na 1a vez)

1) Baixa da PokeAPI (gratis, sem cadastro) a arte oficial e os sprites dos
   jogos de cada Pokemon e guarda em imagens/<classe>/.
2) Converte TUDO que estiver em imagens/ para data/<classe>/<grupo>__<n>.jpg,
   o mesmo formato que o extrair_frames.py gera. Depois e so rodar train.py.

Cada imagem (ou GIF animado) vira um "grupo", que faz o papel de um video:
o train.py manda o grupo inteiro OU para treino OU para validacao.

Quer mais variedade? Coloque fotos suas (Google, prints do anime, fotos de
pelucia...) em imagens/<classe>/ e rode de novo. O que ja foi baixado nao e
baixado outra vez.
"""
import json
import os
import re
import shutil
import urllib.request

import cv2
import numpy as np

from utils import letterbox, ler_imagem, IMG_EXT

# Os 3 escolhidos: cor dominante e silhueta bem diferentes entre si.
# classe -> Pokemon (da PokeAPI) cujas imagens entram nela.
CLASSES = {
    "gengar": ["gengar"],
    "charmander": ["charmander"],
    "squirtle": ["squirtle"],
}
IMAGENS_DIR = "imagens"
DATA_DIR = "data"
QUADROS_POR_GIF = 4  # quadros aproveitados de cada GIF animado
DIFF_MINIMA = 4.0  # descarta imagem quase igual a outra ja salva (0 = desliga)
# shiny muda a cor (Charmander shiny e amarelo, Gengar shiny e quase preto), gray e
# preto e branco, female e quase igual ao normal, icons sao minusculos e
# dream_world e .svg (o Pillow nao abre).
IGNORAR = ("shiny", "gray", "female", "icons", "dream_world")
API = "https://pokeapi.co/api/v2/pokemon/"
HEADERS = {"User-Agent": "reconhecimento-pokemons/1.0"}
SEED = 42


def baixar(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as r:
        return r.read()


def urls_dos_sprites(nome):
    """Todos os links de imagem do JSON da PokeAPI, com o caminho como nome."""
    sprites = json.loads(baixar(API + nome))["sprites"]
    achados = []

    def andar(no, caminho):
        if isinstance(no, dict):
            for k, v in no.items():
                andar(v, caminho + [k])
        elif isinstance(no, str) and no.startswith("http"):
            ext = os.path.splitext(no)[1].lower()
            texto = "-".join(caminho)
            if ext in IMG_EXT and not any(p in texto for p in IGNORAR):
                achados.append((texto.replace("versions-", ""), no, ext))

    andar(sprites, [])
    return achados


def baixar_classe(cls, pokemons, raiz=IMAGENS_DIR, fontes=None):
    """fontes: regex opcional; so baixa sprites cujo caminho na PokeAPI casa com ela."""
    pasta = os.path.join(raiz, cls)
    os.makedirs(pasta, exist_ok=True)
    for nome in pokemons:
        try:
            achados = urls_dos_sprites(nome)
        except Exception as e:
            print(f"  ! nao consegui listar {nome}: {e}")
            continue
        if fontes is not None:
            achados = [a for a in achados if fontes.match(a[0])]
        novos = 0
        for texto, url, ext in achados:
            destino = os.path.join(pasta, f"{nome}_{texto}{ext}")
            if os.path.exists(destino):
                continue
            try:
                with open(destino, "wb") as f:
                    f.write(baixar(url))
                novos += 1
            except Exception as e:
                print(f"  ! falhou {url}: {e}")
        print(f"  {cls}/{nome}: {len(achados)} imagens ({novos} novas)")


def criar_vazias(pasta):
    """Imagens lisas (branco, cor, degrade, ruido) usadas como cenario."""
    os.makedirs(pasta, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for i in range(12):
        destino = os.path.join(pasta, f"vazio_{i:02d}.png")
        if os.path.exists(destino):
            continue
        cor = rng.integers(0, 256, 3)
        if i == 0:
            img = np.full((300, 300, 3), 255, np.uint8)  # branco puro
        elif i % 3 == 1:
            img = np.full((300, 300, 3), cor, np.uint8)
        elif i % 3 == 2:
            t = np.linspace(0, 1, 300)[:, None, None]
            img = (cor * t + rng.integers(0, 256, 3) * (1 - t)).astype(np.uint8)
            img = np.repeat(img, 300, axis=1)
        else:
            img = cv2.resize(rng.integers(0, 256, (8, 8, 3)).astype(np.uint8), (300, 300))
        cv2.imwrite(destino, img)


def converter():
    """imagens/<classe>/* -> data/<classe>/<grupo>__<n>.jpg (com letterbox)."""
    classes = sorted(d for d in os.listdir(IMAGENS_DIR)
                     if os.path.isdir(os.path.join(IMAGENS_DIR, d)))
    if os.path.isdir(DATA_DIR):
        outras = sorted(set(os.listdir(DATA_DIR)) - set(classes))
        if outras:
            print(f"ATENCAO: data/ tambem tem {outras}. O train.py vai treinar essas "
                  "classes junto. Apague essas pastas se nao quiser.")
    resumo = {}
    for cls in classes:
        origem, saida = os.path.join(IMAGENS_DIR, cls), os.path.join(DATA_DIR, cls)
        shutil.rmtree(saida, ignore_errors=True)  # refaz do zero: sem frame antigo sobrando
        os.makedirs(saida)
        vistos, salvos = [], 0
        for arq in sorted(f for f in os.listdir(origem) if f.lower().endswith(IMG_EXT)):
            grupo = re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9-]", "_", os.path.splitext(arq)[0]))
            try:
                quadros = ler_imagem(os.path.join(origem, arq), QUADROS_POR_GIF)
            except Exception as e:
                print(f"  ! pulei {cls}/{arq}: {e}")
                continue
            for k, quadro in enumerate(quadros):
                img = letterbox(quadro)
                mini = cv2.resize(img, (32, 32)).astype(np.float32)
                if vistos and np.abs(np.stack(vistos) - mini).mean(axis=(1, 2, 3)).min() < DIFF_MINIMA:
                    continue  # repetida (ex.: mesmo sprite em dois jogos)
                vistos.append(mini)
                cv2.imwrite(os.path.join(saida, f"{grupo}__{k:06d}.jpg"), img)
                salvos += 1
        resumo[cls] = salvos
        print(f"  {cls}: {salvos} imagens")
    return resumo


if __name__ == "__main__":
    print("Baixando da PokeAPI...")
    for cls, pokemons in CLASSES.items():
        baixar_classe(cls, pokemons)

    print("\nConvertendo imagens/ -> data/ ...")
    resumo = converter()
    print("\nTotal por classe:", resumo)
    menor, maior = min(resumo.values()), max(resumo.values())
    if maior > 1.5 * max(menor, 1):
        print("ATENCAO: classes desbalanceadas. Ponha mais imagens na classe menor.")
    print("Pronto! Agora rode: python train.py")
