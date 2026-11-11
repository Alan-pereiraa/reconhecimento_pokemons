"""
Gera data/ SO com modelos 3D dos Pokemon, colados sobre cenarios variados.
Uso: python baixar_modelos.py      (precisa de internet so na 1a vez)

1) Baixa da PokeAPI os renders 3D de cada Pokemon (Pokemon HOME e os jogos
   em 3D: Let's Go, Sword/Shield, BDSP, Legends Arceus, Scarlet/Violet,
   Legends Z-A e Champions), de frente e de costas, para modelos_3d/<classe>/.
   Arte oficial e sprites 2D ficam de fora.
2) Baixa fotos aleatorias (picsum.photos) para cenarios/ e cria alguns
   cenarios lisos (branco, cor, degrade, ruido).
3) Cola cada modelo sobre um cenario sorteado, com tamanho, posicao e
   formato de tela aleatorios, e salva em data/<classe>/<grupo>__<n>.jpg.
   Depois e so rodar train.py.

Cada render (todos os seus quadros e cenarios) e um "grupo": vai inteiro
OU para treino OU para validacao.

Quer mais variedade?
- modelos_3d/<classe>/: PNG transparente e colado nos cenarios; print de
  jogo ou foto (sem transparencia) entra do jeito que esta.
- cenarios/: fotos do lugar onde o modelo vai ser usado ajudam muito.
"""
import os
import re
import shutil

import cv2
import numpy as np

from baixar_pokemons import CLASSES, SEED, baixar, baixar_classe, criar_vazias
from utils import IMG_EXT, letterbox, ler_imagem, ler_rgba

MODELOS_DIR = "modelos_3d"
CENARIOS_DIR = "cenarios"
DATA_DIR = "data"
# Caminhos da PokeAPI que sao renders de modelos 3D. O resto (arte oficial,
# geracoes I a V) e 2D. X/Y, ORAS e USUM tambem sao 3D, mas tem ~60px: o
# MIN_LADO descarta.
FONTES_3D = re.compile(r"^(other-home|generation-(vii|viii|ix))-")
MIN_LADO = 200  # descarta renders minusculos (ficam pixelados ao ampliar)
QUADROS_POR_ANIMACAO = 6  # quadros usados de cada render animado
N_FOTOS_CENARIO = 150  # fotos baixadas do picsum.photos
IMAGENS_POR_CLASSE = 400
ALTURA = 360  # altura da "tela" antes do letterbox para 224x224
FORMATOS = (1.0, 4 / 3, 16 / 9, 3 / 4)  # largura/altura: foto, webcam, video, celular


def baixar_cenarios():
    os.makedirs(CENARIOS_DIR, exist_ok=True)
    novos = 0
    for i in range(N_FOTOS_CENARIO):
        destino = os.path.join(CENARIOS_DIR, f"picsum_{i:03d}.jpg")
        if os.path.exists(destino):
            continue
        try:
            dados = baixar(f"https://picsum.photos/seed/cenario{i}/640/480")
            with open(destino, "wb") as f:
                f.write(dados)
            novos += 1
        except Exception as e:
            print(f"  ! falhou cenario {i}: {e}")
    criar_vazias(CENARIOS_DIR)
    print(f"  cenarios: {N_FOTOS_CENARIO} fotos ({novos} novas) + cenarios lisos")


def listar(pasta):
    return sorted(f for f in os.listdir(pasta) if f.lower().endswith(IMG_EXT))


def carregar_modelos(cls):
    """[(grupo, [quadros BGRA])] de modelos_3d/<cls>/."""
    pasta = os.path.join(MODELOS_DIR, cls)
    fontes = []
    for arq in listar(pasta):
        try:
            quadros = ler_rgba(os.path.join(pasta, arq), QUADROS_POR_ANIMACAO)
        except Exception as e:
            print(f"  ! pulei {cls}/{arq}: {e}")
            continue
        quadros = [q for q in quadros if max(q.size) >= MIN_LADO]
        if quadros:
            grupo = re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9-]", "_", os.path.splitext(arq)[0]))
            fontes.append((grupo, [cv2.cvtColor(np.array(q), cv2.COLOR_RGBA2BGRA) for q in quadros]))
    return fontes


def recortar(cenario, h, w, rng):
    """Pedaco h x w do cenario, com zoom aleatorio."""
    ch, cw = cenario.shape[:2]
    s = max(h / ch, w / cw) * rng.uniform(1.0, 1.5)
    img = cv2.resize(cenario, (int(np.ceil(cw * s)), int(np.ceil(ch * s))))
    y, x = rng.integers(0, img.shape[0] - h + 1), rng.integers(0, img.shape[1] - w + 1)
    return img[y:y + h, x:x + w].copy()


def compor(modelo, cenario, rng):
    """Cola o modelo (BGRA) num cenario com formato de tela sorteado e aplica o
    mesmo letterbox da deteccao."""
    if modelo[..., 3].min() == 255:
        return letterbox(modelo[..., :3])  # sem transparencia: usa como esta
    h = ALTURA
    w = int(h * FORMATOS[rng.integers(len(FORMATOS))])
    tela = recortar(cenario, h, w, rng)
    oh, ow = modelo.shape[:2]
    s = rng.uniform(0.35, 0.95) * min(h, w) / max(oh, ow)
    nh, nw = max(1, int(oh * s)), max(1, int(ow * s))
    obj = cv2.resize(modelo, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
    y, x = rng.integers(0, h - nh + 1), rng.integers(0, w - nw + 1)
    a = obj[..., 3:] / 255.0
    roi = tela[y:y + nh, x:x + nw]
    tela[y:y + nh, x:x + nw] = (obj[..., :3] * a + roi * (1 - a)).astype(np.uint8)
    if rng.random() < 0.3:  # camera fora de foco / print redimensionado
        tela = cv2.GaussianBlur(tela, (5, 5), 0)
    return letterbox(tela)


def gerar():
    cenarios = []
    for arq in listar(CENARIOS_DIR):
        try:
            cenarios.append(ler_imagem(os.path.join(CENARIOS_DIR, arq))[0])
        except Exception as e:
            print(f"  ! pulei cenario {arq}: {e}")
    if not cenarios:
        raise SystemExit("Nenhum cenario em cenarios/. Confira a internet e rode de novo.")

    classes = sorted(CLASSES)
    if os.path.isdir(DATA_DIR):
        outras = sorted(set(os.listdir(DATA_DIR)) - set(classes))
        if outras:
            print(f"ATENCAO: data/ tambem tem {outras}. O train.py vai treinar essas "
                  "classes junto. Apague essas pastas se nao quiser.")

    rng = np.random.default_rng(SEED)
    resumo = {}
    for cls in classes:
        fontes = carregar_modelos(cls)
        if not fontes:
            print(f"  ! {cls}: nenhum modelo 3D, pulei")
            continue
        saida = os.path.join(DATA_DIR, cls)
        shutil.rmtree(saida, ignore_errors=True)  # refaz do zero
        os.makedirs(saida)
        for i in range(IMAGENS_POR_CLASSE):
            grupo, quadros = fontes[i % len(fontes)]
            modelo = quadros[rng.integers(len(quadros))]
            cenario = cenarios[rng.integers(len(cenarios))]
            cv2.imwrite(os.path.join(saida, f"{grupo}__{i:06d}.jpg"),
                        compor(modelo, cenario, rng))
        resumo[cls] = f"{IMAGENS_POR_CLASSE} imagens de {len(fontes)} modelos"
        print(f"  {cls}: {resumo[cls]}")
    return resumo


if __name__ == "__main__":
    print("Baixando modelos 3D da PokeAPI...")
    for cls, pokemons in CLASSES.items():
        baixar_classe(cls, pokemons, MODELOS_DIR, FONTES_3D)
    print("\nBaixando cenarios...")
    baixar_cenarios()

    print("\nColando os modelos nos cenarios -> data/ ...")
    gerar()
    print("Pronto! Agora rode: python train.py")
