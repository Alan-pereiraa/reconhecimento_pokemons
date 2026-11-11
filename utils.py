"""
utils.py - funcoes compartilhadas entre extracao, treino e deteccao.
REGRA DE OURO: a imagem deve ser preparada EXATAMENTE do mesmo jeito
no treino e na deteccao. Por isso esta funcao fica num arquivo so.
"""
import cv2
import numpy as np
from PIL import Image

IMG_SIZE = 224
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")

# Cor de cada classe na tela (BGR). Classe fora da lista aparece em branco.
CORES = {"gengar": (200, 90, 150), "charmander": (40, 140, 255), "squirtle": (230, 190, 70)}
LIMIAR = 0.6  # abaixo disso o nome aparece cinza com "?" (o modelo esta em duvida)
FONTE = cv2.FONT_HERSHEY_DUPLEX


def escrever(img, texto, org, escala, cor):
    """Texto com contorno preto: fica legivel sobre qualquer fundo."""
    esp = max(1, round(2 * escala))
    cv2.putText(img, texto, org, FONTE, escala, (0, 0, 0), esp + 3, cv2.LINE_AA)
    cv2.putText(img, texto, org, FONTE, escala, cor, esp, cv2.LINE_AA)


def escurecer(img, x0, y0, x1, y1):
    """Faixa semitransparente atras do texto."""
    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.35).astype(np.uint8)


def desenhar(frame, probs, class_names):
    """Desenha o resultado sobre o frame (BGR, alterado no lugar). Tudo escala
    com a largura do frame, para ficar legivel de 480p a 4K.
    - faixa no topo: nome do Pokemon grande, na cor da classe, e a confianca;
    - canto de baixo: uma barra de probabilidade por classe."""
    h, w = frame.shape[:2]
    s = w / 640
    i = int(np.argmax(probs))
    certo = probs[i] >= LIMIAR
    cor = CORES.get(class_names[i], (255, 255, 255)) if certo else (170, 170, 170)

    topo = int(64 * s)
    escurecer(frame, 0, 0, w, topo)
    cv2.rectangle(frame, (0, topo - max(2, int(5 * s))), (w, topo), cor, -1)
    nome = class_names[i].upper() + ("" if certo else "?")
    escrever(frame, nome, (int(16 * s), int(44 * s)), 1.3 * s, cor)
    pct = f"{probs[i]:.0%}"
    (tw, _), _ = cv2.getTextSize(pct, FONTE, 1.0 * s, max(1, round(2 * s)))
    escrever(frame, pct, (w - tw - int(16 * s), int(42 * s)), 1.0 * s, (255, 255, 255))

    lin, larg, alt = int(26 * s), int(150 * s), int(16 * s)
    x, y0 = int(12 * s), h - lin * len(class_names) - int(8 * s)
    maior = max(cv2.getTextSize(f"{c} 100%", FONTE, 0.5 * s, max(1, round(s)) + 3)[0][0]
                for c in class_names)
    escurecer(frame, 0, y0 - int(10 * s), x + larg + int(20 * s) + maior, h)
    for j, c in enumerate(class_names):
        y = y0 + j * lin
        cv2.rectangle(frame, (x, y), (x + larg, y + alt), (70, 70, 70), -1)
        cv2.rectangle(frame, (x, y), (x + int(larg * probs[j]), y + alt),
                      CORES.get(c, (255, 255, 255)), -1)
        escrever(frame, f"{c} {probs[j]:.0%}", (x + larg + int(10 * s), y + alt - int(2 * s)),
                 0.5 * s, (255, 255, 255))


def letterbox(frame, size=IMG_SIZE):
    """Redimensiona mantendo a proporcao e completa com bordas pretas.
    Assim o objeto nao fica 'esticado' nem cortado."""
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def ler_rgba(caminho, max_quadros=1):
    """Abre uma imagem e devolve uma LISTA de quadros RGBA (Pillow), cada um
    cortado no contorno do que nao e transparente.
    - GIF/WEBP animado: devolve ate max_quadros quadros espalhados pela animacao."""
    quadros = []
    with Image.open(caminho) as im:
        n = getattr(im, "n_frames", 1)
        for i in sorted(set(np.linspace(0, n - 1, min(n, max_quadros)).astype(int))):
            im.seek(int(i))
            rgba = im.convert("RGBA")
            caixa = rgba.getchannel("A").getbbox()  # None = quadro todo transparente
            if caixa is not None:
                quadros.append(rgba.crop(caixa))
    return quadros


def ler_imagem(caminho, max_quadros=1):
    """Abre uma foto (jpg, png, webp, gif...) e devolve uma LISTA de quadros
    em BGR, o mesmo formato que o OpenCV le da webcam.
    - PNG/WEBP transparente: corta o espaco vazio em volta e pinta o fundo
      de branco (sprites de jogo e artes oficiais vem assim).
    - GIF animado: devolve ate max_quadros quadros espalhados pela animacao.
    Usada no treino (baixar_pokemons.py) e na deteccao (detectar_imagem.py)."""
    quadros = []
    for rgba in ler_rgba(caminho, max_quadros):
        branco = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        rgb = Image.alpha_composite(branco, rgba).convert("RGB")
        quadros.append(cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR))
    return quadros
