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


def ler_imagem(caminho, max_quadros=1):
    """Abre uma foto (jpg, png, webp, gif...) e devolve uma LISTA de quadros
    em BGR, o mesmo formato que o OpenCV le da webcam.
    - PNG/WEBP transparente: corta o espaco vazio em volta e pinta o fundo
      de branco (sprites de jogo e artes oficiais vem assim).
    - GIF animado: devolve ate max_quadros quadros espalhados pela animacao.
    Usada no treino (baixar_pokemons.py) e na deteccao (detectar_imagem.py)."""
    quadros = []
    with Image.open(caminho) as im:
        n = getattr(im, "n_frames", 1)
        for i in sorted(set(np.linspace(0, n - 1, min(n, max_quadros)).astype(int))):
            im.seek(int(i))
            rgba = im.convert("RGBA")
            caixa = rgba.getchannel("A").getbbox()  # None = quadro todo transparente
            if caixa is None:
                continue
            rgba = rgba.crop(caixa)
            branco = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(branco, rgba).convert("RGB")
            quadros.append(cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR))
    return quadros
