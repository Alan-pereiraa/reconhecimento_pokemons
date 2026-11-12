"""
Passo 0 - Grava os videos dos Pokemon a partir dos modelos 3D da
Pokemon 3D API (https://pokemon-3d-api.github.io/Showcase/).

Em vez de filmar uma pelucia com a webcam, o script abre cada modelo 3D
(.glb) no <model-viewer> (o mesmo visualizador do Showcase), dentro de um
Chrome sem janela, e "filma" o modelo girando a camera em volta dele.
Cada video mistura todos os angulos pedidos na atividade:
    frente, costas, lados, de cima (~45 graus), de baixo, perto e longe.

Uso:
    python gerar_videos.py                    -> videos de treino + video de teste
    python gerar_videos.py costas pikachu     -> teste so com o Pikachu de costas   (pergunta 3)
    python gerar_videos.py sem-costas pikachu -> refaz os videos de treino do Pikachu
                                                 SEM nenhum quadro de costas         (pergunta 3)
    python gerar_videos.py desconhecido 1     -> teste com um Pokemon fora do treino
                                                 (numero da Pokedex, 1 = Bulbasaur)  (pergunta 4)
    python gerar_videos.py dupla              -> teste com 2 Pokemon no mesmo quadro (pergunta 5)

Saida:
    videos/<classe>/v01_claro.mp4 ...    (treino: 4 videos por classe, 2 claros e 2 escuros)
    videos/fundo/v01_claro.mp4 ...       (cenario sem Pokemon, 8 videos)
    videos_teste/teste.mp4               (os 3 Pokemon, um de cada vez, num cenario novo)

Depois: python extrair_frames.py  e  python train.py

Precisa de internet (modelos 3D e o <model-viewer>) e do Playwright:
    pip install playwright
O script usa o Google Chrome instalado. Sem ele, rode: playwright install chromium
"""
import math
import os
import random
import sys
import urllib.request
from multiprocessing import Pool

import cv2
import numpy as np

from utils import abrir_video

# Classe -> numero na Pokedex. Pikachu e Psyduck sao ambos amarelos (sugestao
# da atividade: deixa o problema mais interessante).
POKEMON = {"pikachu": 25, "psyduck": 54, "charmander": 4}

URL_MODELO = "https://raw.githubusercontent.com/Pokemon-3D-api/assets/main/models/opt/regular/{}.glb"
MODELOS_DIR = "modelos_3d"  # cache dos .glb baixados
VIDEOS_DIR = "videos"
TESTE_DIR = "videos_teste"

LARGURA, ALTURA = 640, 480  # resolucao de webcam comum
FPS = 8  # o extrair_frames.py aproveita 4 por segundo
SEGUNDOS = 30  # duracao de cada video de treino
VIDEOS_POR_CLASSE = 4  # metade com cenario claro, metade escuro
VIDEOS_FUNDO = 8  # o fundo gera menos fotos por video, entao grava mais videos
FOV = 30  # campo de visao da camera, em graus
GIRO = 72  # graus por segundo que a camera anda em volta do Pokemon (1 volta a cada 5 s)
PROCESSOS = 3  # videos gerados ao mesmo tempo (cada um abre um Chrome)
SEED = 42

# Cenarios de treino ("claro" e "escuro") sao sorteados em sortear_fundo().
# O de teste (toalha xadrez) nunca aparece no treino ("lugar diferente").
CENARIOS = {
    "teste": [
        "linear-gradient(#cfe3d0 0 50%, transparent 50%), "
        "repeating-conic-gradient(#b5543c 0 25%, #f3e3c8 0 50%) 0 0 / 56px 56px",
    ],
}
# Cores dos "objetos de mesa" (caixas e circulos) espalhados pelo cenario.
CORES_OBJETOS = ["#5b7c99", "#8c5a3c", "#3f6b4f", "#7a6f9b", "#9a9a9a", "#b0413e", "#2f4858"]

HTML = """<!DOCTYPE html><html><head>
<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js"></script>
<style>
html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; }
#cena { position: absolute; inset: -15%; }
#cena div { position: absolute; }
model-viewer { position: absolute; top: 0; height: 100%; background: transparent;
               --poster-color: transparent; --progress-bar-height: 0; }
</style></head><body>
<div id="cena"></div>
<model-viewer id="mv0" style="left:0;width:100%" field-of-view="{fov}deg" min-field-of-view="10deg"
  max-field-of-view="90deg" min-camera-orbit="-Infinity 0deg 0m" max-camera-orbit="Infinity 180deg 1000m"
  interaction-prompt="none" disable-zoom disable-pan shadow-intensity="1" autoplay></model-viewer>
<model-viewer id="mv1" style="left:50%;width:50%;display:none" field-of-view="{fov}deg" min-field-of-view="10deg"
  max-field-of-view="90deg" min-camera-orbit="-Infinity 0deg 0m" max-camera-orbit="Infinity 180deg 1000m"
  interaction-prompt="none" disable-zoom disable-pan shadow-intensity="1" autoplay></model-viewer>
<script>
const proximo = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
async function carregar(mv, src) {
  if (mv.getAttribute('src') === src && mv.loaded) return;
  const ok = new Promise(r => mv.addEventListener('load', r, {once: true}));
  mv.setAttribute('src', src);
  await ok;
  // Alguns modelos (ex.: Pikachu) so ficam em pe com a animacao aplicada, mas
  // a animacao deles e um ataque (ele pula). Congela no primeiro quadro.
  if (mv.availableAnimations.length) { mv.pause(); mv.currentTime = 0; }
  await proximo();
}
// Aplica o estado de um quadro: cenario, posicao do cenario e camera de cada modelo.
window.aplicar = async (st) => {
  const cena = document.getElementById('cena');
  if (st.cena !== undefined) cena.innerHTML = st.cena;
  if (st.fundo !== undefined) cena.style.background = st.fundo;
  cena.style.visibility = st.esconder_cena ? 'hidden' : 'visible';
  cena.style.transform = `translate(${st.px}%, ${st.py}%) scale(${st.zoom})`;
  for (let i = 0; i < 2; i++) {
    const mv = document.getElementById('mv' + i), m = st.mvs[i];
    if (!m) { mv.style.visibility = 'hidden'; continue; }
    mv.style.visibility = 'visible';
    await carregar(mv, m.src);
    mv.exposure = m.exposicao;
    mv.shadowIntensity = m.sombra;
    mv.cameraTarget = `${m.alvo[0]}m ${m.alvo[1]}m ${m.alvo[2]}m`;
    mv.cameraOrbit = `${m.az}deg ${m.pol}deg ${m.raio}m`;
    mv.jumpCameraToGoal();
  }
  await proximo();
};
window.medidas = (i) => {
  const mv = document.getElementById('mv' + i);
  const c = mv.getBoundingBoxCenter(), d = mv.getDimensions();
  return [[c.x, c.y, c.z], Math.max(d.x, d.y, d.z)];
};
</script></body></html>""".replace("{fov}", str(FOV))


# ---- Download dos modelos 3D --------------------------------------------
def baixar_modelo(numero):
    os.makedirs(MODELOS_DIR, exist_ok=True)
    caminho = os.path.join(MODELOS_DIR, f"{numero}.glb")
    if not os.path.exists(caminho):
        print(f"  baixando modelo 3D #{numero} ...")
        urllib.request.urlretrieve(URL_MODELO.format(numero), caminho)
    return caminho


# ---- Navegador ----------------------------------------------------------
def abrir_navegador(pw):
    args = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
    try:
        nav = pw.chromium.launch(channel="chrome", args=args)
    except Exception:
        nav = pw.chromium.launch(args=args)  # Chromium do "playwright install"
    pagina = nav.new_page(viewport={"width": LARGURA, "height": ALTURA})

    def servir(rota):
        url = rota.request.url
        if url.endswith("/"):
            rota.fulfill(body=HTML, content_type="text/html")
        else:  # http://pokemon.local/<numero>.glb
            numero = int(url.rsplit("/", 1)[1].split(".")[0])
            rota.fulfill(path=baixar_modelo(numero), content_type="model/gltf-binary")

    pagina.route("http://pokemon.local/**", servir)
    pagina.goto("http://pokemon.local/")
    pagina.wait_for_function("() => window.aplicar !== undefined && !!customElements.get('model-viewer')")
    return nav, pagina


def mascara(pagina):
    """Screenshot com fundo transparente -> caixa (x0, y0, x1, y1) do modelo."""
    png = pagina.screenshot(type="png", omit_background=True)
    alfa = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)[:, :, 3]
    ys, xs = np.nonzero(alfa > 20)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def calibrar(pagina, numero):
    """Acha o centro e o tamanho reais do modelo.
    Os modelos tem esqueleto (skin) e o <model-viewer> calcula a caixa do
    modelo sem ele, entao o centro que ele informa pode ficar longe do
    Pokemon. Aqui o modelo e fotografado de frente e de lado sem cenario e o
    alvo da camera e corrigido ate o Pokemon ficar no centro da imagem."""
    pagina.evaluate("window.aplicar", {"esconder_cena": True, "px": 0, "py": 0, "zoom": 1,
                                       "mvs": [{"src": f"/{numero}.glb", "exposicao": 1, "sombra": 0,
                                                "alvo": [0, 0, 0], "az": 0, "pol": 90, "raio": 5}]})
    alvo, tamanho = pagina.evaluate("window.medidas(0)")
    alvo = list(alvo)
    raio = tamanho * 3
    larg_m = alt_m = prof_m = tamanho
    for _ in range(4):
        for az in (0, 90):
            pagina.evaluate("window.aplicar", {
                "esconder_cena": True, "px": 0, "py": 0, "zoom": 1,
                "mvs": [{"src": f"/{numero}.glb", "exposicao": 1, "sombra": 0,
                         "alvo": alvo, "az": az, "pol": 90, "raio": raio}]})
            caixa = mascara(pagina)
            if caixa is None:
                raio *= 1.5
                continue
            x0, y0, x1, y1 = caixa
            m_por_px = 2 * raio * math.tan(math.radians(FOV / 2)) / ALTURA
            dx = ((x0 + x1) / 2 - LARGURA / 2) * m_por_px
            dy = ((y0 + y1) / 2 - ALTURA / 2) * m_por_px
            alvo[1] -= dy  # tela para baixo = y negativo
            if az == 0:  # camera em +z: direita da tela = +x
                alvo[0] += dx
                larg_m = (x1 - x0) * m_por_px
            else:  # camera em +x: direita da tela = -z
                alvo[2] -= dx
                prof_m = (x1 - x0) * m_por_px
            alt_m = (y1 - y0) * m_por_px
        raio = max(larg_m, alt_m, prof_m) * 2.5
    return alvo, max(larg_m, alt_m, prof_m)


# ---- Cenario ------------------------------------------------------------
def sortear_fundo(rng, tipo):
    """Parede + mesa com cores sorteadas: claras (luz 75-97%) ou escuras (6-30%)."""
    if tipo == "teste":
        return rng.choice(CENARIOS["teste"])
    lum = (75, 97) if tipo == "claro" else (6, 30)

    def cor(delta=0):
        return (f"hsl({rng.uniform(0, 360):.0f} {rng.uniform(0, 30):.0f}% "
                f"{min(max(rng.uniform(*lum) + delta, 3), 98):.0f}%)")
    estilo = rng.random()
    if estilo < 0.6:  # parede e mesa, separadas numa altura sorteada
        meio = rng.uniform(40, 70)
        return f"linear-gradient({cor()} 0 {meio:.0f}%, {cor(-12)} {meio:.0f}%)"
    if estilo < 0.85:  # degrade
        return f"linear-gradient({rng.uniform(0, 360):.0f}deg, {cor()}, {cor(-10)})"
    return f"radial-gradient(circle at {rng.uniform(20, 80):.0f}% 30%, {cor(5)}, {cor(-15)})"


def montar_cenario(rng, tipo, min_objetos=0):
    """Fundo (parede + mesa) e de min_objetos a 3 objetos simples espalhados."""
    fundo = sortear_fundo(rng, tipo)
    objetos = []
    for _ in range(rng.randint(min_objetos, 3)):
        w, h = rng.uniform(4, 12), rng.uniform(5, 16)
        x = rng.choice([rng.uniform(5, 30), rng.uniform(65, 90)])  # longe do centro
        y = rng.uniform(55, 80) - h
        raio = "50%" if rng.random() < 0.3 else "8%"
        objetos.append(f'<div style="left:{x:.1f}%;top:{y:.1f}%;width:{w:.1f}%;height:{h:.1f}%;'
                       f'background:{rng.choice(CORES_OBJETOS)};border-radius:{raio};'
                       f'box-shadow:0 6px 10px rgba(0,0,0,.35)"></div>')
    return fundo, "".join(objetos)


def onda(t, freq, fase):
    return math.sin(2 * math.pi * (freq * t + fase))


def camera(cal, t, fases, modo="todos"):
    """Posicao da camera t segundos depois do inicio do trecho.
    O movimento e rapido de proposito: o extrair_frames.py descarta frames
    quase iguais ao anterior, entao camera parada nao gera fotos novas.
    - az (giro em volta): frente 0, lado 90, costas 180, outro lado 270;
    - pol (inclinacao): 90 = altura do Pokemon, 45 = de cima, ~120 = de baixo;
    - preenche: fracao da altura da tela ocupada (0.9 = perto, 0.33 = longe)."""
    alvo, tamanho = cal
    f_az, f_pol, f_dist, f_alvo = fases
    if modo == "costas":  # so de costas (180 +- 50)
        az = 180 + 50 * onda(t, 0.25, f_az)
    elif modo == "sem-costas":  # nunca passa de 110 graus para cada lado
        az = 110 * onda(t, 0.2, f_az)
    else:  # uma volta completa a cada 360/GIRO segundos
        az = 360 * f_az + GIRO * t
    pol = 82 + 38 * onda(t, 0.17, f_pol)
    preenche = 0.61 + 0.28 * onda(t, 0.13, f_dist)
    raio = tamanho / (preenche * 2 * math.tan(math.radians(FOV / 2)))
    jitter = [0.12 * tamanho * onda(t, 0.2 + 0.07 * k, f_alvo + 0.3 * k) for k in range(3)]
    return {"az": az, "pol": pol, "raio": raio,
            "alvo": [alvo[k] + jitter[k] for k in range(3)]}


def ruido(img, rng_np):
    """Pequeno ruido de sensor para parecer imagem de camera."""
    r = rng_np.normal(0, 3.0, img.shape)
    return np.clip(img.astype(np.float32) + r, 0, 255).astype(np.uint8)


# ---- Gravacao de um video -----------------------------------------------
def gravar(job):
    """job = {saida, segundos, cenas, trechos}. Cada trecho diz qual(is)
    Pokemon aparecem (ou nenhum) e por quanto tempo."""
    from playwright.sync_api import sync_playwright

    rng = random.Random(job["seed"])
    rng_np = np.random.default_rng(job["seed"])
    os.makedirs(os.path.dirname(job["saida"]), exist_ok=True)
    with sync_playwright() as pw:
        nav, pagina = abrir_navegador(pw)
        numeros = {n for tr in job["trechos"] for n in tr["pokemon"]}
        cal = {n: calibrar(pagina, n) for n in numeros}
        if any(len(tr["pokemon"]) == 2 for tr in job["trechos"]):
            pagina.evaluate("""() => { const a = document.getElementById('mv0'),
                b = document.getElementById('mv1'); a.style.width = '50%';
                b.style.display = 'block'; }""")

        writer = abrir_video(job["saida"], FPS, (LARGURA, ALTURA))
        exposicao = rng.uniform(0.8, 1.2)
        fundo, objetos = montar_cenario(rng, job["cenario"])
        pagina.evaluate("window.aplicar", {"cena": objetos, "fundo": fundo, "px": 0, "py": 0,
                                           "zoom": 1, "mvs": []})
        quadro = 0
        for tr in job["trechos"]:
            n_quadros = int(tr["segundos"] * FPS)
            fases = [[rng.random() for _ in range(4)] for _ in tr["pokemon"]]
            if tr.get("novo_cenario"):  # a classe fundo troca de cenario a cada trecho
                fundo, objetos = montar_cenario(rng, job["cenario"], tr.get("min_objetos", 0))
            p_fase = [rng.random(), rng.random()]
            for k in range(n_quadros):
                t = k / FPS  # segundos desde o inicio do trecho
                tg = quadro / FPS  # tempo global: a "mao" que segura a camera
                mov = tr.get("movimento", 1.0)
                # o cenario e 30% maior que a tela: com zoom >= 1 e deslocamento
                # de no maximo ~10% da tela, a borda dele nunca aparece
                st = {"px": mov * 3 * onda(tg, 0.23, p_fase[0]), "py": mov * 2.5 * onda(tg, 0.31, p_fase[1]),
                      "zoom": 1 + mov * 0.04 * (1 + onda(tg, 0.17, p_fase[0])), "mvs": []}
                if k == 0:
                    st["cena"], st["fundo"] = objetos, fundo
                for n, fs in zip(tr["pokemon"], fases):
                    c = camera(cal[n], t, fs, tr.get("modo", "todos"))
                    c.update({"src": f"/{n}.glb", "exposicao": exposicao, "sombra": 1})
                    st["mvs"].append(c)
                pagina.evaluate("window.aplicar", st)
                jpg = pagina.screenshot(type="jpeg", quality=92)
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                writer.write(ruido(img, rng_np))
                quadro += 1
        writer.release()
        nav.close()
    print(f"  pronto: {job['saida']}")
    return job["saida"]


# ---- Listas de videos ---------------------------------------------------
def jobs_treino(classes, modo="todos"):
    jobs = []
    for c_idx, cls in enumerate(classes):
        n_videos = VIDEOS_FUNDO if cls == "fundo" else VIDEOS_POR_CLASSE
        for v in range(n_videos):
            cenario = "claro" if v < n_videos // 2 else "escuro"
            seed = SEED + 100 * c_idx + v
            if cls == "fundo":  # sem Pokemon: varios cenarios e camera mexendo mais
                n = SEGUNDOS  # um cenario novo a cada 1 s
                trechos = [{"pokemon": [], "segundos": SEGUNDOS / n, "novo_cenario": True,
                            "movimento": 3, "min_objetos": 1} for _ in range(n)]
            else:
                trechos = [{"pokemon": [POKEMON[cls]], "segundos": SEGUNDOS,
                            "modo": modo}]
            jobs.append({"saida": os.path.join(VIDEOS_DIR, cls, f"v{v + 1:02d}_{cenario}.mp4"),
                         "segundos": sum(t["segundos"] for t in trechos), "cenario": cenario,
                         "trechos": trechos, "seed": seed})
    return jobs


def job_teste():
    """30 a 60 s, os 3 Pokemon um de cada vez, num cenario que nao esta no treino."""
    trechos = [{"pokemon": [], "segundos": 2}]
    for n in POKEMON.values():
        trechos += [{"pokemon": [n], "segundos": 13}, {"pokemon": [], "segundos": 2.5}]
    return {"saida": os.path.join(TESTE_DIR, "teste.mp4"), "cenario": "teste", "seed": SEED + 999,
            "segundos": sum(t["segundos"] for t in trechos), "trechos": trechos}


def job_simples(nome, pokemon, segundos=15, modo="todos"):
    trechos = [{"pokemon": [], "segundos": 1.5},
               {"pokemon": pokemon, "segundos": segundos, "modo": modo},
               {"pokemon": [], "segundos": 1.5}]
    return {"saida": os.path.join(TESTE_DIR, f"{nome}.mp4"), "cenario": "teste", "seed": SEED + 777,
            "segundos": sum(t["segundos"] for t in trechos), "trechos": trechos}


def rodar(jobs):
    for n in {n for j in jobs for tr in j["trechos"] for n in tr["pokemon"]}:
        baixar_modelo(n)  # baixa antes, para os processos nao baixarem juntos
    print(f"Gerando {len(jobs)} video(s) ({PROCESSOS} por vez)...")
    with Pool(min(PROCESSOS, len(jobs))) as pool:
        pool.map(gravar, jobs, chunksize=1)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        rodar(jobs_treino(list(POKEMON) + ["fundo"]) + [job_teste()])
        print("Pronto! Agora rode: python extrair_frames.py")
    elif args[0] == "costas" and len(args) > 1:
        rodar([job_simples(f"costas_{args[1]}", [POKEMON[args[1]]], modo="costas")])
    elif args[0] == "sem-costas" and len(args) > 1:
        rodar(jobs_treino([args[1]], modo="sem-costas"))
        print("Apague a pasta data/ e rode de novo: python extrair_frames.py")
    elif args[0] == "desconhecido":
        numero = int(args[1]) if len(args) > 1 else 1
        rodar([job_simples(f"desconhecido_{numero}", [numero])])
    elif args[0] == "dupla":
        a, b = list(POKEMON.values())[:2]
        rodar([job_simples("dupla", [a, b])])
    else:
        raise SystemExit(__doc__)
