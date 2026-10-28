# entrenar_modelo.py — genera modelo_riesgo.json (SOLO PC, CPython, sin librerías externas)
# ⚠ Entrena con datos SINTÉTICOS (igual que el prototipo del proyecto).
#   Un AUC alto aquí refleja lo limpio de la simulación, NO precisión clínica.
#   Reemplazar por datos reales (MAX30102 + oxímetro de referencia) antes de usarlo en serio.
# Uso:  python entrenar_modelo.py   → crea modelo_riesgo.json junto a este archivo

import json
import math
import random

import predictor as P

random.seed(42)
N_VENTANAS = 4000
PUNTOS = P.VENTANA_S // P.MUESTREO_S      # 60 muestras
HORIZONTE_MIN = 5.0                       # "próximos minutos"
NOMBRES = list(P.CARACTERISTICAS)


def ventana():
    """Una ventana simulada: (características, etiqueta 1 = SpO2 < 88 en 5 min)."""
    base = random.uniform(90.0, 96.5)
    bpm0 = random.uniform(62.0, 95.0)
    if random.random() < 0.5:
        pend = random.gauss(0.0, 0.08)                 # %/min, estable
    else:
        pend = -random.uniform(0.2, 1.0)               # descenso sostenido
    k = random.uniform(1.0, 3.0)                       # el BPM sube al caer la SpO2
    spo2, bpm = [], []
    for i in range(PUNTOS):
        t = i * P.MUESTREO_S / 60.0
        spo2.append(base + pend * t + random.gauss(0.0, 0.5))
        bpm.append(bpm0 - pend * k * t + random.gauss(0.0, 2.0))
    futuro = base + pend * (PUNTOS * P.MUESTREO_S / 60.0 + HORIZONTE_MIN)
    y = 1 if futuro + random.gauss(0.0, 0.7) < P.UMBRAL_CRISIS else 0
    return P.calcular_caracteristicas(spo2, bpm), y


def auc(y, p):
    pares = sorted(zip(p, y))
    rangos, i = {}, 0
    while i < len(pares):
        j = i
        while j + 1 < len(pares) and pares[j + 1][0] == pares[i][0]:
            j += 1
        for k in range(i, j + 1):
            rangos[k] = (i + j) / 2.0 + 1
        i = j + 1
    pos = sum(1 for _, l in pares if l == 1)
    neg = len(pares) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    s = sum(rangos[k] for k, (_, l) in enumerate(pares) if l == 1)
    return (s - pos * (pos + 1) / 2.0) / (pos * neg)


def main():
    datos = [ventana() for _ in range(N_VENTANAS)]
    corte = int(N_VENTANAS * 0.8)
    X = [[d[0][n] for n in NOMBRES] for d in datos]
    Y = [d[1] for d in datos]
    Xtr, Ytr, Xte, Yte = X[:corte], Y[:corte], X[corte:], Y[corte:]

    m = len(NOMBRES)
    mu = [sum(r[j] for r in Xtr) / len(Xtr) for j in range(m)]
    sd = [math.sqrt(sum((r[j] - mu[j]) ** 2 for r in Xtr) / len(Xtr)) or 1.0
          for j in range(m)]
    Ztr = [[(r[j] - mu[j]) / sd[j] for j in range(m)] for r in Xtr]
    Zte = [[(r[j] - mu[j]) / sd[j] for j in range(m)] for r in Xte]

    w, b, lr, l2 = [0.0] * m, 0.0, 0.2, 0.001
    for _ in range(500):
        gw, gb = [0.0] * m, 0.0
        for z, y in zip(Ztr, Ytr):
            e = 1.0 / (1.0 + math.exp(-max(-30, min(30, b + sum(a * c for a, c in zip(w, z)))))) - y
            gb += e
            for j in range(m):
                gw[j] += e * z[j]
        n = len(Ztr)
        b -= lr * gb / n
        for j in range(m):
            w[j] -= lr * (gw[j] / n + l2 * w[j])

    pred = [1.0 / (1.0 + math.exp(-max(-30, min(30, b + sum(a * c for a, c in zip(w, z))))))
            for z in Zte]
    print("Positivos: %.0f %% | AUC sintético: %.3f" % (100.0 * sum(Y) / len(Y), auc(Yte, pred)))
    print("⚠ SINTÉTICO — no es precisión clínica.")

    modelo = {
        "version": 1,
        "origen": "sintetico - EN DESARROLLO/PRUEBA, no clinico",
        "umbral_crisis": P.UMBRAL_CRISIS,
        "muestreo_s": P.MUESTREO_S,
        "ventana_s": P.VENTANA_S,
        "features": NOMBRES,
        "pesos": [round(x, 5) for x in w],
        "sesgo": round(b, 5),
        "media": [round(x, 5) for x in mu],
        "desv": [round(x, 5) for x in sd],
    }
    with open("modelo_riesgo.json", "w") as f:
        json.dump(modelo, f)
    print("Creado modelo_riesgo.json")


if __name__ == "__main__":
    main()
