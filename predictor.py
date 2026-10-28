# predictor.py — Predictor de riesgo de crisis (regresión logística)
# EN DESARROLLO / PRUEBA — no validado clínicamente.
# Mismo archivo para el ESP32 (MicroPython) y para el PC (CPython).
# Solo usa math y json. Los pesos se leen de modelo_riesgo.json.

import math
try:
    import ujson as json      # MicroPython
except ImportError:
    import json               # CPython

MODELO_PATH = "modelo_riesgo.json"
UMBRAL_CRISIS = 88            # SpO2 % que define "crisis"
VENTANA_S = 300               # ventana de historial: 5 min
MUESTREO_S = 5                # una muestra cada 5 s (60 máx.)
MIN_MUESTRAS = 12             # mínimo ~1 min de historial
UMBRAL_MODERADO = 0.35        # ajustar con datos reales
UMBRAL_ALTO = 0.65            # ajustar con datos reales

CARACTERISTICAS = ("spo2", "bpm", "spo2_media", "bpm_media",
                   "spo2_pendiente", "bpm_pendiente", "spo2_std", "bpm_std")


def _clave(d, *nombres):
    for n in nombres:
        if n in d:
            return d[n]
    return None


def _stats(v):
    """(media, desviación estándar, pendiente por minuto)."""
    n = len(v)
    m = sum(v) / n
    var = 0.0
    for x in v:
        var += (x - m) * (x - m)
    sd = math.sqrt(var / n)
    tm = (n - 1) / 2.0
    num = 0.0
    den = 0.0
    for i, x in enumerate(v):
        num += (i - tm) * (x - m)
        den += (i - tm) * (i - tm)
    pend = (num / den) * (60.0 / MUESTREO_S) if den else 0.0
    return m, sd, pend


def calcular_caracteristicas(spo2, bpm):
    """Dict con las características calculables. spo2 puede ir vacío."""
    v = {}
    mb, sb, pb = _stats(bpm)
    v["bpm"] = bpm[-1]
    v["bpm_media"] = mb
    v["bpm_std"] = sb
    v["bpm_pendiente"] = pb
    if spo2:
        ms, ss, ps = _stats(spo2)
        v["spo2"] = spo2[-1]
        v["spo2_media"] = ms
        v["spo2_std"] = ss
        v["spo2_pendiente"] = ps
    return v


class PredictorRiesgo:
    def __init__(self, ruta=MODELO_PATH):
        self.max_n = VENTANA_S // MUESTREO_S
        self.spo2 = []
        self.bpm = []
        self.cargado = False
        self.err = ""
        self.feats = []
        self.w = []
        self.b = 0.0
        self.mu = None
        self.sd = None
        self.prob = None
        self.cat = "sin_modelo"
        self._cargar(ruta)

    def _cargar(self, ruta):
        try:
            with open(ruta) as f:
                d = json.load(f)
        except OSError:
            self.err = "falta_json"
            return
        except ValueError:
            self.err = "json_invalido"
            return
        feats = _clave(d, "features", "caracteristicas", "nombres")
        w = _clave(d, "pesos", "weights", "coef")
        b = _clave(d, "sesgo", "bias", "intercept")
        mu = _clave(d, "media", "mean")
        sd = _clave(d, "desv", "std", "scale")
        if not feats or not w or b is None:
            self.err = "json_incompleto"
            return
        if len(feats) != len(w):
            self.err = "largo_distinto"
            return
        for nombre in feats:
            if nombre not in CARACTERISTICAS:
                self.err = "feature_desconocida"
                return
        norm = mu is not None and sd is not None
        if norm:
            if len(mu) != len(w) or len(sd) != len(w):
                self.err = "norm_distinta"
                return
            for x in sd:
                if x == 0:
                    self.err = "desv_cero"
                    return
        self.feats = list(feats)
        self.w = list(w)
        self.b = float(b)
        self.mu = list(mu) if norm else None
        self.sd = list(sd) if norm else None
        self.cargado = True
        self.err = ""
        self.cat = "recopilando"

    def reset(self):
        self.spo2 = []
        self.bpm = []
        self.prob = None
        if self.cargado:
            self.cat = "recopilando"

    def registrar(self, spo2, bpm):
        if bpm and bpm > 0:
            self.bpm.append(float(bpm))
            if len(self.bpm) > self.max_n:
                self.bpm.pop(0)
        if spo2 is not None:
            self.spo2.append(float(spo2))
            if len(self.spo2) > self.max_n:
                self.spo2.pop(0)

    def probabilidad(self, v):
        """Probabilidad a partir de un dict de características."""
        z = self.b
        for i, nombre in enumerate(self.feats):
            x = v[nombre]
            if self.mu is not None:
                x = (x - self.mu[i]) / self.sd[i]
            z += self.w[i] * x
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def evaluar(self):
        if not self.cargado:
            self.cat = "sin_modelo"
            self.prob = None
            return
        usa_spo2 = False
        for f in self.feats:
            if f.startswith("spo2"):
                usa_spo2 = True
        if usa_spo2 and len(self.spo2) == 0:
            self.cat = "sin_spo2"
            self.prob = None
            return
        if len(self.bpm) < MIN_MUESTRAS or (usa_spo2 and len(self.spo2) < MIN_MUESTRAS):
            self.cat = "recopilando"
            self.prob = None
            return
        p = self.probabilidad(calcular_caracteristicas(
            self.spo2 if usa_spo2 else [], self.bpm))
        self.prob = p
        if p >= UMBRAL_ALTO:
            self.cat = "alto"
        elif p >= UMBRAL_MODERADO:
            self.cat = "moderado"
        else:
            self.cat = "bajo"

    def json(self):
        p = "null" if self.prob is None else "%.3f" % self.prob
        return ('{"ok":%s,"p":%s,"cat":"%s","n":%d,"err":"%s"}'
                % ("true" if self.cargado else "false", p, self.cat,
                   len(self.bpm), self.err))
