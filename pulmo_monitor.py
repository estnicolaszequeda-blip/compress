# pulmo_monitor.py — Ventana Tkinter básica (oscura). Se abre al correr en el PC.
# Uso:  python pulmo_monitor.py   (conectado al Wi-Fi "ESP32-Biomedico")
# EN DESARROLLO / PRUEBA — no validado clínicamente. Sin modo demo.

import base64, json, os, socket, struct, sys, threading, time
import tkinter as tk

HOST = sys.argv[1] if len(sys.argv) > 1 else "192.168.4.1"
PORT = 81
BG, FG, MUTED = "#0f1115", "#e6e6e6", "#8b93a1"
GREEN, AMBER, RED = "#3ddc84", "#ffb020", "#ff4d4f"
CATS = {"bajo": ("BAJO", GREEN), "moderado": ("MODERADO", AMBER),
        "alto": ("ALTO", RED), "recopilando": ("recopilando…", MUTED),
        "sin_spo2": ("falta SpO2", AMBER), "sin_modelo": ("sin modelo", AMBER)}

datos = {"d": None, "t": 0.0, "estado": "Conectando…"}


def recibir():
    """Hilo: WebSocket mínimo, reconecta solo."""
    while True:
        try:
            s = socket.create_connection((HOST, PORT), timeout=5)
            key = base64.b64encode(os.urandom(16)).decode()
            s.sendall(("GET / HTTP/1.1\r\nHost: %s\r\nUpgrade: websocket\r\n"
                       "Connection: Upgrade\r\nSec-WebSocket-Key: %s\r\n"
                       "Sec-WebSocket-Version: 13\r\n\r\n" % (HOST, key)).encode())
            buf = b""
            while b"\r\n\r\n" not in buf:
                buf += s.recv(1024)
            buf = buf.split(b"\r\n\r\n", 1)[1]

            def need(n):
                nonlocal buf
                while len(buf) < n:
                    c = s.recv(4096)
                    if not c:
                        raise OSError("cerrado")
                    buf += c
                out, buf = buf[:n], buf[n:]
                return out

            datos["estado"] = "Conectado"
            while True:
                b0, b1 = need(2)
                n = b1 & 0x7F
                if n == 126:
                    n = struct.unpack(">H", need(2))[0]
                payload = need(n)
                if b0 & 0x0F == 1:
                    datos["d"] = json.loads(payload.decode())
                    datos["t"] = time.time()
        except (OSError, ValueError):
            datos["estado"] = "Sin conexión — reintentando…"
            time.sleep(2)


root = tk.Tk()
root.title("PulmoAlert")
root.configure(bg=BG)
root.geometry("340x300")


def fila(titulo):
    tk.Label(root, text=titulo, bg=BG, fg=MUTED, font=("Arial", 10)).pack(pady=(10, 0))
    lbl = tk.Label(root, text="—", bg=BG, fg=FG, font=("Arial", 24, "bold"))
    lbl.pack()
    return lbl


l_estado = tk.Label(root, text="", bg=BG, fg=MUTED, font=("Arial", 10))
l_estado.pack(pady=(8, 0))
l_bpm, l_spo2, l_riesgo = fila("BPM"), fila("SpO2 %"), fila("Riesgo")


def actualizar():
    d = datos["d"]
    if d is None or time.time() - datos["t"] > 3:
        l_estado.config(text="● " + datos["estado"], fg=RED)
        for l in (l_bpm, l_spo2, l_riesgo):
            l.config(text="—", fg=FG)
    else:
        l_estado.config(text="● En vivo", fg=GREEN)
        bpm = d.get("pulse", {}).get("bpm", 0)
        l_bpm.config(text=str(bpm) if bpm > 0 else "—")
        sp = d.get("spo2")
        l_spo2.config(text="—" if sp is None else "%.0f" % sp)
        txt, col = CATS.get(d.get("risk", {}).get("cat"), ("—", FG))
        l_riesgo.config(text=txt, fg=col)
    root.after(200, actualizar)


threading.Thread(target=recibir, daemon=True).start()
actualizar()
root.mainloop()
