# main.py — Telemetría biomédica ESP32 (MicroPython)
# Pulso PPG (GPIO34) + MPU-6050 (I2C) + GPS NEO-6M (UART2)
# Punto de acceso Wi-Fi + HTTP (80) + WebSocket (81)
# Requiere el archivo index.html en la raíz del sistema de archivos del ESP32.
# + Predictor de riesgo de crisis (regresión logística, pesos en modelo_riesgo.json)
#   ESTADO: EN DESARROLLO / PRUEBA — no validado clínicamente.

import time
import struct
import gc
import hashlib
import binascii
import network
import uasyncio as asyncio
from machine import Pin, I2C, UART, ADC

# ── Configuración Wi-Fi (Modo Access Point) ─────────────────
AP_SSID = "ESP32-Biomedico"
AP_PASS = "12345678"

# ── Pines ───────────────────────────────────────────────────
SDA_PIN = 21
SCL_PIN = 22
GPS_TX_PIN = 16
GPS_RX_PIN = 17
PULSE_PIN = 34

# ── MPU-6050 ────────────────────────────────────────────────
MPU_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
ACCEL_CONFIG = 0x1C
GYRO_CONFIG = 0x1B
ACCEL_SCALE = 16384.0
GYRO_SCALE = 131.0
TEMP_SCALE = 340.0

# ── Parámetros Sensor de Pulso ──────────────────────────────
SAMPLE_INTERVAL_MS = 10
BUFFER_SIZE = 50
RR_HISTORY = 4
PEAK_MIN_DISTANCE_MS = 350
PEAK_MIN_DISTANCE_MS_MAX = 1500
THRESHOLD_RATIO = 0.6
EMA_ALPHA = 0.25
FINGER_MIN_AMPLITUDE = 80

# ── Hardware ────────────────────────────────────────────────
i2c = I2C(0, sda=Pin(SDA_PIN), scl=Pin(SCL_PIN), freq=400000, timeout=50000)
gps_uart = UART(2, baudrate=9600, bits=8, parity=None, stop=1,
                rx=GPS_RX_PIN, tx=GPS_TX_PIN, rxbuf=512)
adc = ADC(Pin(PULSE_PIN))
adc.atten(ADC.ATTN_11DB)  # rango completo ~0-3.3 V


# ═══════════════════════════════════════════════════════════
#  Wi-Fi
# ═══════════════════════════════════════════════════════════
def wifi_ap():
    ap = network.WLAN(network.AP_IF)
    ap.active(True)
    ap.config(essid=AP_SSID, password=AP_PASS,
              authmode=network.AUTH_WPA_WPA2_PSK)
    while not ap.active():
        time.sleep_ms(50)
    print("\n[Wi-Fi] Red iniciada. IP del servidor:", ap.ifconfig()[0])


# ═══════════════════════════════════════════════════════════
#  DRIVER MPU-6050
# ═══════════════════════════════════════════════════════════
def mpu_init():
    i2c.writeto_mem(MPU_ADDR, PWR_MGMT_1, b'\x00')
    time.sleep_ms(100)
    i2c.writeto_mem(MPU_ADDR, ACCEL_CONFIG, b'\x00')  # ±2 g
    i2c.writeto_mem(MPU_ADDR, GYRO_CONFIG, b'\x00')   # ±250 °/s


def mpu_read():
    """Devuelve (ax, ay, az, gx, gy, gz, temp). Ceros si falla el I2C."""
    try:
        raw = i2c.readfrom_mem(MPU_ADDR, ACCEL_XOUT_H, 14)
    except OSError:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ax, ay, az, tp, gx, gy, gz = struct.unpack('>7h', raw)
    return (ax / ACCEL_SCALE, ay / ACCEL_SCALE, az / ACCEL_SCALE,
            gx / GYRO_SCALE, gy / GYRO_SCALE, gz / GYRO_SCALE,
            tp / TEMP_SCALE + 36.53)


# ═══════════════════════════════════════════════════════════
#  SENSOR DE PULSO (detección de picos, 100 Hz)
# ═══════════════════════════════════════════════════════════
class Pulse:
    def __init__(self):
        self.buf = [2048] * BUFFER_SIZE
        self.idx = 0
        self.peak = False
        self.last_peak = None
        self.rr = [0] * RR_HISTORY
        self.rr_idx = 0
        self.rr_count = 0
        self.ema = 2048.0
        # Salidas (persisten entre latidos)
        self.finger = False
        self.raw = 0
        self.filtered = 0
        self.bpm = 0
        self.rr_last = 0

    def update(self, raw, now):
        self.raw = raw
        self.ema = EMA_ALPHA * raw + (1.0 - EMA_ALPHA) * self.ema
        f = int(self.ema)
        self.filtered = f

        self.buf[self.idx] = f
        self.idx = (self.idx + 1) % BUFFER_SIZE

        lo = min(self.buf)
        hi = max(self.buf)
        amp = hi - lo
        self.finger = amp >= FINGER_MIN_AMPLITUDE

        if not self.finger:
            self.rr_count = 0
            self.peak = False
            self.bpm = 0
            self.rr_last = 0
            return

        thr = lo + int(amp * THRESHOLD_RATIO)
        if self.last_peak is None:
            dt = PEAK_MIN_DISTANCE_MS_MAX + 1
        else:
            dt = time.ticks_diff(now, self.last_peak)

        if not self.peak and f > thr and dt > PEAK_MIN_DISTANCE_MS:
            self.peak = True
            if self.last_peak is not None and dt < PEAK_MIN_DISTANCE_MS_MAX:
                self.rr[self.rr_idx] = dt
                self.rr_idx = (self.rr_idx + 1) % RR_HISTORY
                if self.rr_count < RR_HISTORY:
                    self.rr_count += 1

                start = (self.rr_idx - self.rr_count) % RR_HISTORY
                total = 0
                for i in range(self.rr_count):
                    total += self.rr[(start + i) % RR_HISTORY]
                self.bpm = int(60000 / (total / self.rr_count))
                self.rr_last = self.rr[(self.rr_idx - 1) % RR_HISTORY]
            self.last_peak = now

        if self.peak and f < thr:
            self.peak = False


pulse = Pulse()


async def pulse_task():
    nxt = time.ticks_add(time.ticks_ms(), SAMPLE_INTERVAL_MS)
    while True:
        wait = time.ticks_diff(nxt, time.ticks_ms())
        if wait < -5 * SAMPLE_INTERVAL_MS:       # nos atrasamos: resincronizar
            nxt = time.ticks_ms()
            wait = 0
        await asyncio.sleep_ms(wait if wait > 0 else 0)
        nxt = time.ticks_add(nxt, SAMPLE_INTERVAL_MS)
        pulse.update(adc.read_u16() >> 4, time.ticks_ms())  # 12 bits


# ═══════════════════════════════════════════════════════════
#  PREDICTOR DE RIESGO DE CRISIS  (EN DESARROLLO / PRUEBA)
#  La lógica vive en predictor.py y los pesos en modelo_riesgo.json:
#  copia AMBOS archivos a la raíz del ESP32 junto a este main.py.
#  Sin modelo_riesgo.json el predictor NO inventa nada: "sin_modelo".
# ═══════════════════════════════════════════════════════════
from predictor import PredictorRiesgo, MUESTREO_S

SIN_DEDO_RESET = 3       # ciclos sin dedo antes de borrar el historial

# Fuente de SpO2. El sensor PPG analógico (GPIO34) solo da BPM.
# Mientras no exista un oxímetro real (MAX30102) queda en None
# y el predictor responde "sin_spo2" en vez de adivinar.
spo2_actual = None


def leer_spo2():
    """Punto único donde conectar un sensor de SpO2 real. Devuelve % o None."""
    return spo2_actual


riesgo = PredictorRiesgo()


async def riesgo_task():
    sin_dedo = 0
    while True:
        await asyncio.sleep(MUESTREO_S)
        if pulse.finger and pulse.bpm > 0:
            sin_dedo = 0
            riesgo.registrar(leer_spo2(), pulse.bpm)
        else:
            sin_dedo += 1
            if sin_dedo >= SIN_DEDO_RESET:
                riesgo.reset()      # evita mezclar lecturas separadas por un hueco
        riesgo.evaluar()


# ═══════════════════════════════════════════════════════════
#  GPS (NMEA GGA)
# ═══════════════════════════════════════════════════════════
gps = {"valid": False, "lat": 0.0, "lng": 0.0, "sat": 0, "alt": 0.0,
       "tim": ""}


def nmea_to_decimal(val, d):
    if len(val) < 4:
        return 0.0
    raw = float(val)
    deg = int(raw / 100)
    dec = deg + (raw - deg * 100) / 60.0
    return -dec if d in ("S", "W") else dec


def nmea_checksum_ok(line):
    if not line.startswith("$") or "*" not in line:
        return False
    body, cs = line[1:].split("*", 1)
    x = 0
    for ch in body:
        x ^= ord(ch)
    try:
        return x == int(cs[:2], 16)
    except ValueError:
        return False


def parse_gga(line):
    if len(line) < 10:
        return
    if not (line.startswith("$GPGGA") or line.startswith("$GNGGA")):
        return
    if not nmea_checksum_ok(line):
        return
    t = line.split(",")
    try:
        if len(t) < 10 or not t[6] or int(t[6]) <= 0:
            return
        lat = nmea_to_decimal(t[2], t[3][0] if t[3] else "N")
        lng = nmea_to_decimal(t[4], t[5][0] if t[5] else "E")
        sat = int(t[7]) if t[7] else 0
        alt = float(t[9]) if t[9] else 0.0
    except ValueError:
        return
    gps["valid"] = True
    gps["lat"] = lat
    gps["lng"] = lng
    gps["sat"] = sat
    gps["alt"] = alt
    if len(t[1]) >= 6:
        gps["tim"] = "%s:%s:%s" % (t[1][0:2], t[1][2:4], t[1][4:6])


async def gps_task():
    reader = asyncio.StreamReader(gps_uart)
    while True:
        raw = await reader.readline()
        try:
            parse_gga(raw.decode().strip())
        except (UnicodeError, ValueError):
            pass


# ═══════════════════════════════════════════════════════════
#  WebSocket mínimo (servidor, puerto 81)
# ═══════════════════════════════════════════════════════════
WS_GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
clients = set()


def ws_frame(payload, opcode=0x1):
    n = len(payload)
    if n < 126:
        head = bytes([0x80 | opcode, n])
    elif n < 65536:
        head = bytes([0x80 | opcode, 126]) + n.to_bytes(2, "big")
    else:
        head = bytes([0x80 | opcode, 127]) + n.to_bytes(8, "big")
    return head + payload


async def ws_handler(reader, writer):
    try:
        key = None
        while True:
            line = await reader.readline()
            if not line or line == b"\r\n":
                break
            if line.lower().startswith(b"sec-websocket-key:"):
                key = line.split(b":", 1)[1].strip()
        if key is None:
            return

        accept = binascii.b2a_base64(
            hashlib.sha1(key + WS_GUID).digest()).strip()
        writer.write(b"HTTP/1.1 101 Switching Protocols\r\n"
                     b"Upgrade: websocket\r\nConnection: Upgrade\r\n"
                     b"Sec-WebSocket-Accept: " + accept + b"\r\n\r\n")
        await writer.drain()
        clients.add(writer)

        # El navegador casi no envía nada; solo atendemos ping/close
        while True:
            hdr = await reader.readexactly(2)
            opcode = hdr[0] & 0x0F
            n = hdr[1] & 0x7F
            if n == 126:
                n = int.from_bytes(await reader.readexactly(2), "big")
            elif n == 127:
                n = int.from_bytes(await reader.readexactly(8), "big")
            mask = await reader.readexactly(4) if hdr[1] & 0x80 else None
            if n > 1024:
                break
            data = await reader.readexactly(n) if n else b""
            if mask:
                data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
            if opcode == 0x8:      # close
                break
            if opcode == 0x9:      # ping -> pong
                writer.write(ws_frame(data, 0xA))
                await writer.drain()
    except (OSError, EOFError):
        pass
    finally:
        clients.discard(writer)
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def broadcast_task():
    n = 0
    while True:
        await asyncio.sleep_ms(100)          # 10 Hz
        if not clients:
            continue
        ax, ay, az, gx, gy, gz, tp = mpu_read()
        sp = leer_spo2()
        sp_txt = "null" if sp is None else "%.1f" % sp
        msg = ('{"pulse":{"bpm":%d,"rr":%d,"flt":%d},'
               '"mpu":{"ax":%.2f,"ay":%.2f,"az":%.2f,"gx":%.1f,"tp":%.1f},'
               '"gps":{"vld":%s,"lat":%.6f,"lng":%.6f,"sat":%d,"alt":%.1f,'
               '"tim":"%s"},"spo2":%s,"risk":%s}') % (
            pulse.bpm, pulse.rr_last, pulse.filtered,
            ax, ay, az, gx, tp,
            "true" if gps["valid"] else "false",
            gps["lat"], gps["lng"], gps["sat"], gps["alt"], gps["tim"],
            sp_txt, riesgo.json())
        frame = ws_frame(msg.encode())
        for w in list(clients):
            try:
                w.write(frame)
                await asyncio.wait_for_ms(w.drain(), 500)
            except Exception:                # cliente caído o muy lento
                clients.discard(w)
                try:
                    w.close()
                except Exception:
                    pass
        n += 1
        if n % 100 == 0:
            gc.collect()


# ═══════════════════════════════════════════════════════════
#  Servidor HTTP (puerto 80) — sirve index.html desde flash
# ═══════════════════════════════════════════════════════════
async def http_handler(reader, writer):
    try:
        req = await reader.readline()
        while True:
            h = await reader.readline()
            if not h or h == b"\r\n":
                break
        if req.startswith(b"GET / ") or req.startswith(b"GET /index"):
            writer.write(b"HTTP/1.0 200 OK\r\n"
                         b"Content-Type: text/html; charset=utf-8\r\n"
                         b"Connection: close\r\n\r\n")
            with open("index.html", "rb") as f:
                while True:
                    chunk = f.read(512)
                    if not chunk:
                        break
                    writer.write(chunk)
                    await writer.drain()
        else:
            writer.write(b"HTTP/1.0 404 Not Found\r\n\r\n")
        await writer.drain()
    except OSError:
        pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════
async def main():
    wifi_ap()
    try:
        mpu_init()
    except OSError:
        print("[MPU] No responde en I2C 0x68, revisa cableado")

    await asyncio.start_server(http_handler, "0.0.0.0", 80)
    await asyncio.start_server(ws_handler, "0.0.0.0", 81)

    asyncio.create_task(pulse_task())
    asyncio.create_task(gps_task())
    asyncio.create_task(riesgo_task())
    asyncio.create_task(broadcast_task())

    while True:
        await asyncio.sleep(3600)


try:
    asyncio.run(main())
finally:
    asyncio.new_event_loop()
