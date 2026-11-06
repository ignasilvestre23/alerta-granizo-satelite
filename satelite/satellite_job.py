"""
Alerta Granizo · proceso de satélite (GOES-East, NOAA).

Cada 10 minutos:
  1. Lee los rayos (GLM) de los últimos 20 min y los agrupa en celdas.
  2. Lee la temperatura de las nubes (ABI banda 13) y busca topes fríos.
  3. Compara con el barrido de ~30 min antes para ver qué tormenta crece.
  4. Publica todo en Supabase (tabla sat_snapshots) para que la app lo lea.

Los datos de NOAA son públicos y gratuitos (bucket noaa-goes19 en AWS).
"""
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np

# Región: todo lo que ve GOES-East (América y Atlántico). Para achicar y que corra
# más rápido, bajá estos valores (por ej. Sudamérica: -56, 15, -85, -30).
LAT_MIN, LAT_MAX = -58.0, 58.0
LON_MIN, LON_MAX = -140.0, -20.0
STRIDE = int(os.environ.get("SAT_STRIDE", "2"))   # 2 = píxeles de 4 km (más rápido)

BUCKETS = [("noaa-goes19", "G19"), ("noaa-goes16", "G16")]
REQ = 6378137.0
RPOL = 6356752.31414
E2 = 1.0 - (RPOL ** 2) / (REQ ** 2)

COLD_BT = 225.0     # K: tope frío = nube muy alta (≈ -48 °C)
MIN_PIXELS = 5      # tamaño mínimo de una celda fría (con STRIDE=2)
BIN_DEG = 0.15      # ~16 km: tamaño de la celda de rayos


# ───────────────────── utilidades de tiempo ─────────────────────
def parse_start(key):
    """OR_..._s20262070914200_e... -> datetime UTC (año, día juliano, hhmmss)."""
    m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})\d", key)
    if not m:
        return None
    y, doy, hh, mm, ss = (int(g) for g in m.groups())
    return datetime(y, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=doy - 1, hours=hh, minutes=mm, seconds=ss)


def hour_prefixes(product, t0, t1):
    out, t = [], t0.replace(minute=0, second=0, microsecond=0)
    while t <= t1:
        out.append(f"{product}/{t.year}/{t.timetuple().tm_yday:03d}/{t.hour:02d}/")
        t += timedelta(hours=1)
    return out


def list_keys(fs, bucket, product, t0, t1, contains=None):
    keys = []
    for p in hour_prefixes(product, t0, t1):
        try:
            for k in fs.ls(f"{bucket}/{p}", detail=False):
                if contains and contains not in k:
                    continue
                st = parse_start(k)
                if st and t0 <= st <= t1:
                    keys.append((st, k))
        except FileNotFoundError:
            pass
    keys.sort()
    return keys


# ───────────────────── navegación GOES (PUG) ─────────────────────
def nav_inverse(x, y, h, lon0_deg):
    """Ángulos de escaneo (rad) -> latitud, longitud (grados)."""
    H = h + REQ
    a = np.sin(x) ** 2 + np.cos(x) ** 2 * (np.cos(y) ** 2 + (REQ ** 2 / RPOL ** 2) * np.sin(y) ** 2)
    b = -2.0 * H * np.cos(x) * np.cos(y)
    c = H ** 2 - REQ ** 2
    disc = b ** 2 - 4.0 * a * c
    with np.errstate(invalid="ignore"):
        rs = (-b - np.sqrt(disc)) / (2.0 * a)
        sx = rs * np.cos(x) * np.cos(y)
        sy = -rs * np.sin(x)
        sz = rs * np.cos(x) * np.sin(y)
        lat = np.degrees(np.arctan((REQ ** 2 / RPOL ** 2) * sz / np.sqrt((H - sx) ** 2 + sy ** 2)))
        lon = np.degrees(math.radians(lon0_deg) - np.arctan(sy / (H - sx)))
    return lat, lon


def nav_forward(lat_deg, lon_deg, h, lon0_deg):
    """Latitud, longitud (grados) -> ángulos de escaneo (rad)."""
    H = h + REQ
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    phi_c = np.arctan((RPOL ** 2 / REQ ** 2) * np.tan(lat))
    rc = RPOL / np.sqrt(1.0 - E2 * np.cos(phi_c) ** 2)
    sx = H - rc * np.cos(phi_c) * np.cos(lon - math.radians(lon0_deg))
    sy = -rc * np.cos(phi_c) * np.sin(lon - math.radians(lon0_deg))
    sz = rc * np.sin(phi_c)
    y = np.arctan(sz / sx)
    x = np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2))
    return x, y


def region_slices(xs, ys, h, lon0_deg):
    """Índices (ix0, ix1, iy0, iy1) que cubren la región en la grilla del satélite."""
    la = np.linspace(LAT_MIN, LAT_MAX, 25)
    lo = np.linspace(LON_MIN, LON_MAX, 25)
    gla, glo = np.meshgrid(la, lo)
    x, y = nav_forward(gla, glo, h, lon0_deg)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    ix0, ix1 = int(np.abs(xs - x.min()).argmin()), int(np.abs(xs - x.max()).argmin())
    iy0, iy1 = int(np.abs(ys - y.max()).argmin()), int(np.abs(ys - y.min()).argmin())
    ix0, ix1 = sorted((ix0, ix1))
    iy0, iy1 = sorted((iy0, iy1))
    return max(ix0 - 2, 0), min(ix1 + 3, len(xs)), max(iy0 - 2, 0), min(iy1 + 3, len(ys))


# ───────────────────── distancias ─────────────────────
def km_between(lat1, lon1, lat2, lon2):
    p = math.pi / 180.0
    dlat, dlon = (lat2 - lat1) * p, (lon2 - lon1) * p
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


# ───────────────────── rayos ─────────────────────
def cluster_lightning(w1, w0):
    """
    w1 / w0: (lat[], lon[]) de los últimos 10 min y de los 10 min previos.
    Devuelve celdas: lat, lon, n (rayos en 20 km, últimos 10 min), p (ídem, 10 min
    previos), vx, vy (km/h hacia el este y hacia el norte, o None).
    """
    lat1, lon1 = np.asarray(w1[0], dtype="float64"), np.asarray(w1[1], dtype="float64")
    lat0, lon0 = np.asarray(w0[0], dtype="float64"), np.asarray(w0[1], dtype="float64")
    if len(lat1) == 0:
        return []
    kx = 111.32

    # 1) celdas candidatas: casilleros de ~16 km con al menos 2 rayos
    bins = {}
    for la, lo in zip(lat1, lon1):
        k = (int(math.floor(la / BIN_DEG)), int(math.floor(lo / BIN_DEG)))
        b = bins.setdefault(k, [0.0, 0.0, 0])
        b[0] += la
        b[1] += lo
        b[2] += 1

    out = []
    for sla, slo, cnt in bins.values():
        if cnt < 2:
            continue
        clat, clon = sla / cnt, slo / cnt
        cosl = math.cos(math.radians(clat))
        d1 = np.hypot((lat1 - clat) * 111.0, (lon1 - clon) * kx * cosl)
        m1 = d1 <= 20.0
        n = int(m1.sum())
        if n < 3:
            continue
        p, vx, vy = 0, None, None
        if len(lat0):
            d0 = np.hypot((lat0 - clat) * 111.0, (lon0 - clon) * kx * cosl)
            p = int((d0 <= 20.0).sum())
            w = d0 <= 30.0
            if n >= 8 and int(w.sum()) >= 8:
                # movimiento = cuánto se corrió el centro de los rayos en 10 min
                dy = (lat1[d1 <= 30.0].mean() - lat0[w].mean()) * 111.0
                dx = (lon1[d1 <= 30.0].mean() - lon0[w].mean()) * kx * cosl
                sp = math.hypot(dx * 6.0, dy * 6.0)
                if 5.0 <= sp <= 120.0:
                    vx, vy = round(dx * 6.0, 1), round(dy * 6.0, 1)
        out.append({"lat": round(float(clat), 3), "lon": round(float(clon), 3),
                    "n": n, "p": p, "vx": vx, "vy": vy})
    out.sort(key=lambda c: -c["n"])
    return out[:1500]


def read_glm(fs, key):
    import xarray as xr
    with fs.open(key, "rb") as f:
        ds = xr.open_dataset(f, engine="h5netcdf")
        if "flash_lat" not in ds:
            return np.array([]), np.array([])
        lat = ds["flash_lat"].values.astype("float64")
        lon = ds["flash_lon"].values.astype("float64")
        if "flash_quality_flag" in ds:
            ok = ds["flash_quality_flag"].values == 0
            lat, lon = lat[ok], lon[ok]
        box = (lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= LON_MAX)
        return lat[box], lon[box]


def lightning_cells(fs, bucket, now):
    t0 = now - timedelta(minutes=20)
    files = list_keys(fs, bucket, "GLM-L2-LCFA", t0, now)
    if not files:
        return None
    cut = now - timedelta(minutes=10)
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(lambda sk: (sk[0], read_glm(fs, sk[1])), files))
    w1 = [r for st, r in res if st >= cut]
    w0 = [r for st, r in res if st < cut]

    def cat(parts):
        if not parts:
            return np.array([]), np.array([])
        return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))

    return cluster_lightning(cat(w1), cat(w0)), files[-1][0]


# ───────────────────── nubes frías (ABI banda 13) ─────────────────────
def est_top_km(bt):
    """Altura estimada del tope a partir de su temperatura (atmósfera estándar)."""
    if bt >= 216.65:
        return max(0.0, (288.15 - bt) / 6.5)
    return min(18.0, 11.0 + (216.65 - bt) * 0.2)


def find_cold_cells(bt, lat, lon, prev_bt=None, ht_km=None):
    from scipy import ndimage
    inside = (lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= LON_MAX)
    mask = np.isfinite(bt) & (bt <= COLD_BT) & inside
    lab, n = ndimage.label(mask)
    cells = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        ys, xs = np.nonzero(lab[sl] == i)
        ys, xs = ys + sl[0].start, xs + sl[1].start
        if len(ys) < MIN_PIXELS:
            continue
        vals = bt[ys, xs]
        j = int(vals.argmin())
        bt_min = float(vals[j])
        cy, cx = int(ys[j]), int(xs[j])
        la, lo = float(lat[ys, xs].mean()), float(lon[ys, xs].mean())
        if not (math.isfinite(la) and math.isfinite(lo)):
            continue
        top, est = None, True
        if ht_km is not None:
            h = ht_km[ys, xs]
            h = h[np.isfinite(h)]
            if h.size >= 3:
                top, est = float(np.percentile(h, 95)), False
        if top is None:
            top = est_top_km(bt_min)
        g = None
        if prev_bt is not None:
            y0, y1 = max(cy - 8, 0), min(cy + 9, bt.shape[0])
            x0, x1 = max(cx - 8, 0), min(cx + 9, bt.shape[1])
            win = prev_bt[y0:y1, x0:x1]
            if np.isfinite(win).any():
                g = round(float(np.nanmin(win)) - bt_min, 1)   # >0: hoy está más fría
        cells.append({"lat": round(la, 3), "lon": round(lo, 3), "bt": round(bt_min, 1),
                      "px": int(len(ys)), "top": round(top, 1), "te": est, "g": g})
    cells.sort(key=lambda c: c["bt"])
    return cells[:600]


def read_cmi(fs, key, with_latlon=True):
    import xarray as xr
    with fs.open(key, "rb") as f:
        ds = xr.open_dataset(f, engine="h5netcdf")
        proj = ds["goes_imager_projection"].attrs
        h = float(proj["perspective_point_height"])
        lon0 = float(proj["longitude_of_projection_origin"])
        xs, ys = ds["x"].values.astype("float64"), ds["y"].values.astype("float64")
        ix0, ix1, iy0, iy1 = region_slices(xs, ys, h, lon0)
        bt = ds["CMI"].isel(y=slice(iy0, iy1, STRIDE), x=slice(ix0, ix1, STRIDE)).values.astype("float32")
        lat = lon = None
        if with_latlon:
            gx, gy = np.meshgrid(xs[ix0:ix1:STRIDE], ys[iy0:iy1:STRIDE])
            lat, lon = nav_inverse(gx, gy, h, lon0)
        return bt, lat, lon, (ix0, ix1, iy0, iy1)


def read_acha(fs, key, sl):
    import xarray as xr
    ix0, ix1, iy0, iy1 = sl
    with fs.open(key, "rb") as f:
        ds = xr.open_dataset(f, engine="h5netcdf")
        ht = ds["HT"].isel(y=slice(iy0, iy1, STRIDE), x=slice(ix0, ix1, STRIDE)).values.astype("float32")
        return ht / 1000.0     # m -> km


def cold_cells(fs, bucket, sat, now):
    cm = list_keys(fs, bucket, "ABI-L2-CMIPF", now - timedelta(minutes=60), now, contains=f"C13_{sat}")
    if not cm:
        return None
    t_now, k_now = cm[-1]
    prev = [c for c in cm if abs((t_now - c[0]).total_seconds() - 1800) <= 420]
    bt, lat, lon, sl = read_cmi(fs, k_now)
    prev_bt = read_cmi(fs, prev[-1][1], with_latlon=False)[0] if prev else None
    ht = None
    try:
        ac = list_keys(fs, bucket, "ABI-L2-ACHAF", t_now - timedelta(minutes=12), t_now + timedelta(minutes=12))
        if ac:
            ht = read_acha(fs, min(ac, key=lambda a: abs((a[0] - t_now).total_seconds()))[1], sl)
            if ht.shape != bt.shape:
                ht = None
    except Exception as e:                   # noqa: BLE001
        print("ACHA no disponible:", e)
    return find_cold_cells(bt, lat, lon, prev_bt, ht), t_now


# ───────────────────── publicación ─────────────────────
def publish(payload, observed):
    import requests
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_KEY"]
    headers = {"apikey": key, "Content-Type": "application/json", "Prefer": "return=minimal"}
    if key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {key}"
    body = {"observed_at": observed.isoformat(), "payload": payload}
    r = requests.post(f"{url}/rest/v1/sat_snapshots", headers=headers, data=json.dumps(body), timeout=30)
    print("Supabase:", r.status_code, r.text[:200])
    r.raise_for_status()


def main():
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    now = datetime.now(timezone.utc)
    for bucket, sat in BUCKETS:
        payload = {"sat": sat, "t": now.isoformat(timespec="seconds"),
                   "lightning": [], "cells": [], "glm_t": None, "abi_t": None}
        got = False
        try:
            res = lightning_cells(fs, bucket, now)
            if res:
                payload["lightning"], t = res
                payload["glm_t"] = t.isoformat(timespec="seconds")
                got = True
                print("Rayos:", len(payload["lightning"]), "celdas")
        except Exception as e:               # noqa: BLE001
            print("Rayos falló:", repr(e))
        try:
            res = cold_cells(fs, bucket, sat, now)
            if res:
                payload["cells"], t = res
                payload["abi_t"] = t.isoformat(timespec="seconds")
                got = True
                print("Nubes frías:", len(payload["cells"]), "celdas")
        except Exception as e:               # noqa: BLE001
            print("Nubes falló:", repr(e))
        if got:
            publish(payload, now)
            return 0
    print("No se pudo leer ningún dato del satélite.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
