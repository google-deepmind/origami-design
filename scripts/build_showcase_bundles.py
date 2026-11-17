#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds static 3D origami showcase bundles for the GitHub Pages web studio.

Zero external runtime dependencies (pure Python 3 standard library) so
contributors can package any `bp-pipeline` output directory into
`docs/models/<slug>/` with a single command:

  python3 scripts/build_showcase_bundles.py \
      --input outputs/my_model \
      --slug my-model \
      --title "My Origami Model"

Each generated `docs/models/<slug>/` directory contains:
  - `stick_figure.json`: Canonical 3D StickFigure tree definition.
  - `packing_cp.svg`: Box-pleated circle/river packing layout SVG.
  - `solving_cp.svg`: Globally flat-foldable base crease pattern SVG.
  - `shaped_cp.svg`: Final 3D shaped crease pattern SVG.
  - `shaped.fold`: Standard FOLD v1.2 file (2D CP + upright 3D folded frame).
  - `bundle.json`: Pre-indexed web topology, two-stage dihedral fold angles,
    upright SO(3) quaternion, and semantic flap region labels for instant
    (< 10ms) 60fps browser rendering.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import shutil
from typing import Any
import xml.etree.ElementTree as ET


# ------------------------------------------------------------------------------
# Pure-stdlib 3D linear algebra & Jacobi 3x3 SVD / SO(3) Procrustes alignment
# ------------------------------------------------------------------------------

Vec3 = tuple[float, float, float]
Mat3 = list[list[float]]
Mat4 = list[list[float]]


def _mat3_identity() -> Mat3:
  return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _mat3_transpose(a: Mat3) -> Mat3:
  return [[a[j][i] for j in range(3)] for i in range(3)]


def _mat3_mul(a: Mat3, b: Mat3) -> Mat3:
  return [
      [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
      for i in range(3)
  ]


def _mat3_vec(a: Mat3, v: Vec3) -> Vec3:
  return (
      a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
      a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
      a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
  )


def _mat3_det(a: Mat3) -> float:
  return (
      a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
      - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
      + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
  )


def _jacobi_eigen_sym3(m_in: Mat3) -> tuple[list[float], Mat3]:
  """Exact symmetric 3x3 eigendecomposition `M = V @ diag(d) @ V^T`."""
  a = [row[:] for row in m_in]
  v = _mat3_identity()
  for _ in range(40):
    p, q = 0, 1
    max_val = abs(a[0][1])
    for i, j in ((0, 2), (1, 2)):
      if abs(a[i][j]) > max_val:
        max_val = abs(a[i][j])
        p, q = i, j
    if max_val < 1e-14:
      break
    diff = a[q][q] - a[p][p]
    if abs(a[p][q]) < 1e-30 * abs(diff):
      t = a[p][q] / diff
    else:
      phi = diff / (2.0 * a[p][q])
      t = 1.0 / (abs(phi) + math.sqrt(phi * phi + 1.0))
      if phi < 0.0:
        t = -t
    c = 1.0 / math.sqrt(1.0 + t * t)
    s = t * c
    tau = s / (1.0 + c)
    apq = a[p][q]
    a[p][q] = 0.0
    a[q][p] = 0.0
    a[p][p] -= t * apq
    a[q][q] += t * apq
    for r in range(3):
      if r != p and r != q:
        arp = a[r][p]
        arq = a[r][q]
        a[r][p] = a[p][r] = arp - s * (arq + tau * arp)
        a[r][q] = a[q][r] = arq + s * (arp - tau * arq)
    for r in range(3):
      vrp = v[r][p]
      vrq = v[r][q]
      v[r][p] = vrp - s * (vrq + tau * vrp)
      v[r][q] = vrq + s * (vrp - tau * vrq)
  vals = [a[0][0], a[1][1], a[2][2]]
  order = sorted(range(3), key=lambda idx: -vals[idx])
  vals_sorted = [vals[idx] for idx in order]
  v_sorted = [[v[r][order[c]] for c in range(3)] for r in range(3)]
  if _mat3_det(v_sorted) < 0.0:
    for r in range(3):
      v_sorted[r][2] = -v_sorted[r][2]
  return vals_sorted, v_sorted


def _svd3(h: Mat3) -> tuple[Mat3, list[float], Mat3]:
  """3x3 SVD `H = U @ diag(S) @ V^T` with `U, V in SO(3)` up to sign."""
  ht_h = _mat3_mul(_mat3_transpose(h), h)
  eigvals, v = _jacobi_eigen_sym3(ht_h)
  s = [math.sqrt(max(0.0, ev)) for ev in eigvals]
  hv = _mat3_mul(h, v)
  u_cols: list[list[float]] = []
  for j in range(3):
    col = [hv[0][j], hv[1][j], hv[2][j]]
    norm = math.sqrt(col[0] ** 2 + col[1] ** 2 + col[2] ** 2)
    if norm > 1e-10:
      u_cols.append([col[0] / norm, col[1] / norm, col[2] / norm])
    else:
      u_cols.append([0.0, 0.0, 0.0])

  # Complete orthonormal basis if rank < 3
  for j in range(3):
    if sum(x * x for x in u_cols[j]) < 0.5:
      for cand in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]):
        vec = cand[:]
        for k in range(j):
          if sum(x * x for x in u_cols[k]) > 0.5:
            dot = sum(vec[d] * u_cols[k][d] for d in range(3))
            vec = [vec[d] - dot * u_cols[k][d] for d in range(3)]
        nrm = math.sqrt(sum(x * x for x in vec))
        if nrm > 1e-6:
          u_cols[j] = [vec[d] / nrm for d in range(3)]
          break

  u = [[u_cols[c][r] for c in range(3)] for r in range(3)]
  if _mat3_det(u) < 0.0:
    for r in range(3):
      u[r][2] = -u[r][2]
    s[2] = -s[2]
  return u, s, _mat3_transpose(v)


def _fit_proper_similarity(
    source: list[Vec3],
    target: list[Vec3],
    weights: list[float] | None = None,
) -> tuple[Mat3, float, Vec3]:
  """Weighted proper (`SO(3)`, `det = +1`) Umeyama similarity alignment."""
  n = len(source)
  if n == 0:
    return _mat3_identity(), 1.0, (0.0, 0.0, 0.0)
  w = [1.0] * n if weights is None else list(weights)
  w_sum = sum(w) or 1.0
  w = [wi / w_sum for wi in w]

  sx = sum(wi * p[0] for wi, p in zip(w, source))
  sy = sum(wi * p[1] for wi, p in zip(w, source))
  sz = sum(wi * p[2] for wi, p in zip(w, source))
  tx = sum(wi * p[0] for wi, p in zip(w, target))
  ty = sum(wi * p[1] for wi, p in zip(w, target))
  tz = sum(wi * p[2] for wi, p in zip(w, target))

  sc = [(p[0] - sx, p[1] - sy, p[2] - sz) for p in source]
  tc = [(p[0] - tx, p[1] - ty, p[2] - tz) for p in target]

  cov = [[0.0, 0.0, 0.0] for _ in range(3)]
  src_var = 0.0
  for wi, sp, tp in zip(w, sc, tc):
    src_var += wi * (sp[0] ** 2 + sp[1] ** 2 + sp[2] ** 2)
    for r in range(3):
      for c in range(3):
        cov[r][c] += wi * tp[r] * sp[c]

  u, singular, vt = _svd3(cov)
  rot = _mat3_mul(u, vt)
  det_sign = 1.0 if _mat3_det(rot) >= 0.0 else -1.0
  if det_sign < 0.0:
    d_mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
    rot = _mat3_mul(_mat3_mul(u, d_mat), vt)
  trace_s = singular[0] + singular[1] + det_sign * singular[2]
  scale = trace_s / src_var if src_var > 1e-12 else 1.0
  rs = _mat3_vec(rot, (sx, sy, sz))
  trans = (tx - scale * rs[0], ty - scale * rs[1], tz - scale * rs[2])
  return rot, scale, trans


def _fit_robust_upright_rotation(
    source: list[Vec3], target: list[Vec3]
) -> Mat3:
  """Cauchy-reweighted proper SO(3) rotation aligning folded -> figure frame."""
  if len(source) < 3:
    return _mat3_identity()
  xs = [p[0] for p in target]
  ys = [p[1] for p in target]
  zs = [p[2] for p in target]
  fig_size = max(
      math.sqrt(
          (max(xs) - min(xs)) ** 2
          + (max(ys) - min(ys)) ** 2
          + (max(zs) - min(zs)) ** 2
      ),
      1e-6,
  )
  weights = [1.0] * len(source)
  rot, scale, trans = _fit_proper_similarity(source, target, weights)
  for _ in range(8):
    new_w = []
    for sp, tp in zip(source, target):
      rp = _mat3_vec(rot, sp)
      dx = scale * rp[0] + trans[0] - tp[0]
      dy = scale * rp[1] + trans[1] - tp[1]
      dz = scale * rp[2] + trans[2] - tp[2]
      res = math.sqrt(dx * dx + dy * dy + dz * dz) / fig_size
      new_w.append(1.0 / (1.0 + (res / 0.25) ** 2))
    weights = new_w
    rot, scale, trans = _fit_proper_similarity(source, target, weights)
  return rot


def _mat3_to_quaternion(m: Mat3) -> list[float]:
  """Converts a proper 3x3 rotation matrix to unit quaternion `[x, y, z, w]`."""
  tr = m[0][0] + m[1][1] + m[2][2]
  if tr > 0.0:
    s = math.sqrt(tr + 1.0) * 2.0
    qw = 0.25 * s
    qx = (m[2][1] - m[1][2]) / s
    qy = (m[0][2] - m[2][0]) / s
    qz = (m[1][0] - m[0][1]) / s
  elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
    s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
    qw = (m[2][1] - m[1][2]) / s
    qx = 0.25 * s
    qy = (m[0][1] + m[1][0]) / s
    qz = (m[0][2] + m[2][0]) / s
  elif m[1][1] > m[2][2]:
    s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
    qw = (m[0][2] - m[2][0]) / s
    qx = (m[0][1] + m[1][0]) / s
    qy = 0.25 * s
    qz = (m[1][2] + m[2][1]) / s
  else:
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
    qw = (m[1][0] - m[0][1]) / s
    qx = (m[0][2] + m[2][0]) / s
    qy = (m[1][2] + m[2][1]) / s
    qz = 0.25 * s
  nrm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
  if qw < 0.0:
    nrm = -nrm
  return [
      round(qx / nrm, 6),
      round(qy / nrm, 6),
      round(qz / nrm, 6),
      round(qw / nrm, 6),
  ]


# ------------------------------------------------------------------------------
# Fast SVG crease pattern parser, O(E) planar topology & 3D BFS folder
# ------------------------------------------------------------------------------

_PRECEDENCE = {'M': 3, 'V': 3, 'F': 2, 'B': 1, 'U': 0}


def _parse_crease_color(stroke: str) -> tuple[str, float]:
  c = stroke.strip().lower().replace(' ', '')
  if c in ('red', '#ff0000', '#f00') or c.startswith('rgb(255,0,0)'):
    return 'M', -180.0
  if c in ('blue', '#0000ff', '#00f') or c.startswith('rgb(0,0,255)'):
    return 'V', 180.0
  if c in ('black', '#000000', '#000') or c.startswith('rgb(0,0,0)'):
    return 'B', 0.0
  if c in (
      'yellow',
      '#ffff00',
      '#ff0',
      'green',
      '#00ff00',
      '#0f0',
      '#808080',
      'gray',
      'grey',
  ) or c.startswith('rgb(255,255,0)'):
    return 'F', 0.0
  return 'U', 0.0


def parse_svg_segments(
    svg_text: str, tol: float = 5e-5
) -> tuple[list[tuple[float, float]], list[tuple[int, int, str, float]]]:
  """Parses SVG crease lines into normalized `[-0.5, 0.5]^2` vertices & edges."""
  root = ET.fromstring(svg_text)
  vb = root.attrib.get('viewBox', '0 0 1000 1000').strip().split()
  if len(vb) == 4:
    x_min, y_min, width, height = (float(v) for v in vb)
  else:
    x_min, y_min, width, height = 0.0, 0.0, 1000.0, 1000.0
  scale = max(width if width > 0 else 1000.0, height if height > 0 else 1000.0)

  # Propagate <g> stroke/opacity attributes
  for elem in root.iter():
    if elem.tag.split('}')[-1] == 'g':
      if elem.attrib.get('class') == 'bp-grid-lines':
        continue
      for child in elem:
        for attr in ('stroke', 'opacity', 'stroke-opacity'):
          if attr in elem.attrib and attr not in child.attrib:
            child.set(attr, elem.attrib[attr])

  vertices: list[tuple[float, float]] = []
  cell_size = tol * 2.0
  grid: dict[tuple[int, int], list[int]] = {}

  def get_vid(x: float, y: float) -> int:
    gx = int(math.floor(x / cell_size))
    gy = int(math.floor(y / cell_size))
    best_i = -1
    best_d2 = tol * tol
    for dx in (-1, 0, 1):
      for dy in (-1, 0, 1):
        for idx in grid.get((gx + dx, gy + dy), ()):
          vx, vy = vertices[idx]
          d2 = (vx - x) ** 2 + (vy - y) ** 2
          if d2 < best_d2:
            best_d2 = d2
            best_i = idx
    if best_i >= 0:
      return best_i
    new_idx = len(vertices)
    vertices.append((x, y))
    grid.setdefault((gx, gy), []).append(new_idx)
    return new_idx

  raw_map: dict[tuple[int, int], tuple[int, int, str, float]] = {}

  def add_raw(rx1: float, ry1: float, rx2: float, ry2: float, stroke: str, op: float) -> None:
    x1 = (rx1 - x_min) / scale - 0.5
    y1 = (ry1 - y_min) / scale - 0.5
    x2 = (rx2 - x_min) / scale - 0.5
    y2 = (ry2 - y_min) / scale - 0.5
    if math.hypot(x2 - x1, y2 - y1) < 1e-5:
      return
    assign, base_angle = _parse_crease_color(stroke)
    if assign == 'U':
      return
    v1 = get_vid(x1, y1)
    v2 = get_vid(x2, y2)
    if v1 == v2:
      return
    key = (v1, v2) if v1 < v2 else (v2, v1)
    prev = raw_map.get(key)
    if prev is None or _PRECEDENCE[assign] > _PRECEDENCE[prev[2]]:
      raw_map[key] = (key[0], key[1], assign, base_angle * op)

  elems = list(root.iter())
  for target_assign in ('B', 'M', 'V', 'F'):
    for el in elems:
      tag = el.tag.split('}')[-1]
      if tag == 'line':
        stroke = el.attrib.get('stroke', '')
        assign, _ = _parse_crease_color(stroke)
        if assign != target_assign:
          continue
        op = float(
            el.attrib.get('stroke-opacity') or el.attrib.get('opacity') or '1'
        )
        add_raw(
            float(el.attrib.get('x1', 0)),
            float(el.attrib.get('y1', 0)),
            float(el.attrib.get('x2', 0)),
            float(el.attrib.get('y2', 0)),
            stroke,
            op,
        )
      elif tag == 'rect' and target_assign == 'B':
        stroke = el.attrib.get('stroke', 'black')
        op = float(
            el.attrib.get('stroke-opacity') or el.attrib.get('opacity') or '1'
        )
        rx = float(el.attrib.get('x', 0))
        ry = float(el.attrib.get('y', 0))
        rw = float(el.attrib.get('width', 0))
        rh = float(el.attrib.get('height', 0))
        if rw > 0 and rh > 0:
          add_raw(rx, ry, rx + rw, ry, stroke, op)
          add_raw(rx + rw, ry, rx + rw, ry + rh, stroke, op)
          add_raw(rx + rw, ry + rh, rx, ry + rh, stroke, op)
          add_raw(rx, ry + rh, rx, ry, stroke, op)

  raw_edges = list(raw_map.values())

  # Insert vertices at any strict interior segment-segment X-crossings so the
  # subsequent collinear pass splits both crossing creases cleanly (matching
  # folder_3d.find_intersections).
  edge_bucket_size = 0.05
  e_buckets: dict[tuple[int, int], list[int]] = {}
  for e_idx, (v1, v2, _, _) in enumerate(raw_edges):
    ax, ay = vertices[v1]
    bx, by = vertices[v2]
    gx0 = int(math.floor(min(ax, bx) / edge_bucket_size))
    gx1 = int(math.floor(max(ax, bx) / edge_bucket_size))
    gy0 = int(math.floor(min(ay, by) / edge_bucket_size))
    gy1 = int(math.floor(max(ay, by) / edge_bucket_size))
    for gx in range(gx0, gx1 + 1):
      for gy in range(gy0, gy1 + 1):
        e_buckets.setdefault((gx, gy), []).append(e_idx)

  checked_pairs: set[tuple[int, int]] = set()
  for cell_edges in e_buckets.values():
    m_ce = len(cell_edges)
    for i in range(m_ce):
      e1 = cell_edges[i]
      v1, v2, _, _ = raw_edges[e1]
      ax, ay = vertices[v1]
      bx, by = vertices[v2]
      rx, ry = bx - ax, by - ay
      x1_min, x1_max = min(ax, bx), max(ax, bx)
      y1_min, y1_max = min(ay, by), max(ay, by)
      for j in range(i + 1, m_ce):
        e2 = cell_edges[j]
        pk = (e1, e2) if e1 < e2 else (e2, e1)
        if pk in checked_pairs:
          continue
        checked_pairs.add(pk)
        u1, u2, _, _ = raw_edges[e2]
        if v1 in (u1, u2) or v2 in (u1, u2):
          continue
        cx, cy = vertices[u1]
        dx, dy = vertices[u2]
        if (
            max(cx, dx) < x1_min - tol
            or min(cx, dx) > x1_max + tol
            or max(cy, dy) < y1_min - tol
            or min(cy, dy) > y1_max + tol
        ):
          continue
        sx, sy = dx - cx, dy - cy
        denom = rx * sy - ry * sx
        if abs(denom) < 1e-10:
          continue
        qpx, qpy = cx - ax, cy - ay
        t_param = (qpx * sy - qpy * sx) / denom
        u_param = (qpx * ry - qpy * rx) / denom
        if 1e-4 < t_param < 1.0 - 1e-4 and 1e-4 < u_param < 1.0 - 1e-4:
          get_vid(ax + t_param * rx, ay + t_param * ry)

  # Single-pass O(E) spatial-bucket collinear splitting (100x faster than O(V*E^2))
  bucket_size = 0.025
  v_buckets: dict[tuple[int, int], list[int]] = {}
  for idx, (vx, vy) in enumerate(vertices):
    bx = int(math.floor(vx / bucket_size))
    by = int(math.floor(vy / bucket_size))
    v_buckets.setdefault((bx, by), []).append(idx)

  split_map: dict[tuple[int, int], tuple[int, int, str, float]] = {}
  for v1, v2, assign, angle in raw_edges:
    ax, ay = vertices[v1]
    bx, by = vertices[v2]
    abx = bx - ax
    aby = by - ay
    l2 = abx * abx + aby * aby
    seg_len = math.sqrt(l2)
    if seg_len < 1e-7:
      continue
    ix0 = int(math.floor((min(ax, bx) - tol) / bucket_size))
    ix1 = int(math.floor((max(ax, bx) + tol) / bucket_size))
    iy0 = int(math.floor((min(ay, by) - tol) / bucket_size))
    iy1 = int(math.floor((max(ay, by) + tol) / bucket_size))

    interior: list[tuple[float, int]] = []
    seen_cand: set[int] = {v1, v2}
    for gx in range(ix0, ix1 + 1):
      for gy in range(iy0, iy1 + 1):
        for cand in v_buckets.get((gx, gy), ()):
          if cand in seen_cand:
            continue
          seen_cand.add(cand)
          px, py = vertices[cand]
          avx = px - ax
          avy = py - ay
          dist = abs(abx * avy - aby * avx) / seg_len
          if dist < tol:
            dot = abx * avx + aby * avy
            limit = tol * seg_len
            if limit < dot < l2 - limit:
              interior.append((dot / l2, cand))

    chain = [v1] + [vid for _, vid in sorted(interior)] + [v2]
    for k in range(len(chain) - 1):
      u, w = chain[k], chain[k + 1]
      if u == w:
        continue
      ek = (u, w) if u < w else (w, u)
      prev = split_map.get(ek)
      if prev is None or _PRECEDENCE[assign] > _PRECEDENCE[prev[2]]:
        split_map[ek] = (ek[0], ek[1], assign, angle)

  return vertices, list(split_map.values())


def _triangulate_polygon_2d(
    poly: list[int], vertices: list[tuple[float, float]]
) -> list[int]:
  """Robust 2D ear-clipping triangulation returning local face indices."""
  m = len(poly)
  if m < 3:
    return []
  if m == 3:
    return [0, 1, 2]

  pts = [vertices[v] for v in poly]
  signed_area = 0.0
  for i in range(m):
    x1, y1 = pts[i]
    x2, y2 = pts[(i + 1) % m]
    signed_area += x1 * y2 - x2 * y1
  ccw = signed_area > 0.0

  idxs = list(range(m))
  tris: list[int] = []

  def cross2(ia: int, ib: int, ic: int) -> float:
    ax, ay = pts[ia]
    bx, by = pts[ib]
    cx, cy = pts[ic]
    val = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    return val if ccw else -val

  def point_in_tri(ip: int, ia: int, ib: int, ic: int) -> bool:
    c1 = cross2(ia, ib, ip)
    c2 = cross2(ib, ic, ip)
    c3 = cross2(ic, ia, ip)
    return c1 >= -1e-11 and c2 >= -1e-11 and c3 >= -1e-11

  guard = 0
  while len(idxs) > 3 and guard < m * m + 10:
    guard += 1
    ear_found = False
    k_len = len(idxs)
    for i in range(k_len):
      ia = idxs[(i - 1) % k_len]
      ib = idxs[i]
      ic = idxs[(i + 1) % k_len]
      if cross2(ia, ib, ic) <= 1e-11:
        continue
      if any(
          point_in_tri(idxs[j], ia, ib, ic)
          for j in range(k_len)
          if idxs[j] not in (ia, ib, ic)
      ):
        continue
      tris.extend([ia, ib, ic])
      idxs.pop(i)
      ear_found = True
      break
    if not ear_found:
      # Fallback fan for nearly collinear edges
      ia, ib, ic = idxs[0], idxs[1], idxs[2]
      tris.extend([ia, ib, ic])
      idxs.pop(1)

  if len(idxs) == 3:
    tris.extend([idxs[0], idxs[1], idxs[2]])
  return tris


def build_planar_model(
    svg_text: str, solving_svg_text: str | None = None
) -> dict[str, Any]:
  """Builds clean planar face topology, triangulations, and 2-stage fold angles."""
  vertices, split_edges = parse_svg_segments(svg_text, tol=5e-5)
  edge_lookup = {(e[0], e[1]): (e[2], e[3]) for e in split_edges}

  # Prune dangling degree-1 vertices before face extraction so hairline
  # overlaps never break interior face cycles.
  adj_set: dict[int, set[int]] = {i: set() for i in range(len(vertices))}
  for u, v, _, _ in split_edges:
    adj_set[u].add(v)
    adj_set[v].add(u)
  deg1_queue = collections.deque(
      [v for v, nbs in adj_set.items() if len(nbs) == 1]
  )
  while deg1_queue:
    leaf = deg1_queue.popleft()
    if len(adj_set[leaf]) != 1:
      continue
    nb = next(iter(adj_set[leaf]))
    adj_set[leaf].clear()
    adj_set[nb].discard(leaf)
    if len(adj_set[nb]) == 1:
      deg1_queue.append(nb)

  adj: list[list[int]] = [list(adj_set[v]) for v in range(len(vertices))]
  for v in range(len(vertices)):
    if not adj[v]:
      continue
    x0, y0 = vertices[v]
    adj[v].sort(key=lambda nb: math.atan2(vertices[nb][1] - y0, vertices[nb][0] - x0))

  next_map: dict[tuple[int, int], int] = {}
  for v, nbs in enumerate(adj):
    n = len(nbs)
    for i, u in enumerate(nbs):
      next_map[(u, v)] = nbs[(i - 1) % n]

  faces_raw: list[list[int]] = []
  visited_he: set[tuple[int, int]] = set()
  for start_uv in list(next_map):
    if start_uv in visited_he:
      continue
    u, v = start_uv
    face = [u, v]
    visited_he.add((u, v))
    w = next_map.get((u, v))
    steps = 0
    broken = False
    while w != start_uv[0]:
      if w is None or steps > 2000:
        broken = True
        break
      face.append(w)
      visited_he.add((v, w))
      u, v = v, w
      w = next_map.get((u, v))
      steps += 1
    if not broken and len(face) >= 3:
      visited_he.add((face[-1], face[0]))
      area2 = 0.0
      flen = len(face)
      for i in range(flen):
        p1 = vertices[face[i]]
        p2 = vertices[face[(i + 1) % flen]]
        area2 += p1[0] * p2[1] - p2[0] * p1[1]
      if area2 > 1e-9:
        faces_raw.append(face[::-1])

  if not faces_raw:
    raise ValueError('No closed faces found in crease pattern.')

  active = sorted({v for f in faces_raw for v in f})
  old_to_new = {old: new for new, old in enumerate(active)}
  clean_verts = [vertices[idx] for idx in active]
  clean_faces = [[old_to_new[idx] for idx in f] for f in faces_raw]

  edge_dict: dict[tuple[int, int], tuple[int, int]] = {}
  for f in clean_faces:
    flen = len(f)
    for i in range(flen):
      u, v = f[i], f[(i + 1) % flen]
      ek = (u, v) if u < v else (v, u)
      if ek not in edge_dict:
        ou, ov = active[ek[0]], active[ek[1]]
        oek = (ou, ov) if ou < ov else (ov, ou)
        edge_dict[ek] = oek

  edges: list[tuple[int, int]] = []
  assignments: list[str] = []
  shaping_angles: list[float] = []
  for (u, v), oek in edge_dict.items():
    edges.append((u, v))
    assign, ang = edge_lookup.get(oek, ('B', 0.0))
    assignments.append(assign)
    shaping_angles.append(ang)

  # Central start face (closest to paper origin (0, 0))
  best_face = 0
  best_d2 = float('inf')
  for f_idx, f in enumerate(clean_faces):
    fx = sum(clean_verts[v][0] for v in f) / len(f)
    fy = sum(clean_verts[v][1] for v in f) / len(f)
    d2 = fx * fx + fy * fy
    if d2 < best_d2:
      best_d2 = d2
      best_face = f_idx

  # Two-stage edge classification against solving_cp.svg
  edge_stage = [1] * len(edges)
  solving_angles = list(shaping_angles)
  if solving_svg_text:
    sol_verts, sol_edges = parse_svg_segments(solving_svg_text, tol=5e-5)
    sol_segs = []
    for su, sv, s_assign, _ in sol_edges:
      if s_assign in ('B', 'U'):
        continue
      p1 = sol_verts[su]
      p2 = sol_verts[sv]
      dx, dy = p2[0] - p1[0], p2[1] - p1[1]
      slen = math.hypot(dx, dy)
      if slen > 1e-5:
        sol_segs.append((p1[0], p1[1], p2[0], p2[1], dx / slen, dy / slen, slen * slen))

    for i, (u, v) in enumerate(edges):
      if assignments[i] == 'B':
        edge_stage[i] = 1
        solving_angles[i] = 0.0
        continue
      v1, v2 = clean_verts[u], clean_verts[v]
      dx, dy = v2[0] - v1[0], v2[1] - v1[1]
      elen = math.hypot(dx, dy)
      if elen < 1e-6:
        edge_stage[i] = 2
        solving_angles[i] = 0.0
        continue
      dir_x, dir_y = dx / elen, dy / elen
      mx, my = 0.5 * (v1[0] + v2[0]), 0.5 * (v1[1] + v2[1])
      matched = False
      for x1, y1, x2, y2, sdx, sdy, sl2 in sol_segs:
        if abs(dir_x * sdx + dir_y * sdy) < 0.88:
          continue
        t = ((mx - x1) * (x2 - x1) + (my - y1) * (y2 - y1)) / sl2
        if -0.02 <= t <= 1.02:
          dist = abs((mx - x1) * (y2 - y1) - (my - y1) * (x2 - x1)) / math.sqrt(sl2)
          if dist < 1.2e-3:
            matched = True
            break
      if matched:
        edge_stage[i] = 1
        solving_angles[i] = shaping_angles[i]
      else:
        edge_stage[i] = 2
        solving_angles[i] = 0.0

  triangulations = [_triangulate_polygon_2d(f, clean_verts) for f in clean_faces]

  return {
      'vertices': clean_verts,
      'faces': clean_faces,
      'edges': edges,
      'assignments': assignments,
      'shaping_angles': shaping_angles,
      'solving_angles': solving_angles,
      'edge_stage': edge_stage,
      'central_face': best_face,
      'triangulations': triangulations,
  }


def _rot_axis_mat4(
    px: float, py: float, pz: float, ax: float, ay: float, az: float, rad: float
) -> Mat4:
  """4x4 transform rotating around line through `(px, py, pz)` along `(ax, ay, az)`."""
  nrm = math.sqrt(ax * ax + ay * ay + az * az)
  if nrm < 1e-12 or abs(rad) < 1e-12:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
  ux, uy, uz = ax / nrm, ay / nrm, az / nrm
  # Match folder_3d.py / Three.js makeRotationAxis right-handed rotation:
  c = math.cos(rad)
  s = math.sin(rad)
  t = 1.0 - c
  r00 = t * ux * ux + c
  r01 = t * ux * uy - s * uz
  r02 = t * ux * uz + s * uy
  r10 = t * ux * uy + s * uz
  r11 = t * uy * uy + c
  r12 = t * uy * uz - s * ux
  r20 = t * ux * uz - s * uy
  r21 = t * uy * uz + s * ux
  r22 = t * uz * uz + c
  tx = px - (r00 * px + r01 * py + r02 * pz)
  ty = py - (r10 * px + r11 * py + r12 * pz)
  tz = pz - (r20 * px + r21 * py + r22 * pz)
  return [
      [r00, r01, r02, tx],
      [r10, r11, r12, ty],
      [r20, r21, r22, tz],
      [0.0, 0.0, 0.0, 1.0],
  ]


def _mat4_mul(a: Mat4, b: Mat4) -> Mat4:
  return [
      [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
      for i in range(4)
  ]


def solve_fold_3d(
    model: dict[str, Any], angles_deg: list[float] | None = None
) -> tuple[list[Mat4], list[Vec3], float]:
  """Solves BFS hinge face matrices, 3D vertices, and mean axial vertex error (%)."""
  faces = model['faces']
  verts = model['vertices']
  edges = model['edges']
  angles = model['shaping_angles'] if angles_deg is None else angles_deg
  start_face = model['central_face']

  edge_idx_map = {(u, v) if u < v else (v, u): i for i, (u, v) in enumerate(edges)}
  edge_to_faces: dict[tuple[int, int], list[int]] = {}
  for f_idx, f in enumerate(faces):
    flen = len(f)
    for i in range(flen):
      u, v = f[i], f[(i + 1) % flen]
      ek = (u, v) if u < v else (v, u)
      edge_to_faces.setdefault(ek, []).append(f_idx)

  mats: list[Mat4 | None] = [None] * len(faces)
  ident: Mat4 = [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0],
  ]
  mats[start_face] = ident
  visited = {start_face}
  q = collections.deque([start_face])

  while q:
    cf = q.popleft()
    cm = mats[cf]
    assert cm is not None
    fverts = faces[cf]
    flen = len(fverts)
    for i in range(flen):
      v1, v2 = fverts[i], fverts[(i + 1) % flen]
      ek = (v1, v2) if v1 < v2 else (v2, v1)
      for nf in edge_to_faces.get(ek, ()):
        if nf != cf and nf not in visited:
          e_idx = edge_idx_map.get(ek)
          deg = angles[e_idx] if e_idx is not None else 0.0
          x1, y1 = verts[v1]
          x2, y2 = verts[v2]
          p1x = cm[0][0] * x1 + cm[0][1] * y1 + cm[0][3]
          p1y = cm[1][0] * x1 + cm[1][1] * y1 + cm[1][3]
          p1z = cm[2][0] * x1 + cm[2][1] * y1 + cm[2][3]
          p2x = cm[0][0] * x2 + cm[0][1] * y2 + cm[0][3]
          p2y = cm[1][0] * x2 + cm[1][1] * y2 + cm[1][3]
          p2z = cm[2][0] * x2 + cm[2][1] * y2 + cm[2][3]
          rot_t = _rot_axis_mat4(
              p1x, p1y, p1z, p2x - p1x, p2y - p1y, p2z - p1z, math.radians(deg)
          )
          mats[nf] = _mat4_mul(rot_t, cm)
          visited.add(nf)
          q.append(nf)

  final_mats: list[Mat4] = [m if m is not None else ident for m in mats]

  sums = [[0.0, 0.0, 0.0] for _ in range(len(verts))]
  counts = [0] * len(verts)
  for f_idx, f in enumerate(faces):
    m = final_mats[f_idx]
    for v_idx in f:
      vx, vy = verts[v_idx]
      sums[v_idx][0] += m[0][0] * vx + m[0][1] * vy + m[0][3]
      sums[v_idx][1] += m[1][0] * vx + m[1][1] * vy + m[1][3]
      sums[v_idx][2] += m[2][0] * vx + m[2][1] * vy + m[2][3]
      counts[v_idx] += 1

  verts_3d: list[Vec3] = []
  for i, (vx, vy) in enumerate(verts):
    c = counts[i]
    if c > 0:
      verts_3d.append((sums[i][0] / c, sums[i][1] / c, sums[i][2] / c))
    else:
      verts_3d.append((vx, vy, 0.0))

  # Compute mean per-vertex axial strain error (%) matching folder_3d.py
  v_err_sum = [0.0] * len(verts)
  v_err_cnt = [0] * len(verts)
  for u, v in edges:
    l2d = math.hypot(verts[u][0] - verts[v][0], verts[u][1] - verts[v][1])
    if l2d < 1e-9:
      continue
    dx = verts_3d[u][0] - verts_3d[v][0]
    dy = verts_3d[u][1] - verts_3d[v][1]
    dz = verts_3d[u][2] - verts_3d[v][2]
    l3d = math.sqrt(dx * dx + dy * dy + dz * dz)
    strain = abs(l3d / l2d - 1.0)
    v_err_sum[u] += strain
    v_err_sum[v] += strain
    v_err_cnt[u] += 1
    v_err_cnt[v] += 1

  mean_v_err = (
      100.0
      * sum(
          v_err_sum[i] / v_err_cnt[i]
          for i in range(len(verts))
          if v_err_cnt[i] > 0
      )
      / max(1, len(verts))
  )
  return final_mats, verts_3d, mean_v_err


# ------------------------------------------------------------------------------
# StickFigure + PackingOutput upright SO(3) alignment & semantic flap regions
# ------------------------------------------------------------------------------


def compute_stick_figure_joints(sf: dict[str, Any]) -> dict[str, Vec3]:
  """Computes 3D StickFigure joint coordinates (`x` fwd, `y` left, `z` up)."""
  sticks = {s['name']: s for s in sf.get('children', [])}
  root_name = sf.get('root')
  if not root_name or root_name not in sticks:
    return {}
  root = sticks[root_name]
  adj: dict[str, list[dict[str, Any]]] = {}
  for s in sticks.values():
    adj.setdefault(s['start'], []).append(s)
    adj.setdefault(s['end'], []).append(s)

  positions: dict[str, Vec3] = {root['start']: (0.0, 0.0, 0.0)}
  visited_sticks: set[str] = set()
  q = collections.deque([root['start']])

  # Process root first from root.start -> root.end
  az = math.radians(float(root['azimuth_angle']))
  el = math.radians(float(root['elevation_angle']))
  ln = float(root['length'])
  positions[root['end']] = (
      ln * math.cos(el) * math.cos(az),
      ln * math.cos(el) * math.sin(az),
      ln * math.sin(el),
  )
  visited_sticks.add(root_name)
  q.append(root['end'])

  while q:
    j = q.popleft()
    jx, jy, jz = positions[j]
    for s in adj.get(j, ()):
      if s['name'] in visited_sticks:
        continue
      visited_sticks.add(s['name'])
      s_az = math.radians(float(s['azimuth_angle']))
      s_el = math.radians(float(s['elevation_angle']))
      s_ln = float(s['length'])
      vx = s_ln * math.cos(s_el) * math.cos(s_az)
      vy = s_ln * math.cos(s_el) * math.sin(s_az)
      vz = s_ln * math.sin(s_el)
      if s['start'] == j:
        far = s['end']
        positions[far] = (jx + vx, jy + vy, jz + vz)
      else:
        far = s['start']
        positions[far] = (jx - vx, jy - vy, jz - vz)
      q.append(far)
  return positions


def _point_in_poly_2d(
    px: float, py: float, poly: list[int], verts: list[tuple[float, float]]
) -> bool:
  inside = False
  n = len(poly)
  j = n - 1
  for i in range(n):
    xi, yi = verts[poly[i]]
    xj, yj = verts[poly[j]]
    if ((yi > py) != (yj > py)) and (
        px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi
    ):
      inside = not inside
    j = i
  return inside


def _locate_paper_points_3d(
    pts_unit: list[tuple[float, float]],
    model: dict[str, Any],
    face_mats: list[Mat4],
) -> Vec3 | None:
  """Evaluates mean folded 3D position of `[0, 1]^2` unit-paper sample points."""
  verts = model['vertices']
  faces = model['faces']
  acc = [0.0, 0.0, 0.0]
  cnt = 0
  for ux, uy in pts_unit:
    px, py = ux - 0.5, uy - 0.5
    chosen_f = -1
    for f_idx, f in enumerate(faces):
      xs = [verts[v][0] for v in f]
      ys = [verts[v][1] for v in f]
      if min(xs) - 1e-5 <= px <= max(xs) + 1e-5 and min(ys) - 1e-5 <= py <= max(ys) + 1e-5:
        if _point_in_poly_2d(px, py, f, verts):
          chosen_f = f_idx
          break
    if chosen_f < 0:
      # Nearest vertex fallback along boundary/crease
      best_d2 = 1e-3
      for f_idx, f in enumerate(faces):
        for v in f:
          d2 = (verts[v][0] - px) ** 2 + (verts[v][1] - py) ** 2
          if d2 < best_d2:
            best_d2 = d2
            chosen_f = f_idx
    if chosen_f >= 0:
      m = face_mats[chosen_f]
      acc[0] += m[0][0] * px + m[0][1] * py + m[0][3]
      acc[1] += m[1][0] * px + m[1][1] * py + m[1][3]
      acc[2] += m[2][0] * px + m[2][1] * py + m[2][3]
      cnt += 1
  if cnt == 0:
    return None
  return (acc[0] / cnt, acc[1] / cnt, acc[2] / cnt)


def _rect_perimeter_samples(
    x0: float, x1: float, y0: float, y1: float, grid_size: float
) -> list[tuple[float, float]]:
  w = max(0.0, x1 - x0)
  h = max(0.0, y1 - y0)
  pts: list[tuple[float, float]] = []
  if w < 1e-6 and h < 1e-6:
    pts.append((x0, y0))
  elif w < 1e-6 or h < 1e-6:
    steps = max(2, int(math.ceil(max(w, h) / 0.5)))
    for k in range(steps + 1):
      t = k / steps
      pts.append((x0 + t * w, y0 + t * h))
  else:
    nx = max(2, int(math.ceil(w / 0.5)))
    ny = max(2, int(math.ceil(h / 0.5)))
    for k in range(nx):
      t = k / nx
      pts.append((x0 + t * w, y0))
      pts.append((x1 - t * w, y1))
    for k in range(ny):
      t = k / ny
      pts.append((x1, y0 + t * h))
      pts.append((x0, y1 - t * h))
  out: list[tuple[float, float]] = []
  for x, y in pts:
    if -1e-5 <= x <= grid_size + 1e-5 and -1e-5 <= y <= grid_size + 1e-5:
      out.append((
          min(1.0, max(0.0, x / grid_size)),
          min(1.0, max(0.0, y / grid_size)),
      ))
  return out


def align_and_annotate_regions(
    model: dict[str, Any],
    face_mats: list[Mat4],
    sf: dict[str, Any],
    packing_raw: dict[str, Any] | None,
) -> tuple[Mat3, list[dict[str, Any]], list[int]]:
  """Computes Three.js upright SO(3) rotation and semantic face region labels."""
  # Permutes StickFigure (x=fwd, y=left, z=up) -> Three.js (X=left, Y=up, Z=fwd)
  # Wait: det([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) = +1 (pure SO(3) rotation!)
  fig_to_three: Mat3 = [
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0],
      [1.0, 0.0, 0.0],
  ]

  if not packing_raw:
    return fig_to_three, [], [-1] * len(model['faces'])

  flaps_json = packing_raw.get('flap_rectangles', '[]')
  flaps = json.loads(flaps_json) if isinstance(flaps_json, str) else flaps_json
  rivers_json = packing_raw.get('packing_rivers', '{}')
  rivers = json.loads(rivers_json) if isinstance(rivers_json, str) else rivers_json

  # Determine grid_size
  grid_size = 0
  for r in rivers.values():
    if isinstance(r, dict) and r.get('grid_size'):
      grid_size = int(r['grid_size'])
      break
  if grid_size <= 0:
    max_c = 0.0
    for fl in flaps:
      b = fl.get('bounds', [0, 0, 0, 0])
      for cc in fl.get('circle_centers', []):
        max_c = max(max_c, float(cc))
      max_c = max(max_c, b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)
    grid_size = max(1, int(round(max_c)))

  design_joints = compute_stick_figure_joints(sf)
  sticks_by_name = {s['name']: s for s in sf.get('children', [])}

  # Map merged flap names (e.g. "neck_head_beak_beak_hook" or "wing_L") to
  # their proximal and distal StickFigure joints!
  # Build stick topology order from root:
  root_name = sf.get('root', '')
  topo_prox: dict[str, str] = {}
  topo_dist: dict[str, str] = {}
  if root_name in sticks_by_name:
    r_stick = sticks_by_name[root_name]
    topo_prox[root_name] = r_stick['start']
    topo_dist[root_name] = r_stick['end']
    visited = {root_name}
    known_joints = {r_stick['start'], r_stick['end']}
    changed = True
    while changed:
      changed = False
      for sname, s in sticks_by_name.items():
        if sname in visited:
          continue
        if s['start'] in known_joints:
          topo_prox[sname] = s['start']
          topo_dist[sname] = s['end']
          known_joints.add(s['end'])
          visited.add(sname)
          changed = True
        elif s['end'] in known_joints:
          topo_prox[sname] = s['end']
          topo_dist[sname] = s['start']
          known_joints.add(s['start'])
          visited.add(sname)
          changed = True

  def match_chain_sticks(merged_name: str) -> list[str]:
    if merged_name in sticks_by_name:
      return [merged_name]
    # Greedy longest-prefix decomposition over stick names separated by '_'
    memo: dict[int, list[str] | None] = {}

    def dfs(pos: int) -> list[str] | None:
      if pos == len(merged_name):
        return []
      if pos in memo:
        return memo[pos]
      for sname in sorted(sticks_by_name, key=len, reverse=True):
        if merged_name.startswith(sname, pos):
          end_pos = pos + len(sname)
          if end_pos == len(merged_name):
            memo[pos] = [sname]
            return [sname]
          if merged_name[end_pos] == '_':
            rest = dfs(end_pos + 1)
            if rest is not None:
              memo[pos] = [sname] + rest
              return memo[pos]
      memo[pos] = None
      return None

    res = dfs(0)
    return res or []

  joint_samples: dict[str, list[tuple[float, float]]] = {}
  flap_tip_3d: dict[str, Vec3] = {}

  for idx, fl in enumerate(flaps):
    fname = fl['name']
    bx, by, bw, bh = (float(v) for v in fl['bounds'])
    chain = match_chain_sticks(fname)
    tip_pts = _rect_perimeter_samples(
        bx + min(bw, bh) / 2.0,
        bx + bw - min(bw, bh) / 2.0,
        by + min(bw, bh) / 2.0,
        by + bh - min(bw, bh) / 2.0,
        float(grid_size),
    )
    tip_3d = _locate_paper_points_3d(tip_pts, model, face_mats)
    if tip_3d is not None:
      flap_tip_3d[fname] = tip_3d

    if not chain:
      continue
    total_len = sum(float(sticks_by_name[s]['length']) for s in chain) or 1.0
    # Tip joint = distal joint of last stick in chain
    tip_joint = topo_dist.get(chain[-1])
    if tip_joint:
      joint_samples.setdefault(tip_joint, []).extend(tip_pts)

    # Intermediate joints in merged chain
    dist_from_tip = 0.0
    for sname in reversed(chain):
      dist_from_tip += float(sticks_by_name[sname]['length'])
      frac = dist_from_tip / total_len
      jname = topo_prox.get(sname)
      if not jname:
        continue
      if frac < 0.999:
        rad = min(bw, bh) / 2.0
        inset = rad * (1.0 - frac)
        c_pts = _rect_perimeter_samples(
            bx + inset, bx + bw - inset, by + inset, by + bh - inset, float(grid_size)
        )
        joint_samples.setdefault(jname, []).extend(c_pts)
      else:
        # Base ring just outside rectangle, avoiding other flaps
        margin = 0.05
        ring = _rect_perimeter_samples(
            bx - margin,
            bx + bw + margin,
            by - margin,
            by + bh + margin,
            float(grid_size),
        )
        filtered = []
        for ux, uy in ring:
          gx, gy = ux * grid_size, uy * grid_size
          in_other = False
          for j_idx, other in enumerate(flaps):
            if j_idx == idx:
              continue
            obx, oby, obw, obh = (float(v) for v in other['bounds'])
            if obx < gx < obx + obw and oby < gy < oby + obh:
              in_other = True
              break
          if not in_other:
            filtered.append((ux, uy))
        if len(filtered) < 4:
          filtered = _rect_perimeter_samples(bx, bx + bw, by, by + bh, float(grid_size))
        joint_samples.setdefault(jname, []).extend(filtered)

  src_pts: list[Vec3] = []
  tgt_pts: list[Vec3] = []
  for jname, samples in joint_samples.items():
    if jname not in design_joints or not samples:
      continue
    loc = _locate_paper_points_3d(samples, model, face_mats)
    if loc is not None:
      src_pts.append(loc)
      tgt_pts.append(design_joints[jname])

  if len(src_pts) >= 3:
    r_fig = _fit_robust_upright_rotation(src_pts, tgt_pts)
    r_three = _mat3_mul(fig_to_three, r_fig)
  else:
    r_three = fig_to_three

  # Semantic regions list (flaps first, then rivers)
  regions: list[dict[str, Any]] = []
  flap_bounds_unit: list[tuple[int, float, float, float, float]] = []
  for fl in flaps:
    fname = fl['name']
    bx, by, bw, bh = (float(v) for v in fl['bounds'])
    rid = len(regions)
    chain = match_chain_sticks(fname)
    pretty = chain[-1] if chain else fname
    tip3 = flap_tip_3d.get(fname)
    regions.append({
        'id': rid,
        'name': fname,
        'label': pretty.replace('_', ' '),
        'kind': 'flap',
        'box': [
            round(max(0.0, bx / grid_size), 4),
            round(max(0.0, by / grid_size), 4),
            round(min(1.0, (bx + bw) / grid_size), 4),
            round(min(1.0, (by + bh) / grid_size), 4),
        ],
        'tip3d_raw': [round(c, 5) for c in tip3] if tip3 else None,
    })
    flap_bounds_unit.append((
        rid,
        bx / grid_size,
        by / grid_size,
        (bx + bw) / grid_size,
        (by + bh) / grid_size,
    ))

  river_cells_unit: list[tuple[int, float, float, float, float]] = []
  for rname, rdata in rivers.items():
    if not isinstance(rdata, dict):
      continue
    rid = len(regions)
    regions.append({
        'id': rid,
        'name': rname,
        'label': rname.replace('_', ' '),
        'kind': 'river',
    })
    for cell in rdata.get('rectangles', []):
      cb = cell.get('bounds', [0, 0, 0, 0])
      river_cells_unit.append((
          rid,
          cb[0] / grid_size,
          cb[1] / grid_size,
          (cb[0] + cb[2]) / grid_size,
          (cb[1] + cb[3]) / grid_size,
      ))

  verts = model['vertices']
  face_regions: list[int] = []
  for f in model['faces']:
    ux = sum(verts[v][0] for v in f) / len(f) + 0.5
    uy = sum(verts[v][1] for v in f) / len(f) + 0.5
    assigned_rid = -1
    for rid, x0, y0, x1, y1 in river_cells_unit:
      if x0 - 1e-4 <= ux <= x1 + 1e-4 and y0 - 1e-4 <= uy <= y1 + 1e-4:
        assigned_rid = rid
        break
    if assigned_rid < 0:
      for rid, x0, y0, x1, y1 in flap_bounds_unit:
        if x0 - 1e-4 <= ux <= x1 + 1e-4 and y0 - 1e-4 <= uy <= y1 + 1e-4:
          assigned_rid = rid
          break
    face_regions.append(assigned_rid)

  return r_three, regions, face_regions


def export_fold_v12(
    title: str,
    description: str,
    model: dict[str, Any],
    verts_3d_upright: list[Vec3],
) -> dict[str, Any]:
  """Builds a standard FOLD v1.2 dictionary with 2D CP + 3D folded frame."""
  return {
      'file_spec': 1.2,
      'file_creator': 'google-deepmind/origami-design (bp-pipeline)',
      'file_title': title,
      'file_description': description,
      'file_classes': ['singleModel'],
      'frame_classes': ['creasePattern'],
      'frame_attributes': ['2D'],
      'vertices_coords': [
          [round(vx + 0.5, 6), round(vy + 0.5, 6)]
          for vx, vy in model['vertices']
      ],
      'edges_vertices': [[u, v] for u, v in model['edges']],
      'edges_assignment': model['assignments'],
      'edges_foldAngle': [round(a, 4) for a in model['shaping_angles']],
      'faces_vertices': model['faces'],
      'file_frames': [{
          'frame_title': f'{title} (3D Shaped)',
          'frame_classes': ['foldedForm'],
          'frame_attributes': ['3D'],
          'frame_inherit': True,
          'frame_parent': 0,
          'vertices_coords': [
              [round(x, 6), round(y, 6), round(z, 6)]
              for x, y, z in verts_3d_upright
          ],
      }],
  }


# ------------------------------------------------------------------------------
# Curated Showcase Manifest (Round 14 Champions + Iconic Multi-Limb Designs)
# ------------------------------------------------------------------------------

CURATED_SHOWCASE: list[dict[str, Any]] = [
    {
        'slug': 'golden-eagle',
        'source_dir': 'Eagle_v3',
        'title': 'Golden Eagle',
        'subtitle': 'Aquila chrysaetos',
        'category': 'Birds & Reptiles',
        'grid_size': 28,
        'fidelity': 8.85,
        'color_top': '#9E3329',
        'color_bottom': '#F3E7CE',
        'camera_azimuth_deg': -42,
        'camera_elevation_deg': 16,
        'techniques': [
            '3.4× Fanned Wings',
            '2.7× Squash-Fold Tail',
            '0.28× Clip-Narrowed Talons',
            '3D Hooked Claws',
        ],
        'summary': (
            'Stooping golden eagle sculpted on an economical 28×28 box-pleated '
            'grid with expansive fanned primary feathers, a bilaterally '
            'squash-folded tail fan, hooked beak, and razor-thin talons.'
        ),
        'highlights': (
            'Combines fan_out_fold on both wings (3.41×/3.34× width), '
            'squash_fold(tail, open=0.60) for a 2.67× tail fan, straight '
            'clip_pattern_fold narrowing on talon_L/R (0.28×), and '
            'crimp_fold_3d hooked claws with 0.17% vertex strain.'
        ),
    },
    {
        'slug': 'winged-pegasus',
        'source_dir': 'Pegasus',
        'title': 'Winged Pegasus',
        'subtitle': 'Mythological Winged Stallion',
        'category': 'Mythic & Dinosaurs',
        'grid_size': 44,
        'fidelity': 8.61,
        'color_top': '#244B7A',
        'color_bottom': '#F5EFE2',
        'camera_azimuth_deg': 38,
        'camera_elevation_deg': 18,
        'techniques': [
            '3.2× Fanned Great Wings',
            '1.31× Squash-Fold Skull',
            '7× Clip-Narrowed Limbs & Neck',
        ],
        'summary': (
            'Rearing winged stallion combining broad pleated pinions, a '
            'sculpted 3D equine cheek wedge, pricked ears, and seven '
            'straight clip-narrowed limb and neck segments.'
        ),
        'highlights': (
            'Applies 7 straight clip_pattern_fold narrowings across forelegs '
            '(0.51×), hindlegs (0.56×), muzzle, tail plume, and neck river '
            'before 3.19×/2.98× wing fans and squash_fold(head, 0.50).'
        ),
    },
    {
        'slug': 'triceratops',
        'source_dir': 'Triceratops_v4_squash',
        'title': 'Triceratops',
        'subtitle': 'Triceratops horridus',
        'category': 'Mythic & Dinosaurs',
        'grid_size': 44,
        'fidelity': 8.31,
        'color_top': '#3B6B54',
        'color_bottom': '#F1E8D4',
        'camera_azimuth_deg': 36,
        'camera_elevation_deg': 20,
        'techniques': [
            '1.95× Corner-Fanned Frill',
            'Squash-Fold 3D Rostrum',
            '4× Clip-Narrowed Feet',
            '3D Folded Spine',
        ],
        'summary': (
            'Three-horned ceratopsian dinosaur featuring a broad corner-packed '
            'cranial frill shield, twin brow horns, nasal horn, 3D squashed '
            'beak rostrum, and four articulated legs.'
        ),
        'highlights': (
            'Expands frill_L/R to 1.95×/1.92× via graded K=4 corner fans, '
            'narrows all four feet to 0.86× via straight clip_pattern_fold, '
            'and sculpts a 1.19× 3D ceratopsian snout wedge via squash_fold.'
        ),
    },
    {
        'slug': 'wyrmlord-dragon',
        'source_dir': 'Dragon',
        'title': 'Western Dragon',
        'subtitle': 'Draco occidentalis',
        'category': 'Mythic & Dinosaurs',
        'grid_size': 46,
        'fidelity': 7.78,
        'color_top': '#8F2432',
        'color_bottom': '#F0DFC2',
        'camera_azimuth_deg': 35,
        'camera_elevation_deg': 20,
        'techniques': [
            '2.0× Fanned Corner Wings',
            'Clip-Narrowed Neck River',
            'Symmetric Swept Horns',
            'Roaring Jaws & S-Tail',
        ],
        'summary': (
            'Articulated 26-stick dragon with twin swept horns, open roaring '
            'jaws, straight-narrowed neck river, fanned bat-like wings, four '
            'clawed legs, and a sinuous whip tail.'
        ),
        'highlights': (
            'Straight-narrows the neck U-river on the base (+2,934 creases, '
            '2,408/2,408 flat-foldable vertices) so the horned head separates '
            'cleanly from the shoulders alongside 2.00× corner wing fans.'
        ),
    },
    {
        'slug': 'king-cobra',
        'source_dir': 'Cobra',
        'title': 'King Cobra',
        'subtitle': 'Ophiophagus hannah',
        'category': 'Birds & Reptiles',
        'grid_size': 42,
        'fidelity': 7.67,
        'color_top': '#B65D28',
        'color_bottom': '#F5EBCE',
        'camera_azimuth_deg': 40,
        'camera_elevation_deg': 22,
        'techniques': [
            '360° Spiral Body Coil',
            '1.59× Squash-Fold Neck',
            '1.66× Flared Hood',
            'Bifurcated Forked Tongue',
        ],
        'summary': (
            'Strike-posture cobra rising from a 6-segment 360° coiled base '
            'into a squash-flared vertical neck, wide spectacled hood, open '
            'lower jaw, and narrowed forked tongue.'
        ),
        'highlights': (
            'Applies squash_fold(neck_rise, open=0.50) to flare the vertical '
            'neck column 1.59× directly into the 1.66×/1.41× open hood while '
            'tapering the tail tip and forked tongue.'
        ),
    },
    {
        'slug': 'emperor-scorpion',
        'source_dir': 'Scorpion',
        'title': 'Emperor Scorpion',
        'subtitle': 'Pandinus imperator',
        'category': 'Arthropods',
        'grid_size': 58,
        'fidelity': 8.78,
        'color_top': '#2E3440',
        'color_bottom': '#EAE1CE',
        'camera_azimuth_deg': 38,
        'camera_elevation_deg': 26,
        'techniques': [
            '8 Articulated Walking Legs',
            'Segmented Overhead Tail Arch',
            'Bifurcated Pedipalp Pincers',
        ],
        'summary': (
            'High-complexity 38-stick arachnid packing eight multi-jointed '
            'walking legs, twin open chela pincers, and a five-segment '
            'metasoma tail arching forward to a terminal stinger.'
        ),
        'highlights': (
            'Demonstrates high-branching box-pleated river routing on a 58×58 '
            'grid with 2,753 flat-foldable vertices and 0.002% axial strain.'
        ),
    },
    {
        'slug': 'boxer-crab',
        'source_dir': 'BoxerCrab',
        'title': 'Boxer Crab',
        'subtitle': 'Lybia tessellata',
        'category': 'Arthropods',
        'grid_size': 50,
        'fidelity': 8.19,
        'color_top': '#C94A38',
        'color_bottom': '#F7EFE0',
        'camera_azimuth_deg': -38,
        'camera_elevation_deg': 32,
        'techniques': [
            'Dual Sea-Anemone Claws',
            '8 Lateral Walking Legs',
            'Inscribed Carapace Volume',
        ],
        'summary': (
            'Defensive marine crab thrusting two anemone-tipped chelipeds '
            'forward from a wide carapace flanked by eight bent walking legs.'
        ),
        'highlights': (
            'Combines symmetric leg narrowing across 8 lateral limbs with '
            'branched distal anemone tufts on a 50×50 grid.'
        ),
    },
    {
        'slug': 'bull-moose',
        'source_dir': 'BullMoose',
        'title': 'Bull Moose',
        'subtitle': 'Alces alces',
        'category': 'Mammals',
        'grid_size': 34,
        'fidelity': 8.05,
        'color_top': '#6E4730',
        'color_bottom': '#F2E7D3',
        'camera_azimuth_deg': 38,
        'camera_elevation_deg': 18,
        'techniques': [
            '6-Point Branched Antlers',
            'Squash-Fold Head Wedge',
            'Quadruped Stance',
        ],
        'summary': (
            'Forest ungulate crowned with multi-tined palmate antlers, '
            'sloped shoulders, squash-sculpted snout, and four slender legs '
            'packed on a compact 34×34 grid.'
        ),
        'highlights': (
            'Routes six symmetric antler tines from a single cranial junction '
            'with 0.02% vertex strain across 4,129 flat-foldable vertices.'
        ),
    },
    {
        'slug': 'indian-peacock',
        'source_dir': 'Peacock',
        'title': 'Indian Peacock',
        'subtitle': 'Pavo cristatus',
        'category': 'Birds & Reptiles',
        'grid_size': 48,
        'fidelity': 8.12,
        'color_top': '#1D6A78',
        'color_bottom': '#F4E9CC',
        'camera_azimuth_deg': 36,
        'camera_elevation_deg': 20,
        'techniques': [
            '11-Feather Radial Train Fan',
            'Upright S-Curved Neck',
            'Bipedal Stance',
        ],
        'summary': (
            'Displaying peacock raising an eleven-ray semicircular tail train '
            'behind an upright neck, crested head, and balanced two-legged '
            'stance.'
        ),
        'highlights': (
            'Showcases high-degree pelvic river fan-out (degree 12 junction) '
            'into radially splayed tail feathers on a 48×48 grid.'
        ),
    },
    {
        'slug': 'crested-hoopoe',
        'source_dir': 'Hoopoe',
        'title': 'Crested Hoopoe',
        'subtitle': 'Upupa epops',
        'category': 'Birds & Reptiles',
        'grid_size': 45,
        'fidelity': 8.04,
        'color_top': '#C96A32',
        'color_bottom': '#F5ECDC',
        'camera_azimuth_deg': 36,
        'camera_elevation_deg': 18,
        'techniques': [
            '8-Feather Crown Crest Fan',
            'Decurved Foraging Bill',
            'Striding Bipedal Pose',
        ],
        'summary': (
            'Walking hoopoe featuring an iconic eight-feather radial crown '
            'crest, long decurved bill, swept wings, and mid-stride legs.'
        ),
        'highlights': (
            'Packs an 8-flap cranial fan alongside wings, tail, and segmented '
            'legs on an asymmetric 45×45 grid.'
        ),
    },
]


def build_single_bundle(
    input_dir: Path,
    output_dir: Path,
    meta: dict[str, Any],
) -> dict[str, Any]:
  """Packages a single model directory into `output_dir` and returns catalog entry."""
  output_dir.mkdir(parents=True, exist_ok=True)

  sf_path = input_dir / 'stick_figure.json'
  packing_svg_path = input_dir / 'packing_cp.svg'
  solving_svg_path = input_dir / 'solving_cp.svg'
  shaping_svg_path = input_dir / 'shaping_cp.svg'
  if not shaping_svg_path.exists():
    shaping_svg_path = input_dir / 'shaped_cp.svg'
  packing_json_path = input_dir / 'packing_output.json'

  with sf_path.open('r', encoding='utf-8') as f:
    sf = json.load(f)
  shaping_svg = shaping_svg_path.read_text(encoding='utf-8')
  solving_svg = (
      solving_svg_path.read_text(encoding='utf-8')
      if solving_svg_path.exists()
      else None
  )
  packing_raw = (
      json.loads(packing_json_path.read_text(encoding='utf-8'))
      if packing_json_path.exists()
      else None
  )

  model = build_planar_model(shaping_svg, solving_svg)
  face_mats, verts_3d_raw, vertex_err_pct = solve_fold_3d(model)
  r_three, regions, face_regions = align_and_annotate_regions(
      model, face_mats, sf, packing_raw
  )

  # Center raw 3D vertices at centroid, then apply upright SO(3) rotation r_three
  n_v = len(verts_3d_raw)
  cx = sum(p[0] for p in verts_3d_raw) / n_v
  cy = sum(p[1] for p in verts_3d_raw) / n_v
  cz = sum(p[2] for p in verts_3d_raw) / n_v

  verts_3d_upright: list[Vec3] = []
  for px, py, pz in verts_3d_raw:
    ux, uy, uz = _mat3_vec(r_three, (px - cx, py - cy, pz - cz))
    verts_3d_upright.append((ux, uy, uz))

  # Shift upright Y so lowest vertex rests cleanly on pedestal y = 0
  min_y = min(p[1] for p in verts_3d_upright)
  verts_3d_upright = [(p[0], p[1] - min_y, p[2]) for p in verts_3d_upright]

  # Transform region 3D tip anchors into the same upright pedestal space
  for reg in regions:
    raw_tip = reg.pop('tip3d_raw', None)
    if raw_tip is not None:
      tx, ty, tz = _mat3_vec(
          r_three, (raw_tip[0] - cx, raw_tip[1] - cy, raw_tip[2] - cz)
      )
      reg['tip3d'] = [round(tx, 4), round(ty - min_y, 4), round(tz, 4)]

  upright_quat = _mat3_to_quaternion(r_three)

  # Copy clean canonical artifacts
  with (output_dir / 'stick_figure.json').open('w', encoding='utf-8') as f:
    json.dump(sf, f, indent=2)
    f.write('\n')
  if packing_svg_path.exists():
    shutil.copyfile(packing_svg_path, output_dir / 'packing_cp.svg')
  if solving_svg_path.exists():
    shutil.copyfile(solving_svg_path, output_dir / 'solving_cp.svg')
  (output_dir / 'shaped_cp.svg').write_text(shaping_svg, encoding='utf-8')

  fold_doc = export_fold_v12(
      meta['title'], meta['summary'], model, verts_3d_upright
  )
  with (output_dir / 'shaped.fold').open('w', encoding='utf-8') as f:
    json.dump(fold_doc, f, separators=(',', ':'))

  # Compact web bundle (< 10ms parse & render in Three.js)
  flat_tris: list[list[int]] = model['triangulations']
  bundle = {
      'slug': meta['slug'],
      'title': meta['title'],
      'subtitle': meta.get('subtitle', ''),
      'category': meta.get('category', 'Showcase'),
      'colorTop': meta['color_top'],
      'colorBottom': meta['color_bottom'],
      'cameraAzimuthDeg': meta.get('camera_azimuth_deg', 36),
      'cameraElevationDeg': meta.get('camera_elevation_deg', 20),
      'uprightQuat': upright_quat,
      'centralFace': model['central_face'],
      'vertices2d': [
          [round(vx, 5), round(vy, 5)] for vx, vy in model['vertices']
      ],
      'faces': model['faces'],
      'triangulations': flat_tris,
      'edges': [[u, v] for u, v in model['edges']],
      'assignments': ''.join(model['assignments']),
      'solvingAngles': [round(a, 2) for a in model['solving_angles']],
      'shapingAngles': [round(a, 2) for a in model['shaping_angles']],
      'edgeStage': model['edge_stage'],
      'faceRegions': face_regions,
      'regions': regions,
      'stickFigure': sf,
  }
  bundle_path = output_dir / 'bundle.json'
  with bundle_path.open('w', encoding='utf-8') as f:
    json.dump(bundle, f, separators=(',', ':'))

  num_creases = sum(1 for a in model['assignments'] if a in ('M', 'V'))
  num_sticks = len(sf.get('children', []))
  num_flaps = sum(1 for r in regions if r.get('kind') == 'flap')
  det_r = _mat3_det(r_three)

  catalog_item = {
      'slug': meta['slug'],
      'title': meta['title'],
      'subtitle': meta.get('subtitle', ''),
      'category': meta.get('category', 'Showcase'),
      'gridSize': meta.get('grid_size', 0),
      'sticks': num_sticks,
      'flaps': num_flaps or num_sticks,
      'vertices': len(model['vertices']),
      'faces': len(model['faces']),
      'creases': num_creases,
      'vertexErrorPct': round(vertex_err_pct, 2),
      'fidelity': meta.get('fidelity'),
      'colorTop': meta['color_top'],
      'colorBottom': meta['color_bottom'],
      'techniques': meta.get('techniques', []),
      'summary': meta.get('summary', ''),
      'highlights': meta.get('highlights', ''),
      'preview': f"models/{meta['slug']}/preview.png",
      'bundleUrl': f"models/{meta['slug']}/bundle.json",
      'files': {
          'fold': f"models/{meta['slug']}/shaped.fold",
          'shapedSvg': f"models/{meta['slug']}/shaped_cp.svg",
          'solvingSvg': f"models/{meta['slug']}/solving_cp.svg",
          'packingSvg': f"models/{meta['slug']}/packing_cp.svg",
          'stickFigure': f"models/{meta['slug']}/stick_figure.json",
      },
      '_det': round(det_r, 5),
      '_bundleKB': round(bundle_path.stat().st_size / 1024.0, 1),
  }
  return catalog_item


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--samples-root',
      type=Path,
      help='Build all curated showcase models from a samples root directory.',
  )
  parser.add_argument(
      '--input',
      type=Path,
      help='Single bp-pipeline model directory to package.',
  )
  parser.add_argument('--slug', type=str, help='URL slug for --input mode.')
  parser.add_argument('--title', type=str, help='Display title for --input mode.')
  parser.add_argument(
      '--subtitle', type=str, default='Contributed Model', help='Subtitle/taxon.'
  )
  parser.add_argument(
      '--category', type=str, default='Community', help='Gallery category.'
  )
  parser.add_argument(
      '--color-top', type=str, default='#2B5B84', help='Hex dorsal paper color.'
  )
  parser.add_argument(
      '--color-bottom',
      type=str,
      default='#F4ECDC',
      help='Hex ventral paper color.',
  )
  parser.add_argument(
      '--docs-dir',
      type=Path,
      default=Path(__file__).resolve().parents[1] / 'docs',
      help='Path to docs/ directory (default: <repo>/docs).',
  )
  args = parser.parse_args()

  models_dir = args.docs_dir / 'models'
  models_dir.mkdir(parents=True, exist_ok=True)
  catalog_path = models_dir / 'catalog.json'

  if args.samples_root:
    catalog: list[dict[str, Any]] = []
    for meta in CURATED_SHOWCASE:
      src = args.samples_root / meta['source_dir']
      if not src.exists():
        raise FileNotFoundError(f'Missing sample directory: {src}')
      entry = build_single_bundle(src, models_dir / meta['slug'], meta)
      det = entry.pop('_det')
      b_kb = entry.pop('_bundleKB')
      print(
          f"[OK] {entry['slug']:18s} | grid={entry['gridSize']:2d} | "
          f"V={entry['vertices']:4d} F={entry['faces']:4d} C={entry['creases']:4d} | "
          f"v_err={entry['vertexErrorPct']:5.2f}% | det(R)={det:+.4f} | "
          f"bundle={b_kb:6.1f}KB"
      )
      catalog.append(entry)
    with catalog_path.open('w', encoding='utf-8') as f:
      json.dump(catalog, f, indent=2)
      f.write('\n')
    print(f'Wrote {len(catalog)} showcase models to {catalog_path}')
    return

  if args.input and args.slug and args.title:
    meta = {
        'slug': args.slug,
        'title': args.title,
        'subtitle': args.subtitle,
        'category': args.category,
        'grid_size': 0,
        'fidelity': None,
        'color_top': args.color_top,
        'color_bottom': args.color_bottom,
        'techniques': [],
        'summary': f'Box-pleated origami {args.title}.',
        'highlights': '',
    }
    entry = build_single_bundle(args.input, models_dir / args.slug, meta)
    entry.pop('_det', None)
    entry.pop('_bundleKB', None)
    existing: list[dict[str, Any]] = []
    if catalog_path.exists():
      existing = json.loads(catalog_path.read_text(encoding='utf-8'))
    existing = [e for e in existing if e.get('slug') != args.slug] + [entry]
    with catalog_path.open('w', encoding='utf-8') as f:
      json.dump(existing, f, indent=2)
      f.write('\n')
    print(f"Added {args.slug} to {catalog_path}")
    return

  parser.error('Pass either --samples-root or (--input --slug --title).')


if __name__ == '__main__':
  main()
