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

"""Width-2 sacrificial-pocket level shifters and exact axial elevations."""

import collections
from collections.abc import Collection, Mapping, Sequence
import dataclasses
import fractions
import functools
from typing import Any

from bp_pipeline import bus
from bp_pipeline import plugins
from bp_pipeline.crease_pattern import crease as crease_lib
from bp_pipeline.crease_pattern import crease_pattern as cp_lib
from bp_pipeline.crease_pattern import point as point_lib
from bp_pipeline.packing import level_shifter as packing_ls
from bp_pipeline.packing import packing_output as packing_output_lib
from bp_pipeline.solving import hinges as hinge_utils
from bp_pipeline.solving import utils_ls

Fraction = fractions.Fraction
RatPoint = tuple[Fraction, Fraction]
RatMat2 = tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]

pocket_depth = packing_ls.pocket_depth
pocket_margin = packing_ls.pocket_margin

_VALID_SHIFT_DIRECTIONS = ('auto', 'up', 'down')
_IDENTITY_MAT: RatMat2 = (
    (Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(1)),
)
_ZERO_VEC: RatPoint = (Fraction(0), Fraction(0))


def _to_rat_point(pt: point_lib.Point, grid_size: int) -> RatPoint:
  """Converts normalized `pt` to exact rational grid coordinates."""
  return (
      Fraction(pt.x * grid_size).limit_denominator(1024),
      Fraction(pt.y * grid_size).limit_denominator(1024),
  )


def _apply_rat_iso(mat: RatMat2, off: RatPoint, pt: RatPoint) -> RatPoint:
  """Applies exact 2D rational isometry `mat @ pt + off`."""
  return (
      mat[0][0] * pt[0] + mat[0][1] * pt[1] + off[0],
      mat[1][0] * pt[0] + mat[1][1] * pt[1] + off[1],
  )


def _reflect_2d_exact(
    mat: RatMat2,
    off: RatPoint,
    p1: RatPoint,
    p2: RatPoint,
) -> tuple[RatMat2, RatPoint]:
  """Composes exact rational 2D isometry with reflection across `p1-p2`."""
  q1 = _apply_rat_iso(mat, off, p1)
  q2 = _apply_rat_iso(mat, off, p2)
  dx = q2[0] - q1[0]
  dy = q2[1] - q1[1]
  norm2 = dx * dx + dy * dy
  if norm2 == 0:
    return mat, off
  r00 = (dx * dx - dy * dy) / norm2
  r01 = (2 * dx * dy) / norm2
  r11 = (dy * dy - dx * dx) / norm2
  new_mat: RatMat2 = (
      (r00 * mat[0][0] + r01 * mat[1][0], r00 * mat[0][1] + r01 * mat[1][1]),
      (r01 * mat[0][0] + r11 * mat[1][0], r01 * mat[0][1] + r11 * mat[1][1]),
  )
  diff0 = off[0] - q1[0]
  diff1 = off[1] - q1[1]
  new_off: RatPoint = (
      q1[0] + r00 * diff0 + r01 * diff1,
      q1[1] + r01 * diff0 + r11 * diff1,
  )
  return new_mat, new_off


def _ccw_compare(d1: RatPoint, d2: RatPoint) -> int:
  """Compares two non-zero rational direction vectors in CCW order."""
  h1 = 0 if (d1[1] > 0 or (d1[1] == 0 and d1[0] > 0)) else 1
  h2 = 0 if (d2[1] > 0 or (d2[1] == 0 and d2[0] > 0)) else 1
  if h1 != h2:
    return -1 if h1 < h2 else 1
  cross = d1[0] * d2[1] - d1[1] * d2[0]
  if cross > 0:
    return -1
  if cross < 0:
    return 1
  return 0


def _segment_intersection_rat(
    a1: RatPoint, a2: RatPoint, b1: RatPoint, b2: RatPoint
) -> RatPoint | None:
  """Returns exact interior intersection of non-collinear segments, if any."""
  dax, day = a2[0] - a1[0], a2[1] - a1[1]
  dbx, dby = b2[0] - b1[0], b2[1] - b1[1]
  denom = dax * dby - day * dbx
  if denom == 0:
    return None
  rx, ry = b1[0] - a1[0], b1[1] - a1[1]
  t = (rx * dby - ry * dbx) / denom
  u = (rx * day - ry * dax) / denom
  if Fraction(0) <= t <= Fraction(1) and Fraction(0) <= u <= Fraction(1):
    return (a1[0] + t * dax, a1[1] + t * day)
  return None


def _fold_faces_exact(
    creases: Sequence[crease_lib.Crease],
    grid_size: int,
) -> tuple[
    list[RatPoint],
    dict[int, list[tuple[int, int]]],
    dict[tuple[int, int], int],
    dict[int, RatMat2],
    dict[int, RatPoint],
    dict[int, int],
]:
  """Extracts planar faces and folds them via exact rational 2D reflections."""
  raw_segs: list[tuple[RatPoint, RatPoint, crease_lib.Fold]] = []
  point_set: set[RatPoint] = set()
  for c in creases:
    p1 = _to_rat_point(c.p1, grid_size)
    p2 = _to_rat_point(c.p2, grid_size)
    if p1 == p2:
      continue
    raw_segs.append((p1, p2, c.fold))
    point_set.add(p1)
    point_set.add(p2)

  n_raw = len(raw_segs)
  for i in range(n_raw):
    a1, a2, _ = raw_segs[i]
    for j in range(i + 1, n_raw):
      b1, b2, _ = raw_segs[j]
      hit = _segment_intersection_rat(a1, a2, b1, b2)
      if hit is not None:
        point_set.add(hit)

  vertices = sorted(point_set)
  vert_index = {pt: i for i, pt in enumerate(vertices)}

  edge_fold: dict[tuple[int, int], crease_lib.Fold] = {}
  adj: dict[int, set[int]] = collections.defaultdict(set)
  crease_sub_edges: dict[int, list[tuple[int, int]]] = collections.defaultdict(
      list
  )

  for idx, c in enumerate(creases):
    p1 = _to_rat_point(c.p1, grid_size)
    p2 = _to_rat_point(c.p2, grid_size)
    if p1 == p2:
      continue
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    len2 = dx * dx + dy * dy
    x_lo, x_hi = min(p1[0], p2[0]), max(p1[0], p2[0])
    y_lo, y_hi = min(p1[1], p2[1]), max(p1[1], p2[1])
    on_seg: list[tuple[Fraction, int]] = []
    for pt, vid in vert_index.items():
      if not (x_lo <= pt[0] <= x_hi and y_lo <= pt[1] <= y_hi):
        continue
      rx, ry = pt[0] - p1[0], pt[1] - p1[1]
      if dx * ry == dy * rx:
        dot = rx * dx + ry * dy
        if Fraction(0) <= dot <= len2:
          on_seg.append((dot, vid))
    on_seg.sort(key=lambda item: item[0])
    for (_, u), (_, v) in zip(on_seg, on_seg[1:]):
      if u == v:
        continue
      adj[u].add(v)
      adj[v].add(u)
      key = (min(u, v), max(u, v))
      if key not in edge_fold or edge_fold[key] != crease_lib.Fold.BORDER:
        edge_fold[key] = c.fold
      crease_sub_edges[idx].append((u, v))

  ordered_out: dict[int, list[int]] = {}
  pos_in_out: dict[tuple[int, int], int] = {}
  for u, nbrs in adj.items():
    ux, uy = vertices[u]
    sorted_nbrs = sorted(
        nbrs,
        key=functools.cmp_to_key(
            lambda w1, w2, ox=ux, oy=uy: _ccw_compare(
                (vertices[w1][0] - ox, vertices[w1][1] - oy),
                (vertices[w2][0] - ox, vertices[w2][1] - oy),
            )
        ),
    )
    ordered_out[u] = sorted_nbrs
    for i, w in enumerate(sorted_nbrs):
      pos_in_out[(u, w)] = i

  visited_he: set[tuple[int, int]] = set()
  faces: list[list[int]] = []
  he_to_face: dict[tuple[int, int], int] = {}
  for u, nbrs in ordered_out.items():
    for v in nbrs:
      if (u, v) in visited_he:
        continue
      cycle: list[int] = []
      cur_u, cur_v = u, v
      valid = True
      while (cur_u, cur_v) not in visited_he:
        visited_he.add((cur_u, cur_v))
        cycle.append(cur_u)
        out_v = ordered_out.get(cur_v)
        if not out_v:
          valid = False
          break
        rev_idx = pos_in_out[(cur_v, cur_u)]
        next_w = out_v[(rev_idx - 1) % len(out_v)]
        cur_u, cur_v = cur_v, next_w
      if not valid or len(cycle) < 3:
        continue
      area2 = Fraction(0)
      for i, vid in enumerate(cycle):
        nx_vid = cycle[(i + 1) % len(cycle)]
        area2 += (
            vertices[vid][0] * vertices[nx_vid][1]
            - vertices[nx_vid][0] * vertices[vid][1]
        )
      if area2 <= 0:
        continue
      fid = len(faces)
      faces.append(cycle)
      for i, vid in enumerate(cycle):
        nx_vid = cycle[(i + 1) % len(cycle)]
        he_to_face[(vid, nx_vid)] = fid

  iso_mat: dict[int, RatMat2] = {}
  iso_off: dict[int, RatPoint] = {}
  face_comp: dict[int, int] = {}
  num_comps = 0
  for start_fid in range(len(faces)):
    if start_fid in iso_mat:
      continue
    comp_id = num_comps
    num_comps += 1
    iso_mat[start_fid] = _IDENTITY_MAT
    iso_off[start_fid] = _ZERO_VEC
    face_comp[start_fid] = comp_id
    queue = collections.deque([start_fid])
    while queue:
      fid = queue.popleft()
      mat = iso_mat[fid]
      off = iso_off[fid]
      f_verts = faces[fid]
      for i, u in enumerate(f_verts):
        v = f_verts[(i + 1) % len(f_verts)]
        opp_fid = he_to_face.get((v, u))
        if opp_fid is None or opp_fid in iso_mat:
          continue
        fold = edge_fold.get((min(u, v), max(u, v)), crease_lib.Fold.BORDER)
        if fold == crease_lib.Fold.BORDER:
          continue
        if fold in (crease_lib.Fold.MOUNTAIN, crease_lib.Fold.VALLEY):
          n_mat, n_off = _reflect_2d_exact(mat, off, vertices[u], vertices[v])
        else:
          n_mat, n_off = mat, off
        iso_mat[opp_fid] = n_mat
        iso_off[opp_fid] = n_off
        face_comp[opp_fid] = comp_id
        queue.append(opp_fid)

  return vertices, crease_sub_edges, he_to_face, iso_mat, iso_off, face_comp


def compute_axial_elevations(
    pattern: cp_lib.CreasePattern,
    *,
    unshifted_reference: cp_lib.CreasePattern | None = None,
) -> cp_lib.CreasePattern:
  """Computes exact integer `elevation` on axial creases from folded coordinates.

  Folds planar faces of `pattern` across all interior mountain/valley creases
  in exact `fractions.Fraction` arithmetic, determines the uniaxial level axis
  and `[0, 1]` unit reference frame, and sets `Crease.elevation` exclusively on
  axial creases (leaving hinges, ridges, and borders with `elevation=None`).

  Requires an intact central crease (`remove_central_crease=False`); bases with
  the central crease removed have opened half-sheet frames that are out of scope
  for bare-crease elevation reconstruction.

  Args:
    pattern: Flat-foldable box-pleated crease pattern (`remove_central_crease`
      must be `False`).
    unshifted_reference: Optional pre-shift `CreasePattern` whose `[0, 1]`
      axial creases anchor the level sign and origin on shifted components.

  Returns:
    A new `CreasePattern` with exact integer `elevation` on axial creases only.
  """
  grid_size = pattern.grid_size if pattern.grid_size > 0 else 1
  creases = list(pattern.creases)
  if not creases or grid_size <= 0:
    return pattern

  vertices, sub_edges, he_to_face, iso_mat, iso_off, face_comp = (
      _fold_faces_exact(creases, grid_size)
  )
  if not iso_mat:
    return pattern

  folded_sub_segs: dict[int, list[tuple[int, RatPoint, RatPoint]]] = {}
  for idx in range(len(creases)):
    subs = sub_edges.get(idx)
    if not subs:
      continue
    sub_list: list[tuple[int, RatPoint, RatPoint]] = []
    for u, v in subs:
      fid = he_to_face.get((u, v))
      if fid is None:
        fid = he_to_face.get((v, u))
      if fid is None or fid not in iso_mat:
        continue
      q1 = _apply_rat_iso(iso_mat[fid], iso_off[fid], vertices[u])
      q2 = _apply_rat_iso(iso_mat[fid], iso_off[fid], vertices[v])
      sub_list.append((face_comp[fid], q1, q2))
    if sub_list and len(sub_list) == len(subs):
      folded_sub_segs[idx] = sub_list

  has_explicit_axials = any(
      c.mountain_or_valley() and c.elevation is not None for c in creases
  )
  ref_level_by_seg: dict[tuple[RatPoint, RatPoint], int] = {}
  src_for_ref = (
      unshifted_reference.creases
      if unshifted_reference is not None
      else creases
  )
  for rc in src_for_ref:
    if rc.mountain_or_valley() and rc.elevation in (0, 1):
      rp1 = _to_rat_point(rc.p1, grid_size)
      rp2 = _to_rat_point(rc.p2, grid_size)
      key = (rp1, rp2) if rp1 <= rp2 else (rp2, rp1)
      ref_level_by_seg[key] = int(rc.elevation)

  comp_ids: set[int] = set()
  for sub_list in folded_sub_segs.values():
    for comp_id, _, _ in sub_list:
      comp_ids.add(comp_id)
  comp_axis: dict[int, int] = {}
  comp_sign_origin: dict[int, tuple[int, int]] = {}

  def _const_coord_on_axis(
      idx: int, axis: int, cid: int
  ) -> Fraction | None:
    sub_list = folded_sub_segs.get(idx)
    if not sub_list:
      return None
    val: Fraction | None = None
    for c_cid, q1, q2 in sub_list:
      if c_cid != cid or q1[axis] != q2[axis] or q1[axis].denominator != 1:
        return None
      if val is None:
        val = q1[axis]
      elif val != q1[axis]:
        return None
    return val

  for cid in comp_ids:
    if has_explicit_axials:
      cand_idxs = [
          idx
          for idx, sub_list in folded_sub_segs.items()
          if sub_list[0][0] == cid
          and creases[idx].mountain_or_valley()
          and creases[idx].elevation is not None
      ]
    else:
      cand_idxs = [
          idx
          for idx, sub_list in folded_sub_segs.items()
          if sub_list[0][0] == cid and creases[idx].mountain_or_valley()
      ]
    if not cand_idxs:
      continue

    const_counts = [0, 0]
    spans: list[Fraction] = [Fraction(0), Fraction(0)]
    for axis in (0, 1):
      qs: list[Fraction] = []
      for sub_list in folded_sub_segs.values():
        for c_cid, q1, q2 in sub_list:
          if c_cid == cid:
            qs.extend((q1[axis], q2[axis]))
      if qs:
        spans[axis] = max(qs) - min(qs)
      for idx in cand_idxs:
        if _const_coord_on_axis(idx, axis, cid) is not None:
          const_counts[axis] += 1

    if has_explicit_axials:
      level_axis = 0 if const_counts[0] >= const_counts[1] else 1
    else:
      if const_counts[0] > 0 and const_counts[1] == 0:
        level_axis = 0
      elif const_counts[1] > 0 and const_counts[0] == 0:
        level_axis = 1
      else:
        level_axis = 0 if spans[0] <= spans[1] else 1
    comp_axis[cid] = level_axis

    q_for_zero: int | None = None
    q_for_one: int | None = None
    for idx in folded_sub_segs:
      q_val = _const_coord_on_axis(idx, level_axis, cid)
      if q_val is None:
        continue
      c = creases[idx]
      p1 = _to_rat_point(c.p1, grid_size)
      p2 = _to_rat_point(c.p2, grid_size)
      key = (p1, p2) if p1 <= p2 else (p2, p1)
      ref_el = ref_level_by_seg.get(key)
      if ref_el == 0 and q_for_zero is None:
        q_for_zero = int(q_val)
      elif ref_el == 1 and q_for_one is None:
        q_for_one = int(q_val)
      if (
          q_for_zero is not None
          and q_for_one is not None
          and abs(q_for_one - q_for_zero) == 1
      ):
        break

    if (
        q_for_zero is not None
        and q_for_one is not None
        and abs(q_for_one - q_for_zero) == 1
    ):
      sgn = q_for_one - q_for_zero
      comp_sign_origin[cid] = (sgn, sgn * q_for_zero)
    else:
      border_qs: list[int] = []
      for idx in folded_sub_segs:
        if creases[idx].fold != crease_lib.Fold.BORDER:
          continue
        qv = _const_coord_on_axis(idx, level_axis, cid)
        if qv is not None:
          border_qs.append(int(qv))
      mv_qs: list[int] = []
      for idx in cand_idxs:
        qv = _const_coord_on_axis(idx, level_axis, cid)
        if qv is not None:
          mv_qs.append(int(qv))
      origin_q = min(border_qs) if border_qs else (min(mv_qs) if mv_qs else 0)
      comp_sign_origin[cid] = (1, origin_q)

  updated: list[crease_lib.Crease] = []
  for idx, c in enumerate(creases):
    if not c.mountain_or_valley() or idx not in folded_sub_segs:
      updated.append(
          dataclasses.replace(c, elevation=None)
          if c.elevation is not None
          else c
      )
      continue
    cid = folded_sub_segs[idx][0][0]
    if cid not in comp_axis or cid not in comp_sign_origin:
      updated.append(dataclasses.replace(c, elevation=None))
      continue
    level_axis = comp_axis[cid]
    q_val = _const_coord_on_axis(idx, level_axis, cid)
    if q_val is None:
      updated.append(dataclasses.replace(c, elevation=None))
      continue
    sgn, origin = comp_sign_origin[cid]
    elev = sgn * int(q_val) - origin
    updated.append(dataclasses.replace(c, elevation=elev))

  return cp_lib.CreasePattern(
      creases=updated,
      grid_size=pattern.grid_size,
      light_mode=True,
      tolerance=pattern.tolerance,
  )


def _strip_flap_boundary_hinge_elevations(
    creases: Sequence[crease_lib.Crease],
    elevated_flaps: Collection[utils_ls.LevelShiftable],
) -> list[crease_lib.Crease]:
  """Clears `elevation` on non-border boundary hinges of `elevated_flaps`."""
  boundary_edges: list[crease_lib.Crease] = []
  crimp_edges: list[crease_lib.Crease] = []
  for item in elevated_flaps:
    atoms: Sequence[Any] = (
        (item.original, item.copy)
        if isinstance(item, utils_ls.MirroredElevatedFlapPair)
        else (item,)
    )
    for ef in atoms:
      boundary_edges.extend(ef.main_flap.edges)
      for ls_f in ef.ls_flaps:
        crimp_edges.append(ls_f.crimp_hinge)
  if not boundary_edges:
    return list(creases)
  cleaned: list[crease_lib.Crease] = []
  for c in creases:
    if (
        c.fold != crease_lib.Fold.BORDER
        and c.elevation is not None
        and any(c.is_overlapping(be) for be in boundary_edges)
        and not any(c.is_overlapping(ce) for ce in crimp_edges)
    ):
      cleaned.append(dataclasses.replace(c, elevation=None))
    else:
      cleaned.append(c)
  return cleaned


def _repair_central_crease_polarities(
    raw_creases: list[crease_lib.Crease],
    tol: float = 1e-6,
) -> list[crease_lib.Crease]:
  """Repairs `x = 0.5` M/V polarities after mirrored horizontal shifts."""
  vert_idxs: list[int] = []
  split_ys: set[float] = set()
  for idx, c in enumerate(raw_creases):
    if not c.mountain_or_valley():
      continue
    if abs(c.p1.x - 0.5) <= tol and abs(c.p2.x - 0.5) <= tol:
      vert_idxs.append(idx)
      split_ys.add(round(float(c.p1.y), 6))
      split_ys.add(round(float(c.p2.y), 6))
    else:
      for pt in (c.p1, c.p2):
        if abs(pt.x - 0.5) <= tol:
          split_ys.add(round(float(pt.y), 6))
      if (
          abs(c.p1.y - c.p2.y) <= tol
          and min(c.p1.x, c.p2.x) < 0.5 - tol
          and max(c.p1.x, c.p2.x) > 0.5 + tol
      ):
        split_ys.add(round(float(c.p1.y), 6))

  if not vert_idxs:
    return raw_creases

  vert_set = set(vert_idxs)
  other_creases = [c for i, c in enumerate(raw_creases) if i not in vert_set]
  sorted_ys = sorted(split_ys)

  central_segs: list[crease_lib.Crease] = []
  for idx in vert_idxs:
    vc = raw_creases[idx]
    ylo, yhi = min(vc.p1.y, vc.p2.y), max(vc.p1.y, vc.p2.y)
    sub_ys = [ylo] + [y for y in sorted_ys if ylo + tol < y < yhi - tol] + [yhi]
    for ya, yb in zip(sub_ys, sub_ys[1:]):
      central_segs.append(
          crease_lib.Crease(
              point_lib.Point(x=0.5, y=ya, require_on_paper=False),
              point_lib.Point(x=0.5, y=yb, require_on_paper=False),
              fold=vc.fold,
              fold_percent=vc.fold_percent,
              elevation=vc.elevation,
          )
      )
  central_segs.sort(key=lambda s: (min(s.p1.y, s.p2.y), max(s.p1.y, s.p2.y)))
  if len(central_segs) < 2:
    return raw_creases

  for i in range(len(central_segs) - 1):
    below = central_segs[i]
    above = central_segs[i + 1]
    y_join = max(below.p1.y, below.p2.y)
    if abs(min(above.p1.y, above.p2.y) - y_join) > tol:
      continue
    inc_folds: list[crease_lib.Fold] = []
    for oc in other_creases:
      if not oc.mountain_or_valley():
        continue
      touches = (
          abs(oc.p1.x - 0.5) <= tol and abs(oc.p1.y - y_join) <= tol
      ) or (abs(oc.p2.x - 0.5) <= tol and abs(oc.p2.y - y_join) <= tol)
      if touches:
        inc_folds.append(oc.fold)
        continue
      x1, y1 = float(oc.p1.x), float(oc.p1.y)
      x2, y2 = float(oc.p2.x), float(oc.p2.y)
      if min(x1, x2) < 0.5 - tol and max(x1, x2) > 0.5 + tol:
        t_cross = (0.5 - x1) / (x2 - x1)
        y_cross = y1 + t_cross * (y2 - y1)
        if abs(y_cross - y_join) <= tol:
          inc_folds.extend((oc.fold, oc.fold))
    if len(inc_folds) < 2 or len(inc_folds) % 2 != 0:
      continue
    m_cnt = sum(1 for f in inc_folds if f == crease_lib.Fold.MOUNTAIN)
    v_cnt = len(inc_folds) - m_cnt
    b_m = 1 if below.fold == crease_lib.Fold.MOUNTAIN else -1
    a_m = 1 if above.fold == crease_lib.Fold.MOUNTAIN else -1
    if (
        abs((m_cnt - v_cnt) + b_m + a_m) != 2
        and abs((m_cnt - v_cnt) + b_m - a_m) == 2
    ):
      central_segs[i + 1] = dataclasses.replace(above, fold=above.fold.flip())

  return other_creases + central_segs


def _extract_applied_shifts_for_flap(
    cp: cp_lib.CreasePattern,
    flap: utils_ls.LevelShiftable,
    partner_by_name: Mapping[str, str] | None = None,
) -> tuple[bus.AppliedLevelShift, ...]:
  """Extracts `AppliedLevelShift` records for `flap` from `cp`."""
  grid_size = cp.grid_size
  atoms = (
      (flap.original, flap.copy)
      if isinstance(flap, utils_ls.MirroredElevatedFlapPair)
      else (flap,)
  )
  by_stick_and_level: dict[tuple[str, int], list[bus.GridSegment]] = (
      collections.defaultdict(list)
  )
  stick_order: list[str] = []
  for ef in atoms:
    raw_name = ef.main_flap.name
    if raw_name.endswith('_copy'):
      base = raw_name.removesuffix('_copy')
      stick_name = partner_by_name.get(base, base) if partner_by_name else base
    else:
      stick_name = raw_name
    if stick_name not in stick_order:
      stick_order.append(stick_name)
    for c in cp.creases:
      if (
          c.mountain_or_valley()
          and c.elevation is not None
          and int(c.elevation) not in (0, 1)
          and ef.main_flap.crease_is_within(c, strict=False)
      ):
        seg = bus.GridSegment.from_normalized_crease(c, grid_size)
        by_stick_and_level[(stick_name, int(c.elevation))].append(seg)
  records: list[bus.AppliedLevelShift] = []
  for stick_name in stick_order:
    levels = sorted(
        lvl for s_key, lvl in by_stick_and_level if s_key == stick_name
    )
    for lvl in levels:
      segs = tuple(
          sorted(
              set(by_stick_and_level[(stick_name, lvl)]),
              key=lambda s: (s.p1.x, s.p1.y),
          )
      )
      records.append(
          bus.AppliedLevelShift(stick=stick_name, level=lvl, axial_creases=segs)
      )
  return tuple(records)


def apply_pocket_level_shifter_with_record(
    pattern: cp_lib.CreasePattern,
    flap: utils_ls.LevelShiftable,
    position_percent: float = 0.5,
    *,
    direction: str = 'auto',
    check_global_flat_foldability: bool = True,
    partner_by_name: Mapping[str, str] | None = None,
) -> tuple[cp_lib.CreasePattern, tuple[bus.AppliedLevelShift, ...]]:
  """Applies sacrificial-pocket level shifter(s) and verifies foldability.

  Args:
    pattern: `LS_TEMPLATE` crease pattern containing sacrificial pocket creases.
    flap: Target `ElevatedFlap` or `MirroredElevatedFlapPair`.
    position_percent: Relative position in `[0, 1]` along valid pocket slots.
    direction: One of `'auto'`, `'up'`, or `'down'`.
    check_global_flat_foldability: Whether to require global layer-ordering
      flat-foldability via `hinges.valid_crease_pattern`.
    partner_by_name: Optional symmetric partner mapping for `_copy` rivers.

  Returns:
    `(shifted_cp, applied_shifts)` with exact `Crease.elevation` on axial
    creases only.

  Raises:
    ValueError: If `direction` is invalid or no pocket position produces a
      globally flat-foldable crease pattern.
  """
  if direction not in _VALID_SHIFT_DIRECTIONS:
    raise ValueError(
        f'direction must be one of {_VALID_SHIFT_DIRECTIONS}, got'
        f' {direction!r}.'
    )

  cleaned_in = cp_lib.CreasePattern(
      creases=_strip_flap_boundary_hinge_elevations(pattern.creases, [flap]),
      grid_size=pattern.grid_size,
      light_mode=True,
      tolerance=pattern.tolerance,
  )
  prepared = compute_axial_elevations(cleaned_in)
  grid_size = prepared.grid_size

  axial_map = flap.axial_paths(prepared.creases)
  valid = sorted(flap.valid_positions(axial_map.keys()))
  if not valid:
    raise ValueError(
        f'No valid level-shifter positions found for {flap.main_flap.name}.'
    )

  def _pos_dir(pos: utils_ls.LSPosition) -> str:
    segs = axial_map.get(pos, [])
    if segs and segs[0].elevation is not None:
      return 'up' if int(segs[0].elevation) <= 0 else 'down'
    raise ValueError(
        f'Axial crease at position {pos} on {flap.main_flap.name} has no'
        ' computed elevation.'
    )

  def _finalize_and_check(
      raw_creases: list[crease_lib.Crease],
  ) -> cp_lib.CreasePattern | None:
    merged_cp = cp_lib.CreasePattern(
        creases=raw_creases,
        grid_size=grid_size,
        light_mode=False,
        merge_trivial=True,
        tolerance=pattern.tolerance,
    )
    if not merged_cp.local_flat_foldability():
      repaired_raw = _repair_central_crease_polarities(
          raw_creases, tol=max(pattern.tolerance * 10.0, 1e-6)
      )
      if repaired_raw is not raw_creases:
        repaired_cp = cp_lib.CreasePattern(
            creases=repaired_raw,
            grid_size=grid_size,
            light_mode=False,
            merge_trivial=True,
            tolerance=pattern.tolerance,
        )
        if repaired_cp.local_flat_foldability():
          merged_cp = repaired_cp
    if not hinge_utils.valid_crease_pattern(
        merged_cp,
        grid_size,
        local_flat_foldability=True,
        global_flat_foldability=check_global_flat_foldability,
    ):
      return None
    return compute_axial_elevations(merged_cp, unshifted_reference=prepared)

  dir_valid = (
      valid
      if direction == 'auto'
      else [p for p in valid if _pos_dir(p) == direction]
  )
  if not dir_valid:
    raise ValueError(
        f'No {direction!r} level-shifter positions found for'
        f' {flap.main_flap.name} among valid positions {valid}.'
    )

  chosen = utils_ls.get_ls_position_from_percent(
      dir_valid, float(position_percent)
  )
  chosen_idx = dir_valid.index(chosen) if chosen in dir_valid else 0
  ordered_cands = sorted(
      range(len(dir_valid)), key=lambda i: (abs(i - chosen_idx), i)
  )
  last_err = None
  for cand_idx in ordered_cands:
    pos_cand = dir_valid[cand_idx]
    try:
      cand_creases = flap.apply_level_shifter(list(prepared.creases), pos_cand)
      cand_cp = _finalize_and_check(cand_creases)
      if cand_cp is not None:
        return cand_cp, _extract_applied_shifts_for_flap(
            cand_cp, flap, partner_by_name
        )
    except (NotImplementedError, ValueError, KeyError, IndexError) as exc:
      last_err = exc

  raise ValueError(
      f'All {direction!r} level-shifter positions {dir_valid} failed'
      f' flat-foldability for {flap.main_flap.name}: {last_err}'
  ) from last_err


def _rebuild_widened_final_state(
    sol: hinge_utils.SolverResult,
    widened_cp: cp_lib.CreasePattern,
    packing_cp: cp_lib.CreasePattern,
    packing_out: packing_output_lib.PackingOutput,
) -> hinge_utils.SolverStateSnapshot:
  """Rebuilds `hinges`, `pleats` (with integer `elevation`), and `ridges`."""
  hinge_lines = [
      h.coordinate
      for h in (packing_out.hinges or ())
      if h.coordinate is not None
  ]
  hinge_lines.extend(
      c for c in packing_cp.creases if c.fold == crease_lib.Fold.HINGE
  )
  prev_hinges = list(sol.final_state.hinges)
  widened_hinges: list[crease_lib.Crease] = []
  widened_pleats: list[crease_lib.Crease] = []
  widened_ridges: list[crease_lib.Crease] = []
  for c in widened_cp.creases:
    if c.fold == crease_lib.Fold.BORDER:
      continue
    if c.is_diagonal or (not c.is_horizontal and not c.is_vertical):
      widened_ridges.append(c)
    elif any(
        c.is_within(hl, match_fold=False) for hl in hinge_lines
    ) or any(c.is_overlapping(ph) for ph in prev_hinges):
      widened_hinges.append(c)
    elif c.elevation is None:
      widened_ridges.append(c)
    else:
      widened_pleats.append(c)
  return dataclasses.replace(
      sol.final_state,
      hinges=widened_hinges,
      pleats=widened_pleats,
      ridges=widened_ridges,
  )


def solve_with_pocket_level_shifters(
    packing: bus.PackedCreasePattern | packing_output_lib.PackingOutput,
    solver_config: hinge_utils.SolverConfig,
    *,
    scale_factor: int = 1,
    resolved_solver: str | None = None,
    record_stages: bool = False,
) -> list[
    tuple[
        hinge_utils.SolverResult,
        hinge_utils.SolverConfig,
        tuple[bus.AppliedLevelShift, ...],
        tuple[bus.StageSnapshot, ...],
    ]
]:
  """Shared two-stage pocket level-shifter solver for `api.py` & `pipeline.py`.

  Solves sacrificial preassignments into `LS_TEMPLATE` bases and widens every
  elevated stick into a globally flat-foldable `LEVEL_SHIFTED` base with exact
  axial elevations.

  Note on effective `SolverConfig` overrides: sacrificial crimp/ridge
  preassignments conflict with stick-figure hinge preassignments, so the solver
  runs with `hinges_from_stick_figure=False`,
  `allow_misaligned_symmetric_hinges=True`, and `max_time_limit` capped at
  `20.0`s; the returned `used_config` records these effective values truthfully.

  Args:
    packing: `PackedCreasePattern` or legacy `PackingOutput` with pockets.
    solver_config: Requested `SolverConfig` (`remove_central_crease=False`
      required).
    scale_factor: Grid scale factor (must be `1` for pocket level shifters).
    resolved_solver: Resolved solver plugin name (`'builtin'` required).
    record_stages: If True, returns the 7 standard solver stages plus
      `'LS Template CP'` in each result's `stages` tuple.

  Returns:
    List of `(widened_result, used_config, applied_shifts, stages)` tuples.

  Raises:
    ValueError: If `resolved_solver != 'builtin'`, `scale_factor != 1`,
      `solver_config.remove_central_crease` is True, or pocket metadata is
      invalid.
  """
  eff_solver = resolved_solver or plugins.resolve_solver(solver_config.solver)
  if eff_solver != 'builtin':
    raise ValueError(
        "Level-shifter pocket solving requires solver='builtin', got"
        f' {eff_solver!r}.'
    )
  if scale_factor != 1:
    raise ValueError(
        'Level-shifter pocket solving requires scale_factor=1, got'
        f' {scale_factor}.'
    )
  if solver_config.remove_central_crease:
    raise ValueError(
        'Level-shifter pocket solving does not support'
        ' remove_central_crease=True.'
    )

  if isinstance(packing, bus.PackedCreasePattern):
    packing_out = packing.to_packing_output()
    packing_cp = packing.creases
    grid_size = packing.grid_size
    sym_pairs: dict[str, str] | None = None
    if packing.design is not None and packing.design.symmetric_partner:
      sym_pairs = {
          k: v
          for k, v in packing.design.symmetric_partner.items()
          if v and k != v
      }
    elif packing.packed_tree is not None:
      sym_pairs = {
          s.name: s.symmetric_stick_name
          for s in packing.packed_tree.children
          if s.symmetric_stick_name is not None
          and s.symmetric_stick_name != s.name
      }
  else:
    packing_out = packing
    if packing_out.packing_cp is None:
      raise ValueError('PackingOutput.packing_cp is None.')
    packing_cp = packing_out.packing_cp
    grid_size = packing_cp.grid_size
    sym_pairs = None
    if packing_out.stick_figure is not None:
      sym_pairs = {
          s.name: s.symmetric_stick_name
          for s in packing_out.stick_figure.children
          if s.symmetric_stick_name is not None
          and s.symmetric_stick_name != s.name
      }

  elevated = utils_ls.get_elevated_flaps(
      packing_out.flap_rectangles or [],
      packing_out.sacrificial_flaps or [],
      grid_size,
      symmetric_pairs=sym_pairs,
  )
  collected: list[hinge_utils.SolverResult] = []
  ext_cb = solver_config.solution_callback

  def _collect(res: hinge_utils.SolverResult) -> hinge_utils.SolverResult:
    if res not in collected:
      collected.append(res)
    return res

  ls_time_limit = min(float(solver_config.max_time_limit or 20.0), 20.0)
  base_ls_cfg = dataclasses.replace(
      solver_config,
      grid_size=grid_size,
      max_time_limit=ls_time_limit,
      packing_information=None,
      hinges_from_stick_figure=False,
      allow_misaligned_symmetric_hinges=True,
      solution_callback=_collect,
  )
  used_cfg = base_ls_cfg
  initial_stages: list[tuple[str, cp_lib.CreasePattern]] = []
  for use_part in dict.fromkeys((base_ls_cfg.use_partition_order, False)):
    try_cfg = dataclasses.replace(base_ls_cfg, use_partition_order=use_part)
    used_cfg = try_cfg
    try:
      utils_ls.solve_with_ls_templates(
          try_cfg,
          packing_cp,
          elevated,
          plot_stages=False,
          record_stages=record_stages,
          initial_stages_out=initial_stages if record_stages else None,
      )
    except hinge_utils.NoSolutionError:
      pass
    if collected:
      break

  if not collected:
    return []

  processed = used_cfg.process_solutions(collected)
  results: list[
      tuple[
          hinge_utils.SolverResult,
          hinge_utils.SolverConfig,
          tuple[bus.AppliedLevelShift, ...],
          tuple[bus.StageSnapshot, ...],
      ]
  ] = []
  for sol in processed:
    ls_template_cp = compute_axial_elevations(sol.crease_pattern)
    cur_cp = ls_template_cp
    all_shifts: list[bus.AppliedLevelShift] = []
    ok = True
    for ef in elevated:
      try:
        cur_cp, shifts = apply_pocket_level_shifter_with_record(
            cur_cp,
            ef,
            position_percent=0.5,
            direction='auto',
            check_global_flat_foldability=True,
            partner_by_name=sym_pairs,
        )
        all_shifts.extend(shifts)
      except ValueError:
        ok = False
        break
    if not ok:
      continue
    if not used_cfg.is_valid_crease_pattern(cur_cp):
      continue
    widened_state = _rebuild_widened_final_state(
        sol, cur_cp, packing_cp, packing_out
    )
    widened_sol = dataclasses.replace(
        sol,
        crease_pattern=cur_cp,
        final_state=widened_state,
        base_type=hinge_utils.BaseType.LEVEL_SHIFTED,
        is_globally_flat_foldable=True,
    )
    if ext_cb is not None:
      cb_res = ext_cb(widened_sol)
      if cb_res is None:
        continue
      widened_sol = cb_res
    stages_out: tuple[bus.StageSnapshot, ...] = ()
    if record_stages:
      stage_pairs = [
          *initial_stages,
          (bus.LS_TEMPLATE_STAGE_NAME, ls_template_cp),
          ('(7) Final Solving CP', widened_sol.crease_pattern),
      ]
      stages_out = tuple(
          bus.StageSnapshot(name=s_name, creases=s_cp, grid_size=grid_size)
          for s_name, s_cp in stage_pairs
      )
    results.append((widened_sol, used_cfg, tuple(all_shifts), stages_out))
  return results
