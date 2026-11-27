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

"""Matplotlib multi-panel visualizers for stick figures and pipeline stages."""

from __future__ import annotations

from collections.abc import Sequence
import pathlib
import typing
import xml.etree.ElementTree as ET

from . import bus
from . import render_3d
from .crease_pattern import crease_pattern as cp_lib
from .shaping import geometric_simple_fold
from .third_party.flat_folder import fold_cp
from .stick_figures import stick_figure as sf_lib

if typing.TYPE_CHECKING:
  from matplotlib import axes  # pylint: disable=g-import-not-at-top,g-bad-import-order


def save_stick_figure_views(
    target: sf_lib.StickFigure | bus.DesignTree,
    out_path: str | pathlib.Path,
    *,
    figsize: int = 5,
    dpi: int = 150,
) -> pathlib.Path:
  """Saves a 4-view (`top`, `side`, `front`, `isometric`) skeleton sheet PNG.

  Args:
    target: `StickFigure` or `DesignTree` to render.
    out_path: Destination PNG path.
    figsize: Subplot width/height in inches for each of the 4 panels.
    dpi: Output PNG resolution in dots per inch.

  Returns:
    The resolved `pathlib.Path` written to disk.
  """
  tree = target.source if isinstance(target, bus.DesignTree) else target
  dest = pathlib.Path(out_path)
  dest.parent.mkdir(parents=True, exist_ok=True)
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  from .stick_figures import visualize as sf_visualize  # pylint: disable=g-import-not-at-top

  fig = sf_visualize.visualize_stick_from_all_views(tree, figsize=figsize)
  try:
    fig.savefig(dest, dpi=dpi, bbox_inches='tight')
  finally:
    plt.close(fig)
  return dest


def _draw_folded_svg_on_ax(
    ax: axes.Axes,
    svg_str: str | None,
    title: str,
    conflict_reason: str | None = None,
) -> None:
  """Renders a FlatFolder SVG string onto a Matplotlib `Axes`."""
  ax.set_facecolor('#f1f5f9')
  ax.set_aspect('equal', adjustable='box')
  ax.set_title(title, fontsize=11, fontweight='bold')
  ax.set_xticks([])
  ax.set_yticks([])

  if not svg_str or '<polygon' not in svg_str:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    msg = conflict_reason or 'No flat-folded state available'
    ax.text(
        0.5,
        0.5,
        f'Flat-folder:\n{msg}',
        ha='center',
        va='center',
        fontsize=9,
        color='#b91c1c',
        wrap=True,
    )
    return

  root = ET.fromstring(svg_str)
  from matplotlib import patches  # pylint: disable=g-import-not-at-top

  for elem in root.iter():
    tag = elem.tag.split('}')[-1]
    if tag == 'polygon':
      pts_str = elem.attrib.get('points', '').strip()
      if not pts_str:
        continue
      coords: list[tuple[float, float]] = []
      for pair in pts_str.split():
        x_s, y_s = pair.split(',')
        coords.append((float(x_s), float(y_s)))
      fill = elem.attrib.get('fill', '#AAA')
      stroke = elem.attrib.get('stroke', '#000')
      lw = float(elem.attrib.get('stroke-width', '0.5'))
      poly = patches.Polygon(
          coords,
          closed=True,
          facecolor=fill,
          edgecolor=stroke,
          linewidth=lw,
      )
      ax.add_patch(poly)
    elif tag == 'line':
      x1 = float(elem.attrib.get('x1', '0'))
      y1 = float(elem.attrib.get('y1', '0'))
      x2 = float(elem.attrib.get('x2', '0'))
      y2 = float(elem.attrib.get('y2', '0'))
      stroke = elem.attrib.get('stroke', '#000')
      lw = float(elem.attrib.get('stroke-width', '1.0'))
      ax.plot([x1, x2], [y1, y2], color=stroke, linewidth=lw)

  ax.set_xlim(-40, 1040)
  ax.set_ylim(1040, -40)


def _draw_packed_cp_on_ax(
    ax: axes.Axes,
    packed: bus.PackedCreasePattern,
    lw: float = 0.65,
) -> None:
  """Draws a `PackedCreasePattern` with colored flap and river overlays."""
  from matplotlib import patches  # pylint: disable=g-import-not-at-top
  from .crease_pattern import visualize as cp_visualize  # pylint: disable=g-import-not-at-top

  gs = float(packed.grid_size) if packed.grid_size > 0 else 1.0
  cp_visualize.draw_cp(
      packed.creases,
      ax=ax,
      grid_size=packed.grid_size,
      show_only_not_flat_foldable_vertices=False,
      show_vertices=False,
      annotate_vertices=False,
      annotate_flat_foldability=False,
      lw=lw,
  )
  ax.set_xticks([])
  ax.set_yticks([])
  for river in packed.rivers.values():
    for rect in river.rects:
      ax.add_patch(
          patches.Rectangle(
              (rect.x / gs, rect.y / gs),
              rect.w / gs,
              rect.h / gs,
              facecolor='#f59e0b',
              edgecolor='#d97706',
              alpha=0.22,
              linewidth=1.0,
          )
      )
    if river.rects:
      mid = river.rects[len(river.rects) // 2]
      ax.text(
          (mid.x + mid.w / 2.0) / gs,
          (mid.y + mid.h / 2.0) / gs,
          f'{river.id}\n(w={river.width})',
          ha='center',
          va='center',
          fontsize=6.5,
          color='#92400e',
          bbox={
              'boxstyle': 'round,pad=0.15',
              'fc': 'white',
              'ec': 'none',
              'alpha': 0.75,
          },
      )

  for flap in packed.flaps.values():
    rect = flap.rect
    ax.add_patch(
        patches.Rectangle(
            (rect.x / gs, rect.y / gs),
            rect.w / gs,
            rect.h / gs,
            facecolor='#3b82f6',
            edgecolor='#1d4ed8',
            alpha=0.22,
            linewidth=1.2,
        )
    )
    ax.text(
        (rect.x + rect.w / 2.0) / gs,
        (rect.y + rect.h / 2.0) / gs,
        f'{flap.id}\n({rect.w}x{rect.h})',
        ha='center',
        va='center',
        fontsize=7.0,
        fontweight='bold',
        color='#1e3a8a',
        bbox={
            'boxstyle': 'round,pad=0.15',
            'fc': 'white',
            'ec': 'none',
            'alpha': 0.8,
        },
    )
  ax.set_title(
      f'1. Packed CP (Grid {packed.grid_size}x{packed.grid_size}, '
      f'{len(packed.flaps)} flaps, {len(packed.rivers)} rivers)',
      fontsize=11,
      fontweight='bold',
  )


def _draw_kinematic_folded_cp_on_ax(
    ax: axes.Axes,
    cp: cp_lib.CreasePattern,
    title: str,
    *,
    color_top: Sequence[int] | None = None,
    color_bottom: Sequence[int] | None = None,
) -> None:
  """Renders the 2D kinematic folded state of a `CreasePattern` onto `ax`."""
  ax.set_facecolor('#f1f5f9')
  ax.set_aspect('equal', adjustable='box')
  ax.set_title(title, fontsize=11, fontweight='bold')
  ax.set_xticks([])
  ax.set_yticks([])

  try:
    graph_data = geometric_simple_fold.prepare_graph_data(cp)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    ax.text(
        0.5, 0.5, f'Kinematic fold failed:\n{exc}', ha='center', va='center'
    )
    return

  folded_verts = graph_data['folded_vertices']
  face_verts = graph_data['face_vertices']
  face_orients = graph_data['face_orientations']
  edge_verts = graph_data['edge_vertices']

  valid_pts = [p for p in folded_verts if p is not None]
  if not valid_pts:
    ax.text(0.5, 0.5, 'No folded vertices', ha='center', va='center')
    return

  top_rgb = (
      tuple(c / 255.0 for c in color_top)
      if color_top and len(color_top) == 3
      else (0.58, 0.64, 0.72)
  )
  bot_rgb = (
      tuple(c / 255.0 for c in color_bottom)
      if color_bottom and len(color_bottom) == 3
      else (0.95, 0.96, 0.98)
  )

  for f_idx, v_indices in enumerate(face_verts):
    from matplotlib import patches  # pylint: disable=g-import-not-at-top

    pts: list[tuple[float, float]] = []
    for v_idx in v_indices:
      if v_idx < len(folded_verts) and folded_verts[v_idx] is not None:
        pts.append((folded_verts[v_idx][0], folded_verts[v_idx][1]))
    if len(pts) < 3:
      continue
    orient = face_orients[f_idx] if f_idx < len(face_orients) else True
    fc = top_rgb if orient else bot_rgb
    poly = patches.Polygon(
        pts,
        closed=True,
        facecolor=fc,
        edgecolor='#1e293b',
        linewidth=0.6,
        alpha=0.85,
    )
    ax.add_patch(poly)

  for v1_idx, v2_idx in edge_verts:
    if v1_idx >= len(folded_verts) or v2_idx >= len(folded_verts):
      continue
    p1 = folded_verts[v1_idx]
    p2 = folded_verts[v2_idx]
    if p1 is None or p2 is None:
      continue
    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        color='#0f172a',
        linewidth=0.8,
        alpha=0.75,
    )

  xs = [p[0] for p in valid_pts]
  ys = [p[1] for p in valid_pts]
  min_x, max_x = min(xs), max(xs)
  min_y, max_y = min(ys), max(ys)
  span = max(max_x - min_x, max_y - min_y, 1e-3)
  pad = span * 0.08
  cx = (min_x + max_x) / 2.0
  cy = (min_y + max_y) / 2.0
  half = span / 2.0 + pad
  ax.set_xlim(cx - half, cx + half)
  ax.set_ylim(cy + half, cy - half)


def save_pipeline_summary(
    out_path: str | pathlib.Path,
    *,
    packed: bus.PackedCreasePattern,
    solved: bus.SolvedCreasePattern | None = None,
    shaped: bus.ShapedCreasePattern | None = None,
    name: str | None = None,
    solved_ff: fold_cp.FlatFolderFOLD | None = None,
    shaped_ff: fold_cp.FlatFolderFOLD | None = None,
    shaped_mesh_3d: render_3d.FoldedMesh3D | None = None,
    color_top: Sequence[int] | None = None,
    color_bottom: Sequence[int] | None = None,
    lw: float = 0.6,
    dpi: int = 180,
) -> pathlib.Path:
  """Saves a 5-panel pipeline summary PNG across packing, solving, and shaping.

  Panels:
    1. Packed CP with flap and river overlays.
    2. Solved Base CP with flat-foldability badge.
    3. 2D Flat-Folded Base (uniaxial Flat-Folder SVG or kinematic fallback).
    4. Shaped CP with partial-fold opacity creases.
    5. Shaded 3D Shaped Figure (isometric hinge-folded mesh, or 2D fallback).

  Args:
    out_path: Destination PNG path.
    packed: `PackedCreasePattern` stage record.
    solved: Optional `SolvedCreasePattern` stage record.
    shaped: Optional `ShapedCreasePattern` stage record.
    name: Optional model title (inferred from `packed.design` if omitted).
    solved_ff: Optional precomputed `FlatFolderFOLD` for `solved`.
    shaped_ff: Optional precomputed `FlatFolderFOLD` for `shaped`.
    shaped_mesh_3d: Optional precomputed `FoldedMesh3D` for `shaped` (built
      automatically when `shaped` is provided and `shaped_mesh_3d` is `None`).
    color_top: Optional RGB `[0..255]` paper top color.
    color_bottom: Optional RGB `[0..255]` paper bottom color.
    lw: Crease line width on crease-pattern panels.
    dpi: Output PNG resolution in dots per inch.

  Returns:
    The resolved `pathlib.Path` written to disk.
  """
  source_tree = None
  if packed.design is not None:
    source_tree = packed.design.source
  elif packed.packed_tree is not None:
    source_tree = packed.packed_tree

  resolved_name = (
      name
      or (source_tree.name if source_tree and source_tree.name else None)
      or 'model'
  )
  resolved_top = color_top
  if resolved_top is None and source_tree is not None:
    resolved_top = getattr(source_tree, 'color_top', None)
  resolved_bottom = color_bottom
  if resolved_bottom is None and source_tree is not None:
    resolved_bottom = getattr(source_tree, 'color_bottom', None)

  resolved_mesh = shaped_mesh_3d
  if resolved_mesh is None and shaped is not None:
    try:
      resolved_mesh = render_3d.build_folded_mesh_3d(shaped)
    except ValueError:
      resolved_mesh = None

  dest = pathlib.Path(out_path)
  dest.parent.mkdir(parents=True, exist_ok=True)
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  from .crease_pattern import visualize as cp_visualize  # pylint: disable=g-import-not-at-top

  fig, axs = plt.subplots(1, 5, figsize=(26, 5.4))
  fig.suptitle(
      f'Origami Pipeline Summary: {resolved_name} '
      f'(Grid {packed.grid_size}x{packed.grid_size})',
      fontsize=14,
      fontweight='bold',
      y=1.02,
  )

  _draw_packed_cp_on_ax(axs[0], packed, lw=lw)

  if solved is not None:
    cp_visualize.draw_cp(
        solved.creases,
        ax=axs[1],
        grid_size=solved.creases.grid_size,
        show_only_not_flat_foldable_vertices=True,
        show_vertices=False,
        annotate_vertices=False,
        annotate_flat_foldability=True,
        lw=lw,
    )
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    ff_tag = (
        'Global FF: YES'
        if solved.is_globally_flat_foldable
        else 'Global FF: NO'
    )
    cc_tag = 'no-CC' if solved.central_crease_removed else 'with-CC'
    axs[1].set_title(
        f'2. Solved Base CP ({ff_tag}, {cc_tag})',
        fontsize=11,
        fontweight='bold',
    )
  else:
    axs[1].axis('off')
    axs[1].set_title(
        '2. Solved Base CP (Skipped/Failed)', fontsize=11, fontweight='bold'
    )

  if solved is not None:
    if (
        solved_ff is not None
        and solved_ff.folded_state_svg
        and '<polygon' in solved_ff.folded_state_svg
    ):
      _draw_folded_svg_on_ax(
          axs[2],
          solved_ff.folded_state_svg,
          '3. Folded Base (Uniaxial)',
          conflict_reason=solved_ff.conflict_reason,
      )
    else:
      _draw_kinematic_folded_cp_on_ax(
          axs[2],
          solved.creases,
          '3. Folded Base (Kinematic)',
          color_top=resolved_top,
          color_bottom=resolved_bottom,
      )
  else:
    axs[2].axis('off')
    axs[2].set_title('3. Folded Base (N/A)', fontsize=11, fontweight='bold')

  if shaped is not None:
    cp_visualize.draw_cp(
        shaped.creases,
        ax=axs[3],
        grid_size=shaped.creases.grid_size,
        show_only_not_flat_foldable_vertices=True,
        show_vertices=False,
        annotate_vertices=False,
        annotate_flat_foldability=True,
        lw=lw,
    )
    axs[3].set_xticks([])
    axs[3].set_yticks([])
    axs[3].set_title(
        f'4. Shaped CP (strat={shaped.strategy.value})',
        fontsize=11,
        fontweight='bold',
    )
  else:
    axs[3].axis('off')
    axs[3].set_title(
        '4. Shaped CP (Skipped/Failed)', fontsize=11, fontweight='bold'
    )

  if resolved_mesh is not None:
    render_3d.draw_folded_mesh_on_ax(
        axs[4],
        resolved_mesh,
        '5. 3D Shaped Figure (Isometric)',
        color_top=resolved_top,
    )
  elif shaped is not None:
    if (
        shaped_ff is not None
        and shaped_ff.folded_state_svg
        and '<polygon' in shaped_ff.folded_state_svg
    ):
      _draw_folded_svg_on_ax(
          axs[4],
          shaped_ff.folded_state_svg,
          '5. Folded Shaped Model (Global FF)',
      )
    else:
      _draw_kinematic_folded_cp_on_ax(
          axs[4],
          shaped.creases,
          '5. Folded Shaped Model (2D Projection)',
          color_top=resolved_top,
          color_bottom=resolved_bottom,
      )
  else:
    axs[4].axis('off')
    axs[4].set_title(
        '5. 3D Shaped Figure (N/A)', fontsize=11, fontweight='bold'
    )

  try:
    fig.tight_layout()
    fig.savefig(dest, dpi=dpi, bbox_inches='tight')
  finally:
    plt.close(fig)
  return dest


render_pipeline_summary_png = save_pipeline_summary
