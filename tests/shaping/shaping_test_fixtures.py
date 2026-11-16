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

"""Shaping crease-pattern SVG test fixtures for unit tests."""

from collections.abc import Sequence
import dataclasses
import functools
import math
from typing import Any, Optional

from bp_pipeline import rectangle as shaping_utils_rect
from bp_pipeline.crease_pattern import crease as crease_utils
from bp_pipeline.crease_pattern import crease_pattern
from bp_pipeline.crease_pattern import point as point_utils
from bp_pipeline.crease_pattern.parser import svg as parser
from bp_pipeline.shaping import crimp_fold_shaping
from bp_pipeline.shaping import fan_out_shaping
from bp_pipeline.shaping import inscribed_rectangle_shaping
from bp_pipeline.shaping import reverse_fold_shaping
from bp_pipeline.shaping import shaping_elements
from bp_pipeline.shaping import shaping_utils
from bp_pipeline.shaping import types as shaping_types
from bp_pipeline.shaping import zigzag_shaping
from bp_pipeline.stick_figures import stick_figure as stick_figure_utils
from bp_pipeline.stick_figures.examples import example_trees as stick_figure_examples
import cp_test_fixtures
from cp_test_fixtures import *  # pylint: disable=wildcard-import,unused-wildcard-import
import immutabledict
from bp_pipeline import bps_types
from bp_pipeline.packing import river as river_utils
from bp_pipeline.packing import types as packing_types
import numpy as np

BoxPleatingTreeData = bps_types.BoxPleatingTreeData


@dataclasses.dataclass(frozen=True, repr=False)
class BoxPleatingCreasePatternExample:
  """Contains a single variant of a problem.

  Attributes:
    name: The name of the example.
    packing_svg: The SVG string of the packing crease pattern.
    grid_size: The gride size of the box-pleating grid.
    tree_data: The tree data used to generate the packing crease pattern.
    stick_figure: The stick figure used to generate the packing crease pattern.
    unmerged_stick_figure: The unmerged stick figure.
    assigned_pleats_svg: The SVG string of the assigned pleats crease pattern.
    assigned_pleats_cp: The assigned pleats crease pattern.
    assigned_ridges_svg: The SVG string of the assigned ridges crease pattern.
    packing_cp: The packing crease pattern.
    assigned_ridges_cp: The assigned ridges crease pattern.
    base_svg: The SVG string of the base crease pattern.
    base_cp: a base crease pattern corresponding to packing_cp.  # for shaping
      tests
    central_crease_removed: Whether the central flap is removed from the
      base_cp.  For zigzag shaping tests:
      rectangles: Rectangles defining flaps.
      rivers: river shaping objects corresponding to rivers on the cp.
      target_zigzag_depths: The target zigzag depths. If None,
        target_flap_anglesmust be specified.
      target_flap_angles: The target flap angles. If None, target_zigzag_depths
        must be specified.
      fold_orders: a sequence of folds detailing the fold order; for example a
        fold order of [MOUNTAIN, VALLEY] will have the zigzag crease starting
        with a mountain fold and then alternating valley and mountain
        orientations, whereas a fold order of [VALLEY] will have zigzag creases
        be all valley folds.
      even_phases: boolean determining whether zigzags should be in even phase
        or odd phase. Phase determines the phase shift of the zigzags.
      zigzag_shaping_svg_sha: The shaped crease pattern svg sha hash value.  For
        inscribed rectangle shaping tests:
      zigzag_shaping_svg: The svg string of the zigzagshaped crease pattern.
      inscribed_rectangle_shaping_params: params determining incribed rectangle
        placement for each flap.
      inscribed_rectangle_shaping_svg_sha: The incscribe rectangle shaped crease
        pattern svg sha hash value.
      fan_out_shaping_svg: The svg string of the fan out shaped crease pattern.
      fan_out_shaping_svg_sha: The fan out shaped crease pattern svg sha hash
        value.
      zigzag_creases: The list of creases that are part of the zigzag.
  """

  name: str
  packing_svg: str
  grid_size: int
  tree_data: BoxPleatingTreeData | None = None
  stick_figure: stick_figure_utils.StickFigure | None = None
  unmerged_stick_figure: stick_figure_utils.StickFigure | None = None
  assigned_pleats_svg: str | None = None
  assigned_ridges_svg: str | None = None
  base_svg: str | None = None
  central_crease_removed: bool = False

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}(name={self.name!r})"

  @functools.cached_property
  def packing_cp(self) -> crease_pattern.CreasePattern:
    return parser.parse_svg(self.packing_svg, grid_size=self.grid_size)

  @functools.cached_property
  def assigned_pleats_cp(self) -> crease_pattern.CreasePattern | None:
    if self.assigned_pleats_svg is None:
      return None
    return parser.parse_svg(self.assigned_pleats_svg, grid_size=self.grid_size)

  @functools.cached_property
  def assigned_ridges_cp(self) -> crease_pattern.CreasePattern | None:
    if self.assigned_ridges_svg is None:
      return None
    return parser.parse_svg(self.assigned_ridges_svg, grid_size=self.grid_size)

  @functools.cached_property
  def base_cp(self) -> crease_pattern.CreasePattern | None:
    if self.base_svg is None:
      return None
    return parser.parse_svg(self.base_svg, grid_size=self.grid_size)

  @functools.cached_property
  def base_creases(self) -> list[crease_utils.Crease]:
    """Returns the creases of the base crease pattern."""
    if self.base_cp is None:
      return []
    return self.base_cp.creases

  # for zigzag shaping tests
  rectangles: (
      list[shaping_utils.Rectangle] | list[shaping_elements.ShapingFlap] | None
  ) = None
  rivers: list[shaping_elements.ShapingRiver] | None = None
  target_zigzag_depths: list[float | None] | None = None
  target_flap_angles: list[float | None] | None = None
  fold_orders: list[list[crease_utils.Fold]] | None = None
  even_phases: list[bool] | None = None
  zigzag_shaping_svg_sha: str | None = None

  # for inscribed rectangle shaping tests
  zigzag_shaping_svg: str | None = None
  inscribed_rectangle_shaping_params: (
      list[inscribed_rectangle_shaping.InscribedRectangleFlapShapingParams]
      | None
  ) = None
  inscribed_rectangle_shaping_svg_sha: str | None = None

  # for fan out shaping tests
  fan_out_shaping_svg: str | None = None
  fan_out_shaping_svg_sha: str | None = None

  # for narrow flap shaping tests
  zigzag_creases: list[crease_utils.Crease] | None = None

  @functools.cached_property
  def zigzag_shaping_cp(self) -> crease_pattern.CreasePattern | None:
    """Returns the zigzag shaped crease pattern."""
    if self.zigzag_shaping_svg is None:
      return None
    return parser.parse_svg(self.zigzag_shaping_svg)

  def get_zigzag_shaping_params(
      self,
  ) -> tuple[
      list[zigzag_shaping.ZigzagShapingParamsForElement],
      list[shaping_utils.Rectangle],
  ]:
    """Creates the zigzag shaping params from other params.

    Recomputes the rectangles from circle centers.

    Returns:
      A tuple of zigzag shaping params and the recomputed rectangles.
    """
    assert (
        self.target_flap_angles
        and self.target_zigzag_depths
        and self.fold_orders
        and self.even_phases
        and self.rectangles
    ), "All zigzag shaping params must be specified."
    zigzag_shaping_params = []
    new_rectangles = []
    for (
        target_flap_angle,
        target_zigzag_depth,
        fold_order,
        even_phase,
        rect,
    ) in zip(
        self.target_flap_angles,
        self.target_zigzag_depths,
        self.fold_orders,
        self.even_phases,
        self.rectangles,
        strict=True,
    ):
      if (
          not target_flap_angle
          or not target_zigzag_depth
          or not fold_order
          or not even_phase
          or not rect
      ):
        continue
      name = rect.name
      try:
        new_rectangles.append(
            shaping_utils.Rectangle.from_centers(
                rect.circle_centers, rect.radius, name
            )
        )
      except ValueError:
        continue
      zigzag_params = shaping_types.ZigzagParams(
          target_flap_angle=target_flap_angle,
          target_zigzag_depth=target_zigzag_depth,
          fold_order=fold_order,
          even_phase=even_phase,
      )
      zigzag_shaping_params.append(
          zigzag_shaping.ZigzagShapingParamsForElement(
              element_name=name, zigzag_params=zigzag_params
          )
      )
    return zigzag_shaping_params, new_rectangles


BASIC_NARROW_22P5_ZIGZAG_BASE = BoxPleatingCreasePatternExample(
    name="basic_narrow_22p5_zigzag",
    packing_svg="",
    grid_size=4,
    base_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="250" y2="0"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="500"/>
  <line stroke="red" x1="250" y1="0" x2="250" y2="375"/>
  <line stroke="black" x1="250" y1="0" x2="500" y2="0"/>
  <line stroke="red" x1="250" y1="375" x2="0" y2="500"/>
  <line stroke="blue" x1="250" y1="375" x2="500" y2="500"/>
  <line stroke="red" x1="250" y1="375" x2="250" y2="1000"/>
  <line stroke="black" x1="0" y1="500" x2="0" y2="1000"/>
  <line stroke="blue" x1="500" y1="0" x2="500" y2="500"/>
  <line stroke="black" x1="500" y1="0" x2="750" y2="0"/>
  <line stroke="red" x1="500" y1="500" x2="750" y2="375"/>
  <line stroke="blue" x1="500" y1="500" x2="500" y2="1000"/>
  <line stroke="red" x1="750" y1="0" x2="750" y2="375"/>
  <line stroke="black" x1="750" y1="0" x2="1000" y2="0"/>
  <line stroke="blue" x1="750" y1="375" x2="1000" y2="500"/>
  <line stroke="red" x1="750" y1="375" x2="750" y2="1000"/>
  <line stroke="black" x1="0" y1="1000" x2="250" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="500"/>
  <line stroke="black" x1="250" y1="1000" x2="500" y2="1000"/>
  <line stroke="black" x1="500" y1="1000" x2="750" y2="1000"/>
  <line stroke="black" x1="1000" y1="500" x2="1000" y2="1000"/>
  <line stroke="black" x1="750" y1="1000" x2="1000" y2="1000"/>
</svg>""",
    zigzag_creases=[
        crease_utils.Crease(
            point_utils.Point(x=0.0, y=0.5),
            point_utils.Point(x=0.25, y=0.375),
            fold=crease_utils.Fold.MOUNTAIN,
        ),
        crease_utils.Crease(
            point_utils.Point(x=0.25, y=0.375),
            point_utils.Point(x=0.5, y=0.5),
            fold=crease_utils.Fold.VALLEY,
        ),
        crease_utils.Crease(
            point_utils.Point(x=0.5, y=0.5),
            point_utils.Point(x=0.75, y=0.375),
            fold=crease_utils.Fold.MOUNTAIN,
        ),
        crease_utils.Crease(
            point_utils.Point(x=0.75, y=0.375),
            point_utils.Point(x=1.0, y=0.5),
            fold=crease_utils.Fold.VALLEY,
        ),
    ],
)

BASIC_NARROW_45_ZIGZAG_BASE = BoxPleatingCreasePatternExample(
    name="basic_narrow_45_zigzag",
    packing_svg="",
    grid_size=4,
    base_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="250" y2="0"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="500"/>
  <line stroke="red" x1="250" y1="0" x2="250" y2="250"/>
  <line stroke="black" x1="250" y1="0" x2="500" y2="0"/>
  <line stroke="red" x1="250" y1="250" x2="0" y2="500"/>
  <line stroke="blue" x1="250" y1="250" x2="500" y2="500"/>
  <line stroke="red" x1="250" y1="250" x2="250" y2="1000"/>
  <line stroke="black" x1="0" y1="500" x2="0" y2="1000"/>
  <line stroke="blue" x1="500" y1="0" x2="500" y2="500"/>
  <line stroke="black" x1="500" y1="0" x2="750" y2="0"/>
  <line stroke="red" x1="500" y1="500" x2="750" y2="250"/>
  <line stroke="blue" x1="500" y1="500" x2="500" y2="1000"/>
  <line stroke="red" x1="750" y1="0" x2="750" y2="250"/>
  <line stroke="black" x1="750" y1="0" x2="1000" y2="0"/>
  <line stroke="blue" x1="750" y1="250" x2="1000" y2="500"/>
  <line stroke="red" x1="750" y1="250" x2="750" y2="1000"/>
  <line stroke="black" x1="0" y1="1000" x2="250" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="500"/>
  <line stroke="black" x1="250" y1="1000" x2="500" y2="1000"/>
  <line stroke="black" x1="500" y1="1000" x2="750" y2="1000"/>
  <line stroke="black" x1="1000" y1="500" x2="1000" y2="1000"/>
  <line stroke="black" x1="750" y1="1000" x2="1000" y2="1000"/>
</svg>""",
    zigzag_creases=[
        crease_utils.Crease(
            point_utils.Point(x=0.0, y=0.5),
            point_utils.Point(x=0.25, y=0.25),
            fold=crease_utils.Fold.MOUNTAIN,
        ),
        crease_utils.Crease(
            point_utils.Point(x=0.25, y=0.25),
            point_utils.Point(x=0.5, y=0.5),
            fold=crease_utils.Fold.VALLEY,
        ),
        crease_utils.Crease(
            point_utils.Point(x=0.5, y=0.5),
            point_utils.Point(x=0.75, y=0.25),
            fold=crease_utils.Fold.MOUNTAIN,
        ),
        crease_utils.Crease(
            point_utils.Point(x=0.75, y=0.25),
            point_utils.Point(x=1.0, y=0.5),
            fold=crease_utils.Fold.VALLEY,
        ),
    ],
)

BASIC_NARROW_56P31_ZIGZAG_BASE = BoxPleatingCreasePatternExample(
    name="basic_narrow_22p5_zigzag",
    packing_svg="",
    grid_size=4,
    base_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="250" y2="0"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="500"/>
  <line stroke="red" x1="250" y1="0" x2="250" y2="125"/>
  <line stroke="black" x1="250" y1="0" x2="500" y2="0"/>
  <line stroke="red" x1="250" y1="125" x2="0" y2="500"/>
  <line stroke="blue" x1="250" y1="125" x2="500" y2="500"/>
  <line stroke="red" x1="250" y1="125" x2="250" y2="1000"/>
  <line stroke="black" x1="0" y1="500" x2="0" y2="1000"/>
  <line stroke="blue" x1="500" y1="0" x2="500" y2="500"/>
  <line stroke="black" x1="500" y1="0" x2="750" y2="0"/>
  <line stroke="red" x1="500" y1="500" x2="750" y2="125"/>
  <line stroke="blue" x1="500" y1="500" x2="500" y2="1000"/>
  <line stroke="red" x1="750" y1="0" x2="750" y2="125"/>
  <line stroke="black" x1="750" y1="0" x2="1000" y2="0"/>
  <line stroke="blue" x1="750" y1="125" x2="1000" y2="500"/>
  <line stroke="red" x1="750" y1="125" x2="750" y2="1000"/>
  <line stroke="black" x1="0" y1="1000" x2="250" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="500"/>
  <line stroke="black" x1="250" y1="1000" x2="500" y2="1000"/>
  <line stroke="black" x1="500" y1="1000" x2="750" y2="1000"/>
  <line stroke="black" x1="1000" y1="500" x2="1000" y2="1000"/>
  <line stroke="black" x1="750" y1="1000" x2="1000" y2="1000"/>
</svg>""",
    zigzag_creases=[
        crease_utils.Crease(
            point_utils.Point(x=0.0, y=0.5),
            point_utils.Point(x=0.25, y=0.125),
            fold=crease_utils.Fold.MOUNTAIN,
        ),
        crease_utils.Crease(
            point_utils.Point(x=0.25, y=0.125),
            point_utils.Point(x=0.5, y=0.5),
            fold=crease_utils.Fold.VALLEY,
        ),
        crease_utils.Crease(
            point_utils.Point(x=0.5, y=0.5),
            point_utils.Point(x=0.75, y=0.125),
            fold=crease_utils.Fold.MOUNTAIN,
        ),
        crease_utils.Crease(
            point_utils.Point(x=0.75, y=0.125),
            point_utils.Point(x=1.0, y=0.5),
            fold=crease_utils.Fold.VALLEY,
        ),
    ],
)

HUMAN = BoxPleatingCreasePatternExample(
    name="human",
    grid_size=8,
    central_crease_removed=True,
    tree_data={
        "version": "0.7",
        "design": {
            "title": "human",
            "mode": "tree",
            "description": "",
            "layout": {
                "sheet": {"type": "rect", "height": 16, "width": 16},
                "flaps": [
                    {"id": 2, "x": 8, "y": 6, "width": 0, "height": 0},
                    {"id": 3, "x": 10, "y": 3, "width": 0, "height": 0},
                    {"id": 4, "x": 8, "y": 12, "width": 0, "height": 0},
                    {"id": 5, "x": 11, "y": 9, "width": 0, "height": 0},
                    {"id": 6, "x": 4, "y": 9, "width": 0, "height": 0},
                ],
                "stretches": [],
            },
            "tree": {
                "sheet": {"type": "rect", "height": 20, "width": 20},
                "nodes": [
                    {"id": 0, "name": "", "x": 10, "y": 10},
                    {"id": 1, "name": "", "x": 10, "y": 13},
                    {"id": 2, "name": "", "x": 7, "y": 4},
                    {"id": 3, "name": "", "x": 13, "y": 4},
                    {"id": 4, "name": "", "x": 10, "y": 15},
                    {"id": 5, "name": "", "x": 14, "y": 11},
                    {"id": 6, "name": "", "x": 6, "y": 11},
                ],
                "edges": [
                    {"n1": 0, "n2": 1, "length": 2},
                    {"n1": 0, "n2": 2, "length": 8},
                    {"n1": 0, "n2": 3, "length": 8},
                    {"n1": 1, "n2": 6, "length": 6},
                    {"n1": 1, "n2": 5, "length": 6},
                    {"n1": 1, "n2": 4, "length": 2},
                ],
            },
        },
    },
    packing_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8">
  <line stroke="black" x1="0" y1="0" x2="0" y2="3"/>
  <line stroke="black" x1="0" y1="0" x2="3" y2="0"/>
  <line stroke="red" x1="0" y1="0" x2="3" y2="3"/>
  <line stroke="black" x1="0" y1="3" x2="0" y2="4"/>
  <line stroke="green" x1="0" y1="3" x2="3" y2="3"/>
  <line stroke="black" x1="3" y1="0" x2="4" y2="0"/>
  <line stroke="green" x1="3" y1="0" x2="3" y2="3"/>
  <line stroke="green" x1="0" y1="4" x2="4" y2="4"/>
  <line stroke="black" x1="0" y1="4" x2="0" y2="8"/>
  <line stroke="red" x1="4" y1="0" x2="4" y2="2"/>
  <line stroke="black" x1="4" y1="0" x2="5" y2="0"/>
  <line stroke="red" x1="3" y1="3" x2="4" y2="2"/>
  <line stroke="green" x1="3" y1="3" x2="5" y2="3"/>
  <line stroke="red" x1="4" y1="2" x2="5" y2="3"/>
  <line stroke="green" x1="5" y1="0" x2="5" y2="3"/>
  <line stroke="black" x1="5" y1="0" x2="8" y2="0"/>
  <line stroke="red" x1="4" y1="4" x2="0" y2="8"/>
  <line stroke="green" x1="4" y1="4" x2="4" y2="8"/>
  <line stroke="green" x1="4" y1="4" x2="8" y2="4"/>
  <line stroke="red" x1="4" y1="4" x2="8" y2="8"/>
  <line stroke="red" x1="5" y1="3" x2="8" y2="0"/>
  <line stroke="green" x1="5" y1="3" x2="8" y2="3"/>
  <line stroke="black" x1="0" y1="8" x2="4" y2="8"/>
  <line stroke="black" x1="8" y1="0" x2="8" y2="3"/>
  <line stroke="black" x1="8" y1="3" x2="8" y2="4"/>
  <line stroke="black" x1="4" y1="8" x2="8" y2="8"/>
  <line stroke="black" x1="8" y1="4" x2="8" y2="8"/>
</svg>""",
    base_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="red" x1="0" y1="0" x2="125" y2="125"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="375"/>
  <line stroke="black" x1="0" y1="0" x2="500" y2="0"/>
  <line stroke="blue" x1="125" y1="125" x2="250" y2="250"/>
  <line stroke="red" x1="125" y1="125" x2="125" y2="375"/>
  <line stroke="red" x1="125" y1="125" x2="375" y2="125"/>
  <line stroke="blue" x1="250" y1="250" x2="250" y2="375"/>
  <line stroke="red" x1="250" y1="250" x2="375" y2="375"/>
  <line stroke="blue" x1="250" y1="250" x2="500" y2="250"/>
  <line stroke="blue" x1="0" y1="375" x2="125" y2="375"/>
  <line stroke="black" x1="0" y1="375" x2="0" y2="1000"/>
  <line stroke="red" x1="125" y1="375" x2="250" y2="375"/>
  <line stroke="red" x1="125" y1="375" x2="125" y2="875"/>
  <line stroke="blue" x1="375" y1="125" x2="500" y2="0"/>
  <line stroke="red" x1="375" y1="125" x2="500" y2="250"/>
  <line stroke="red" x1="375" y1="125" x2="625" y2="125"/>
  <line stroke="blue" x1="250" y1="375" x2="375" y2="375"/>
  <line stroke="blue" x1="250" y1="375" x2="250" y2="750"/>
  <line stroke="blue" x1="500" y1="0" x2="625" y2="125"/>
  <line stroke="black" x1="500" y1="0" x2="1000" y2="0"/>
  <line stroke="red" x1="375" y1="375" x2="500" y2="250"/>
  <line stroke="red" x1="375" y1="375" x2="375" y2="625"/>
  <line stroke="red" x1="500" y1="250" x2="625" y2="125"/>
  <line stroke="red" x1="500" y1="250" x2="625" y2="375"/>
  <line stroke="blue" x1="500" y1="250" x2="750" y2="250"/>
  <line stroke="red" x1="625" y1="125" x2="875" y2="125"/>
  <line stroke="red" x1="375" y1="625" x2="250" y2="750"/>
  <line stroke="red" x1="375" y1="625" x2="625" y2="625"/>
  <line stroke="blue" x1="375" y1="625" x2="500" y2="750"/>
  <line stroke="red" x1="625" y1="375" x2="750" y2="250"/>
  <line stroke="blue" x1="625" y1="375" x2="750" y2="375"/>
  <line stroke="red" x1="625" y1="375" x2="625" y2="625"/>
  <line stroke="blue" x1="250" y1="750" x2="125" y2="875"/>
  <line stroke="blue" x1="250" y1="750" x2="500" y2="750"/>
  <line stroke="blue" x1="750" y1="250" x2="750" y2="375"/>
  <line stroke="blue" x1="750" y1="250" x2="875" y2="125"/>
  <line stroke="red" x1="750" y1="375" x2="875" y2="375"/>
  <line stroke="blue" x1="750" y1="375" x2="750" y2="750"/>
  <line stroke="red" x1="125" y1="875" x2="375" y2="875"/>
  <line stroke="red" x1="125" y1="875" x2="0" y2="1000"/>
  <line stroke="blue" x1="625" y1="625" x2="500" y2="750"/>
  <line stroke="red" x1="625" y1="625" x2="750" y2="750"/>
  <line stroke="red" x1="875" y1="125" x2="875" y2="375"/>
  <line stroke="red" x1="875" y1="125" x2="1000" y2="0"/>
  <line stroke="red" x1="500" y1="750" x2="375" y2="875"/>
  <line stroke="blue" x1="500" y1="750" x2="750" y2="750"/>
  <line stroke="red" x1="500" y1="750" x2="625" y2="875"/>
  <line stroke="red" x1="375" y1="875" x2="625" y2="875"/>
  <line stroke="blue" x1="375" y1="875" x2="500" y2="1000"/>
  <line stroke="blue" x1="875" y1="375" x2="1000" y2="375"/>
  <line stroke="red" x1="875" y1="375" x2="875" y2="875"/>
  <line stroke="black" x1="0" y1="1000" x2="500" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="375"/>
  <line stroke="blue" x1="750" y1="750" x2="875" y2="875"/>
  <line stroke="black" x1="1000" y1="375" x2="1000" y2="1000"/>
  <line stroke="blue" x1="625" y1="875" x2="500" y2="1000"/>
  <line stroke="red" x1="625" y1="875" x2="875" y2="875"/>
  <line stroke="black" x1="500" y1="1000" x2="1000" y2="1000"/>
  <line stroke="red" x1="875" y1="875" x2="1000" y2="1000"/>
</svg>""",
    rectangles=[
        shaping_utils.Rectangle(
            name="left_arm",
            x_start=0,
            y_start=0,
            x_len=0.375,
            y_len=0.375,
        ),
        shaping_utils.Rectangle(
            name="head",
            x_start=0.375,
            y_start=0,
            x_len=0.25,
            y_len=0.375,
        ),
        shaping_utils.Rectangle(
            name="right_arm",
            x_start=0.375 + 0.25,
            y_start=0,
            x_len=0.375,
            y_len=0.375,
        ),
        shaping_utils.Rectangle(
            name="left_leg",
            x_start=0,
            y_start=0.5,
            x_len=0.5,
            y_len=0.5,
        ),
        shaping_utils.Rectangle(
            name="right_leg",
            x_start=0.5,
            y_start=0.5,
            x_len=0.5,
            y_len=0.5,
        ),
    ],
    target_zigzag_depths=[
        1,
        None,
        1,
        None,
        None,
    ],
    target_flap_angles=[
        None,
        None,
        None,
        None,
        None,
    ],
    fold_orders=[
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
    ],
    even_phases=[
        False,
        True,
        True,
        True,
        True,
    ],
    zigzag_shaping_svg_sha=(
        "9c7fb4b90c201ddf121bf53e04ddd47b94458b2004ce6f8367de1019e907d3da"
    ),
)

JAGUAR = BoxPleatingCreasePatternExample(
    name="jaguar",
    grid_size=14,
    central_crease_removed=False,
    packing_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14">
  <rect stroke="black" x="0" y="0" width="14" height="14"/>
  <line stroke="red" x1="2" y1="0" x2="5" y2="3"/>
  <line stroke="green" x1="0" y1="3" x2="5" y2="3"/>
  <line stroke="green" x1="5" y1="0" x2="5" y2="3"/>
  <line stroke="red" x1="5" y1="3" x2="7" y2="1"/>
  <line stroke="green" x1="5" y1="3" x2="9" y2="3"/>
  <line stroke="red" x1="7" y1="0" x2="7" y2="1"/>
  <line stroke="red" x1="7" y1="1" x2="9" y2="3"/>
  <line stroke="green" x1="0" y1="8" x2="3" y2="8"/>
  <line stroke="red" x1="3" y1="8" x2="0" y2="11"/>
  <line stroke="green" x1="3" y1="8" x2="11" y2="8"/>
  <line stroke="red" x1="3" y1="8" x2="7" y2="12"/>
  <line stroke="green" x1="3" y1="8" x2="3" y2="14"/>
  <line stroke="green" x1="9" y1="0" x2="9" y2="3"/>
  <line stroke="red" x1="9" y1="3" x2="12" y2="0"/>
  <line stroke="green" x1="9" y1="3" x2="14" y2="3"/>
  <line stroke="red" x1="11" y1="8" x2="7" y2="12"/>
  <line stroke="green" x1="11" y1="8" x2="14" y2="8"/>
  <line stroke="green" x1="11" y1="8" x2="11" y2="14"/>
  <line stroke="red" x1="11" y1="8" x2="14" y2="11"/>
  <line stroke="red" x1="7" y1="12" x2="7" y2="14"/>
</svg>""",
    base_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14">
  <rect stroke="black" x="0" y="0" width="14" height="14"/>
  <line stroke="red" x1="1" y1="0" x2="1" y2="8" elevation="1"/>
  <line stroke="red" x1="2" y1="0" x2="3" y2="1"/>
  <line stroke="blue" x1="2" y1="0" x2="2" y2="8" elevation="0"/>
  <line stroke="blue" x1="3" y1="1" x2="4" y2="2"/>
  <line stroke="red" x1="3" y1="1" x2="5" y2="1" elevation="1"/>
  <line stroke="red" x1="3" y1="1" x2="3" y2="8" elevation="1"/>
  <line stroke="blue" x1="4" y1="2" x2="5" y2="2" elevation="0"/>
  <line stroke="red" x1="4" y1="2" x2="5" y2="3"/>
  <line stroke="blue" x1="4" y1="2" x2="4" y2="9" elevation="0"/>
  <line stroke="blue" x1="5" y1="0" x2="5" y2="1"/>
  <line stroke="red" x1="5" y1="1" x2="5" y2="2"/>
  <line stroke="red" x1="5" y1="1" x2="7" y2="1" elevation="1"/>
  <line stroke="blue" x1="5" y1="2" x2="5" y2="3"/>
  <line stroke="blue" x1="5" y1="2" x2="6" y2="2" elevation="0"/>
  <line stroke="red" x1="5" y1="3" x2="6" y2="2"/>
  <line stroke="red" x1="5" y1="3" x2="5" y2="10" elevation="1"/>
  <line stroke="blue" x1="6" y1="2" x2="7" y2="1"/>
  <line stroke="blue" x1="6" y1="2" x2="6" y2="11" elevation="0"/>
  <line stroke="red" x1="7" y1="0" x2="7" y2="1"/>
  <line stroke="blue" x1="7" y1="1" x2="8" y2="2"/>
  <line stroke="red" x1="7" y1="1" x2="9" y2="1" elevation="1"/>
  <line stroke="red" x1="7" y1="1" x2="7" y2="12" elevation="1"/>
  <line stroke="blue" x1="0" y1="8" x2="1" y2="8"/>
  <line stroke="red" x1="1" y1="8" x2="2" y2="8"/>
  <line stroke="red" x1="1" y1="8" x2="1" y2="10" elevation="1"/>
  <line stroke="blue" x1="2" y1="8" x2="3" y2="8"/>
  <line stroke="blue" x1="2" y1="8" x2="2" y2="9" elevation="0"/>
  <line stroke="blue" x1="8" y1="2" x2="9" y2="2" elevation="0"/>
  <line stroke="red" x1="8" y1="2" x2="9" y2="3"/>
  <line stroke="blue" x1="8" y1="2" x2="8" y2="11" elevation="0"/>
  <line stroke="red" x1="3" y1="8" x2="2" y2="9"/>
  <line stroke="red" x1="3" y1="8" x2="4" y2="9"/>
  <line stroke="blue" x1="9" y1="0" x2="9" y2="1"/>
  <line stroke="red" x1="9" y1="1" x2="9" y2="2"/>
  <line stroke="red" x1="9" y1="1" x2="11" y2="1" elevation="1"/>
  <line stroke="blue" x1="2" y1="9" x2="4" y2="9" elevation="0"/>
  <line stroke="blue" x1="2" y1="9" x2="1" y2="10"/>
  <line stroke="blue" x1="9" y1="2" x2="9" y2="3"/>
  <line stroke="blue" x1="9" y1="2" x2="10" y2="2" elevation="0"/>
  <line stroke="red" x1="9" y1="3" x2="10" y2="2"/>
  <line stroke="red" x1="9" y1="3" x2="9" y2="10" elevation="1"/>
  <line stroke="blue" x1="4" y1="9" x2="5" y2="10"/>
  <line stroke="red" x1="1" y1="10" x2="0" y2="11"/>
  <line stroke="red" x1="1" y1="10" x2="5" y2="10" elevation="1"/>
  <line stroke="blue" x1="10" y1="2" x2="11" y2="1"/>
  <line stroke="blue" x1="10" y1="2" x2="10" y2="9" elevation="0"/>
  <line stroke="blue" x1="0" y1="11" x2="6" y2="11" elevation="0"/>
  <line stroke="red" x1="11" y1="1" x2="12" y2="0"/>
  <line stroke="red" x1="11" y1="1" x2="11" y2="8" elevation="1"/>
  <line stroke="red" x1="5" y1="10" x2="6" y2="11"/>
  <line stroke="red" x1="0" y1="12" x2="7" y2="12" elevation="1"/>
  <line stroke="blue" x1="12" y1="0" x2="12" y2="8" elevation="0"/>
  <line stroke="blue" x1="6" y1="11" x2="7" y2="12"/>
  <line stroke="blue" x1="0" y1="13" x2="7" y2="13" elevation="0"/>
  <line stroke="red" x1="13" y1="0" x2="13" y2="8" elevation="1"/>
  <line stroke="blue" x1="9" y1="10" x2="10" y2="9"/>
  <line stroke="red" x1="9" y1="10" x2="8" y2="11"/>
  <line stroke="red" x1="9" y1="10" x2="13" y2="10" elevation="1"/>
  <line stroke="red" x1="10" y1="9" x2="11" y2="8"/>
  <line stroke="blue" x1="10" y1="9" x2="12" y2="9" elevation="0"/>
  <line stroke="blue" x1="8" y1="11" x2="7" y2="12"/>
  <line stroke="blue" x1="8" y1="11" x2="14" y2="11" elevation="0"/>
  <line stroke="blue" x1="11" y1="8" x2="12" y2="8"/>
  <line stroke="red" x1="11" y1="8" x2="12" y2="9"/>
  <line stroke="red" x1="7" y1="12" x2="7" y2="13"/>
  <line stroke="red" x1="7" y1="12" x2="14" y2="12" elevation="1"/>
  <line stroke="blue" x1="12" y1="8" x2="12" y2="9" elevation="0"/>
  <line stroke="red" x1="12" y1="8" x2="13" y2="8"/>
  <line stroke="blue" x1="7" y1="13" x2="7" y2="14"/>
  <line stroke="blue" x1="7" y1="13" x2="14" y2="13" elevation="0"/>
  <line stroke="blue" x1="12" y1="9" x2="13" y2="10"/>
  <line stroke="blue" x1="13" y1="8" x2="14" y2="8"/>
  <line stroke="red" x1="13" y1="8" x2="13" y2="10" elevation="1"/>
  <line stroke="red" x1="13" y1="10" x2="14" y2="11"/>
</svg>""",
)

SCORPION_WITH_ELEVATION = BoxPleatingCreasePatternExample(
    name="scorpion_with_elevation",
    grid_size=32,
    central_crease_removed=True,
    packing_svg=SCORPION.packing_svg,
    base_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <line stroke="red" x1="0" y1="0" x2="1" y2="1"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="4"/>
  <line stroke="black" x1="0" y1="0" x2="8" y2="0"/>
  <line stroke="blue" x1="1" y1="1" x2="2" y2="2"/>
  <line stroke="red" x1="1" y1="1" x2="1" y2="4" elevation="1"/>
  <line stroke="red" x1="1" y1="1" x2="7" y2="1" elevation="1"/>
  <line stroke="red" x1="2" y1="2" x2="3" y2="3"/>
  <line stroke="blue" x1="2" y1="2" x2="2" y2="4" elevation="0"/>
  <line stroke="blue" x1="2" y1="2" x2="6" y2="2" elevation="0"/>
  <line stroke="blue" x1="0" y1="4" x2="1" y2="4"/>
  <line stroke="black" x1="0" y1="4" x2="0" y2="8"/>
  <line stroke="red" x1="1" y1="4" x2="2" y2="4"/>
  <line stroke="red" x1="1" y1="4" x2="1" y2="8" elevation="1"/>
  <line stroke="red" x1="3" y1="3" x2="3" y2="4" elevation="1"/>
  <line stroke="blue" x1="3" y1="3" x2="4" y2="4"/>
  <line stroke="red" x1="3" y1="3" x2="5" y2="3" elevation="1"/>
  <line stroke="blue" x1="2" y1="4" x2="3" y2="4"/>
  <line stroke="blue" x1="2" y1="4" x2="2" y2="8" elevation="0"/>
  <line stroke="red" x1="3" y1="4" x2="4" y2="4"/>
  <line stroke="red" x1="3" y1="4" x2="3" y2="8" elevation="1"/>
  <line stroke="blue" x1="4" y1="4" x2="5" y2="3"/>
  <line stroke="blue" x1="4" y1="4" x2="4" y2="8" elevation="0"/>
  <line stroke="red" x1="5" y1="3" x2="6" y2="2"/>
  <line stroke="red" x1="5" y1="3" x2="5" y2="8" elevation="1"/>
  <line stroke="blue" x1="6" y1="2" x2="7" y2="1"/>
  <line stroke="blue" x1="6" y1="2" x2="6" y2="8" elevation="0"/>
  <line stroke="red" x1="7" y1="1" x2="8" y2="0"/>
  <line stroke="red" x1="7" y1="1" x2="7" y2="8" elevation="1"/>
  <line stroke="blue" x1="0" y1="8" x2="1" y2="8"/>
  <line stroke="black" x1="0" y1="8" x2="0" y2="12"/>
  <line stroke="red" x1="8" y1="0" x2="9" y2="1"/>
  <line stroke="blue" x1="8" y1="0" x2="8" y2="8" elevation="0"/>
  <line stroke="black" x1="8" y1="0" x2="16" y2="0"/>
  <line stroke="red" x1="1" y1="8" x2="2" y2="8"/>
  <line stroke="red" x1="1" y1="8" x2="1" y2="11" elevation="1"/>
  <line stroke="blue" x1="2" y1="8" x2="3" y2="8"/>
  <line stroke="blue" x1="2" y1="8" x2="2" y2="10" elevation="0"/>
  <line stroke="red" x1="3" y1="8" x2="4" y2="8"/>
  <line stroke="red" x1="3" y1="8" x2="3" y2="9" elevation="1"/>
  <line stroke="red" x1="4" y1="8" x2="5" y2="8"/>
  <line stroke="blue" x1="4" y1="8" x2="3" y2="9"/>
  <line stroke="blue" x1="4" y1="8" x2="4" y2="9"/>
  <line stroke="blue" x1="4" y1="8" x2="5" y2="9"/>
  <line stroke="blue" x1="9" y1="1" x2="10" y2="2"/>
  <line stroke="red" x1="9" y1="1" x2="9" y2="8" elevation="1"/>
  <line stroke="red" x1="9" y1="1" x2="15" y2="1" elevation="1"/>
  <line stroke="blue" x1="5" y1="8" x2="6" y2="8"/>
  <line stroke="red" x1="5" y1="8" x2="5" y2="9" elevation="1"/>
  <line stroke="red" x1="3" y1="9" x2="4" y2="9" elevation="1"/>
  <line stroke="red" x1="3" y1="9" x2="2" y2="10"/>
  <line stroke="red" x1="4" y1="9" x2="5" y2="9" elevation="1"/>
  <line stroke="red" x1="4" y1="9" x2="4" y2="10"/>
  <line stroke="red" x1="6" y1="8" x2="7" y2="8"/>
  <line stroke="blue" x1="6" y1="8" x2="6" y2="10" elevation="0"/>
  <line stroke="blue" x1="2" y1="10" x2="4" y2="10" elevation="0"/>
  <line stroke="blue" x1="2" y1="10" x2="1" y2="11"/>
  <line stroke="red" x1="10" y1="2" x2="11" y2="3"/>
  <line stroke="blue" x1="10" y1="2" x2="10" y2="8" elevation="0"/>
  <line stroke="blue" x1="10" y1="2" x2="16" y2="2" elevation="0"/>
  <line stroke="red" x1="5" y1="9" x2="6" y2="10"/>
  <line stroke="blue" x1="7" y1="8" x2="8" y2="8"/>
  <line stroke="red" x1="7" y1="8" x2="7" y2="11" elevation="1"/>
  <line stroke="blue" x1="4" y1="10" x2="6" y2="10" elevation="0"/>
  <line stroke="blue" x1="4" y1="10" x2="4" y2="11"/>
  <line stroke="red" x1="1" y1="11" x2="4" y2="11" elevation="1"/>
  <line stroke="red" x1="1" y1="11" x2="0" y2="12"/>
  <line stroke="blue" x1="8" y1="8" x2="9" y2="8"/>
  <line stroke="red" x1="8" y1="8" x2="8" y2="12" elevation="0"/>
  <line stroke="blue" x1="11" y1="3" x2="12" y2="4"/>
  <line stroke="red" x1="11" y1="3" x2="11" y2="8" elevation="1"/>
  <line stroke="red" x1="11" y1="3" x2="15" y2="3" elevation="1"/>
  <line stroke="blue" x1="6" y1="10" x2="7" y2="11"/>
  <line stroke="red" x1="4" y1="11" x2="4" y2="12"/>
  <line stroke="red" x1="4" y1="11" x2="7" y2="11" elevation="1"/>
  <line stroke="blue" x1="0" y1="12" x2="4" y2="12" elevation="0"/>
  <line stroke="red" x1="0" y1="12" x2="1" y2="13"/>
  <line stroke="black" x1="0" y1="12" x2="0" y2="16"/>
  <line stroke="red" x1="9" y1="8" x2="10" y2="8"/>
  <line stroke="red" x1="9" y1="8" x2="9" y2="11" elevation="1"/>
  <line stroke="blue" x1="4" y1="12" x2="4" y2="13"/>
  <line stroke="blue" x1="4" y1="12" x2="8" y2="12" elevation="0"/>
  <line stroke="red" x1="12" y1="4" x2="13" y2="5"/>
  <line stroke="blue" x1="12" y1="4" x2="12" y2="8" elevation="0"/>
  <line stroke="blue" x1="12" y1="4" x2="16" y2="4" elevation="0"/>
  <line stroke="blue" x1="10" y1="8" x2="11" y2="8"/>
  <line stroke="blue" x1="10" y1="8" x2="10" y2="10" elevation="0"/>
  <line stroke="red" x1="1" y1="13" x2="4" y2="13" elevation="1"/>
  <line stroke="blue" x1="1" y1="13" x2="2" y2="14"/>
  <line stroke="red" x1="1" y1="13" x2="1" y2="16" elevation="1"/>
  <line stroke="red" x1="7" y1="11" x2="8" y2="12"/>
  <line stroke="red" x1="4" y1="13" x2="4" y2="14"/>
  <line stroke="red" x1="4" y1="13" x2="7" y2="13" elevation="1"/>
  <line stroke="red" x1="11" y1="8" x2="11" y2="9" elevation="1"/>
  <line stroke="red" x1="11" y1="8" x2="12" y2="8"/>
  <line stroke="blue" x1="13" y1="5" x2="14" y2="6"/>
  <line stroke="red" x1="13" y1="5" x2="13" y2="9" elevation="1"/>
  <line stroke="red" x1="13" y1="5" x2="15" y2="5" elevation="1"/>
  <line stroke="blue" x1="2" y1="14" x2="4" y2="14" elevation="0"/>
  <line stroke="red" x1="2" y1="14" x2="3" y2="15"/>
  <line stroke="blue" x1="2" y1="14" x2="2" y2="16" elevation="0"/>
  <line stroke="blue" x1="10" y1="10" x2="9" y2="11"/>
  <line stroke="red" x1="10" y1="10" x2="11" y2="9"/>
  <line stroke="blue" x1="10" y1="10" x2="12" y2="10" elevation="0"/>
  <line stroke="red" x1="9" y1="11" x2="8" y2="12"/>
  <line stroke="red" x1="9" y1="11" x2="13" y2="11" elevation="1"/>
  <line stroke="blue" x1="11" y1="9" x2="12" y2="8"/>
  <line stroke="red" x1="11" y1="9" x2="13" y2="9" elevation="1"/>
  <line stroke="red" x1="8" y1="12" x2="7" y2="13"/>
  <line stroke="red" x1="8" y1="12" x2="9" y2="13"/>
  <line stroke="blue" x1="8" y1="12" x2="12" y2="12" elevation="0"/>
  <line stroke="blue" x1="8" y1="12" x2="8" y2="18" elevation="0"/>
  <line stroke="blue" x1="12" y1="8" x2="13" y2="9"/>
  <line stroke="blue" x1="4" y1="14" x2="6" y2="14" elevation="0"/>
  <line stroke="blue" x1="4" y1="14" x2="4" y2="15"/>
  <line stroke="blue" x1="7" y1="13" x2="6" y2="14"/>
  <line stroke="red" x1="7" y1="13" x2="7" y2="18" elevation="1"/>
  <line stroke="blue" x1="15" y1="1" x2="16" y2="0"/>
  <line stroke="red" x1="15" y1="1" x2="16" y2="2"/>
  <line stroke="red" x1="15" y1="1" x2="17" y2="1" elevation="1"/>
  <line stroke="red" x1="6" y1="14" x2="5" y2="15"/>
  <line stroke="blue" x1="6" y1="14" x2="6" y2="18" elevation="0"/>
  <line stroke="red" x1="14" y1="6" x2="15" y2="7"/>
  <line stroke="blue" x1="14" y1="6" x2="14" y2="9" elevation="0"/>
  <line stroke="blue" x1="14" y1="6" x2="16" y2="6" elevation="0"/>
  <line stroke="red" x1="3" y1="15" x2="4" y2="15" elevation="1"/>
  <line stroke="red" x1="3" y1="15" x2="3" y2="16" elevation="1"/>
  <line stroke="blue" x1="3" y1="15" x2="4" y2="16"/>
  <line stroke="blue" x1="15" y1="3" x2="16" y2="2"/>
  <line stroke="red" x1="15" y1="3" x2="16" y2="4"/>
  <line stroke="red" x1="15" y1="3" x2="17" y2="3" elevation="1"/>
  <line stroke="red" x1="4" y1="15" x2="5" y2="15" elevation="1"/>
  <line stroke="red" x1="4" y1="15" x2="4" y2="16"/>
  <line stroke="blue" x1="12" y1="10" x2="13" y2="9"/>
  <line stroke="red" x1="12" y1="10" x2="13" y2="10"/>
  <line stroke="blue" x1="12" y1="10" x2="13" y2="11"/>
  <line stroke="blue" x1="5" y1="15" x2="4" y2="16"/>
  <line stroke="red" x1="5" y1="15" x2="5" y2="17" elevation="1"/>
  <line stroke="blue" x1="9" y1="13" x2="10" y2="14"/>
  <line stroke="red" x1="9" y1="13" x2="13" y2="13" elevation="1"/>
  <line stroke="red" x1="9" y1="13" x2="9" y2="18" elevation="1"/>
  <line stroke="red" x1="13" y1="9" x2="13" y2="10" elevation="1"/>
  <line stroke="red" x1="13" y1="9" x2="14" y2="9"/>
  <line stroke="blue" x1="15" y1="5" x2="16" y2="4"/>
  <line stroke="red" x1="15" y1="5" x2="16" y2="6"/>
  <line stroke="red" x1="15" y1="5" x2="17" y2="5" elevation="1"/>
  <line stroke="blue" x1="0" y1="16" x2="1" y2="16"/>
  <line stroke="black" x1="0" y1="16" x2="0" y2="20"/>
  <line stroke="blue" x1="16" y1="0" x2="17" y2="1"/>
  <line stroke="black" x1="16" y1="0" x2="24" y2="0"/>
  <line stroke="red" x1="1" y1="16" x2="2" y2="16"/>
  <line stroke="red" x1="1" y1="16" x2="1" y2="19" elevation="1"/>
  <line stroke="blue" x1="2" y1="16" x2="3" y2="16"/>
  <line stroke="blue" x1="2" y1="16" x2="2" y2="18" elevation="0"/>
  <line stroke="red" x1="16" y1="2" x2="17" y2="1"/>
  <line stroke="blue" x1="16" y1="2" x2="17" y2="3"/>
  <line stroke="blue" x1="16" y1="2" x2="22" y2="2" elevation="0"/>
  <line stroke="red" x1="3" y1="16" x2="4" y2="16"/>
  <line stroke="red" x1="3" y1="16" x2="3" y2="17" elevation="1"/>
  <line stroke="red" x1="13" y1="10" x2="13" y2="11" elevation="1"/>
  <line stroke="blue" x1="13" y1="10" x2="14" y2="10"/>
  <line stroke="blue" x1="4" y1="16" x2="3" y2="17"/>
  <line stroke="blue" x1="4" y1="16" x2="5" y2="17"/>
  <line stroke="red" x1="16" y1="4" x2="17" y2="3"/>
  <line stroke="blue" x1="16" y1="4" x2="17" y2="5"/>
  <line stroke="blue" x1="16" y1="4" x2="20" y2="4" elevation="0"/>
  <line stroke="blue" x1="15" y1="7" x2="16" y2="6"/>
  <line stroke="red" x1="15" y1="7" x2="15" y2="9" elevation="1"/>
  <line stroke="red" x1="15" y1="7" x2="17" y2="7" elevation="1"/>
  <line stroke="blue" x1="14" y1="9" x2="14" y2="10" elevation="0"/>
  <line stroke="blue" x1="14" y1="9" x2="15" y2="9"/>
  <line stroke="blue" x1="12" y1="12" x2="13" y2="11"/>
  <line stroke="red" x1="12" y1="12" x2="13" y2="12"/>
  <line stroke="blue" x1="12" y1="12" x2="13" y2="13"/>
  <line stroke="red" x1="13" y1="11" x2="13" y2="12" elevation="1"/>
  <line stroke="red" x1="13" y1="11" x2="14" y2="11"/>
  <line stroke="red" x1="17" y1="1" x2="23" y2="1" elevation="1"/>
  <line stroke="red" x1="16" y1="6" x2="17" y2="5"/>
  <line stroke="blue" x1="16" y1="6" x2="17" y2="7"/>
  <line stroke="blue" x1="16" y1="6" x2="18" y2="6" elevation="0"/>
  <line stroke="blue" x1="10" y1="14" x2="12" y2="14" elevation="0"/>
  <line stroke="red" x1="10" y1="14" x2="11" y2="15"/>
  <line stroke="blue" x1="10" y1="14" x2="10" y2="18" elevation="0"/>
  <line stroke="blue" x1="14" y1="10" x2="14" y2="11" elevation="0"/>
  <line stroke="red" x1="14" y1="10" x2="15" y2="10"/>
  <line stroke="red" x1="3" y1="17" x2="5" y2="17" elevation="1"/>
  <line stroke="red" x1="3" y1="17" x2="2" y2="18"/>
  <line stroke="red" x1="17" y1="3" x2="21" y2="3" elevation="1"/>
  <line stroke="red" x1="15" y1="9" x2="15" y2="10" elevation="1"/>
  <line stroke="red" x1="15" y1="9" x2="17" y2="9"/>
  <line stroke="red" x1="13" y1="12" x2="13" y2="13" elevation="1"/>
  <line stroke="blue" x1="13" y1="12" x2="14" y2="12"/>
  <line stroke="red" x1="5" y1="17" x2="6" y2="18"/>
  <line stroke="red" x1="17" y1="5" x2="19" y2="5" elevation="1"/>
  <line stroke="blue" x1="14" y1="11" x2="14" y2="12" elevation="0"/>
  <line stroke="blue" x1="14" y1="11" x2="15" y2="11"/>
  <line stroke="red" x1="15" y1="10" x2="15" y2="11" elevation="1"/>
  <line stroke="blue" x1="15" y1="10" x2="17" y2="10"/>
  <line stroke="blue" x1="2" y1="18" x2="6" y2="18" elevation="0"/>
  <line stroke="blue" x1="2" y1="18" x2="1" y2="19"/>
  <line stroke="blue" x1="13" y1="13" x2="12" y2="14"/>
  <line stroke="red" x1="13" y1="13" x2="13" y2="14" elevation="1"/>
  <line stroke="red" x1="13" y1="13" x2="14" y2="13"/>
  <line stroke="red" x1="17" y1="7" x2="18" y2="6"/>
  <line stroke="red" x1="17" y1="7" x2="17" y2="9" elevation="1"/>
  <line stroke="red" x1="12" y1="14" x2="13" y2="14"/>
  <line stroke="blue" x1="12" y1="14" x2="13" y2="15"/>
  <line stroke="blue" x1="14" y1="12" x2="14" y2="13" elevation="0"/>
  <line stroke="red" x1="14" y1="12" x2="15" y2="12"/>
  <line stroke="red" x1="11" y1="15" x2="13" y2="15" elevation="1"/>
  <line stroke="blue" x1="11" y1="15" x2="12" y2="16"/>
  <line stroke="red" x1="11" y1="15" x2="11" y2="18" elevation="1"/>
  <line stroke="red" x1="15" y1="11" x2="15" y2="12" elevation="1"/>
  <line stroke="red" x1="15" y1="11" x2="17" y2="11"/>
  <line stroke="blue" x1="6" y1="18" x2="7" y2="18"/>
  <line stroke="red" x1="6" y1="18" x2="5" y2="19"/>
  <line stroke="blue" x1="6" y1="18" x2="6" y2="20" elevation="0"/>
  <line stroke="blue" x1="18" y1="6" x2="19" y2="5"/>
  <line stroke="blue" x1="18" y1="6" x2="18" y2="9" elevation="0"/>
  <line stroke="red" x1="1" y1="19" x2="5" y2="19" elevation="1"/>
  <line stroke="red" x1="1" y1="19" x2="0" y2="20"/>
  <line stroke="blue" x1="13" y1="14" x2="14" y2="14"/>
  <line stroke="red" x1="13" y1="14" x2="13" y2="15" elevation="1"/>
  <line stroke="blue" x1="14" y1="13" x2="14" y2="14" elevation="0"/>
  <line stroke="blue" x1="14" y1="13" x2="15" y2="13"/>
  <line stroke="red" x1="15" y1="12" x2="15" y2="13" elevation="1"/>
  <line stroke="blue" x1="15" y1="12" x2="17" y2="12"/>
  <line stroke="red" x1="17" y1="9" x2="17" y2="10" elevation="1"/>
  <line stroke="blue" x1="17" y1="9" x2="18" y2="9"/>
  <line stroke="red" x1="7" y1="18" x2="8" y2="18"/>
  <line stroke="red" x1="7" y1="18" x2="7" y2="20" elevation="1"/>
  <line stroke="blue" x1="5" y1="19" x2="4" y2="20"/>
  <line stroke="red" x1="5" y1="19" x2="5" y2="20" elevation="1"/>
  <line stroke="red" x1="19" y1="5" x2="20" y2="4"/>
  <line stroke="red" x1="19" y1="5" x2="19" y2="9" elevation="1"/>
  <line stroke="blue" x1="8" y1="18" x2="9" y2="18"/>
  <line stroke="blue" x1="8" y1="18" x2="8" y2="20" elevation="0"/>
  <line stroke="red" x1="17" y1="10" x2="17" y2="11" elevation="1"/>
  <line stroke="red" x1="17" y1="10" x2="18" y2="10"/>
  <line stroke="blue" x1="14" y1="14" x2="14" y2="15" elevation="0"/>
  <line stroke="red" x1="14" y1="14" x2="15" y2="14"/>
  <line stroke="blue" x1="13" y1="15" x2="12" y2="16"/>
  <line stroke="red" x1="13" y1="15" x2="14" y2="15"/>
  <line stroke="red" x1="13" y1="15" x2="13" y2="16" elevation="1"/>
  <line stroke="red" x1="15" y1="13" x2="15" y2="14" elevation="1"/>
  <line stroke="red" x1="15" y1="13" x2="17" y2="13"/>
  <line stroke="blue" x1="0" y1="20" x2="4" y2="20" elevation="0"/>
  <line stroke="red" x1="0" y1="20" x2="1" y2="21"/>
  <line stroke="black" x1="0" y1="20" x2="0" y2="24"/>
  <line stroke="red" x1="12" y1="16" x2="13" y2="16"/>
  <line stroke="blue" x1="12" y1="16" x2="12" y2="18" elevation="0"/>
  <line stroke="red" x1="9" y1="18" x2="10" y2="18"/>
  <line stroke="red" x1="9" y1="18" x2="9" y2="20" elevation="1"/>
  <line stroke="blue" x1="18" y1="9" x2="18" y2="10" elevation="0"/>
  <line stroke="red" x1="18" y1="9" x2="19" y2="9"/>
  <line stroke="red" x1="17" y1="11" x2="17" y2="12" elevation="1"/>
  <line stroke="blue" x1="17" y1="11" x2="18" y2="11"/>
  <line stroke="red" x1="4" y1="20" x2="5" y2="20"/>
  <line stroke="blue" x1="4" y1="20" x2="5" y2="21"/>
  <line stroke="blue" x1="20" y1="4" x2="21" y2="3"/>
  <line stroke="blue" x1="20" y1="4" x2="20" y2="8" elevation="0"/>
  <line stroke="blue" x1="14" y1="15" x2="15" y2="15"/>
  <line stroke="blue" x1="14" y1="15" x2="14" y2="16" elevation="0"/>
  <line stroke="red" x1="15" y1="14" x2="15" y2="15" elevation="1"/>
  <line stroke="blue" x1="15" y1="14" x2="17" y2="14"/>
  <line stroke="blue" x1="10" y1="18" x2="11" y2="18"/>
  <line stroke="blue" x1="10" y1="18" x2="10" y2="20" elevation="0"/>
  <line stroke="blue" x1="18" y1="10" x2="18" y2="11" elevation="0"/>
  <line stroke="blue" x1="18" y1="10" x2="19" y2="10"/>
  <line stroke="blue" x1="5" y1="20" x2="6" y2="20"/>
  <line stroke="red" x1="5" y1="20" x2="5" y2="21" elevation="1"/>
  <line stroke="blue" x1="13" y1="16" x2="14" y2="16"/>
  <line stroke="red" x1="13" y1="16" x2="13" y2="18" elevation="1"/>
  <line stroke="red" x1="17" y1="12" x2="17" y2="13" elevation="1"/>
  <line stroke="red" x1="17" y1="12" x2="18" y2="12"/>
  <line stroke="red" x1="6" y1="20" x2="7" y2="20"/>
  <line stroke="blue" x1="6" y1="20" x2="6" y2="22" elevation="0"/>
  <line stroke="red" x1="1" y1="21" x2="5" y2="21" elevation="1"/>
  <line stroke="blue" x1="1" y1="21" x2="2" y2="22"/>
  <line stroke="red" x1="1" y1="21" x2="1" y2="24" elevation="1"/>
  <line stroke="red" x1="19" y1="9" x2="19" y2="10" elevation="1"/>
  <line stroke="blue" x1="19" y1="9" x2="20" y2="8"/>
  <line stroke="blue" x1="19" y1="9" x2="20" y2="10"/>
  <line stroke="red" x1="19" y1="9" x2="21" y2="9" elevation="1"/>
  <line stroke="red" x1="11" y1="18" x2="12" y2="18"/>
  <line stroke="red" x1="11" y1="18" x2="11" y2="20" elevation="1"/>
  <line stroke="blue" x1="18" y1="11" x2="18" y2="12" elevation="0"/>
  <line stroke="red" x1="18" y1="11" x2="19" y2="11"/>
  <line stroke="blue" x1="7" y1="20" x2="8" y2="20"/>
  <line stroke="red" x1="7" y1="20" x2="7" y2="23" elevation="1"/>
  <line stroke="red" x1="15" y1="15" x2="15" y2="16" elevation="1"/>
  <line stroke="red" x1="15" y1="15" x2="17" y2="15"/>
  <line stroke="red" x1="21" y1="3" x2="22" y2="2"/>
  <line stroke="red" x1="21" y1="3" x2="21" y2="8" elevation="1"/>
  <line stroke="red" x1="14" y1="16" x2="15" y2="16"/>
  <line stroke="blue" x1="14" y1="16" x2="14" y2="18" elevation="0"/>
  <line stroke="red" x1="17" y1="13" x2="17" y2="14" elevation="1"/>
  <line stroke="blue" x1="17" y1="13" x2="18" y2="13"/>
  <line stroke="red" x1="19" y1="10" x2="19" y2="11" elevation="1"/>
  <line stroke="red" x1="19" y1="10" x2="20" y2="10"/>
  <line stroke="red" x1="8" y1="20" x2="9" y2="20"/>
  <line stroke="blue" x1="8" y1="20" x2="8" y2="24" elevation="0"/>
  <line stroke="red" x1="20" y1="8" x2="21" y2="8"/>
  <line stroke="blue" x1="20" y1="8" x2="21" y2="9"/>
  <line stroke="red" x1="5" y1="21" x2="6" y2="22"/>
  <line stroke="blue" x1="12" y1="18" x2="13" y2="18"/>
  <line stroke="blue" x1="12" y1="18" x2="12" y2="20" elevation="0"/>
  <line stroke="blue" x1="18" y1="12" x2="18" y2="13" elevation="0"/>
  <line stroke="blue" x1="18" y1="12" x2="19" y2="12"/>
  <line stroke="blue" x1="9" y1="20" x2="10" y2="20"/>
  <line stroke="red" x1="9" y1="20" x2="9" y2="25" elevation="1"/>
  <line stroke="blue" x1="15" y1="16" x2="17" y2="16"/>
  <line stroke="red" x1="15" y1="16" x2="15" y2="18" elevation="1"/>
  <line stroke="blue" x1="19" y1="11" x2="20" y2="10"/>
  <line stroke="red" x1="19" y1="11" x2="19" y2="12" elevation="1"/>
  <line stroke="blue" x1="19" y1="11" x2="20" y2="12"/>
  <line stroke="red" x1="19" y1="11" x2="23" y2="11" elevation="1"/>
  <line stroke="red" x1="17" y1="14" x2="17" y2="15" elevation="1"/>
  <line stroke="red" x1="17" y1="14" x2="18" y2="14"/>
  <line stroke="blue" x1="2" y1="22" x2="6" y2="22" elevation="0"/>
  <line stroke="red" x1="2" y1="22" x2="3" y2="23"/>
  <line stroke="blue" x1="2" y1="22" x2="2" y2="24" elevation="0"/>
  <line stroke="blue" x1="22" y1="2" x2="23" y2="1"/>
  <line stroke="blue" x1="22" y1="2" x2="22" y2="8" elevation="0"/>
  <line stroke="red" x1="13" y1="18" x2="14" y2="18"/>
  <line stroke="red" x1="13" y1="18" x2="13" y2="20" elevation="1"/>
  <line stroke="blue" x1="18" y1="13" x2="18" y2="14" elevation="0"/>
  <line stroke="red" x1="18" y1="13" x2="19" y2="13"/>
  <line stroke="red" x1="10" y1="20" x2="11" y2="20"/>
  <line stroke="blue" x1="10" y1="20" x2="10" y2="26" elevation="0"/>
  <line stroke="blue" x1="20" y1="10" x2="22" y2="10" elevation="0"/>
  <line stroke="red" x1="19" y1="12" x2="19" y2="13" elevation="1"/>
  <line stroke="red" x1="19" y1="12" x2="20" y2="12"/>
  <line stroke="red" x1="21" y1="8" x2="21" y2="9" elevation="1"/>
  <line stroke="blue" x1="21" y1="8" x2="22" y2="8"/>
  <line stroke="red" x1="17" y1="15" x2="17" y2="16" elevation="1"/>
  <line stroke="blue" x1="17" y1="15" x2="18" y2="15"/>
  <line stroke="blue" x1="6" y1="22" x2="7" y2="23"/>
  <line stroke="blue" x1="14" y1="18" x2="15" y2="18"/>
  <line stroke="blue" x1="14" y1="18" x2="14" y2="20" elevation="0"/>
  <line stroke="blue" x1="18" y1="14" x2="18" y2="15" elevation="0"/>
  <line stroke="blue" x1="18" y1="14" x2="19" y2="14"/>
  <line stroke="blue" x1="11" y1="20" x2="12" y2="20"/>
  <line stroke="red" x1="11" y1="20" x2="11" y2="27" elevation="1"/>
  <line stroke="red" x1="21" y1="9" x2="22" y2="10"/>
  <line stroke="blue" x1="19" y1="13" x2="20" y2="12"/>
  <line stroke="red" x1="19" y1="13" x2="19" y2="14" elevation="1"/>
  <line stroke="blue" x1="19" y1="13" x2="20" y2="14"/>
  <line stroke="red" x1="19" y1="13" x2="23" y2="13" elevation="1"/>
  <line stroke="red" x1="23" y1="1" x2="24" y2="0"/>
  <line stroke="red" x1="23" y1="1" x2="23" y2="8" elevation="1"/>
  <line stroke="red" x1="3" y1="23" x2="7" y2="23" elevation="1"/>
  <line stroke="red" x1="3" y1="23" x2="3" y2="24" elevation="1"/>
  <line stroke="blue" x1="3" y1="23" x2="4" y2="24"/>
  <line stroke="red" x1="12" y1="20" x2="13" y2="20"/>
  <line stroke="blue" x1="12" y1="20" x2="12" y2="28" elevation="0"/>
  <line stroke="blue" x1="20" y1="12" x2="24" y2="12" elevation="0"/>
  <line stroke="red" x1="17" y1="16" x2="18" y2="16"/>
  <line stroke="red" x1="17" y1="16" x2="17" y2="18" elevation="1"/>
  <line stroke="blue" x1="22" y1="8" x2="22" y2="10" elevation="0"/>
  <line stroke="red" x1="22" y1="8" x2="23" y2="8"/>
  <line stroke="red" x1="15" y1="18" x2="17" y2="18"/>
  <line stroke="red" x1="15" y1="18" x2="15" y2="20" elevation="1"/>
  <line stroke="blue" x1="18" y1="15" x2="18" y2="16" elevation="0"/>
  <line stroke="red" x1="18" y1="15" x2="19" y2="15"/>
  <line stroke="red" x1="19" y1="14" x2="19" y2="15" elevation="1"/>
  <line stroke="red" x1="19" y1="14" x2="20" y2="14"/>
  <line stroke="blue" x1="13" y1="20" x2="14" y2="20"/>
  <line stroke="red" x1="13" y1="20" x2="13" y2="29" elevation="1"/>
  <line stroke="blue" x1="0" y1="24" x2="1" y2="24"/>
  <line stroke="black" x1="0" y1="24" x2="0" y2="28"/>
  <line stroke="red" x1="24" y1="0" x2="25" y2="1"/>
  <line stroke="blue" x1="24" y1="0" x2="24" y2="8" elevation="0"/>
  <line stroke="black" x1="24" y1="0" x2="32" y2="0"/>
  <line stroke="red" x1="1" y1="24" x2="2" y2="24"/>
  <line stroke="red" x1="1" y1="24" x2="1" y2="27" elevation="1"/>
  <line stroke="red" x1="7" y1="23" x2="8" y2="24"/>
  <line stroke="blue" x1="2" y1="24" x2="3" y2="24"/>
  <line stroke="blue" x1="2" y1="24" x2="2" y2="26" elevation="0"/>
  <line stroke="blue" x1="18" y1="16" x2="19" y2="16"/>
  <line stroke="blue" x1="18" y1="16" x2="18" y2="18" elevation="0"/>
  <line stroke="blue" x1="22" y1="10" x2="23" y2="11"/>
  <line stroke="red" x1="3" y1="24" x2="4" y2="24"/>
  <line stroke="red" x1="3" y1="24" x2="3" y2="25" elevation="1"/>
  <line stroke="blue" x1="19" y1="15" x2="20" y2="14"/>
  <line stroke="red" x1="19" y1="15" x2="19" y2="16" elevation="1"/>
  <line stroke="blue" x1="19" y1="15" x2="20" y2="16"/>
  <line stroke="red" x1="19" y1="15" x2="21" y2="15" elevation="1"/>
  <line stroke="blue" x1="4" y1="24" x2="3" y2="25"/>
  <line stroke="blue" x1="4" y1="24" x2="8" y2="24" elevation="0"/>
  <line stroke="blue" x1="23" y1="8" x2="24" y2="8"/>
  <line stroke="red" x1="23" y1="8" x2="23" y2="11" elevation="1"/>
  <line stroke="red" x1="14" y1="20" x2="15" y2="20"/>
  <line stroke="blue" x1="14" y1="20" x2="14" y2="30" elevation="0"/>
  <line stroke="blue" x1="20" y1="14" x2="22" y2="14" elevation="0"/>
  <line stroke="blue" x1="17" y1="18" x2="18" y2="18"/>
  <line stroke="red" x1="17" y1="18" x2="17" y2="20" elevation="1"/>
  <line stroke="red" x1="19" y1="16" x2="20" y2="16"/>
  <line stroke="red" x1="19" y1="16" x2="19" y2="18" elevation="1"/>
  <line stroke="blue" x1="15" y1="20" x2="17" y2="20"/>
  <line stroke="red" x1="15" y1="20" x2="15" y2="31" elevation="1"/>
  <line stroke="blue" x1="25" y1="1" x2="26" y2="2"/>
  <line stroke="red" x1="25" y1="1" x2="25" y2="8" elevation="1"/>
  <line stroke="red" x1="25" y1="1" x2="31" y2="1" elevation="1"/>
  <line stroke="red" x1="3" y1="25" x2="2" y2="26"/>
  <line stroke="red" x1="3" y1="25" x2="9" y2="25" elevation="1"/>
  <line stroke="blue" x1="8" y1="24" x2="9" y2="25"/>
  <line stroke="blue" x1="24" y1="8" x2="25" y2="8"/>
  <line stroke="red" x1="24" y1="8" x2="24" y2="12" elevation="0"/>
  <line stroke="red" x1="18" y1="18" x2="19" y2="18"/>
  <line stroke="blue" x1="18" y1="18" x2="18" y2="20" elevation="0"/>
  <line stroke="red" x1="23" y1="11" x2="24" y2="12"/>
  <line stroke="blue" x1="20" y1="16" x2="21" y2="15"/>
  <line stroke="blue" x1="20" y1="16" x2="20" y2="18" elevation="0"/>
  <line stroke="red" x1="21" y1="15" x2="22" y2="14"/>
  <line stroke="red" x1="21" y1="15" x2="21" y2="18" elevation="1"/>
  <line stroke="blue" x1="2" y1="26" x2="1" y2="27"/>
  <line stroke="blue" x1="2" y1="26" x2="10" y2="26" elevation="0"/>
  <line stroke="blue" x1="22" y1="14" x2="23" y2="13"/>
  <line stroke="blue" x1="22" y1="14" x2="22" y2="18" elevation="0"/>
  <line stroke="red" x1="26" y1="2" x2="27" y2="3"/>
  <line stroke="blue" x1="26" y1="2" x2="26" y2="8" elevation="0"/>
  <line stroke="blue" x1="26" y1="2" x2="30" y2="2" elevation="0"/>
  <line stroke="blue" x1="19" y1="18" x2="20" y2="18"/>
  <line stroke="red" x1="19" y1="18" x2="19" y2="20" elevation="1"/>
  <line stroke="red" x1="17" y1="20" x2="18" y2="20"/>
  <line stroke="red" x1="17" y1="20" x2="17" y2="31" elevation="1"/>
  <line stroke="red" x1="25" y1="8" x2="26" y2="8"/>
  <line stroke="red" x1="25" y1="8" x2="25" y2="11" elevation="1"/>
  <line stroke="red" x1="23" y1="13" x2="24" y2="12"/>
  <line stroke="red" x1="23" y1="13" x2="23" y2="18" elevation="1"/>
  <line stroke="red" x1="9" y1="25" x2="10" y2="26"/>
  <line stroke="red" x1="24" y1="12" x2="25" y2="11"/>
  <line stroke="red" x1="24" y1="12" x2="25" y2="13"/>
  <line stroke="blue" x1="24" y1="12" x2="24" y2="18" elevation="0"/>
  <line stroke="blue" x1="24" y1="12" x2="28" y2="12" elevation="0"/>
  <line stroke="blue" x1="18" y1="20" x2="19" y2="20"/>
  <line stroke="blue" x1="18" y1="20" x2="18" y2="30" elevation="0"/>
  <line stroke="red" x1="20" y1="18" x2="21" y2="18"/>
  <line stroke="blue" x1="20" y1="18" x2="20" y2="20" elevation="0"/>
  <line stroke="red" x1="1" y1="27" x2="0" y2="28"/>
  <line stroke="red" x1="1" y1="27" x2="11" y2="27" elevation="1"/>
  <line stroke="red" x1="27" y1="3" x2="27" y2="8" elevation="1"/>
  <line stroke="blue" x1="27" y1="3" x2="28" y2="4"/>
  <line stroke="red" x1="27" y1="3" x2="29" y2="3" elevation="1"/>
  <line stroke="blue" x1="26" y1="8" x2="26" y2="10" elevation="0"/>
  <line stroke="blue" x1="26" y1="8" x2="27" y2="8"/>
  <line stroke="blue" x1="25" y1="11" x2="26" y2="10"/>
  <line stroke="red" x1="25" y1="11" x2="28" y2="11" elevation="1"/>
  <line stroke="red" x1="19" y1="20" x2="20" y2="20"/>
  <line stroke="red" x1="19" y1="20" x2="19" y2="29" elevation="1"/>
  <line stroke="blue" x1="21" y1="18" x2="22" y2="18"/>
  <line stroke="red" x1="21" y1="18" x2="21" y2="20" elevation="1"/>
  <line stroke="blue" x1="10" y1="26" x2="11" y2="27"/>
  <line stroke="red" x1="26" y1="10" x2="27" y2="9"/>
  <line stroke="blue" x1="26" y1="10" x2="28" y2="10" elevation="0"/>
  <line stroke="red" x1="0" y1="28" x2="1" y2="29"/>
  <line stroke="blue" x1="0" y1="28" x2="12" y2="28" elevation="0"/>
  <line stroke="black" x1="0" y1="28" x2="0" y2="32"/>
  <line stroke="red" x1="27" y1="8" x2="27" y2="9" elevation="1"/>
  <line stroke="red" x1="27" y1="8" x2="28" y2="8"/>
  <line stroke="blue" x1="25" y1="13" x2="26" y2="14"/>
  <line stroke="red" x1="25" y1="13" x2="25" y2="18" elevation="1"/>
  <line stroke="red" x1="25" y1="13" x2="28" y2="13" elevation="1"/>
  <line stroke="blue" x1="20" y1="20" x2="21" y2="20"/>
  <line stroke="blue" x1="20" y1="20" x2="20" y2="28" elevation="0"/>
  <line stroke="blue" x1="28" y1="4" x2="28" y2="8" elevation="0"/>
  <line stroke="blue" x1="28" y1="4" x2="29" y2="3"/>
  <line stroke="red" x1="28" y1="4" x2="29" y2="4"/>
  <line stroke="red" x1="22" y1="18" x2="23" y2="18"/>
  <line stroke="blue" x1="22" y1="18" x2="22" y2="20" elevation="0"/>
  <line stroke="blue" x1="27" y1="9" x2="28" y2="8"/>
  <line stroke="red" x1="27" y1="9" x2="28" y2="9" elevation="1"/>
  <line stroke="red" x1="21" y1="20" x2="22" y2="20"/>
  <line stroke="red" x1="21" y1="20" x2="21" y2="27" elevation="1"/>
  <line stroke="blue" x1="1" y1="29" x2="2" y2="30"/>
  <line stroke="red" x1="1" y1="29" x2="13" y2="29" elevation="1"/>
  <line stroke="red" x1="1" y1="29" x2="1" y2="32" elevation="1"/>
  <line stroke="blue" x1="28" y1="8" x2="28" y2="9"/>
  <line stroke="red" x1="28" y1="8" x2="29" y2="8"/>
  <line stroke="blue" x1="28" y1="8" x2="29" y2="9"/>
  <line stroke="red" x1="11" y1="27" x2="12" y2="28"/>
  <line stroke="red" x1="29" y1="3" x2="29" y2="4" elevation="1"/>
  <line stroke="red" x1="29" y1="3" x2="30" y2="2"/>
  <line stroke="blue" x1="23" y1="18" x2="24" y2="18"/>
  <line stroke="red" x1="23" y1="18" x2="23" y2="20" elevation="1"/>
  <line stroke="red" x1="29" y1="4" x2="29" y2="8" elevation="1"/>
  <line stroke="blue" x1="29" y1="4" x2="30" y2="4"/>
  <line stroke="red" x1="28" y1="9" x2="28" y2="10"/>
  <line stroke="red" x1="28" y1="9" x2="29" y2="9" elevation="1"/>
  <line stroke="red" x1="26" y1="14" x2="27" y2="15"/>
  <line stroke="blue" x1="26" y1="14" x2="28" y2="14" elevation="0"/>
  <line stroke="blue" x1="26" y1="14" x2="26" y2="18" elevation="0"/>
  <line stroke="blue" x1="22" y1="20" x2="23" y2="20"/>
  <line stroke="blue" x1="22" y1="20" x2="22" y2="26" elevation="0"/>
  <line stroke="blue" x1="28" y1="10" x2="28" y2="11"/>
  <line stroke="blue" x1="28" y1="10" x2="30" y2="10" elevation="0"/>
  <line stroke="red" x1="24" y1="18" x2="25" y2="18"/>
  <line stroke="blue" x1="24" y1="18" x2="24" y2="20" elevation="0"/>
  <line stroke="red" x1="2" y1="30" x2="3" y2="31"/>
  <line stroke="blue" x1="2" y1="30" x2="2" y2="32" elevation="0"/>
  <line stroke="blue" x1="2" y1="30" x2="14" y2="30" elevation="0"/>
  <line stroke="blue" x1="30" y1="2" x2="30" y2="4" elevation="0"/>
  <line stroke="blue" x1="30" y1="2" x2="31" y2="1"/>
  <line stroke="red" x1="28" y1="11" x2="28" y2="12"/>
  <line stroke="red" x1="28" y1="11" x2="31" y2="11" elevation="1"/>
  <line stroke="red" x1="29" y1="8" x2="29" y2="9" elevation="1"/>
  <line stroke="blue" x1="29" y1="8" x2="30" y2="8"/>
  <line stroke="blue" x1="30" y1="4" x2="30" y2="8" elevation="0"/>
  <line stroke="red" x1="30" y1="4" x2="31" y2="4"/>
  <line stroke="red" x1="29" y1="9" x2="30" y2="10"/>
  <line stroke="blue" x1="12" y1="28" x2="13" y2="29"/>
  <line stroke="blue" x1="28" y1="12" x2="28" y2="13"/>
  <line stroke="blue" x1="28" y1="12" x2="32" y2="12" elevation="0"/>
  <line stroke="red" x1="23" y1="20" x2="24" y2="20"/>
  <line stroke="red" x1="23" y1="20" x2="23" y2="25" elevation="1"/>
  <line stroke="blue" x1="25" y1="18" x2="26" y2="18"/>
  <line stroke="red" x1="25" y1="18" x2="25" y2="20" elevation="1"/>
  <line stroke="red" x1="28" y1="13" x2="28" y2="14"/>
  <line stroke="red" x1="28" y1="13" x2="31" y2="13" elevation="1"/>
  <line stroke="red" x1="27" y1="15" x2="28" y2="15" elevation="1"/>
  <line stroke="red" x1="27" y1="15" x2="27" y2="17" elevation="1"/>
  <line stroke="blue" x1="27" y1="15" x2="28" y2="16"/>
  <line stroke="red" x1="31" y1="1" x2="31" y2="4" elevation="1"/>
  <line stroke="red" x1="31" y1="1" x2="32" y2="0"/>
  <line stroke="blue" x1="30" y1="8" x2="30" y2="10" elevation="0"/>
  <line stroke="red" x1="30" y1="8" x2="31" y2="8"/>
  <line stroke="red" x1="3" y1="31" x2="3" y2="32" elevation="1"/>
  <line stroke="blue" x1="3" y1="31" x2="4" y2="32"/>
  <line stroke="red" x1="3" y1="31" x2="15" y2="31" elevation="1"/>
  <line stroke="blue" x1="24" y1="20" x2="25" y2="20"/>
  <line stroke="blue" x1="24" y1="20" x2="24" y2="24" elevation="0"/>
  <line stroke="red" x1="31" y1="4" x2="31" y2="8" elevation="1"/>
  <line stroke="blue" x1="31" y1="4" x2="32" y2="4"/>
  <line stroke="blue" x1="28" y1="14" x2="28" y2="15"/>
  <line stroke="blue" x1="28" y1="14" x2="30" y2="14" elevation="0"/>
  <line stroke="red" x1="26" y1="18" x2="27" y2="17"/>
  <line stroke="blue" x1="26" y1="18" x2="26" y2="20" elevation="0"/>
  <line stroke="red" x1="26" y1="18" x2="27" y2="19"/>
  <line stroke="blue" x1="26" y1="18" x2="30" y2="18" elevation="0"/>
  <line stroke="blue" x1="30" y1="10" x2="31" y2="11"/>
  <line stroke="red" x1="28" y1="15" x2="28" y2="16"/>
  <line stroke="red" x1="28" y1="15" x2="29" y2="15" elevation="1"/>
  <line stroke="red" x1="13" y1="29" x2="14" y2="30"/>
  <line stroke="blue" x1="27" y1="17" x2="28" y2="16"/>
  <line stroke="red" x1="27" y1="17" x2="29" y2="17" elevation="1"/>
  <line stroke="black" x1="0" y1="32" x2="1" y2="32"/>
  <line stroke="black" x1="32" y1="0" x2="32" y2="4"/>
  <line stroke="black" x1="1" y1="32" x2="2" y2="32"/>
  <line stroke="red" x1="25" y1="20" x2="26" y2="20"/>
  <line stroke="red" x1="25" y1="20" x2="25" y2="23" elevation="1"/>
  <line stroke="red" x1="31" y1="8" x2="31" y2="11" elevation="1"/>
  <line stroke="blue" x1="31" y1="8" x2="32" y2="8"/>
  <line stroke="black" x1="2" y1="32" x2="3" y2="32"/>
  <line stroke="black" x1="3" y1="32" x2="4" y2="32"/>
  <line stroke="black" x1="4" y1="32" x2="16" y2="32"/>
  <line stroke="blue" x1="28" y1="16" x2="29" y2="15"/>
  <line stroke="red" x1="28" y1="16" x2="29" y2="16"/>
  <line stroke="blue" x1="28" y1="16" x2="29" y2="17"/>
  <line stroke="black" x1="32" y1="4" x2="32" y2="8"/>
  <line stroke="red" x1="29" y1="15" x2="30" y2="14"/>
  <line stroke="red" x1="29" y1="15" x2="29" y2="16" elevation="1"/>
  <line stroke="blue" x1="26" y1="20" x2="27" y2="20"/>
  <line stroke="blue" x1="26" y1="20" x2="26" y2="22" elevation="0"/>
  <line stroke="red" x1="31" y1="11" x2="32" y2="12"/>
  <line stroke="black" x1="32" y1="8" x2="32" y2="12"/>
  <line stroke="red" x1="27" y1="19" x2="27" y2="20" elevation="1"/>
  <line stroke="blue" x1="27" y1="19" x2="28" y2="20"/>
  <line stroke="red" x1="27" y1="19" x2="31" y2="19" elevation="1"/>
  <line stroke="blue" x1="14" y1="30" x2="15" y2="31"/>
  <line stroke="blue" x1="30" y1="14" x2="31" y2="13"/>
  <line stroke="blue" x1="30" y1="14" x2="30" y2="16" elevation="0"/>
  <line stroke="red" x1="29" y1="16" x2="29" y2="17" elevation="1"/>
  <line stroke="blue" x1="29" y1="16" x2="30" y2="16"/>
  <line stroke="red" x1="27" y1="20" x2="27" y2="21" elevation="1"/>
  <line stroke="red" x1="27" y1="20" x2="28" y2="20"/>
  <line stroke="red" x1="29" y1="17" x2="30" y2="18"/>
  <line stroke="red" x1="31" y1="13" x2="32" y2="12"/>
  <line stroke="red" x1="31" y1="13" x2="31" y2="16" elevation="1"/>
  <line stroke="blue" x1="24" y1="24" x2="23" y2="25"/>
  <line stroke="red" x1="24" y1="24" x2="25" y2="23"/>
  <line stroke="blue" x1="24" y1="24" x2="28" y2="24" elevation="0"/>
  <line stroke="red" x1="23" y1="25" x2="22" y2="26"/>
  <line stroke="red" x1="23" y1="25" x2="29" y2="25" elevation="1"/>
  <line stroke="blue" x1="25" y1="23" x2="26" y2="22"/>
  <line stroke="red" x1="25" y1="23" x2="29" y2="23" elevation="1"/>
  <line stroke="red" x1="30" y1="16" x2="31" y2="16"/>
  <line stroke="blue" x1="30" y1="16" x2="30" y2="18" elevation="0"/>
  <line stroke="blue" x1="22" y1="26" x2="21" y2="27"/>
  <line stroke="blue" x1="22" y1="26" x2="30" y2="26" elevation="0"/>
  <line stroke="red" x1="26" y1="22" x2="27" y2="21"/>
  <line stroke="blue" x1="26" y1="22" x2="30" y2="22" elevation="0"/>
  <line stroke="black" x1="32" y1="12" x2="32" y2="16"/>
  <line stroke="red" x1="21" y1="27" x2="20" y2="28"/>
  <line stroke="red" x1="21" y1="27" x2="31" y2="27" elevation="1"/>
  <line stroke="blue" x1="27" y1="21" x2="28" y2="20"/>
  <line stroke="red" x1="27" y1="21" x2="31" y2="21" elevation="1"/>
  <line stroke="blue" x1="20" y1="28" x2="19" y2="29"/>
  <line stroke="blue" x1="20" y1="28" x2="32" y2="28" elevation="0"/>
  <line stroke="blue" x1="28" y1="20" x2="32" y2="20" elevation="0"/>
  <line stroke="red" x1="15" y1="31" x2="16" y2="32"/>
  <line stroke="red" x1="19" y1="29" x2="18" y2="30"/>
  <line stroke="red" x1="19" y1="29" x2="31" y2="29" elevation="1"/>
  <line stroke="blue" x1="31" y1="16" x2="32" y2="16"/>
  <line stroke="red" x1="31" y1="16" x2="31" y2="19" elevation="1"/>
  <line stroke="blue" x1="18" y1="30" x2="17" y2="31"/>
  <line stroke="blue" x1="18" y1="30" x2="30" y2="30" elevation="0"/>
  <line stroke="blue" x1="30" y1="18" x2="31" y2="19"/>
  <line stroke="red" x1="17" y1="31" x2="16" y2="32"/>
  <line stroke="red" x1="17" y1="31" x2="29" y2="31" elevation="1"/>
  <line stroke="black" x1="16" y1="32" x2="28" y2="32"/>
  <line stroke="black" x1="32" y1="16" x2="32" y2="20"/>
  <line stroke="red" x1="31" y1="19" x2="32" y2="20"/>
  <line stroke="blue" x1="28" y1="24" x2="29" y2="23"/>
  <line stroke="red" x1="28" y1="24" x2="29" y2="24"/>
  <line stroke="blue" x1="28" y1="24" x2="29" y2="25"/>
  <line stroke="red" x1="29" y1="23" x2="30" y2="22"/>
  <line stroke="red" x1="29" y1="23" x2="29" y2="24" elevation="1"/>
  <line stroke="blue" x1="30" y1="22" x2="31" y2="21"/>
  <line stroke="blue" x1="30" y1="22" x2="30" y2="24" elevation="0"/>
  <line stroke="red" x1="31" y1="21" x2="32" y2="20"/>
  <line stroke="red" x1="31" y1="21" x2="31" y2="24" elevation="1"/>
  <line stroke="red" x1="29" y1="24" x2="29" y2="25" elevation="1"/>
  <line stroke="blue" x1="29" y1="24" x2="30" y2="24"/>
  <line stroke="black" x1="32" y1="20" x2="32" y2="24"/>
  <line stroke="red" x1="29" y1="25" x2="30" y2="26"/>
  <line stroke="red" x1="30" y1="24" x2="31" y2="24"/>
  <line stroke="blue" x1="30" y1="24" x2="30" y2="26" elevation="0"/>
  <line stroke="blue" x1="31" y1="24" x2="32" y2="24"/>
  <line stroke="red" x1="31" y1="24" x2="31" y2="27" elevation="1"/>
  <line stroke="blue" x1="30" y1="26" x2="31" y2="27"/>
  <line stroke="black" x1="32" y1="24" x2="32" y2="28"/>
  <line stroke="red" x1="31" y1="27" x2="32" y2="28"/>
  <line stroke="red" x1="30" y1="30" x2="29" y2="31"/>
  <line stroke="blue" x1="30" y1="30" x2="31" y2="29"/>
  <line stroke="blue" x1="30" y1="30" x2="30" y2="32" elevation="0"/>
  <line stroke="blue" x1="29" y1="31" x2="28" y2="32"/>
  <line stroke="red" x1="29" y1="31" x2="29" y2="32" elevation="1"/>
  <line stroke="red" x1="31" y1="29" x2="32" y2="28"/>
  <line stroke="red" x1="31" y1="29" x2="31" y2="32" elevation="1"/>
  <line stroke="black" x1="28" y1="32" x2="29" y2="32"/>
  <line stroke="black" x1="32" y1="28" x2="32" y2="32"/>
  <line stroke="black" x1="29" y1="32" x2="30" y2="32"/>
  <line stroke="black" x1="30" y1="32" x2="31" y2="32"/>
  <line stroke="black" x1="31" y1="32" x2="32" y2="32"/>
</svg>""",
)

SNAKE = BoxPleatingCreasePatternExample(
    name="snake",
    grid_size=16,
    central_crease_removed=False,
    packing_svg="""<svg xmlns="http://www.w3.org/2000/svg" version="1.1" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:svgjs="http://svgjs.dev/svgjs" viewBox="-0.5 -0.5 17 17" width="17cm" height="17cm"><g class="crease"><line x1="0" y1="0" x2="16" y2="16" class="M" stroke-opacity="1" data-index="4" stroke-width="0.1" stroke="#ff0000"></line><line x1="0" y1="16" x2="16" y2="16" class="B" stroke-opacity="1" data-index="0" stroke="#000000" stroke-width="0.1"></line><line x1="16" y1="16" x2="16" y2="0" class="B" stroke-opacity="1" data-index="1" stroke="#000000" stroke-width="0.1"></line><line x1="16" y1="0" x2="0" y2="0" class="B" stroke-opacity="1" data-index="2" stroke="#000000" stroke-width="0.1"></line><line x1="0" y1="0" x2="0" y2="16" class="B" stroke-opacity="1" data-index="3" stroke="#000000" stroke-width="0.1"></line></g><style>line { stroke-linecap: round; }</style></svg>""",
    base_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">
  <rect stroke="black" x="0" y="0" width="16" height="16"/>
  <line stroke="blue" x1="0" y1="0" x2="1" y2="1"/>
  <line stroke="red" x1="0" y1="1" x2="1" y2="1" elevation="1"/>
  <line stroke="red" x1="1" y1="0" x2="1" y2="1" elevation="1"/>
  <line stroke="red" x1="1" y1="1" x2="2" y2="2"/>
  <line stroke="blue" x1="0" y1="2" x2="2" y2="2" elevation="0"/>
  <line stroke="blue" x1="2" y1="0" x2="2" y2="2" elevation="0"/>
  <line stroke="blue" x1="2" y1="2" x2="3" y2="3"/>
  <line stroke="red" x1="0" y1="3" x2="3" y2="3" elevation="1"/>
  <line stroke="red" x1="3" y1="0" x2="3" y2="3" elevation="1"/>
  <line stroke="blue" x1="0" y1="4" x2="4" y2="4" elevation="0"/>
  <line stroke="blue" x1="4" y1="0" x2="4" y2="4" elevation="0"/>
  <line stroke="red" x1="3" y1="3" x2="4" y2="4"/>
  <line stroke="red" x1="0" y1="5" x2="5" y2="5" elevation="1"/>
  <line stroke="red" x1="5" y1="0" x2="5" y2="5" elevation="1"/>
  <line stroke="blue" x1="4" y1="4" x2="5" y2="5"/>
  <line stroke="blue" x1="0" y1="6" x2="6" y2="6" elevation="0"/>
  <line stroke="blue" x1="6" y1="0" x2="6" y2="6" elevation="0"/>
  <line stroke="red" x1="0" y1="7" x2="7" y2="7" elevation="1"/>
  <line stroke="red" x1="7" y1="0" x2="7" y2="7" elevation="1"/>
  <line stroke="red" x1="5" y1="5" x2="6" y2="6"/>
  <line stroke="blue" x1="0" y1="8" x2="8" y2="8" elevation="0"/>
  <line stroke="blue" x1="8" y1="0" x2="8" y2="8" elevation="0"/>
  <line stroke="blue" x1="6" y1="6" x2="7" y2="7"/>
  <line stroke="red" x1="0" y1="9" x2="9" y2="9" elevation="1"/>
  <line stroke="red" x1="9" y1="0" x2="9" y2="9" elevation="1"/>
  <line stroke="red" x1="7" y1="7" x2="8" y2="8"/>
  <line stroke="blue" x1="0" y1="10" x2="10" y2="10" elevation="0"/>
  <line stroke="blue" x1="10" y1="0" x2="10" y2="10" elevation="0"/>
  <line stroke="red" x1="0" y1="11" x2="11" y2="11" elevation="1"/>
  <line stroke="red" x1="11" y1="0" x2="11" y2="11" elevation="1"/>
  <line stroke="blue" x1="8" y1="8" x2="9" y2="9"/>
  <line stroke="blue" x1="0" y1="12" x2="12" y2="12" elevation="0"/>
  <line stroke="blue" x1="12" y1="0" x2="12" y2="12" elevation="0"/>
  <line stroke="red" x1="9" y1="9" x2="10" y2="10"/>
  <line stroke="red" x1="0" y1="13" x2="13" y2="13" elevation="1"/>
  <line stroke="red" x1="13" y1="0" x2="13" y2="13" elevation="1"/>
  <line stroke="blue" x1="0" y1="14" x2="14" y2="14" elevation="0"/>
  <line stroke="blue" x1="14" y1="0" x2="14" y2="14" elevation="0"/>
  <line stroke="blue" x1="10" y1="10" x2="11" y2="11"/>
  <line stroke="red" x1="0" y1="15" x2="15" y2="15" elevation="1"/>
  <line stroke="red" x1="15" y1="0" x2="15" y2="15" elevation="1"/>
  <line stroke="red" x1="11" y1="11" x2="12" y2="12"/>
  <line stroke="blue" x1="12" y1="12" x2="13" y2="13"/>
  <line stroke="red" x1="13" y1="13" x2="14" y2="14"/>
  <line stroke="blue" x1="14" y1="14" x2="15" y2="15"/>
  <line stroke="red" x1="15" y1="15" x2="16" y2="16"/>
</svg>""")

VERTICAL_FAN_OUT_SHAPING_TEST1 = BoxPleatingCreasePatternExample(
    name="vertical_fan_out_shaping_test1",
    grid_size=8,
    packing_svg="",
    base_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8">
  <line stroke="black" x1="0" y1="0" x2="2" y2="0"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="4"/>
  <line stroke="black" x1="2" y1="0" x2="4" y2="0"/>
  <line stroke="blue" x1="2" y1="0" x2="2" y2="4"/>
  <line stroke="blue" x1="0" y1="4" x2="2" y2="4"/>
  <line stroke="black" x1="0" y1="4" x2="0" y2="8"/>
  <line stroke="red" x1="4" y1="0" x2="4" y2="4"/>
  <line stroke="black" x1="4" y1="0" x2="6" y2="0"/>
  <line stroke="red" x1="2" y1="4" x2="4" y2="4"/>
  <line stroke="blue" x1="2" y1="4" x2="2" y2="8"/>
  <line stroke="blue" x1="4" y1="4" x2="6" y2="4"/>
  <line stroke="red" x1="4" y1="4" x2="4" y2="8"/>
  <line stroke="blue" x1="6" y1="0" x2="6" y2="4"/>
  <line stroke="black" x1="6" y1="0" x2="8" y2="0"/>
  <line stroke="red" x1="6" y1="4" x2="8" y2="4"/>
  <line stroke="blue" x1="6" y1="4" x2="6" y2="8"/>
  <line stroke="black" x1="0" y1="8" x2="2" y2="8"/>
  <line stroke="black" x1="8" y1="0" x2="8" y2="4"/>
  <line stroke="black" x1="2" y1="8" x2="4" y2="8"/>
  <line stroke="black" x1="4" y1="8" x2="6" y2="8"/>
  <line stroke="black" x1="8" y1="4" x2="8" y2="8"/>
  <line stroke="black" x1="6" y1="8" x2="8" y2="8"/>
</svg>
""",
    rectangles=[
        shaping_utils.Rectangle(
            name="top_half",
            x_start=0,
            y_start=0,
            x_len=1,
            y_len=0.5,
        ),
        shaping_utils.Rectangle(
            name="botom_half",
            x_start=0,
            y_start=0.5,
            x_len=1,
            y_len=0.5,
        ),
    ],
)

VERTICAL_FAN_OUT_SHAPING_TEST2 = BoxPleatingCreasePatternExample(
    name="vertical_fan_out_shaping_test2",
    grid_size=8,
    packing_svg="",
    base_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="0" y2="500"/>
  <line stroke="black" x1="0" y1="0" x2="500" y2="0"/>
  <line stroke="red" x1="0" y1="0" x2="500" y2="500"/>
  <line stroke="red" x1="0" y1="500" x2="500" y2="500"/>
  <line stroke="black" x1="0" y1="500" x2="0" y2="1000"/>
  <line stroke="blue" x1="500" y1="0" x2="500" y2="500"/>
  <line stroke="black" x1="500" y1="0" x2="1000" y2="0"/>
  <line stroke="red" x1="500" y1="500" x2="0" y2="1000"/>
  <line stroke="red" x1="500" y1="500" x2="1000" y2="0"/>
  <line stroke="blue" x1="500" y1="500" x2="500" y2="1000"/>
  <line stroke="blue" x1="500" y1="500" x2="1000" y2="500"/>
  <line stroke="red" x1="500" y1="500" x2="1000" y2="1000"/>
  <line stroke="black" x1="0" y1="1000" x2="500" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="500"/>
  <line stroke="black" x1="500" y1="1000" x2="1000" y2="1000"/>
  <line stroke="black" x1="1000" y1="500" x2="1000" y2="1000"/>
</svg>""",
    rectangles=[
        shaping_utils.Rectangle(
            name="top_half",
            x_start=0,
            y_start=0,
            x_len=1,
            y_len=0.5,
        ),
        shaping_utils.Rectangle(
            name="bottom_half",
            x_start=0,
            y_start=0.5,
            x_len=1,
            y_len=0.5,
        ),
    ],
)


_FOLD_ORDERS = immutabledict.immutabledict({
    "MV": [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
    "VM": [crease_utils.Fold.VALLEY, crease_utils.Fold.MOUNTAIN],
})

@dataclasses.dataclass(frozen=True, kw_only=True)
class BoxPleatingShapingExample:
  """Shaped variations of BoxPleatingCreasePatternExample objects.

  Attributes:
    base_example: The base example to shape.
    stick_figure: The stick figure used to used to heuristically shape the
      example.
    central_crease_removed: Whether the central flap is removed from the
      base_cp.  For zigzag shaping tests:
    rectangles: Rectangles defining flaps.
    rivers: river shaping objects corresponding to rivers on the cp.
    assigned_hinge_name_pairs: Tuple of string names denoting the assigned
      hinges.
    target_zigzag_depths: The target zigzag depths. If None,
      target_flap_anglesmust be specified.
    target_flap_angles: The target flap angles. If None, target_zigzag_depths
      must be specified.
    fold_orders: a sequence of folds detailing the fold order; for example a
      fold order of [MOUNTAIN, VALLEY] will have the zigzag crease starting with
      a mountain fold and then alternating valley and mountain orientations,
      whereas a fold order of [VALLEY] will have zigzag creases be all valley
      folds.
    even_phases: boolean determining whether zigzags should be in even phase or
      odd phase. Phase determines the phase shift of the zigzags.
    zigzag_shaping_params: params determining zigzag placement for each flap.
    zigzag_svg: The svg string of the zigzag shaped crease pattern.
    inscribed_rectangle_shaping_params: params determining incribed rectangle
      placement for each flap.
    inscribed_shaping_params: params determining incribed rectangle placement
      for each flap or river.
    inscribed_rectangle_svg: The svg string of the incribed rectangle shaped
      crease pattern.
    narrow_flap_names: The names of the flaps to narrow. If None, all flaps will
      be narrowed.
    narrowed_svg: The svg string of the crease pattern after narrowing of flaps.
    crimp_fold_shaping_params: params determining crimp fold placement for each
      flap.
    crimp_fold_shaping_svg: The svg string of the crimp fold shaped crease
      pattern.
    reverse_fold_shaping_params: params determining reverse fold placement for
      each flap.
    reverse_fold_shaping_svg: The svg string of the reverse fold shaped crease
      pattern.
    fan_out_shaping_params: params determining fan out shaping for each flap.
    fan_out_svg: The svg string of the fan out shaped crease pattern.
    squash_sink_svg: The svg string of the squash sink shaped crease pattern.
  """

  base_example: BoxPleatingCreasePatternExample
  stick_figure: stick_figure_utils.StickFigure | None = None
  central_crease_removed: bool = True
  rectangles: (
      list[shaping_utils.Rectangle] | list[shaping_elements.ShapingFlap] | None
  ) = None
  rivers: list[shaping_elements.ShapingRiver] | None = None
  assigned_hinge_name_pairs: list[tuple[str, str]] = dataclasses.field(
      default_factory=list
  )

  # zigzag shaping info
  target_zigzag_depths: list[float | None] | None = None
  target_flap_angles: list[float | None] | None = None
  fold_orders: list[list[crease_utils.Fold]] | None = None
  even_phases: list[bool] | None = None

  zigzag_shaping_params: (
      list[zigzag_shaping.ZigzagShapingParamsForElement] | None
  ) = None

  zigzag_svg: str | None = None

  # inscribed rectangle shaping info
  inscribed_rectangle_shaping_params: (
      list[
          inscribed_rectangle_shaping.InscribedRectangleFlapShapingParams
          | inscribed_rectangle_shaping.InscribedShapingParamsForElement
      ]
      | None
  ) = None
  # inscribed shaping params for flaps or rivers
  inscribed_shaping_params: (
      list[inscribed_rectangle_shaping.InscribedShapingParamsForElement] | None
  ) = None
  inscribed_rectangle_svg: str | None = None

  # narrowing flaps shaping info
  # NOTE: Add shaping info for narrowing flaps.
  narrow_flap_names: Sequence[str] | None = None
  narrowed_svg: str | None = None

  # crimp folds shaping info
  crimp_fold_shaping_params: (
      list[crimp_fold_shaping.CrimpFoldShapingParams] | None
  ) = None
  crimp_fold_shaping_svg: str | None = None

  # reverse fold shaping info
  reverse_fold_shaping_params: (
      list[reverse_fold_shaping.ReverseFoldShapingParams] | None
  ) = None
  reverse_fold_shaping_svg: str | None = None

  # fan out shaping info
  fan_out_shaping_params: list[fan_out_shaping.FanOutShapingParams] | None = (
      None
  )
  fan_out_svg: str | None = None

  # squash sink shaping info
  squash_sink_svg: str | None = None

  @functools.cached_property
  def zigzag_cp(self) -> crease_pattern.CreasePattern | None:
    """Returns the zigzag shaped crease pattern."""
    return (
        parser.parse_svg(self.zigzag_svg)
        if self.zigzag_svg is not None
        else None
    )

  @functools.cached_property
  def inscribed_rectangle_cp(self) -> crease_pattern.CreasePattern | None:
    """Returns the inscribed rectangle shaped crease pattern."""
    return (
        parser.parse_svg(self.inscribed_rectangle_svg)
        if self.inscribed_rectangle_svg is not None
        else None
    )

  @functools.cached_property
  def flap_narrowed_cp(self) -> crease_pattern.CreasePattern | None:
    """Returns the flaps narrowed crease pattern."""
    return (
        parser.parse_svg(self.narrowed_svg)
        if self.narrowed_svg is not None
        else None
    )

  @functools.cached_property
  def crimp_fold_cp(self) -> crease_pattern.CreasePattern | None:
    """Returns the crimp folds shaped crease pattern."""
    return (
        parser.parse_svg(self.crimp_fold_shaping_svg)
        if self.crimp_fold_shaping_svg is not None
        else None
    )

  def check_svg_equality(self, svg1: str, svg2: str) -> bool:
    """Checks whether given svg is same as base_svg."""
    # load both to cp and back to svg
    svg1 = parser.to_svg(parser.parse_svg(svg1))
    svg2 = parser.to_svg(parser.parse_svg(svg2))
    return svg1 == svg2

  def check_zigzag_shaping(
      self, cp: crease_pattern.CreasePattern, ndigits: int = 3
  ) -> bool:
    """Checks whether given svg is same as zigzag_svg."""
    print(parser.to_svg(cp, ndigits=ndigits))
    print(self.zigzag_svg)
    return self.check_svg_equality(
        parser.to_svg(cp, ndigits=ndigits), self.zigzag_svg
    )

  def check_narrow_flap_shaping(
      self, cp: crease_pattern.CreasePattern, ndigits: int = 4
  ) -> bool:
    """Checks whether given svg is same as flap_narrowed_shaping_svg."""
    return self.check_svg_equality(
        parser.to_svg(cp, ndigits=ndigits), self.narrowed_svg
    )

  def check_inscribed_rectangle_shaping(
      self, cp: crease_pattern.CreasePattern
  ) -> bool:
    """Checks whether given svg is same as inscribed_rectangle_svg."""
    return self.check_svg_equality(
        parser.to_svg(cp), self.inscribed_rectangle_svg
    )

  def check_fan_out_shaping(
      self, cp: crease_pattern.CreasePattern, ndigits: int = 3
  ) -> bool:
    """Checks whether given svg is same as fan_out_svg."""
    return self.check_svg_equality(
        parser.to_svg(cp, ndigits=ndigits), self.fan_out_svg
    )

  def check_squash_sink_shaping(
      self, cp: crease_pattern.CreasePattern, ndigits: int = 3
  ) -> bool:
    """Checks whether given svg is same as squash_sink_svg."""
    return self.check_svg_equality(
        parser.to_svg(cp, ndigits=ndigits), self.squash_sink_svg
    )

def _make_hinge_crease(
    p1: tuple[float, float],
    p2: tuple[float, float],
    grid_size: int,
) -> crease_utils.Crease:
  """Make a hinge crease from two points."""
  return crease_utils.Crease(
      point_utils.Point(
          x=p1[0] / grid_size, y=p1[1] / grid_size, require_on_paper=False
      ),
      point_utils.Point(
          x=p2[0] / grid_size, y=p2[1] / grid_size, require_on_paper=False
      ),
      fold=crease_utils.Fold.HINGE,
  )

SCORPION_SHAPING_UNCLIPPED = BoxPleatingShapingExample(
    base_example=SCORPION,
    central_crease_removed=True,
    rectangles=[
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(x=0, y=0, require_on_paper=False)
            ],
            radius=4 / 32 * math.sqrt(2),
            name="leftpincer_1",
        ),
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(x=0.25, y=0, require_on_paper=False)
            ],
            radius=4 / 32 * math.sqrt(2),
            name="leftpincer_2",
        ),
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(x=0, y=12 / 32, require_on_paper=False)
            ],
            radius=4 / 32 * math.sqrt(2),
            name="limb5_left",
        ),
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(
                    x=4 / 32 + 4 / 32, y=12 / 32, require_on_paper=False
                )
            ],
            radius=4 / 32 * math.sqrt(2),
            name="limb4_left",
        ),
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(x=0, y=1 / 2 + 4 / 32, require_on_paper=False)
            ],
            radius=4 / 32 * math.sqrt(2),
            name="limb3_left",
        ),
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(x=0, y=1 - 4 / 32, require_on_paper=False)
            ],
            radius=4 / 32 * math.sqrt(2),
            name="limb2_left",
        ),
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(x=1, y=0, require_on_paper=False)
            ],
            radius=4 / 32 * math.sqrt(2),
            name="rightpincer_1",
        ),
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(x=1 - 0.25, y=0, require_on_paper=False)
            ],
            radius=4 / 32 * math.sqrt(2),
            name="rightpincer_2",
        ),
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(x=1, y=12 / 32, require_on_paper=False)
            ],
            radius=4 / 32 * math.sqrt(2),
            name="limb5_right",
        ),
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(
                    x=1 - (4 / 32 + 4 / 32), y=12 / 32, require_on_paper=False
                )
            ],
            radius=4 / 32 * math.sqrt(2),
            name="limb4_right",
        ),
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(x=1, y=1 / 2 + 4 / 32, require_on_paper=False)
            ],
            radius=4 / 32 * math.sqrt(2),
            name="limb3_right",
        ),
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(x=1, y=1 - 4 / 32, require_on_paper=False)
            ],
            radius=4 / 32 * math.sqrt(2),
            name="limb2_right",
        ),
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(x=13 / 32, y=9 / 32, require_on_paper=False),
                point_utils.Point(x=19 / 32, y=9 / 32, require_on_paper=False),
            ],
            radius=1 / 32 * math.sqrt(2),
            name="appendage_1",
        ),
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(x=13 / 32, y=11 / 32, require_on_paper=False),
                point_utils.Point(x=19 / 32, y=11 / 32, require_on_paper=False),
            ],
            radius=1 / 32 * math.sqrt(2),
            name="appendage_2",
        ),
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(x=13 / 32, y=13 / 32, require_on_paper=False),
                point_utils.Point(x=19 / 32, y=13 / 32, require_on_paper=False),
            ],
            radius=1 / 32 * math.sqrt(2),
            name="appendage_3",
        ),
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(x=13 / 32, y=15 / 32, require_on_paper=False),
                point_utils.Point(x=19 / 32, y=15 / 32, require_on_paper=False),
            ],
            radius=1 / 32 * math.sqrt(2),
            name="appendage_4",
        ),
        # Dummy rectangles for the limb1.
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(x=0, y=0, require_on_paper=False),
            ],
            radius=1 / 32 * math.sqrt(2),
            name="limb1_left",
        ),
        # Dummy rectangle for the limb1.
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(x=0, y=0, require_on_paper=False),
            ],
            radius=1 / 32 * math.sqrt(2),
            name="limb1_right",
        ),
        # Dummy limb7.
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(x=0, y=0, require_on_paper=False),
            ],
            radius=1 / 32 * math.sqrt(2),
            name="limb7",
        ),
        shaping_utils.Rectangle(
            name="?",
            x_start=0.125,
            y_start=0.5,
            x_len=0.75,
            y_len=0.125,
            circle_centers=[
                point_utils.Point(
                    x=0.125 + 2 / 32, y=0.5 + 2 / 32, require_on_paper=False
                ),
                point_utils.Point(
                    x=0.875 - 2 / 32, y=0.5 + 2 / 32, require_on_paper=False
                ),
            ],
        ),
        shaping_utils.Rectangle.from_centers(
            name="tail",
            radius=math.sqrt(2) * 12 / 32,
            circle_centers=[
                point_utils.Point(x=0.5, y=1, require_on_paper=False)
            ],
        ),
    ],
    rivers=[
        shaping_elements.ShapingRiver(
            name="limb6_left",
            bank_1=[
                crease_utils.Crease(
                    point_utils.Point(x=0, y=4 / 32),
                    point_utils.Point(x=12 / 32, y=4 / 32),
                    fold=crease_utils.Fold.HINGE,
                ),
                crease_utils.Crease(
                    point_utils.Point(x=12 / 32, y=4 / 32),
                    point_utils.Point(x=12 / 32, y=0),
                    fold=crease_utils.Fold.HINGE,
                ),
            ],
            bank_2=[
                crease_utils.Crease(
                    point_utils.Point(x=0, y=8 / 32),
                    point_utils.Point(x=16 / 32, y=8 / 32),
                    fold=crease_utils.Fold.HINGE,
                ),
                crease_utils.Crease(
                    point_utils.Point(x=16 / 32, y=8 / 32),
                    point_utils.Point(x=16 / 32, y=0),
                    fold=crease_utils.Fold.HINGE,
                ),
            ],
            width=4,
            grid_size=32,
        ).extend_creases(SCORPION_WITH_ELEVATION.base_creases),
        shaping_elements.ShapingRiver(
            name="limb6_right",
            bank_1=[
                crease_utils.Crease(
                    point_utils.Point(x=1 - 0, y=4 / 32),
                    point_utils.Point(x=1 - 12 / 32, y=4 / 32),
                    fold=crease_utils.Fold.HINGE,
                ),
                crease_utils.Crease(
                    point_utils.Point(x=1 - 12 / 32, y=4 / 32),
                    point_utils.Point(x=1 - 12 / 32, y=0),
                    fold=crease_utils.Fold.HINGE,
                ),
            ],
            bank_2=[
                crease_utils.Crease(
                    point_utils.Point(x=1 - 0, y=8 / 32),
                    point_utils.Point(x=1 - 16 / 32, y=8 / 32),
                    fold=crease_utils.Fold.HINGE,
                ),
                crease_utils.Crease(
                    point_utils.Point(x=1 - 16 / 32, y=8 / 32),
                    point_utils.Point(x=1 - 16 / 32, y=0),
                    fold=crease_utils.Fold.HINGE,
                ),
            ],
            width=4,
            grid_size=32,
        ).extend_creases(SCORPION_WITH_ELEVATION.base_creases),
    ],
    zigzag_shaping_params=[
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="leftpincer_1",
            zigzag_params=shaping_types.ZigzagParams(
                target_flap_angle=np.pi / 2,
                fold_order=[
                    crease_utils.Fold.MOUNTAIN,
                    crease_utils.Fold.VALLEY,
                ],
                even_phase=True,
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="rightpincer_1",
            zigzag_params=shaping_types.ZigzagParams(
                target_flap_angle=np.pi / 2,
                fold_order=[
                    crease_utils.Fold.MOUNTAIN,
                    crease_utils.Fold.VALLEY,
                ],
                even_phase=True,
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="limb5_left",
            zigzag_params=shaping_types.ZigzagParams(
                target_flap_angle=np.pi / 3,
                fold_order=[
                    crease_utils.Fold.MOUNTAIN,
                    crease_utils.Fold.VALLEY,
                ],
                even_phase=True,
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="limb5_right",
            zigzag_params=shaping_types.ZigzagParams(
                target_flap_angle=np.pi / 3,
                fold_order=[
                    crease_utils.Fold.MOUNTAIN,
                    crease_utils.Fold.VALLEY,
                ],
                even_phase=True,
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="limb4_left",
            zigzag_params=shaping_types.ZigzagParams(
                target_flap_angle=np.pi / 2,
                fold_order=[
                    crease_utils.Fold.MOUNTAIN,
                    crease_utils.Fold.VALLEY,
                ],
                even_phase=True,
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="limb4_right",
            zigzag_params=shaping_types.ZigzagParams(
                target_flap_angle=np.pi / 2,
                fold_order=[
                    crease_utils.Fold.MOUNTAIN,
                    crease_utils.Fold.VALLEY,
                ],
                even_phase=True,
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="limb3_left",
            zigzag_params=shaping_types.ZigzagParams(
                target_flap_angle=3 * np.pi / 4,
                fold_order=[
                    crease_utils.Fold.MOUNTAIN,
                    crease_utils.Fold.VALLEY,
                ],
                even_phase=True,
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="limb3_right",
            zigzag_params=shaping_types.ZigzagParams(
                target_flap_angle=3 * np.pi / 4,
                fold_order=[
                    crease_utils.Fold.MOUNTAIN,
                    crease_utils.Fold.VALLEY,
                ],
                even_phase=True,
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="limb2_left",
            zigzag_params=shaping_types.ZigzagParams(
                target_flap_angle=5 * np.pi / 6,
                fold_order=[
                    crease_utils.Fold.MOUNTAIN,
                    crease_utils.Fold.VALLEY,
                ],
                even_phase=True,
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="limb2_right",
            zigzag_params=shaping_types.ZigzagParams(
                target_flap_angle=5 * np.pi / 6,
                fold_order=[
                    crease_utils.Fold.MOUNTAIN,
                    crease_utils.Fold.VALLEY,
                ],
                even_phase=True,
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="limb6_left",
            zigzag_params=shaping_types.RiverZigzagParams(
                target_flap_angle=np.pi / 4,
                fold_order=[
                    crease_utils.Fold.MOUNTAIN,
                    crease_utils.Fold.VALLEY,
                ],
                even_phase=True,
                bank_index="SECOND",
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="limb6_right",
            zigzag_params=shaping_types.RiverZigzagParams(
                target_flap_angle=np.pi / 4,
                fold_order=[
                    crease_utils.Fold.MOUNTAIN,
                    crease_utils.Fold.VALLEY,
                ],
                even_phase=True,
                bank_index="SECOND",
            ),
        ),
    ],
    zigzag_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="red" x1="0" y1="0" x2="31.25" y2="31.25"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="125"/>
  <line stroke="black" x1="0" y1="0" x2="125" y2="0"/>
  <line stroke="blue" x1="31.25" y1="31.25" x2="62.5" y2="62.5"/>
  <line stroke="red" x1="31.25" y1="31.25" x2="31.25" y2="93.75"/>
  <line stroke="red" x1="31.25" y1="31.25" x2="93.75" y2="31.25"/>
  <line stroke="red" x1="62.5" y1="62.5" x2="93.75" y2="93.75"/>
  <line stroke="blue" x1="62.5" y1="62.5" x2="62.5" y2="125"/>
  <line stroke="blue" x1="62.5" y1="62.5" x2="125" y2="62.5"/>
  <line stroke="blue" x1="31.25" y1="93.75" x2="0" y2="125" fold-step="1"/>
  <line stroke="red" x1="31.25" y1="93.75" x2="62.5" y2="125" fold-step="1"/>
  <line stroke="red" x1="31.25" y1="93.75" x2="31.25" y2="174.56"/>
  <line stroke="red" x1="93.75" y1="31.25" x2="125" y2="0" fold-step="1"/>
  <line stroke="red" x1="93.75" y1="31.25" x2="125" y2="31.25"/>
  <line stroke="blue" x1="93.75" y1="31.25" x2="125" y2="62.5" fold-step="1"/>
  <line stroke="black" x1="0" y1="125" x2="0" y2="250"/>
  <line stroke="blue" x1="125" y1="0" x2="125" y2="31.25"/>
  <line stroke="black" x1="125" y1="0" x2="250" y2="0"/>
  <line stroke="red" x1="125" y1="31.25" x2="125" y2="62.5"/>
  <line stroke="red" x1="125" y1="31.25" x2="218.75" y2="31.25"/>
  <line stroke="blue" x1="93.75" y1="93.75" x2="62.5" y2="125" fold-step="1"/>
  <line stroke="red" x1="93.75" y1="93.75" x2="125" y2="62.5" fold-step="1"/>
  <line stroke="red" x1="93.75" y1="93.75" x2="125" y2="93.75"/>
  <line stroke="blue" x1="93.75" y1="93.75" x2="125" y2="125"/>
  <line stroke="red" x1="93.75" y1="93.75" x2="93.75" y2="174.56"/>
  <line stroke="blue" x1="62.5" y1="125" x2="62.5" y2="250"/>
  <line stroke="blue" x1="125" y1="62.5" x2="125" y2="93.75"/>
  <line stroke="blue" x1="125" y1="62.5" x2="187.5" y2="62.5"/>
  <line stroke="red" x1="125" y1="93.75" x2="125" y2="125"/>
  <line stroke="red" x1="125" y1="93.75" x2="156.25" y2="93.75"/>
  <line stroke="blue" x1="125" y1="125" x2="156.25" y2="93.75"/>
  <line stroke="blue" x1="125" y1="125" x2="125" y2="250"/>
  <line stroke="red" x1="31.25" y1="174.56" x2="0" y2="250" fold-step="11"/>
  <line stroke="red" x1="31.25" y1="174.56" x2="31.25" y2="250"/>
  <line stroke="blue" x1="31.25" y1="174.56" x2="62.5" y2="250" fold-step="11"/>
  <line stroke="red" x1="156.25" y1="93.75" x2="187.5" y2="62.5"/>
  <line stroke="red" x1="156.25" y1="93.75" x2="156.25" y2="174.56"/>
  <line stroke="blue" x1="187.5" y1="62.5" x2="218.75" y2="31.25"/>
  <line stroke="blue" x1="187.5" y1="62.5" x2="187.5" y2="250"/>
  <line stroke="red" x1="93.75" y1="174.56" x2="62.5" y2="250" fold-step="11"/>
  <line stroke="red" x1="93.75" y1="174.56" x2="93.75" y2="250"/>
  <line stroke="blue" x1="93.75" y1="174.56" x2="125" y2="250" fold-step="11"/>
  <line stroke="red" x1="218.75" y1="31.25" x2="250" y2="0"/>
  <line stroke="red" x1="218.75" y1="31.25" x2="218.75" y2="174.56"/>
  <line stroke="red" x1="156.25" y1="174.56" x2="125" y2="250" fold-step="11"/>
  <line stroke="red" x1="156.25" y1="174.56" x2="156.25" y2="250"/>
  <line stroke="blue" x1="156.25" y1="174.56" x2="187.5" y2="250" fold-step="11"/>
  <line stroke="blue" x1="0" y1="250" x2="31.25" y2="250"/>
  <line stroke="red" x1="0" y1="250" x2="31.25" y2="304.13" fold-step="3"/>
  <line stroke="black" x1="0" y1="250" x2="0" y2="375"/>
  <line stroke="red" x1="250" y1="0" x2="281.25" y2="31.25"/>
  <line stroke="blue" x1="250" y1="0" x2="250" y2="250"/>
  <line stroke="black" x1="250" y1="0" x2="500" y2="0"/>
  <line stroke="red" x1="31.25" y1="250" x2="62.5" y2="250"/>
  <line stroke="red" x1="31.25" y1="250" x2="31.25" y2="304.13"/>
  <line stroke="blue" x1="62.5" y1="250" x2="93.75" y2="250"/>
  <line stroke="red" x1="62.5" y1="250" x2="85.38" y2="289.62" fold-step="3"/>
  <line stroke="blue" x1="62.5" y1="250" x2="31.25" y2="304.13" fold-step="3"/>
  <line stroke="blue" x1="62.5" y1="250" x2="62.5" y2="312.5"/>
  <line stroke="red" x1="93.75" y1="250" x2="125" y2="250"/>
  <line stroke="red" x1="93.75" y1="250" x2="93.75" y2="281.25"/>
  <line stroke="blue" x1="125" y1="250" x2="156.25" y2="250"/>
  <line stroke="blue" x1="125" y1="250" x2="93.75" y2="281.25"/>
  <line stroke="red" x1="125" y1="250" x2="125" y2="281.25"/>
  <line stroke="blue" x1="125" y1="250" x2="156.25" y2="281.25"/>
  <line stroke="red" x1="218.75" y1="174.56" x2="187.5" y2="250" fold-step="11"/>
  <line stroke="red" x1="218.75" y1="174.56" x2="218.75" y2="250"/>
  <line stroke="blue" x1="218.75" y1="174.56" x2="250" y2="250" fold-step="11"/>
  <line stroke="blue" x1="281.25" y1="31.25" x2="312.5" y2="62.5"/>
  <line stroke="red" x1="281.25" y1="31.25" x2="281.25" y2="174.56"/>
  <line stroke="red" x1="281.25" y1="31.25" x2="424.56" y2="31.25"/>
  <line stroke="red" x1="156.25" y1="250" x2="187.5" y2="250"/>
  <line stroke="red" x1="156.25" y1="250" x2="156.25" y2="281.25"/>
  <line stroke="red" x1="93.75" y1="281.25" x2="85.38" y2="289.62"/>
  <line stroke="red" x1="93.75" y1="281.25" x2="125" y2="281.25"/>
  <line stroke="red" x1="85.38" y1="289.62" x2="62.5" y2="312.5"/>
  <line stroke="blue" x1="85.38" y1="289.62" x2="125" y2="312.5" fold-step="3"/>
  <line stroke="red" x1="31.25" y1="304.13" x2="31.25" y2="343.75"/>
  <line stroke="red" x1="125" y1="281.25" x2="156.25" y2="281.25"/>
  <line stroke="blue" x1="125" y1="281.25" x2="125" y2="312.5"/>
  <line stroke="blue" x1="187.5" y1="250" x2="156.25" y2="281.25" fold-step="5"/>
  <line stroke="blue" x1="187.5" y1="250" x2="218.75" y2="250"/>
  <line stroke="red" x1="187.5" y1="250" x2="218.75" y2="281.25" fold-step="5"/>
  <line stroke="blue" x1="187.5" y1="250" x2="187.5" y2="312.5"/>
  <line stroke="blue" x1="62.5" y1="312.5" x2="125" y2="312.5"/>
  <line stroke="blue" x1="62.5" y1="312.5" x2="31.25" y2="343.75"/>
  <line stroke="red" x1="312.5" y1="62.5" x2="343.75" y2="93.75"/>
  <line stroke="blue" x1="312.5" y1="62.5" x2="312.5" y2="250"/>
  <line stroke="blue" x1="312.5" y1="62.5" x2="500" y2="62.5"/>
  <line stroke="red" x1="156.25" y1="281.25" x2="125" y2="312.5" fold-step="5"/>
  <line stroke="red" x1="156.25" y1="281.25" x2="187.5" y2="312.5"/>
  <line stroke="red" x1="281.25" y1="174.56" x2="250" y2="250" fold-step="11"/>
  <line stroke="red" x1="281.25" y1="174.56" x2="281.25" y2="250"/>
  <line stroke="blue" x1="281.25" y1="174.56" x2="312.5" y2="250" fold-step="11"/>
  <line stroke="red" x1="218.75" y1="250" x2="250" y2="250"/>
  <line stroke="red" x1="218.75" y1="250" x2="218.75" y2="281.25"/>
  <line stroke="red" x1="125" y1="312.5" x2="70.87" y2="343.75" fold-step="3"/>
  <line stroke="blue" x1="125" y1="312.5" x2="187.5" y2="312.5"/>
  <line stroke="red" x1="125" y1="312.5" x2="125" y2="343.75"/>
  <line stroke="blue" x1="125" y1="312.5" x2="156.25" y2="343.75" fold-step="5"/>
  <line stroke="red" x1="31.25" y1="343.75" x2="70.87" y2="343.75"/>
  <line stroke="red" x1="31.25" y1="343.75" x2="0" y2="375"/>
  <line stroke="red" x1="70.87" y1="343.75" x2="125" y2="343.75"/>
  <line stroke="blue" x1="70.87" y1="343.75" x2="125" y2="375" fold-step="3"/>
  <line stroke="blue" x1="250" y1="250" x2="218.75" y2="281.25" fold-step="5"/>
  <line stroke="blue" x1="250" y1="250" x2="281.25" y2="250"/>
  <line stroke="red" x1="250" y1="250" x2="281.25" y2="281.25" fold-step="5"/>
  <line stroke="blue" x1="250" y1="250" x2="250" y2="375"/>
  <line stroke="red" x1="218.75" y1="281.25" x2="218.75" y2="343.75"/>
  <line stroke="red" x1="343.75" y1="93.75" x2="343.75" y2="174.56"/>
  <line stroke="blue" x1="343.75" y1="93.75" x2="375" y2="125"/>
  <line stroke="red" x1="343.75" y1="93.75" x2="424.56" y2="93.75"/>
  <line stroke="blue" x1="187.5" y1="312.5" x2="218.75" y2="343.75"/>
  <line stroke="red" x1="125" y1="343.75" x2="156.25" y2="343.75"/>
  <line stroke="blue" x1="125" y1="343.75" x2="125" y2="375"/>
  <line stroke="blue" x1="0" y1="375" x2="125" y2="375"/>
  <line stroke="red" x1="0" y1="375" x2="31.25" y2="406.25"/>
  <line stroke="black" x1="0" y1="375" x2="0" y2="500"/>
  <line stroke="red" x1="281.25" y1="250" x2="281.25" y2="281.25"/>
  <line stroke="red" x1="281.25" y1="250" x2="312.5" y2="250"/>
  <line stroke="red" x1="156.25" y1="343.75" x2="125" y2="375" fold-step="5"/>
  <line stroke="red" x1="156.25" y1="343.75" x2="218.75" y2="343.75"/>
  <line stroke="red" x1="343.75" y1="174.56" x2="312.5" y2="250" fold-step="11"/>
  <line stroke="red" x1="343.75" y1="174.56" x2="343.75" y2="250"/>
  <line stroke="blue" x1="343.75" y1="174.56" x2="375" y2="250" fold-step="11"/>
  <line stroke="red" x1="125" y1="375" x2="70.87" y2="406.25" fold-step="3"/>
  <line stroke="blue" x1="125" y1="375" x2="125" y2="406.25"/>
  <line stroke="blue" x1="125" y1="375" x2="156.25" y2="406.25" fold-step="5"/>
  <line stroke="red" x1="125" y1="375" x2="250" y2="375"/>
  <line stroke="red" x1="375" y1="125" x2="406.25" y2="156.25"/>
  <line stroke="blue" x1="375" y1="125" x2="375" y2="250"/>
  <line stroke="blue" x1="375" y1="125" x2="500" y2="125"/>
  <line stroke="blue" x1="281.25" y1="281.25" x2="312.5" y2="250" fold-step="5"/>
  <line stroke="red" x1="281.25" y1="281.25" x2="281.25" y2="343.75"/>
  <line stroke="blue" x1="312.5" y1="250" x2="343.75" y2="250"/>
  <line stroke="blue" x1="312.5" y1="250" x2="312.5" y2="312.5"/>
  <line stroke="red" x1="312.5" y1="250" x2="343.75" y2="281.25" fold-step="5"/>
  <line stroke="red" x1="31.25" y1="406.25" x2="70.87" y2="406.25"/>
  <line stroke="blue" x1="31.25" y1="406.25" x2="62.5" y2="437.5"/>
  <line stroke="red" x1="31.25" y1="406.25" x2="31.25" y2="445.87"/>
  <line stroke="red" x1="218.75" y1="343.75" x2="250" y2="375"/>
  <line stroke="red" x1="70.87" y1="406.25" x2="125" y2="406.25"/>
  <line stroke="blue" x1="70.87" y1="406.25" x2="125" y2="437.5" fold-step="3"/>
  <line stroke="red" x1="125" y1="406.25" x2="156.25" y2="406.25"/>
  <line stroke="red" x1="125" y1="406.25" x2="125" y2="437.5"/>
  <line stroke="red" x1="343.75" y1="250" x2="343.75" y2="281.25"/>
  <line stroke="red" x1="343.75" y1="250" x2="375" y2="250"/>
  <line stroke="red" x1="424.56" y1="31.25" x2="468.75" y2="31.25"/>
  <line stroke="blue" x1="424.56" y1="31.25" x2="500" y2="0" fold-step="11"/>
  <line stroke="red" x1="424.56" y1="31.25" x2="500" y2="62.5" fold-step="11"/>
  <line stroke="red" x1="424.56" y1="93.75" x2="468.75" y2="93.75"/>
  <line stroke="blue" x1="424.56" y1="93.75" x2="500" y2="62.5" fold-step="11"/>
  <line stroke="red" x1="424.56" y1="93.75" x2="500" y2="125" fold-step="11"/>
  <line stroke="red" x1="156.25" y1="406.25" x2="125" y2="437.5" fold-step="5"/>
  <line stroke="red" x1="156.25" y1="406.25" x2="218.75" y2="406.25"/>
  <line stroke="red" x1="406.25" y1="156.25" x2="406.25" y2="174.56"/>
  <line stroke="red" x1="406.25" y1="156.25" x2="424.56" y2="156.25"/>
  <line stroke="blue" x1="406.25" y1="156.25" x2="437.5" y2="187.5"/>
  <line stroke="blue" x1="62.5" y1="437.5" x2="125" y2="437.5"/>
  <line stroke="red" x1="62.5" y1="437.5" x2="85.38" y2="460.38"/>
  <line stroke="blue" x1="62.5" y1="437.5" x2="62.5" y2="500"/>
  <line stroke="blue" x1="312.5" y1="312.5" x2="281.25" y2="343.75"/>
  <line stroke="red" x1="312.5" y1="312.5" x2="343.75" y2="281.25"/>
  <line stroke="blue" x1="312.5" y1="312.5" x2="375" y2="312.5"/>
  <line stroke="red" x1="406.25" y1="174.56" x2="375" y2="250" fold-step="11"/>
  <line stroke="red" x1="406.25" y1="174.56" x2="406.25" y2="281.25"/>
  <line stroke="blue" x1="406.25" y1="174.56" x2="437.5" y2="250" fold-step="11"/>
  <line stroke="red" x1="281.25" y1="343.75" x2="250" y2="375"/>
  <line stroke="red" x1="281.25" y1="343.75" x2="343.75" y2="343.75"/>
  <line stroke="blue" x1="343.75" y1="281.25" x2="375" y2="250"/>
  <line stroke="blue" x1="343.75" y1="281.25" x2="375" y2="312.5" fold-step="5"/>
  <line stroke="red" x1="343.75" y1="281.25" x2="406.25" y2="281.25"/>
  <line stroke="blue" x1="31.25" y1="445.87" x2="0" y2="500" fold-step="3"/>
  <line stroke="red" x1="31.25" y1="445.87" x2="31.25" y2="500"/>
  <line stroke="red" x1="31.25" y1="445.87" x2="62.5" y2="500" fold-step="3"/>
  <line stroke="red" x1="250" y1="375" x2="218.75" y2="406.25"/>
  <line stroke="red" x1="250" y1="375" x2="281.25" y2="406.25"/>
  <line stroke="blue" x1="250" y1="375" x2="375" y2="375"/>
  <line stroke="blue" x1="250" y1="375" x2="250" y2="500"/>
  <line stroke="blue" x1="375" y1="250" x2="406.25" y2="281.25"/>
  <line stroke="red" x1="424.56" y1="156.25" x2="468.75" y2="156.25"/>
  <line stroke="blue" x1="424.56" y1="156.25" x2="500" y2="125" fold-step="11"/>
  <line stroke="red" x1="424.56" y1="156.25" x2="500" y2="187.5" fold-step="11"/>
  <line stroke="red" x1="125" y1="437.5" x2="85.38" y2="460.38" fold-step="3"/>
  <line stroke="blue" x1="125" y1="437.5" x2="187.5" y2="437.5"/>
  <line stroke="blue" x1="125" y1="437.5" x2="125" y2="468.75"/>
  <line stroke="blue" x1="125" y1="437.5" x2="156.25" y2="468.75" fold-step="5"/>
  <line stroke="blue" x1="218.75" y1="406.25" x2="187.5" y2="437.5"/>
  <line stroke="red" x1="218.75" y1="406.25" x2="218.75" y2="468.75"/>
  <line stroke="red" x1="85.38" y1="460.38" x2="93.75" y2="468.75"/>
  <line stroke="blue" x1="85.38" y1="460.38" x2="62.5" y2="500" fold-step="3"/>
  <line stroke="blue" x1="468.75" y1="31.25" x2="500" y2="0"/>
  <line stroke="red" x1="468.75" y1="31.25" x2="500" y2="62.5"/>
  <line stroke="red" x1="468.75" y1="31.25" x2="531.25" y2="31.25"/>
  <line stroke="red" x1="187.5" y1="437.5" x2="156.25" y2="468.75"/>
  <line stroke="blue" x1="187.5" y1="437.5" x2="187.5" y2="500"/>
  <line stroke="red" x1="437.5" y1="187.5" x2="455.81" y2="205.81"/>
  <line stroke="blue" x1="437.5" y1="187.5" x2="437.5" y2="250"/>
  <line stroke="blue" x1="437.5" y1="187.5" x2="500" y2="187.5"/>
  <line stroke="red" x1="93.75" y1="468.75" x2="125" y2="468.75"/>
  <line stroke="red" x1="93.75" y1="468.75" x2="93.75" y2="500"/>
  <line stroke="blue" x1="93.75" y1="468.75" x2="125" y2="500"/>
  <line stroke="blue" x1="468.75" y1="93.75" x2="500" y2="62.5"/>
  <line stroke="red" x1="468.75" y1="93.75" x2="500" y2="125"/>
  <line stroke="red" x1="468.75" y1="93.75" x2="531.25" y2="93.75"/>
  <line stroke="red" x1="125" y1="468.75" x2="156.25" y2="468.75"/>
  <line stroke="red" x1="125" y1="468.75" x2="125" y2="500"/>
  <line stroke="red" x1="343.75" y1="343.75" x2="375" y2="312.5" fold-step="5"/>
  <line stroke="blue" x1="343.75" y1="343.75" x2="375" y2="375" fold-step="5"/>
  <line stroke="red" x1="343.75" y1="343.75" x2="406.25" y2="343.75"/>
  <line stroke="blue" x1="375" y1="312.5" x2="406.25" y2="281.25"/>
  <line stroke="red" x1="375" y1="312.5" x2="406.25" y2="312.5"/>
  <line stroke="blue" x1="375" y1="312.5" x2="406.25" y2="343.75"/>
  <line stroke="blue" x1="156.25" y1="468.75" x2="125" y2="500"/>
  <line stroke="red" x1="156.25" y1="468.75" x2="187.5" y2="500" fold-step="5"/>
  <line stroke="red" x1="156.25" y1="468.75" x2="156.25" y2="531.25"/>
  <line stroke="red" x1="281.25" y1="406.25" x2="343.75" y2="406.25"/>
  <line stroke="blue" x1="281.25" y1="406.25" x2="312.5" y2="437.5"/>
  <line stroke="red" x1="281.25" y1="406.25" x2="281.25" y2="468.75"/>
  <line stroke="red" x1="406.25" y1="281.25" x2="406.25" y2="312.5"/>
  <line stroke="red" x1="406.25" y1="281.25" x2="437.5" y2="281.25"/>
  <line stroke="blue" x1="468.75" y1="156.25" x2="500" y2="125"/>
  <line stroke="red" x1="468.75" y1="156.25" x2="500" y2="187.5"/>
  <line stroke="red" x1="468.75" y1="156.25" x2="531.25" y2="156.25"/>
  <line stroke="blue" x1="0" y1="500" x2="31.25" y2="500"/>
  <line stroke="red" x1="0" y1="500" x2="31.25" y2="512.94" fold-step="7"/>
  <line stroke="black" x1="0" y1="500" x2="0" y2="625"/>
  <line stroke="blue" x1="500" y1="0" x2="575.44" y2="31.25" fold-step="12"/>
  <line stroke="blue" x1="500" y1="0" x2="531.25" y2="31.25"/>
  <line stroke="black" x1="500" y1="0" x2="750" y2="0"/>
  <line stroke="red" x1="455.81" y1="205.81" x2="437.5" y2="250" fold-step="11"/>
  <line stroke="red" x1="455.81" y1="205.81" x2="468.75" y2="218.75"/>
  <line stroke="blue" x1="455.81" y1="205.81" x2="500" y2="187.5" fold-step="11"/>
  <line stroke="red" x1="31.25" y1="500" x2="62.5" y2="500"/>
  <line stroke="red" x1="31.25" y1="500" x2="31.25" y2="512.94"/>
  <line stroke="blue" x1="62.5" y1="500" x2="93.75" y2="500"/>
  <line stroke="blue" x1="62.5" y1="500" x2="31.25" y2="512.94" fold-step="7"/>
  <line stroke="red" x1="62.5" y1="500" x2="93.75" y2="512.94" fold-step="7"/>
  <line stroke="blue" x1="62.5" y1="500" x2="62.5" y2="562.5"/>
  <line stroke="blue" x1="437.5" y1="250" x2="437.5" y2="281.25"/>
  <line stroke="red" x1="500" y1="62.5" x2="575.44" y2="31.25" fold-step="12"/>
  <line stroke="blue" x1="500" y1="62.5" x2="575.44" y2="93.75" fold-step="12"/>
  <line stroke="red" x1="500" y1="62.5" x2="531.25" y2="31.25"/>
  <line stroke="blue" x1="500" y1="62.5" x2="531.25" y2="93.75"/>
  <line stroke="blue" x1="500" y1="62.5" x2="687.5" y2="62.5"/>
  <line stroke="red" x1="93.75" y1="500" x2="125" y2="500"/>
  <line stroke="red" x1="93.75" y1="500" x2="93.75" y2="512.94"/>
  <line stroke="red" x1="406.25" y1="312.5" x2="406.25" y2="343.75"/>
  <line stroke="blue" x1="406.25" y1="312.5" x2="437.5" y2="312.5"/>
  <line stroke="red" x1="31.25" y1="512.94" x2="31.25" y2="593.75"/>
  <line stroke="blue" x1="125" y1="500" x2="93.75" y2="512.94" fold-step="7"/>
  <line stroke="blue" x1="125" y1="500" x2="93.75" y2="531.25"/>
  <line stroke="red" x1="125" y1="500" x2="112.06" y2="531.25" fold-step="7"/>
  <line stroke="blue" x1="125" y1="500" x2="156.25" y2="531.25"/>
  <line stroke="red" x1="500" y1="125" x2="575.44" y2="93.75" fold-step="12"/>
  <line stroke="blue" x1="500" y1="125" x2="575.44" y2="156.25" fold-step="12"/>
  <line stroke="red" x1="500" y1="125" x2="531.25" y2="93.75"/>
  <line stroke="blue" x1="500" y1="125" x2="531.25" y2="156.25"/>
  <line stroke="blue" x1="500" y1="125" x2="625" y2="125"/>
  <line stroke="blue" x1="218.75" y1="468.75" x2="187.5" y2="500" fold-step="5"/>
  <line stroke="red" x1="218.75" y1="468.75" x2="250" y2="500" fold-step="5"/>
  <line stroke="red" x1="218.75" y1="468.75" x2="218.75" y2="562.5"/>
  <line stroke="blue" x1="468.75" y1="218.75" x2="500" y2="187.5"/>
  <line stroke="red" x1="468.75" y1="218.75" x2="468.75" y2="281.25"/>
  <line stroke="red" x1="468.75" y1="218.75" x2="531.25" y2="218.75"/>
  <line stroke="blue" x1="437.5" y1="281.25" x2="437.5" y2="312.5"/>
  <line stroke="blue" x1="437.5" y1="281.25" x2="468.75" y2="281.25"/>
  <line stroke="red" x1="93.75" y1="512.94" x2="93.75" y2="531.25"/>
  <line stroke="red" x1="375" y1="375" x2="343.75" y2="406.25" fold-step="5"/>
  <line stroke="blue" x1="375" y1="375" x2="406.25" y2="343.75"/>
  <line stroke="red" x1="375" y1="375" x2="406.25" y2="375"/>
  <line stroke="blue" x1="375" y1="375" x2="406.25" y2="406.25"/>
  <line stroke="red" x1="343.75" y1="406.25" x2="406.25" y2="406.25"/>
  <line stroke="blue" x1="343.75" y1="406.25" x2="375" y2="437.5" fold-step="5"/>
  <line stroke="red" x1="406.25" y1="343.75" x2="406.25" y2="375"/>
  <line stroke="red" x1="406.25" y1="343.75" x2="437.5" y2="343.75"/>
  <line stroke="red" x1="531.25" y1="31.25" x2="575.44" y2="31.25"/>
  <line stroke="blue" x1="187.5" y1="500" x2="187.5" y2="562.5"/>
  <line stroke="blue" x1="500" y1="187.5" x2="544.19" y2="205.81" fold-step="12"/>
  <line stroke="red" x1="500" y1="187.5" x2="575.44" y2="156.25" fold-step="12"/>
  <line stroke="red" x1="500" y1="187.5" x2="531.25" y2="156.25"/>
  <line stroke="blue" x1="500" y1="187.5" x2="531.25" y2="218.75"/>
  <line stroke="blue" x1="500" y1="187.5" x2="562.5" y2="187.5"/>
  <line stroke="blue" x1="312.5" y1="437.5" x2="375" y2="437.5"/>
  <line stroke="red" x1="312.5" y1="437.5" x2="343.75" y2="468.75"/>
  <line stroke="blue" x1="312.5" y1="437.5" x2="312.5" y2="500"/>
  <line stroke="blue" x1="437.5" y1="312.5" x2="437.5" y2="343.75"/>
  <line stroke="red" x1="437.5" y1="312.5" x2="468.75" y2="312.5"/>
  <line stroke="red" x1="93.75" y1="531.25" x2="112.06" y2="531.25"/>
  <line stroke="red" x1="93.75" y1="531.25" x2="62.5" y2="562.5"/>
  <line stroke="red" x1="531.25" y1="93.75" x2="575.44" y2="93.75"/>
  <line stroke="red" x1="112.06" y1="531.25" x2="156.25" y2="531.25"/>
  <line stroke="blue" x1="112.06" y1="531.25" x2="125" y2="562.5" fold-step="7"/>
  <line stroke="blue" x1="281.25" y1="468.75" x2="250" y2="500" fold-step="5"/>
  <line stroke="red" x1="281.25" y1="468.75" x2="312.5" y2="500" fold-step="5"/>
  <line stroke="red" x1="281.25" y1="468.75" x2="281.25" y2="562.5"/>
  <line stroke="red" x1="468.75" y1="281.25" x2="468.75" y2="312.5"/>
  <line stroke="red" x1="468.75" y1="281.25" x2="531.25" y2="281.25"/>
  <line stroke="red" x1="406.25" y1="375" x2="406.25" y2="406.25"/>
  <line stroke="blue" x1="406.25" y1="375" x2="437.5" y2="375"/>
  <line stroke="red" x1="156.25" y1="531.25" x2="187.5" y2="562.5"/>
  <line stroke="red" x1="531.25" y1="156.25" x2="575.44" y2="156.25"/>
  <line stroke="blue" x1="437.5" y1="343.75" x2="437.5" y2="375"/>
  <line stroke="blue" x1="437.5" y1="343.75" x2="468.75" y2="343.75"/>
  <line stroke="blue" x1="250" y1="500" x2="250" y2="562.5"/>
  <line stroke="red" x1="468.75" y1="312.5" x2="468.75" y2="343.75"/>
  <line stroke="blue" x1="468.75" y1="312.5" x2="531.25" y2="312.5"/>
  <line stroke="blue" x1="62.5" y1="562.5" x2="125" y2="562.5"/>
  <line stroke="blue" x1="62.5" y1="562.5" x2="31.25" y2="593.75"/>
  <line stroke="blue" x1="406.25" y1="406.25" x2="375" y2="437.5"/>
  <line stroke="red" x1="406.25" y1="406.25" x2="406.25" y2="437.5"/>
  <line stroke="red" x1="406.25" y1="406.25" x2="437.5" y2="406.25"/>
  <line stroke="red" x1="531.25" y1="218.75" x2="544.19" y2="205.81"/>
  <line stroke="red" x1="531.25" y1="218.75" x2="531.25" y2="281.25"/>
  <line stroke="blue" x1="125" y1="562.5" x2="187.5" y2="562.5"/>
  <line stroke="red" x1="125" y1="562.5" x2="112.06" y2="593.75" fold-step="7"/>
  <line stroke="red" x1="375" y1="437.5" x2="343.75" y2="468.75" fold-step="5"/>
  <line stroke="red" x1="375" y1="437.5" x2="406.25" y2="437.5"/>
  <line stroke="blue" x1="375" y1="437.5" x2="406.25" y2="468.75"/>
  <line stroke="blue" x1="437.5" y1="375" x2="437.5" y2="406.25"/>
  <line stroke="red" x1="437.5" y1="375" x2="468.75" y2="375"/>
  <line stroke="red" x1="575.44" y1="31.25" x2="718.75" y2="31.25"/>
  <line stroke="blue" x1="343.75" y1="468.75" x2="312.5" y2="500" fold-step="5"/>
  <line stroke="red" x1="343.75" y1="468.75" x2="406.25" y2="468.75"/>
  <line stroke="blue" x1="343.75" y1="468.75" x2="375" y2="500"/>
  <line stroke="red" x1="343.75" y1="468.75" x2="343.75" y2="562.5"/>
  <line stroke="red" x1="468.75" y1="343.75" x2="468.75" y2="375"/>
  <line stroke="red" x1="468.75" y1="343.75" x2="531.25" y2="343.75"/>
  <line stroke="red" x1="544.19" y1="205.81" x2="562.5" y2="187.5"/>
  <line stroke="red" x1="544.19" y1="205.81" x2="562.5" y2="250" fold-step="12"/>
  <line stroke="red" x1="575.44" y1="93.75" x2="656.25" y2="93.75"/>
  <line stroke="blue" x1="312.5" y1="500" x2="312.5" y2="562.5"/>
  <line stroke="blue" x1="187.5" y1="562.5" x2="218.75" y2="562.5"/>
  <line stroke="red" x1="187.5" y1="562.5" x2="156.25" y2="593.75"/>
  <line stroke="blue" x1="187.5" y1="562.5" x2="187.5" y2="625"/>
  <line stroke="blue" x1="562.5" y1="187.5" x2="593.75" y2="156.25"/>
  <line stroke="blue" x1="562.5" y1="187.5" x2="562.5" y2="250"/>
  <line stroke="red" x1="31.25" y1="593.75" x2="112.06" y2="593.75"/>
  <line stroke="red" x1="31.25" y1="593.75" x2="0" y2="625"/>
  <line stroke="red" x1="575.44" y1="156.25" x2="593.75" y2="156.25"/>
  <line stroke="blue" x1="406.25" y1="437.5" x2="437.5" y2="437.5"/>
  <line stroke="red" x1="406.25" y1="437.5" x2="406.25" y2="468.75"/>
  <line stroke="blue" x1="437.5" y1="406.25" x2="437.5" y2="437.5"/>
  <line stroke="blue" x1="437.5" y1="406.25" x2="468.75" y2="406.25"/>
  <line stroke="red" x1="468.75" y1="375" x2="468.75" y2="406.25"/>
  <line stroke="blue" x1="468.75" y1="375" x2="531.25" y2="375"/>
  <line stroke="red" x1="531.25" y1="281.25" x2="531.25" y2="312.5"/>
  <line stroke="blue" x1="531.25" y1="281.25" x2="562.5" y2="281.25"/>
  <line stroke="red" x1="218.75" y1="562.5" x2="250" y2="562.5"/>
  <line stroke="red" x1="218.75" y1="562.5" x2="218.75" y2="625"/>
  <line stroke="red" x1="112.06" y1="593.75" x2="156.25" y2="593.75"/>
  <line stroke="blue" x1="112.06" y1="593.75" x2="125" y2="625" fold-step="7"/>
  <line stroke="blue" x1="156.25" y1="593.75" x2="125" y2="625"/>
  <line stroke="red" x1="156.25" y1="593.75" x2="156.25" y2="625"/>
  <line stroke="red" x1="593.75" y1="156.25" x2="593.75" y2="174.56"/>
  <line stroke="red" x1="593.75" y1="156.25" x2="625" y2="125"/>
  <line stroke="blue" x1="250" y1="562.5" x2="281.25" y2="562.5"/>
  <line stroke="blue" x1="250" y1="562.5" x2="250" y2="625"/>
  <line stroke="blue" x1="562.5" y1="250" x2="593.75" y2="174.56" fold-step="12"/>
  <line stroke="blue" x1="562.5" y1="250" x2="562.5" y2="281.25"/>
  <line stroke="red" x1="531.25" y1="312.5" x2="531.25" y2="343.75"/>
  <line stroke="red" x1="531.25" y1="312.5" x2="562.5" y2="312.5"/>
  <line stroke="blue" x1="437.5" y1="437.5" x2="437.5" y2="468.75"/>
  <line stroke="red" x1="437.5" y1="437.5" x2="468.75" y2="437.5"/>
  <line stroke="red" x1="593.75" y1="174.56" x2="593.75" y2="281.25"/>
  <line stroke="red" x1="593.75" y1="174.56" x2="625" y2="250" fold-step="12"/>
  <line stroke="blue" x1="406.25" y1="468.75" x2="375" y2="500"/>
  <line stroke="red" x1="406.25" y1="468.75" x2="437.5" y2="468.75"/>
  <line stroke="red" x1="406.25" y1="468.75" x2="406.25" y2="500"/>
  <line stroke="red" x1="468.75" y1="406.25" x2="468.75" y2="437.5"/>
  <line stroke="red" x1="468.75" y1="406.25" x2="531.25" y2="406.25"/>
  <line stroke="blue" x1="0" y1="625" x2="125" y2="625"/>
  <line stroke="red" x1="0" y1="625" x2="31.25" y2="656.25"/>
  <line stroke="black" x1="0" y1="625" x2="0" y2="750"/>
  <line stroke="red" x1="375" y1="500" x2="406.25" y2="500"/>
  <line stroke="blue" x1="375" y1="500" x2="375" y2="562.5"/>
  <line stroke="red" x1="281.25" y1="562.5" x2="312.5" y2="562.5"/>
  <line stroke="red" x1="281.25" y1="562.5" x2="281.25" y2="625"/>
  <line stroke="blue" x1="562.5" y1="281.25" x2="562.5" y2="312.5"/>
  <line stroke="red" x1="562.5" y1="281.25" x2="593.75" y2="281.25"/>
  <line stroke="red" x1="531.25" y1="343.75" x2="531.25" y2="375"/>
  <line stroke="blue" x1="531.25" y1="343.75" x2="562.5" y2="343.75"/>
  <line stroke="red" x1="125" y1="625" x2="156.25" y2="625"/>
  <line stroke="red" x1="125" y1="625" x2="112.06" y2="656.25" fold-step="7"/>
  <line stroke="blue" x1="125" y1="625" x2="156.25" y2="656.25"/>
  <line stroke="blue" x1="625" y1="125" x2="656.25" y2="93.75"/>
  <line stroke="blue" x1="625" y1="125" x2="625" y2="250"/>
  <line stroke="blue" x1="437.5" y1="468.75" x2="468.75" y2="468.75"/>
  <line stroke="blue" x1="437.5" y1="468.75" x2="437.5" y2="500"/>
  <line stroke="red" x1="468.75" y1="437.5" x2="468.75" y2="468.75"/>
  <line stroke="blue" x1="468.75" y1="437.5" x2="531.25" y2="437.5"/>
  <line stroke="blue" x1="312.5" y1="562.5" x2="343.75" y2="562.5"/>
  <line stroke="blue" x1="312.5" y1="562.5" x2="312.5" y2="625"/>
  <line stroke="blue" x1="562.5" y1="312.5" x2="562.5" y2="343.75"/>
  <line stroke="blue" x1="562.5" y1="312.5" x2="593.75" y2="312.5"/>
  <line stroke="blue" x1="156.25" y1="625" x2="187.5" y2="625"/>
  <line stroke="red" x1="156.25" y1="625" x2="156.25" y2="656.25"/>
  <line stroke="blue" x1="406.25" y1="500" x2="437.5" y2="500"/>
  <line stroke="red" x1="406.25" y1="500" x2="406.25" y2="562.5"/>
  <line stroke="red" x1="531.25" y1="375" x2="531.25" y2="406.25"/>
  <line stroke="red" x1="531.25" y1="375" x2="562.5" y2="375"/>
  <line stroke="red" x1="187.5" y1="625" x2="218.75" y2="625"/>
  <line stroke="blue" x1="187.5" y1="625" x2="187.5" y2="687.5"/>
  <line stroke="red" x1="31.25" y1="656.25" x2="112.06" y2="656.25"/>
  <line stroke="blue" x1="31.25" y1="656.25" x2="62.5" y2="687.5"/>
  <line stroke="red" x1="31.25" y1="656.25" x2="31.25" y2="737.06"/>
  <line stroke="red" x1="593.75" y1="281.25" x2="593.75" y2="312.5"/>
  <line stroke="blue" x1="593.75" y1="281.25" x2="625" y2="250"/>
  <line stroke="blue" x1="593.75" y1="281.25" x2="625" y2="312.5"/>
  <line stroke="red" x1="593.75" y1="281.25" x2="656.25" y2="281.25"/>
  <line stroke="red" x1="343.75" y1="562.5" x2="375" y2="562.5"/>
  <line stroke="red" x1="343.75" y1="562.5" x2="343.75" y2="625"/>
  <line stroke="blue" x1="562.5" y1="343.75" x2="562.5" y2="375"/>
  <line stroke="red" x1="562.5" y1="343.75" x2="593.75" y2="343.75"/>
  <line stroke="blue" x1="218.75" y1="625" x2="250" y2="625"/>
  <line stroke="red" x1="218.75" y1="625" x2="218.75" y2="718.75"/>
  <line stroke="red" x1="468.75" y1="468.75" x2="468.75" y2="500"/>
  <line stroke="red" x1="468.75" y1="468.75" x2="531.25" y2="468.75"/>
  <line stroke="red" x1="656.25" y1="93.75" x2="656.25" y2="174.56"/>
  <line stroke="red" x1="656.25" y1="93.75" x2="687.5" y2="62.5"/>
  <line stroke="red" x1="437.5" y1="500" x2="468.75" y2="500"/>
  <line stroke="blue" x1="437.5" y1="500" x2="437.5" y2="562.5"/>
  <line stroke="red" x1="112.06" y1="656.25" x2="156.25" y2="656.25"/>
  <line stroke="blue" x1="112.06" y1="656.25" x2="125" y2="687.5" fold-step="7"/>
  <line stroke="red" x1="531.25" y1="406.25" x2="531.25" y2="437.5"/>
  <line stroke="blue" x1="531.25" y1="406.25" x2="562.5" y2="406.25"/>
  <line stroke="red" x1="593.75" y1="312.5" x2="593.75" y2="343.75"/>
  <line stroke="red" x1="593.75" y1="312.5" x2="625" y2="312.5"/>
  <line stroke="red" x1="250" y1="625" x2="281.25" y2="625"/>
  <line stroke="blue" x1="250" y1="625" x2="250" y2="750"/>
  <line stroke="blue" x1="625" y1="250" x2="656.25" y2="174.56" fold-step="12"/>
  <line stroke="red" x1="625" y1="250" x2="656.25" y2="250"/>
  <line stroke="blue" x1="625" y1="250" x2="656.25" y2="281.25"/>
  <line stroke="red" x1="156.25" y1="656.25" x2="187.5" y2="687.5"/>
  <line stroke="blue" x1="375" y1="562.5" x2="406.25" y2="562.5"/>
  <line stroke="blue" x1="375" y1="562.5" x2="375" y2="625"/>
  <line stroke="blue" x1="562.5" y1="375" x2="562.5" y2="406.25"/>
  <line stroke="blue" x1="562.5" y1="375" x2="593.75" y2="375"/>
  <line stroke="red" x1="656.25" y1="174.56" x2="656.25" y2="250"/>
  <line stroke="red" x1="656.25" y1="174.56" x2="687.5" y2="250" fold-step="12"/>
  <line stroke="blue" x1="281.25" y1="625" x2="312.5" y2="625"/>
  <line stroke="red" x1="281.25" y1="625" x2="281.25" y2="781.25"/>
  <line stroke="blue" x1="468.75" y1="500" x2="531.25" y2="500"/>
  <line stroke="red" x1="468.75" y1="500" x2="468.75" y2="562.5"/>
  <line stroke="blue" x1="593.75" y1="343.75" x2="625" y2="312.5"/>
  <line stroke="red" x1="593.75" y1="343.75" x2="593.75" y2="375"/>
  <line stroke="blue" x1="593.75" y1="343.75" x2="625" y2="375"/>
  <line stroke="red" x1="593.75" y1="343.75" x2="656.25" y2="343.75"/>
  <line stroke="red" x1="531.25" y1="437.5" x2="531.25" y2="468.75"/>
  <line stroke="red" x1="531.25" y1="437.5" x2="562.5" y2="437.5"/>
  <line stroke="blue" x1="62.5" y1="687.5" x2="125" y2="687.5"/>
  <line stroke="red" x1="62.5" y1="687.5" x2="93.75" y2="718.75"/>
  <line stroke="blue" x1="62.5" y1="687.5" x2="62.5" y2="750"/>
  <line stroke="blue" x1="687.5" y1="62.5" x2="718.75" y2="31.25"/>
  <line stroke="blue" x1="687.5" y1="62.5" x2="687.5" y2="250"/>
  <line stroke="red" x1="406.25" y1="562.5" x2="437.5" y2="562.5"/>
  <line stroke="red" x1="406.25" y1="562.5" x2="406.25" y2="625"/>
  <line stroke="blue" x1="562.5" y1="406.25" x2="562.5" y2="437.5"/>
  <line stroke="red" x1="562.5" y1="406.25" x2="593.75" y2="406.25"/>
  <line stroke="blue" x1="125" y1="687.5" x2="187.5" y2="687.5"/>
  <line stroke="red" x1="125" y1="687.5" x2="112.06" y2="718.75" fold-step="7"/>
  <line stroke="red" x1="312.5" y1="625" x2="343.75" y2="625"/>
  <line stroke="blue" x1="312.5" y1="625" x2="312.5" y2="812.5"/>
  <line stroke="red" x1="625" y1="312.5" x2="656.25" y2="281.25" fold-step="6"/>
  <line stroke="blue" x1="625" y1="312.5" x2="656.25" y2="343.75" fold-step="6"/>
  <line stroke="blue" x1="625" y1="312.5" x2="687.5" y2="312.5"/>
  <line stroke="red" x1="593.75" y1="375" x2="593.75" y2="406.25"/>
  <line stroke="red" x1="593.75" y1="375" x2="625" y2="375"/>
  <line stroke="red" x1="656.25" y1="250" x2="656.25" y2="281.25"/>
  <line stroke="blue" x1="656.25" y1="250" x2="687.5" y2="250"/>
  <line stroke="red" x1="531.25" y1="468.75" x2="531.25" y2="500"/>
  <line stroke="blue" x1="531.25" y1="468.75" x2="562.5" y2="468.75"/>
  <line stroke="blue" x1="187.5" y1="687.5" x2="218.75" y2="718.75"/>
  <line stroke="blue" x1="437.5" y1="562.5" x2="468.75" y2="562.5"/>
  <line stroke="blue" x1="437.5" y1="562.5" x2="437.5" y2="625"/>
  <line stroke="blue" x1="562.5" y1="437.5" x2="562.5" y2="468.75"/>
  <line stroke="blue" x1="562.5" y1="437.5" x2="593.75" y2="437.5"/>
  <line stroke="blue" x1="343.75" y1="625" x2="375" y2="625"/>
  <line stroke="red" x1="343.75" y1="625" x2="343.75" y2="843.75"/>
  <line stroke="blue" x1="656.25" y1="281.25" x2="687.5" y2="250" fold-step="6"/>
  <line stroke="red" x1="656.25" y1="281.25" x2="687.5" y2="312.5"/>
  <line stroke="blue" x1="593.75" y1="406.25" x2="625" y2="375"/>
  <line stroke="red" x1="593.75" y1="406.25" x2="593.75" y2="437.5"/>
  <line stroke="blue" x1="593.75" y1="406.25" x2="625" y2="437.5"/>
  <line stroke="red" x1="593.75" y1="406.25" x2="656.25" y2="406.25"/>
  <line stroke="red" x1="718.75" y1="31.25" x2="718.75" y2="174.56"/>
  <line stroke="red" x1="718.75" y1="31.25" x2="750" y2="0"/>
  <line stroke="red" x1="93.75" y1="718.75" x2="112.06" y2="718.75"/>
  <line stroke="red" x1="93.75" y1="718.75" x2="93.75" y2="737.06"/>
  <line stroke="blue" x1="93.75" y1="718.75" x2="125" y2="750"/>
  <line stroke="red" x1="112.06" y1="718.75" x2="218.75" y2="718.75"/>
  <line stroke="blue" x1="112.06" y1="718.75" x2="125" y2="750" fold-step="7"/>
  <line stroke="red" x1="375" y1="625" x2="406.25" y2="625"/>
  <line stroke="blue" x1="375" y1="625" x2="375" y2="875"/>
  <line stroke="red" x1="625" y1="375" x2="656.25" y2="343.75" fold-step="6"/>
  <line stroke="blue" x1="625" y1="375" x2="656.25" y2="406.25" fold-step="6"/>
  <line stroke="blue" x1="625" y1="375" x2="750" y2="375"/>
  <line stroke="red" x1="531.25" y1="500" x2="562.5" y2="500"/>
  <line stroke="red" x1="531.25" y1="500" x2="531.25" y2="562.5"/>
  <line stroke="blue" x1="687.5" y1="250" x2="718.75" y2="174.56" fold-step="12"/>
  <line stroke="blue" x1="687.5" y1="250" x2="687.5" y2="312.5"/>
  <line stroke="red" x1="687.5" y1="250" x2="718.75" y2="250"/>
  <line stroke="red" x1="687.5" y1="250" x2="718.75" y2="281.25" fold-step="6"/>
  <line stroke="red" x1="468.75" y1="562.5" x2="531.25" y2="562.5"/>
  <line stroke="red" x1="468.75" y1="562.5" x2="468.75" y2="625"/>
  <line stroke="blue" x1="562.5" y1="468.75" x2="562.5" y2="500"/>
  <line stroke="red" x1="562.5" y1="468.75" x2="593.75" y2="468.75"/>
  <line stroke="red" x1="593.75" y1="437.5" x2="593.75" y2="468.75"/>
  <line stroke="red" x1="593.75" y1="437.5" x2="625" y2="437.5"/>
  <line stroke="blue" x1="31.25" y1="737.06" x2="0" y2="750" fold-step="7"/>
  <line stroke="red" x1="31.25" y1="737.06" x2="31.25" y2="750"/>
  <line stroke="red" x1="31.25" y1="737.06" x2="62.5" y2="750" fold-step="7"/>
  <line stroke="red" x1="718.75" y1="174.56" x2="718.75" y2="250"/>
  <line stroke="red" x1="718.75" y1="174.56" x2="750" y2="250" fold-step="12"/>
  <line stroke="red" x1="656.25" y1="343.75" x2="718.75" y2="343.75"/>
  <line stroke="blue" x1="93.75" y1="737.06" x2="62.5" y2="750" fold-step="7"/>
  <line stroke="red" x1="93.75" y1="737.06" x2="93.75" y2="750"/>
  <line stroke="red" x1="93.75" y1="737.06" x2="125" y2="750" fold-step="7"/>
  <line stroke="blue" x1="406.25" y1="625" x2="437.5" y2="625"/>
  <line stroke="red" x1="406.25" y1="625" x2="406.25" y2="906.25"/>
  <line stroke="blue" x1="0" y1="750" x2="31.25" y2="750"/>
  <line stroke="red" x1="0" y1="750" x2="31.25" y2="758.37" fold-step="9"/>
  <line stroke="black" x1="0" y1="750" x2="0" y2="875"/>
  <line stroke="red" x1="750" y1="0" x2="781.25" y2="31.25"/>
  <line stroke="blue" x1="750" y1="0" x2="750" y2="250"/>
  <line stroke="black" x1="750" y1="0" x2="875" y2="0"/>
  <line stroke="red" x1="31.25" y1="750" x2="62.5" y2="750"/>
  <line stroke="red" x1="31.25" y1="750" x2="31.25" y2="758.37"/>
  <line stroke="red" x1="218.75" y1="718.75" x2="250" y2="750"/>
  <line stroke="blue" x1="62.5" y1="750" x2="93.75" y2="750"/>
  <line stroke="blue" x1="62.5" y1="750" x2="31.25" y2="758.37" fold-step="9"/>
  <line stroke="red" x1="62.5" y1="750" x2="93.75" y2="758.37" fold-step="9"/>
  <line stroke="blue" x1="62.5" y1="750" x2="62.5" y2="812.5"/>
  <line stroke="blue" x1="562.5" y1="500" x2="593.75" y2="500"/>
  <line stroke="blue" x1="562.5" y1="500" x2="562.5" y2="562.5"/>
  <line stroke="blue" x1="687.5" y1="312.5" x2="718.75" y2="343.75"/>
  <line stroke="red" x1="93.75" y1="750" x2="125" y2="750"/>
  <line stroke="red" x1="93.75" y1="750" x2="93.75" y2="758.37"/>
  <line stroke="blue" x1="593.75" y1="468.75" x2="625" y2="437.5"/>
  <line stroke="red" x1="593.75" y1="468.75" x2="593.75" y2="500"/>
  <line stroke="blue" x1="593.75" y1="468.75" x2="625" y2="500"/>
  <line stroke="red" x1="593.75" y1="468.75" x2="656.25" y2="468.75"/>
  <line stroke="red" x1="31.25" y1="758.37" x2="31.25" y2="843.75"/>
  <line stroke="blue" x1="125" y1="750" x2="93.75" y2="758.37" fold-step="9"/>
  <line stroke="blue" x1="125" y1="750" x2="93.75" y2="781.25"/>
  <line stroke="red" x1="125" y1="750" x2="116.63" y2="781.25" fold-step="9"/>
  <line stroke="blue" x1="125" y1="750" x2="250" y2="750"/>
  <line stroke="red" x1="718.75" y1="250" x2="718.75" y2="281.25"/>
  <line stroke="blue" x1="718.75" y1="250" x2="750" y2="250"/>
  <line stroke="red" x1="437.5" y1="625" x2="468.75" y2="625"/>
  <line stroke="blue" x1="437.5" y1="625" x2="437.5" y2="937.5"/>
  <line stroke="red" x1="625" y1="437.5" x2="656.25" y2="406.25" fold-step="6"/>
  <line stroke="blue" x1="625" y1="437.5" x2="656.25" y2="468.75" fold-step="6"/>
  <line stroke="blue" x1="625" y1="437.5" x2="687.5" y2="437.5"/>
  <line stroke="red" x1="93.75" y1="758.37" x2="93.75" y2="781.25"/>
  <line stroke="red" x1="656.25" y1="406.25" x2="718.75" y2="406.25"/>
  <line stroke="blue" x1="718.75" y1="281.25" x2="750" y2="250" fold-step="6"/>
  <line stroke="red" x1="718.75" y1="281.25" x2="718.75" y2="343.75"/>
  <line stroke="blue" x1="531.25" y1="562.5" x2="562.5" y2="562.5"/>
  <line stroke="red" x1="531.25" y1="562.5" x2="531.25" y2="625"/>
  <line stroke="red" x1="593.75" y1="500" x2="625" y2="500"/>
  <line stroke="red" x1="593.75" y1="500" x2="593.75" y2="562.5"/>
  <line stroke="blue" x1="468.75" y1="625" x2="531.25" y2="625"/>
  <line stroke="red" x1="468.75" y1="625" x2="468.75" y2="968.75"/>
  <line stroke="red" x1="781.25" y1="31.25" x2="781.25" y2="174.56"/>
  <line stroke="blue" x1="781.25" y1="31.25" x2="812.5" y2="62.5"/>
  <line stroke="red" x1="781.25" y1="31.25" x2="875" y2="31.25"/>
  <line stroke="red" x1="93.75" y1="781.25" x2="116.63" y2="781.25"/>
  <line stroke="red" x1="93.75" y1="781.25" x2="62.5" y2="812.5"/>
  <line stroke="blue" x1="116.63" y1="781.25" x2="125" y2="812.5" fold-step="9"/>
  <line stroke="red" x1="116.63" y1="781.25" x2="281.25" y2="781.25"/>
  <line stroke="blue" x1="250" y1="750" x2="281.25" y2="781.25"/>
  <line stroke="blue" x1="750" y1="250" x2="781.25" y2="174.56" fold-step="12"/>
  <line stroke="red" x1="750" y1="250" x2="781.25" y2="250"/>
  <line stroke="red" x1="750" y1="250" x2="781.25" y2="281.25" fold-step="6"/>
  <line stroke="blue" x1="750" y1="250" x2="750" y2="375"/>
  <line stroke="red" x1="562.5" y1="562.5" x2="593.75" y2="562.5"/>
  <line stroke="blue" x1="562.5" y1="562.5" x2="562.5" y2="625"/>
  <line stroke="red" x1="718.75" y1="343.75" x2="750" y2="375"/>
  <line stroke="blue" x1="625" y1="500" x2="656.25" y2="468.75"/>
  <line stroke="blue" x1="625" y1="500" x2="625" y2="562.5"/>
  <line stroke="red" x1="781.25" y1="174.56" x2="781.25" y2="250"/>
  <line stroke="red" x1="781.25" y1="174.56" x2="812.5" y2="250" fold-step="12"/>
  <line stroke="red" x1="656.25" y1="468.75" x2="687.5" y2="437.5"/>
  <line stroke="red" x1="656.25" y1="468.75" x2="687.5" y2="500" fold-step="6"/>
  <line stroke="red" x1="656.25" y1="468.75" x2="656.25" y2="562.5"/>
  <line stroke="blue" x1="62.5" y1="812.5" x2="125" y2="812.5"/>
  <line stroke="blue" x1="62.5" y1="812.5" x2="31.25" y2="843.75"/>
  <line stroke="blue" x1="687.5" y1="437.5" x2="718.75" y2="406.25"/>
  <line stroke="blue" x1="687.5" y1="437.5" x2="687.5" y2="500"/>
  <line stroke="red" x1="812.5" y1="62.5" x2="843.75" y2="93.75"/>
  <line stroke="blue" x1="812.5" y1="62.5" x2="812.5" y2="250"/>
  <line stroke="blue" x1="812.5" y1="62.5" x2="875" y2="62.5"/>
  <line stroke="blue" x1="593.75" y1="562.5" x2="625" y2="562.5"/>
  <line stroke="red" x1="593.75" y1="562.5" x2="593.75" y2="625"/>
  <line stroke="red" x1="531.25" y1="625" x2="562.5" y2="625"/>
  <line stroke="red" x1="531.25" y1="625" x2="531.25" y2="968.75"/>
  <line stroke="red" x1="781.25" y1="250" x2="781.25" y2="281.25"/>
  <line stroke="blue" x1="781.25" y1="250" x2="812.5" y2="250"/>
  <line stroke="red" x1="125" y1="812.5" x2="116.63" y2="843.75" fold-step="9"/>
  <line stroke="blue" x1="125" y1="812.5" x2="312.5" y2="812.5"/>
  <line stroke="red" x1="718.75" y1="406.25" x2="750" y2="375"/>
  <line stroke="red" x1="718.75" y1="406.25" x2="718.75" y2="468.75"/>
  <line stroke="red" x1="281.25" y1="781.25" x2="312.5" y2="812.5"/>
  <line stroke="blue" x1="781.25" y1="281.25" x2="812.5" y2="250" fold-step="6"/>
  <line stroke="red" x1="781.25" y1="281.25" x2="781.25" y2="343.75"/>
  <line stroke="red" x1="750" y1="375" x2="781.25" y2="343.75"/>
  <line stroke="red" x1="750" y1="375" x2="781.25" y2="406.25"/>
  <line stroke="blue" x1="750" y1="375" x2="750" y2="500"/>
  <line stroke="red" x1="750" y1="375" x2="875" y2="375"/>
  <line stroke="blue" x1="562.5" y1="625" x2="593.75" y2="625"/>
  <line stroke="blue" x1="562.5" y1="625" x2="562.5" y2="937.5"/>
  <line stroke="red" x1="625" y1="562.5" x2="656.25" y2="562.5"/>
  <line stroke="blue" x1="625" y1="562.5" x2="625" y2="625"/>
  <line stroke="red" x1="31.25" y1="843.75" x2="116.63" y2="843.75"/>
  <line stroke="red" x1="31.25" y1="843.75" x2="0" y2="875"/>
  <line stroke="red" x1="843.75" y1="93.75" x2="843.75" y2="174.56"/>
  <line stroke="red" x1="843.75" y1="93.75" x2="875" y2="93.75"/>
  <line stroke="blue" x1="843.75" y1="93.75" x2="875" y2="125"/>
  <line stroke="blue" x1="687.5" y1="500" x2="718.75" y2="468.75" fold-step="6"/>
  <line stroke="blue" x1="687.5" y1="500" x2="687.5" y2="562.5"/>
  <line stroke="blue" x1="812.5" y1="250" x2="843.75" y2="174.56" fold-step="12"/>
  <line stroke="blue" x1="812.5" y1="250" x2="812.5" y2="312.5"/>
  <line stroke="red" x1="812.5" y1="250" x2="843.75" y2="250"/>
  <line stroke="red" x1="812.5" y1="250" x2="843.75" y2="281.25" fold-step="6"/>
  <line stroke="blue" x1="116.63" y1="843.75" x2="125" y2="875" fold-step="9"/>
  <line stroke="red" x1="116.63" y1="843.75" x2="343.75" y2="843.75"/>
  <line stroke="blue" x1="781.25" y1="343.75" x2="812.5" y2="312.5"/>
  <line stroke="red" x1="781.25" y1="343.75" x2="843.75" y2="343.75"/>
  <line stroke="red" x1="718.75" y1="468.75" x2="750" y2="500" fold-step="6"/>
  <line stroke="red" x1="718.75" y1="468.75" x2="718.75" y2="562.5"/>
  <line stroke="red" x1="843.75" y1="174.56" x2="843.75" y2="250"/>
  <line stroke="red" x1="843.75" y1="174.56" x2="875" y2="250" fold-step="12"/>
  <line stroke="red" x1="593.75" y1="625" x2="625" y2="625"/>
  <line stroke="red" x1="593.75" y1="625" x2="593.75" y2="906.25"/>
  <line stroke="blue" x1="656.25" y1="562.5" x2="687.5" y2="562.5"/>
  <line stroke="red" x1="656.25" y1="562.5" x2="656.25" y2="625"/>
  <line stroke="blue" x1="312.5" y1="812.5" x2="343.75" y2="843.75"/>
  <line stroke="red" x1="812.5" y1="312.5" x2="843.75" y2="281.25"/>
  <line stroke="blue" x1="812.5" y1="312.5" x2="875" y2="312.5"/>
  <line stroke="blue" x1="0" y1="875" x2="125" y2="875"/>
  <line stroke="red" x1="0" y1="875" x2="31.25" y2="906.25"/>
  <line stroke="black" x1="0" y1="875" x2="0" y2="1000"/>
  <line stroke="blue" x1="875" y1="0" x2="875" y2="31.25"/>
  <line stroke="blue" x1="875" y1="0" x2="906.25" y2="31.25" fold-step="2"/>
  <line stroke="black" x1="875" y1="0" x2="1000" y2="0"/>
  <line stroke="red" x1="875" y1="31.25" x2="875" y2="62.5"/>
  <line stroke="red" x1="875" y1="31.25" x2="906.25" y2="31.25"/>
  <line stroke="blue" x1="875" y1="62.5" x2="875" y2="93.75"/>
  <line stroke="red" x1="875" y1="62.5" x2="906.25" y2="31.25" fold-step="2"/>
  <line stroke="blue" x1="875" y1="62.5" x2="906.25" y2="93.75" fold-step="2"/>
  <line stroke="blue" x1="875" y1="62.5" x2="937.5" y2="62.5"/>
  <line stroke="red" x1="843.75" y1="250" x2="843.75" y2="281.25"/>
  <line stroke="blue" x1="843.75" y1="250" x2="875" y2="250"/>
  <line stroke="red" x1="875" y1="93.75" x2="875" y2="125"/>
  <line stroke="red" x1="875" y1="93.75" x2="906.25" y2="93.75"/>
  <line stroke="red" x1="781.25" y1="406.25" x2="781.25" y2="468.75"/>
  <line stroke="blue" x1="781.25" y1="406.25" x2="812.5" y2="437.5"/>
  <line stroke="red" x1="781.25" y1="406.25" x2="843.75" y2="406.25"/>
  <line stroke="red" x1="125" y1="875" x2="116.63" y2="906.25" fold-step="9"/>
  <line stroke="blue" x1="125" y1="875" x2="375" y2="875"/>
  <line stroke="blue" x1="625" y1="625" x2="656.25" y2="625"/>
  <line stroke="blue" x1="625" y1="625" x2="625" y2="875"/>
  <line stroke="blue" x1="875" y1="125" x2="875" y2="250"/>
  <line stroke="blue" x1="875" y1="125" x2="906.25" y2="93.75"/>
  <line stroke="red" x1="687.5" y1="562.5" x2="718.75" y2="562.5"/>
  <line stroke="blue" x1="687.5" y1="562.5" x2="687.5" y2="625"/>
  <line stroke="blue" x1="843.75" y1="281.25" x2="875" y2="250"/>
  <line stroke="red" x1="843.75" y1="281.25" x2="875" y2="281.25"/>
  <line stroke="blue" x1="843.75" y1="281.25" x2="875" y2="312.5" fold-step="6"/>
  <line stroke="blue" x1="750" y1="500" x2="781.25" y2="468.75" fold-step="6"/>
  <line stroke="blue" x1="750" y1="500" x2="750" y2="562.5"/>
  <line stroke="red" x1="656.25" y1="625" x2="687.5" y2="625"/>
  <line stroke="red" x1="656.25" y1="625" x2="656.25" y2="843.75"/>
  <line stroke="red" x1="31.25" y1="906.25" x2="116.63" y2="906.25"/>
  <line stroke="blue" x1="31.25" y1="906.25" x2="62.5" y2="937.5"/>
  <line stroke="red" x1="31.25" y1="906.25" x2="31.25" y2="991.63"/>
  <line stroke="red" x1="906.25" y1="31.25" x2="968.75" y2="31.25"/>
  <line stroke="red" x1="875" y1="250" x2="875" y2="281.25"/>
  <line stroke="blue" x1="875" y1="250" x2="906.25" y2="174.56" fold-step="12"/>
  <line stroke="red" x1="875" y1="250" x2="906.25" y2="250"/>
  <line stroke="blue" x1="875" y1="250" x2="906.25" y2="281.25"/>
  <line stroke="red" x1="343.75" y1="843.75" x2="375" y2="875"/>
  <line stroke="red" x1="781.25" y1="468.75" x2="812.5" y2="500" fold-step="6"/>
  <line stroke="red" x1="781.25" y1="468.75" x2="781.25" y2="562.5"/>
  <line stroke="red" x1="843.75" y1="343.75" x2="875" y2="312.5" fold-step="6"/>
  <line stroke="red" x1="843.75" y1="343.75" x2="875" y2="343.75"/>
  <line stroke="blue" x1="843.75" y1="343.75" x2="875" y2="375" fold-step="6"/>
  <line stroke="red" x1="906.25" y1="93.75" x2="906.25" y2="174.56"/>
  <line stroke="red" x1="906.25" y1="93.75" x2="937.5" y2="62.5"/>
  <line stroke="red" x1="906.25" y1="93.75" x2="937.5" y2="125" fold-step="2"/>
  <line stroke="blue" x1="718.75" y1="562.5" x2="750" y2="562.5"/>
  <line stroke="red" x1="718.75" y1="562.5" x2="718.75" y2="625"/>
  <line stroke="blue" x1="116.63" y1="906.25" x2="125" y2="937.5" fold-step="9"/>
  <line stroke="red" x1="116.63" y1="906.25" x2="406.25" y2="906.25"/>
  <line stroke="blue" x1="875" y1="281.25" x2="875" y2="312.5"/>
  <line stroke="red" x1="875" y1="281.25" x2="906.25" y2="281.25"/>
  <line stroke="blue" x1="812.5" y1="437.5" x2="812.5" y2="500"/>
  <line stroke="red" x1="812.5" y1="437.5" x2="843.75" y2="468.75"/>
  <line stroke="blue" x1="812.5" y1="437.5" x2="875" y2="437.5"/>
  <line stroke="red" x1="906.25" y1="174.56" x2="906.25" y2="250"/>
  <line stroke="red" x1="906.25" y1="174.56" x2="937.5" y2="250" fold-step="12"/>
  <line stroke="blue" x1="687.5" y1="625" x2="718.75" y2="625"/>
  <line stroke="blue" x1="687.5" y1="625" x2="687.5" y2="812.5"/>
  <line stroke="red" x1="875" y1="312.5" x2="875" y2="343.75"/>
  <line stroke="blue" x1="875" y1="312.5" x2="937.5" y2="312.5"/>
  <line stroke="red" x1="875" y1="312.5" x2="914.62" y2="289.62" fold-step="4"/>
  <line stroke="blue" x1="875" y1="312.5" x2="929.13" y2="343.75" fold-step="4"/>
  <line stroke="red" x1="843.75" y1="406.25" x2="875" y2="375" fold-step="6"/>
  <line stroke="red" x1="843.75" y1="406.25" x2="875" y2="406.25"/>
  <line stroke="blue" x1="843.75" y1="406.25" x2="875" y2="437.5" fold-step="6"/>
  <line stroke="red" x1="750" y1="562.5" x2="781.25" y2="562.5"/>
  <line stroke="blue" x1="750" y1="562.5" x2="750" y2="625"/>
  <line stroke="blue" x1="62.5" y1="937.5" x2="125" y2="937.5"/>
  <line stroke="red" x1="62.5" y1="937.5" x2="93.75" y2="968.75"/>
  <line stroke="blue" x1="62.5" y1="937.5" x2="62.5" y2="1000"/>
  <line stroke="blue" x1="937.5" y1="62.5" x2="937.5" y2="125"/>
  <line stroke="blue" x1="937.5" y1="62.5" x2="968.75" y2="31.25"/>
  <line stroke="blue" x1="875" y1="343.75" x2="875" y2="375"/>
  <line stroke="red" x1="875" y1="343.75" x2="929.13" y2="343.75"/>
  <line stroke="red" x1="906.25" y1="250" x2="906.25" y2="281.25"/>
  <line stroke="blue" x1="906.25" y1="250" x2="937.5" y2="250"/>
  <line stroke="red" x1="125" y1="937.5" x2="116.63" y2="968.75" fold-step="9"/>
  <line stroke="blue" x1="125" y1="937.5" x2="437.5" y2="937.5"/>
  <line stroke="blue" x1="937.5" y1="125" x2="937.5" y2="250"/>
  <line stroke="blue" x1="937.5" y1="125" x2="968.75" y2="93.75" fold-step="2"/>
  <line stroke="red" x1="906.25" y1="281.25" x2="914.62" y2="289.62"/>
  <line stroke="blue" x1="375" y1="875" x2="406.25" y2="906.25"/>
  <line stroke="blue" x1="875" y1="375" x2="875" y2="406.25"/>
  <line stroke="blue" x1="875" y1="375" x2="1000" y2="375"/>
  <line stroke="red" x1="875" y1="375" x2="929.13" y2="343.75" fold-step="4"/>
  <line stroke="blue" x1="875" y1="375" x2="929.13" y2="406.25" fold-step="4"/>
  <line stroke="red" x1="718.75" y1="625" x2="750" y2="625"/>
  <line stroke="red" x1="718.75" y1="625" x2="718.75" y2="781.25"/>
  <line stroke="blue" x1="812.5" y1="500" x2="843.75" y2="468.75" fold-step="6"/>
  <line stroke="blue" x1="812.5" y1="500" x2="812.5" y2="562.5"/>
  <line stroke="blue" x1="914.62" y1="289.62" x2="937.5" y2="250" fold-step="4"/>
  <line stroke="red" x1="914.62" y1="289.62" x2="937.5" y2="312.5"/>
  <line stroke="blue" x1="781.25" y1="562.5" x2="812.5" y2="562.5"/>
  <line stroke="red" x1="781.25" y1="562.5" x2="781.25" y2="625"/>
  <line stroke="red" x1="875" y1="406.25" x2="875" y2="437.5"/>
  <line stroke="red" x1="875" y1="406.25" x2="929.13" y2="406.25"/>
  <line stroke="red" x1="843.75" y1="468.75" x2="875" y2="437.5" fold-step="6"/>
  <line stroke="red" x1="843.75" y1="468.75" x2="875" y2="468.75"/>
  <line stroke="red" x1="843.75" y1="468.75" x2="843.75" y2="531.25"/>
  <line stroke="blue" x1="843.75" y1="468.75" x2="875" y2="500"/>
  <line stroke="red" x1="968.75" y1="31.25" x2="968.75" y2="93.75"/>
  <line stroke="red" x1="968.75" y1="31.25" x2="1000" y2="0"/>
  <line stroke="blue" x1="937.5" y1="250" x2="968.75" y2="174.56" fold-step="12"/>
  <line stroke="blue" x1="937.5" y1="250" x2="937.5" y2="312.5"/>
  <line stroke="red" x1="937.5" y1="250" x2="968.75" y2="250"/>
  <line stroke="red" x1="937.5" y1="250" x2="968.75" y2="304.13" fold-step="4"/>
  <line stroke="red" x1="93.75" y1="968.75" x2="116.63" y2="968.75"/>
  <line stroke="red" x1="93.75" y1="968.75" x2="93.75" y2="991.63"/>
  <line stroke="blue" x1="93.75" y1="968.75" x2="125" y2="1000"/>
  <line stroke="red" x1="968.75" y1="93.75" x2="968.75" y2="174.56"/>
  <line stroke="red" x1="968.75" y1="93.75" x2="1000" y2="125" fold-step="2"/>
  <line stroke="blue" x1="116.63" y1="968.75" x2="125" y2="1000" fold-step="9"/>
  <line stroke="red" x1="116.63" y1="968.75" x2="468.75" y2="968.75"/>
  <line stroke="blue" x1="750" y1="625" x2="781.25" y2="625"/>
  <line stroke="blue" x1="750" y1="625" x2="750" y2="750"/>
  <line stroke="blue" x1="875" y1="437.5" x2="875" y2="468.75"/>
  <line stroke="blue" x1="875" y1="437.5" x2="937.5" y2="437.5"/>
  <line stroke="red" x1="875" y1="437.5" x2="929.13" y2="406.25" fold-step="4"/>
  <line stroke="blue" x1="875" y1="437.5" x2="914.62" y2="460.38" fold-step="4"/>
  <line stroke="red" x1="968.75" y1="174.56" x2="968.75" y2="250"/>
  <line stroke="red" x1="968.75" y1="174.56" x2="1000" y2="250" fold-step="12"/>
  <line stroke="red" x1="812.5" y1="562.5" x2="843.75" y2="531.25"/>
  <line stroke="blue" x1="812.5" y1="562.5" x2="812.5" y2="625"/>
  <line stroke="red" x1="812.5" y1="562.5" x2="843.75" y2="593.75"/>
  <line stroke="blue" x1="812.5" y1="562.5" x2="875" y2="562.5"/>
  <line stroke="blue" x1="937.5" y1="312.5" x2="968.75" y2="343.75"/>
  <line stroke="red" x1="929.13" y1="343.75" x2="968.75" y2="343.75"/>
  <line stroke="blue" x1="31.25" y1="991.63" x2="0" y2="1000" fold-step="9"/>
  <line stroke="red" x1="31.25" y1="991.63" x2="31.25" y2="1000"/>
  <line stroke="red" x1="31.25" y1="991.63" x2="62.5" y2="1000" fold-step="9"/>
  <line stroke="red" x1="875" y1="468.75" x2="875" y2="500"/>
  <line stroke="red" x1="875" y1="468.75" x2="906.25" y2="468.75"/>
  <line stroke="red" x1="406.25" y1="906.25" x2="437.5" y2="937.5"/>
  <line stroke="blue" x1="93.75" y1="991.63" x2="62.5" y2="1000" fold-step="9"/>
  <line stroke="red" x1="93.75" y1="991.63" x2="93.75" y2="1000"/>
  <line stroke="red" x1="93.75" y1="991.63" x2="125" y2="1000" fold-step="9"/>
  <line stroke="blue" x1="843.75" y1="531.25" x2="875" y2="500"/>
  <line stroke="red" x1="843.75" y1="531.25" x2="887.94" y2="531.25"/>
  <line stroke="black" x1="0" y1="1000" x2="31.25" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="125"/>
  <line stroke="black" x1="31.25" y1="1000" x2="62.5" y2="1000"/>
  <line stroke="red" x1="781.25" y1="625" x2="812.5" y2="625"/>
  <line stroke="red" x1="781.25" y1="625" x2="781.25" y2="718.75"/>
  <line stroke="red" x1="968.75" y1="250" x2="968.75" y2="304.13"/>
  <line stroke="blue" x1="968.75" y1="250" x2="1000" y2="250"/>
  <line stroke="black" x1="62.5" y1="1000" x2="93.75" y2="1000"/>
  <line stroke="black" x1="93.75" y1="1000" x2="125" y2="1000"/>
  <line stroke="black" x1="125" y1="1000" x2="500" y2="1000"/>
  <line stroke="blue" x1="875" y1="500" x2="906.25" y2="468.75"/>
  <line stroke="blue" x1="875" y1="500" x2="887.94" y2="531.25" fold-step="8"/>
  <line stroke="red" x1="875" y1="500" x2="906.25" y2="500"/>
  <line stroke="red" x1="875" y1="500" x2="906.25" y2="512.94" fold-step="8"/>
  <line stroke="blue" x1="875" y1="500" x2="906.25" y2="531.25"/>
  <line stroke="black" x1="1000" y1="125" x2="1000" y2="250"/>
  <line stroke="red" x1="929.13" y1="406.25" x2="968.75" y2="406.25"/>
  <line stroke="red" x1="968.75" y1="304.13" x2="968.75" y2="343.75"/>
  <line stroke="blue" x1="968.75" y1="304.13" x2="1000" y2="250" fold-step="4"/>
  <line stroke="red" x1="906.25" y1="468.75" x2="914.62" y2="460.38"/>
  <line stroke="red" x1="906.25" y1="468.75" x2="906.25" y2="500"/>
  <line stroke="red" x1="914.62" y1="460.38" x2="937.5" y2="437.5"/>
  <line stroke="red" x1="914.62" y1="460.38" x2="937.5" y2="500" fold-step="4"/>
  <line stroke="blue" x1="812.5" y1="625" x2="843.75" y2="625"/>
  <line stroke="blue" x1="812.5" y1="625" x2="812.5" y2="687.5"/>
  <line stroke="red" x1="968.75" y1="343.75" x2="1000" y2="375"/>
  <line stroke="black" x1="1000" y1="250" x2="1000" y2="375"/>
  <line stroke="red" x1="843.75" y1="593.75" x2="843.75" y2="625"/>
  <line stroke="red" x1="843.75" y1="593.75" x2="887.94" y2="593.75"/>
  <line stroke="blue" x1="843.75" y1="593.75" x2="875" y2="625"/>
  <line stroke="blue" x1="437.5" y1="937.5" x2="468.75" y2="968.75"/>
  <line stroke="blue" x1="937.5" y1="437.5" x2="968.75" y2="406.25"/>
  <line stroke="blue" x1="937.5" y1="437.5" x2="937.5" y2="500"/>
  <line stroke="red" x1="887.94" y1="531.25" x2="875" y2="562.5" fold-step="8"/>
  <line stroke="red" x1="887.94" y1="531.25" x2="906.25" y2="531.25"/>
  <line stroke="red" x1="906.25" y1="500" x2="906.25" y2="512.94"/>
  <line stroke="blue" x1="906.25" y1="500" x2="937.5" y2="500"/>
  <line stroke="blue" x1="875" y1="562.5" x2="887.94" y2="593.75" fold-step="8"/>
  <line stroke="blue" x1="875" y1="562.5" x2="937.5" y2="562.5"/>
  <line stroke="red" x1="906.25" y1="512.94" x2="906.25" y2="531.25"/>
  <line stroke="blue" x1="906.25" y1="512.94" x2="937.5" y2="500" fold-step="8"/>
  <line stroke="red" x1="843.75" y1="625" x2="843.75" y2="656.25"/>
  <line stroke="red" x1="843.75" y1="625" x2="875" y2="625"/>
  <line stroke="red" x1="906.25" y1="531.25" x2="937.5" y2="562.5"/>
  <line stroke="red" x1="968.75" y1="406.25" x2="968.75" y2="445.87"/>
  <line stroke="red" x1="968.75" y1="406.25" x2="1000" y2="375"/>
  <line stroke="blue" x1="750" y1="750" x2="718.75" y2="781.25"/>
  <line stroke="red" x1="750" y1="750" x2="781.25" y2="718.75"/>
  <line stroke="blue" x1="750" y1="750" x2="875" y2="750"/>
  <line stroke="red" x1="718.75" y1="781.25" x2="687.5" y2="812.5"/>
  <line stroke="red" x1="718.75" y1="781.25" x2="883.37" y2="781.25"/>
  <line stroke="blue" x1="781.25" y1="718.75" x2="812.5" y2="687.5"/>
  <line stroke="red" x1="781.25" y1="718.75" x2="887.94" y2="718.75"/>
  <line stroke="blue" x1="937.5" y1="500" x2="968.75" y2="445.87" fold-step="4"/>
  <line stroke="red" x1="937.5" y1="500" x2="968.75" y2="500"/>
  <line stroke="blue" x1="937.5" y1="500" x2="937.5" y2="562.5"/>
  <line stroke="red" x1="937.5" y1="500" x2="968.75" y2="512.94" fold-step="8"/>
  <line stroke="blue" x1="687.5" y1="812.5" x2="656.25" y2="843.75"/>
  <line stroke="blue" x1="687.5" y1="812.5" x2="875" y2="812.5"/>
  <line stroke="red" x1="812.5" y1="687.5" x2="843.75" y2="656.25"/>
  <line stroke="blue" x1="812.5" y1="687.5" x2="875" y2="687.5"/>
  <line stroke="red" x1="968.75" y1="445.87" x2="968.75" y2="500"/>
  <line stroke="red" x1="968.75" y1="445.87" x2="1000" y2="500" fold-step="4"/>
  <line stroke="black" x1="1000" y1="375" x2="1000" y2="500"/>
  <line stroke="red" x1="887.94" y1="593.75" x2="875" y2="625" fold-step="8"/>
  <line stroke="red" x1="887.94" y1="593.75" x2="968.75" y2="593.75"/>
  <line stroke="red" x1="656.25" y1="843.75" x2="625" y2="875"/>
  <line stroke="red" x1="656.25" y1="843.75" x2="883.37" y2="843.75"/>
  <line stroke="blue" x1="843.75" y1="656.25" x2="875" y2="625"/>
  <line stroke="red" x1="843.75" y1="656.25" x2="887.94" y2="656.25"/>
  <line stroke="blue" x1="625" y1="875" x2="593.75" y2="906.25"/>
  <line stroke="blue" x1="625" y1="875" x2="875" y2="875"/>
  <line stroke="blue" x1="875" y1="625" x2="887.94" y2="656.25" fold-step="8"/>
  <line stroke="blue" x1="875" y1="625" x2="1000" y2="625"/>
  <line stroke="red" x1="468.75" y1="968.75" x2="500" y2="1000"/>
  <line stroke="red" x1="593.75" y1="906.25" x2="562.5" y2="937.5"/>
  <line stroke="red" x1="593.75" y1="906.25" x2="883.37" y2="906.25"/>
  <line stroke="red" x1="968.75" y1="500" x2="968.75" y2="512.94"/>
  <line stroke="blue" x1="968.75" y1="500" x2="1000" y2="500"/>
  <line stroke="blue" x1="562.5" y1="937.5" x2="531.25" y2="968.75"/>
  <line stroke="blue" x1="562.5" y1="937.5" x2="875" y2="937.5"/>
  <line stroke="blue" x1="937.5" y1="562.5" x2="968.75" y2="593.75"/>
  <line stroke="blue" x1="968.75" y1="512.94" x2="1000" y2="500" fold-step="8"/>
  <line stroke="red" x1="968.75" y1="512.94" x2="968.75" y2="593.75"/>
  <line stroke="red" x1="887.94" y1="656.25" x2="875" y2="687.5" fold-step="8"/>
  <line stroke="red" x1="887.94" y1="656.25" x2="968.75" y2="656.25"/>
  <line stroke="red" x1="531.25" y1="968.75" x2="500" y2="1000"/>
  <line stroke="red" x1="531.25" y1="968.75" x2="883.37" y2="968.75"/>
  <line stroke="blue" x1="875" y1="687.5" x2="887.94" y2="718.75" fold-step="8"/>
  <line stroke="blue" x1="875" y1="687.5" x2="937.5" y2="687.5"/>
  <line stroke="black" x1="500" y1="1000" x2="875" y2="1000"/>
  <line stroke="black" x1="1000" y1="500" x2="1000" y2="625"/>
  <line stroke="red" x1="968.75" y1="593.75" x2="1000" y2="625"/>
  <line stroke="red" x1="887.94" y1="718.75" x2="875" y2="750" fold-step="8"/>
  <line stroke="red" x1="887.94" y1="718.75" x2="906.25" y2="718.75"/>
  <line stroke="blue" x1="875" y1="750" x2="906.25" y2="718.75"/>
  <line stroke="blue" x1="875" y1="750" x2="906.25" y2="737.06" fold-step="8"/>
  <line stroke="red" x1="875" y1="750" x2="906.25" y2="750"/>
  <line stroke="blue" x1="875" y1="750" x2="883.37" y2="781.25" fold-step="10"/>
  <line stroke="red" x1="875" y1="750" x2="906.25" y2="758.37" fold-step="10"/>
  <line stroke="blue" x1="875" y1="750" x2="906.25" y2="781.25"/>
  <line stroke="red" x1="906.25" y1="718.75" x2="937.5" y2="687.5"/>
  <line stroke="red" x1="906.25" y1="718.75" x2="906.25" y2="737.06"/>
  <line stroke="blue" x1="937.5" y1="687.5" x2="968.75" y2="656.25"/>
  <line stroke="blue" x1="937.5" y1="687.5" x2="937.5" y2="750"/>
  <line stroke="red" x1="906.25" y1="737.06" x2="906.25" y2="750"/>
  <line stroke="red" x1="906.25" y1="737.06" x2="937.5" y2="750" fold-step="8"/>
  <line stroke="red" x1="968.75" y1="656.25" x2="1000" y2="625"/>
  <line stroke="red" x1="968.75" y1="656.25" x2="968.75" y2="737.06"/>
  <line stroke="red" x1="906.25" y1="750" x2="906.25" y2="758.37"/>
  <line stroke="blue" x1="906.25" y1="750" x2="937.5" y2="750"/>
  <line stroke="black" x1="1000" y1="625" x2="1000" y2="750"/>
  <line stroke="red" x1="883.37" y1="781.25" x2="875" y2="812.5" fold-step="10"/>
  <line stroke="red" x1="883.37" y1="781.25" x2="906.25" y2="781.25"/>
  <line stroke="red" x1="906.25" y1="758.37" x2="906.25" y2="781.25"/>
  <line stroke="blue" x1="906.25" y1="758.37" x2="937.5" y2="750" fold-step="10"/>
  <line stroke="blue" x1="875" y1="812.5" x2="883.37" y2="843.75" fold-step="10"/>
  <line stroke="blue" x1="875" y1="812.5" x2="937.5" y2="812.5"/>
  <line stroke="red" x1="906.25" y1="781.25" x2="937.5" y2="812.5"/>
  <line stroke="blue" x1="937.5" y1="750" x2="968.75" y2="737.06" fold-step="8"/>
  <line stroke="red" x1="937.5" y1="750" x2="968.75" y2="750"/>
  <line stroke="red" x1="937.5" y1="750" x2="968.75" y2="758.37" fold-step="10"/>
  <line stroke="blue" x1="937.5" y1="750" x2="937.5" y2="812.5"/>
  <line stroke="red" x1="968.75" y1="737.06" x2="968.75" y2="750"/>
  <line stroke="red" x1="968.75" y1="737.06" x2="1000" y2="750" fold-step="8"/>
  <line stroke="red" x1="883.37" y1="843.75" x2="875" y2="875" fold-step="10"/>
  <line stroke="red" x1="883.37" y1="843.75" x2="968.75" y2="843.75"/>
  <line stroke="red" x1="968.75" y1="750" x2="968.75" y2="758.37"/>
  <line stroke="blue" x1="968.75" y1="750" x2="1000" y2="750"/>
  <line stroke="blue" x1="968.75" y1="758.37" x2="1000" y2="750" fold-step="10"/>
  <line stroke="red" x1="968.75" y1="758.37" x2="968.75" y2="843.75"/>
  <line stroke="blue" x1="875" y1="875" x2="883.37" y2="906.25" fold-step="10"/>
  <line stroke="blue" x1="875" y1="875" x2="1000" y2="875"/>
  <line stroke="blue" x1="937.5" y1="812.5" x2="968.75" y2="843.75"/>
  <line stroke="black" x1="1000" y1="750" x2="1000" y2="875"/>
  <line stroke="red" x1="883.37" y1="906.25" x2="875" y2="937.5" fold-step="10"/>
  <line stroke="red" x1="883.37" y1="906.25" x2="968.75" y2="906.25"/>
  <line stroke="blue" x1="875" y1="937.5" x2="883.37" y2="968.75" fold-step="10"/>
  <line stroke="blue" x1="875" y1="937.5" x2="937.5" y2="937.5"/>
  <line stroke="red" x1="968.75" y1="843.75" x2="1000" y2="875"/>
  <line stroke="red" x1="883.37" y1="968.75" x2="906.25" y2="968.75"/>
  <line stroke="red" x1="883.37" y1="968.75" x2="875" y2="1000" fold-step="10"/>
  <line stroke="red" x1="937.5" y1="937.5" x2="906.25" y2="968.75"/>
  <line stroke="blue" x1="937.5" y1="937.5" x2="968.75" y2="906.25"/>
  <line stroke="blue" x1="937.5" y1="937.5" x2="937.5" y2="1000"/>
  <line stroke="blue" x1="906.25" y1="968.75" x2="875" y2="1000"/>
  <line stroke="red" x1="906.25" y1="968.75" x2="906.25" y2="991.63"/>
  <line stroke="red" x1="968.75" y1="906.25" x2="1000" y2="875"/>
  <line stroke="red" x1="968.75" y1="906.25" x2="968.75" y2="991.63"/>
  <line stroke="blue" x1="875" y1="1000" x2="906.25" y2="991.63" fold-step="10"/>
  <line stroke="black" x1="875" y1="1000" x2="906.25" y2="1000"/>
  <line stroke="black" x1="1000" y1="875" x2="1000" y2="1000"/>
  <line stroke="red" x1="906.25" y1="991.63" x2="906.25" y2="1000"/>
  <line stroke="red" x1="906.25" y1="991.63" x2="937.5" y2="1000" fold-step="10"/>
  <line stroke="black" x1="906.25" y1="1000" x2="937.5" y2="1000"/>
  <line stroke="blue" x1="937.5" y1="1000" x2="968.75" y2="991.63" fold-step="10"/>
  <line stroke="black" x1="937.5" y1="1000" x2="968.75" y2="1000"/>
  <line stroke="red" x1="968.75" y1="991.63" x2="968.75" y2="1000"/>
  <line stroke="red" x1="968.75" y1="991.63" x2="1000" y2="1000" fold-step="10"/>
  <line stroke="black" x1="968.75" y1="1000" x2="1000" y2="1000"/>
</svg>""",
    narrowed_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="red" x1="0" y1="0" x2="15.63" y2="15.63"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="125"/>
  <line stroke="black" x1="0" y1="0" x2="125" y2="0"/>
  <line stroke="red" x1="15.63" y1="15.62" x2="15.63" y2="109.38"/>
  <line stroke="blue" x1="15.63" y1="15.62" x2="109.38" y2="15.62"/>
  <line stroke="red" x1="15.63" y1="15.63" x2="31.25" y2="31.25"/>
  <line stroke="blue" x1="31.25" y1="31.25" x2="46.88" y2="46.88"/>
  <line stroke="red" x1="31.25" y1="31.25" x2="31.25" y2="93.75"/>
  <line stroke="red" x1="31.25" y1="31.25" x2="93.75" y2="31.25"/>
  <line stroke="blue" x1="46.88" y1="46.88" x2="62.5" y2="62.5"/>
  <line stroke="blue" x1="46.88" y1="46.88" x2="46.88" y2="109.38"/>
  <line stroke="red" x1="46.88" y1="46.88" x2="109.38" y2="46.88"/>
  <line stroke="red" x1="62.5" y1="62.5" x2="78.12" y2="78.12"/>
  <line stroke="blue" x1="62.5" y1="62.5" x2="62.5" y2="125"/>
  <line stroke="blue" x1="62.5" y1="62.5" x2="125" y2="62.5"/>
  <line stroke="blue" x1="31.25" y1="93.75" x2="15.63" y2="109.38" fold-step="1"/>
  <line stroke="red" x1="31.25" y1="93.75" x2="31.25" y2="109.38"/>
  <line stroke="red" x1="31.25" y1="93.75" x2="46.88" y2="109.38" fold-step="1"/>
  <line stroke="red" x1="93.75" y1="31.25" x2="109.38" y2="15.62" fold-step="1"/>
  <line stroke="red" x1="93.75" y1="31.25" x2="109.38" y2="31.25"/>
  <line stroke="blue" x1="93.75" y1="31.25" x2="109.38" y2="46.88" fold-step="1"/>
  <line stroke="blue" x1="15.63" y1="109.38" x2="31.25" y2="109.38"/>
  <line stroke="blue" x1="15.63" y1="109.38" x2="0" y2="125" fold-step="1"/>
  <line stroke="red" x1="78.12" y1="78.12" x2="93.75" y2="93.75"/>
  <line stroke="red" x1="78.12" y1="78.12" x2="109.38" y2="78.12"/>
  <line stroke="blue" x1="78.12" y1="78.12" x2="78.12" y2="109.38"/>
  <line stroke="red" x1="109.38" y1="15.62" x2="125" y2="0" fold-step="1"/>
  <line stroke="red" x1="109.38" y1="15.62" x2="109.38" y2="31.25"/>
  <line stroke="red" x1="31.25" y1="109.38" x2="46.88" y2="109.38"/>
  <line stroke="red" x1="31.25" y1="109.38" x2="31.25" y2="174.56"/>
  <line stroke="blue" x1="109.38" y1="31.25" x2="109.38" y2="46.88"/>
  <line stroke="red" x1="109.38" y1="31.25" x2="125" y2="31.25"/>
  <line stroke="red" x1="46.88" y1="109.38" x2="62.5" y2="125" fold-step="1"/>
  <line stroke="blue" x1="109.38" y1="46.88" x2="125" y2="62.5" fold-step="1"/>
  <line stroke="black" x1="0" y1="125" x2="0" y2="250"/>
  <line stroke="blue" x1="125" y1="0" x2="125" y2="31.25"/>
  <line stroke="black" x1="125" y1="0" x2="250" y2="0"/>
  <line stroke="red" x1="125" y1="31.25" x2="125" y2="62.5"/>
  <line stroke="red" x1="125" y1="31.25" x2="218.75" y2="31.25"/>
  <line stroke="blue" x1="93.75" y1="93.75" x2="78.12" y2="109.37" fold-step="1"/>
  <line stroke="red" x1="93.75" y1="93.75" x2="109.38" y2="78.12" fold-step="1"/>
  <line stroke="red" x1="93.75" y1="93.75" x2="93.75" y2="109.38"/>
  <line stroke="red" x1="93.75" y1="93.75" x2="109.38" y2="93.75"/>
  <line stroke="blue" x1="93.75" y1="93.75" x2="109.38" y2="109.38"/>
  <line stroke="blue" x1="78.12" y1="109.37" x2="62.5" y2="125" fold-step="1"/>
  <line stroke="red" x1="109.38" y1="78.12" x2="125" y2="62.5" fold-step="1"/>
  <line stroke="red" x1="78.12" y1="109.38" x2="93.75" y2="109.38"/>
  <line stroke="blue" x1="109.38" y1="78.12" x2="109.38" y2="93.75"/>
  <line stroke="blue" x1="62.5" y1="125" x2="62.5" y2="250"/>
  <line stroke="blue" x1="125" y1="62.5" x2="125" y2="93.75"/>
  <line stroke="blue" x1="125" y1="62.5" x2="187.5" y2="62.5"/>
  <line stroke="blue" x1="93.75" y1="109.38" x2="109.38" y2="109.38"/>
  <line stroke="red" x1="93.75" y1="109.38" x2="93.75" y2="174.56"/>
  <line stroke="red" x1="109.38" y1="93.75" x2="109.38" y2="109.38"/>
  <line stroke="red" x1="109.38" y1="93.75" x2="125" y2="93.75"/>
  <line stroke="blue" x1="109.38" y1="109.38" x2="125" y2="125"/>
  <line stroke="red" x1="125" y1="93.75" x2="125" y2="125"/>
  <line stroke="red" x1="125" y1="93.75" x2="156.25" y2="93.75"/>
  <line stroke="blue" x1="125" y1="125" x2="156.25" y2="93.75"/>
  <line stroke="blue" x1="125" y1="125" x2="125" y2="250"/>
  <line stroke="red" x1="31.25" y1="174.56" x2="0" y2="250" fold-step="11"/>
  <line stroke="red" x1="31.25" y1="174.56" x2="31.25" y2="250"/>
  <line stroke="blue" x1="31.25" y1="174.56" x2="62.5" y2="250" fold-step="11"/>
  <line stroke="red" x1="156.25" y1="93.75" x2="187.5" y2="62.5"/>
  <line stroke="red" x1="156.25" y1="93.75" x2="156.25" y2="174.56"/>
  <line stroke="blue" x1="187.5" y1="62.5" x2="218.75" y2="31.25"/>
  <line stroke="blue" x1="187.5" y1="62.5" x2="187.5" y2="250"/>
  <line stroke="red" x1="93.75" y1="174.56" x2="62.5" y2="250" fold-step="11"/>
  <line stroke="red" x1="93.75" y1="174.56" x2="93.75" y2="250"/>
  <line stroke="blue" x1="93.75" y1="174.56" x2="125" y2="250" fold-step="11"/>
  <line stroke="red" x1="218.75" y1="31.25" x2="250" y2="0"/>
  <line stroke="red" x1="218.75" y1="31.25" x2="218.75" y2="174.56"/>
  <line stroke="red" x1="156.25" y1="174.56" x2="125" y2="250" fold-step="11"/>
  <line stroke="red" x1="156.25" y1="174.56" x2="156.25" y2="250"/>
  <line stroke="blue" x1="156.25" y1="174.56" x2="187.5" y2="250" fold-step="11"/>
  <line stroke="blue" x1="0" y1="250" x2="31.25" y2="250"/>
  <line stroke="red" x1="0" y1="250" x2="15.63" y2="277.06" fold-step="3"/>
  <line stroke="black" x1="0" y1="250" x2="0" y2="375"/>
  <line stroke="red" x1="250" y1="0" x2="281.25" y2="31.25"/>
  <line stroke="blue" x1="250" y1="0" x2="250" y2="250"/>
  <line stroke="black" x1="250" y1="0" x2="500" y2="0"/>
  <line stroke="red" x1="31.25" y1="250" x2="62.5" y2="250"/>
  <line stroke="red" x1="31.25" y1="250" x2="31.25" y2="286.08"/>
  <line stroke="blue" x1="62.5" y1="250" x2="93.75" y2="250"/>
  <line stroke="blue" x1="62.5" y1="250" x2="46.88" y2="277.06" fold-step="3"/>
  <line stroke="red" x1="62.5" y1="250" x2="78.12" y2="277.06" fold-step="3"/>
  <line stroke="blue" x1="62.5" y1="250" x2="62.5" y2="312.5"/>
  <line stroke="red" x1="93.75" y1="250" x2="125" y2="250"/>
  <line stroke="red" x1="93.75" y1="250" x2="93.75" y2="281.25"/>
  <line stroke="red" x1="15.63" y1="277.06" x2="31.25" y2="286.08"/>
  <line stroke="blue" x1="15.63" y1="277.06" x2="15.63" y2="359.37"/>
  <line stroke="red" x1="15.63" y1="277.06" x2="31.25" y2="304.13" fold-step="3"/>
  <line stroke="blue" x1="125" y1="250" x2="156.25" y2="250"/>
  <line stroke="blue" x1="125" y1="250" x2="93.75" y2="281.25"/>
  <line stroke="blue" x1="125" y1="250" x2="140.62" y2="265.62"/>
  <line stroke="red" x1="125" y1="250" x2="125" y2="281.25"/>
  <line stroke="red" x1="218.75" y1="174.56" x2="187.5" y2="250" fold-step="11"/>
  <line stroke="red" x1="218.75" y1="174.56" x2="218.75" y2="250"/>
  <line stroke="blue" x1="218.75" y1="174.56" x2="250" y2="250" fold-step="11"/>
  <line stroke="blue" x1="46.88" y1="277.06" x2="31.25" y2="304.13" fold-step="3"/>
  <line stroke="red" x1="46.88" y1="277.06" x2="46.88" y2="328.13"/>
  <line stroke="blue" x1="46.87" y1="277.06" x2="31.25" y2="286.08"/>
  <line stroke="blue" x1="281.25" y1="31.25" x2="312.5" y2="62.5"/>
  <line stroke="red" x1="281.25" y1="31.25" x2="281.25" y2="174.56"/>
  <line stroke="red" x1="281.25" y1="31.25" x2="424.56" y2="31.25"/>
  <line stroke="red" x1="31.25" y1="286.08" x2="31.25" y2="304.13"/>
  <line stroke="blue" x1="78.12" y1="277.06" x2="90.69" y2="284.31"/>
  <line stroke="red" x1="78.12" y1="277.06" x2="78.12" y2="296.87"/>
  <line stroke="red" x1="78.12" y1="277.06" x2="85.38" y2="289.62" fold-step="3"/>
  <line stroke="red" x1="156.25" y1="250" x2="156.25" y2="265.62"/>
  <line stroke="red" x1="156.25" y1="250" x2="187.5" y2="250"/>
  <line stroke="red" x1="93.75" y1="281.25" x2="90.69" y2="284.31"/>
  <line stroke="red" x1="93.75" y1="281.25" x2="125" y2="281.25"/>
  <line stroke="red" x1="90.69" y1="284.31" x2="85.38" y2="289.62"/>
  <line stroke="red" x1="90.69" y1="284.31" x2="97.94" y2="296.87"/>
  <line stroke="blue" x1="140.62" y1="265.62" x2="156.25" y2="265.62"/>
  <line stroke="red" x1="140.62" y1="265.62" x2="140.62" y2="281.25"/>
  <line stroke="blue" x1="140.62" y1="265.62" x2="156.25" y2="281.25"/>
  <line stroke="red" x1="85.38" y1="289.62" x2="78.12" y2="296.88"/>
  <line stroke="blue" x1="85.38" y1="289.62" x2="97.94" y2="296.87" fold-step="3"/>
  <line stroke="red" x1="31.25" y1="304.13" x2="31.25" y2="343.75"/>
  <line stroke="blue" x1="78.12" y1="296.87" x2="97.94" y2="296.87"/>
  <line stroke="red" x1="78.12" y1="296.88" x2="62.5" y2="312.5"/>
  <line stroke="red" x1="125" y1="281.25" x2="140.62" y2="281.25"/>
  <line stroke="blue" x1="125" y1="281.25" x2="125" y2="312.5"/>
  <line stroke="red" x1="156.25" y1="265.62" x2="171.88" y2="265.62"/>
  <line stroke="red" x1="156.25" y1="265.62" x2="156.25" y2="281.25"/>
  <line stroke="blue" x1="187.5" y1="250" x2="171.88" y2="265.63" fold-step="5"/>
  <line stroke="blue" x1="187.5" y1="250" x2="218.75" y2="250"/>
  <line stroke="red" x1="187.5" y1="250" x2="203.12" y2="265.62" fold-step="5"/>
  <line stroke="blue" x1="187.5" y1="250" x2="187.5" y2="312.5"/>
  <line stroke="blue" x1="97.94" y1="296.87" x2="125" y2="312.5" fold-step="3"/>
  <line stroke="red" x1="140.62" y1="281.25" x2="156.25" y2="281.25"/>
  <line stroke="blue" x1="140.62" y1="281.25" x2="140.62" y2="296.88"/>
  <line stroke="blue" x1="171.88" y1="265.62" x2="171.88" y2="296.88"/>
  <line stroke="blue" x1="171.88" y1="265.63" x2="156.25" y2="281.25" fold-step="5"/>
  <line stroke="blue" x1="62.5" y1="312.5" x2="46.88" y2="328.12"/>
  <line stroke="blue" x1="62.5" y1="312.5" x2="125" y2="312.5"/>
  <line stroke="red" x1="312.5" y1="62.5" x2="343.75" y2="93.75"/>
  <line stroke="blue" x1="312.5" y1="62.5" x2="312.5" y2="250"/>
  <line stroke="blue" x1="312.5" y1="62.5" x2="500" y2="62.5"/>
  <line stroke="red" x1="156.25" y1="281.25" x2="140.62" y2="296.88" fold-step="5"/>
  <line stroke="red" x1="156.25" y1="281.25" x2="171.88" y2="296.88"/>
  <line stroke="red" x1="140.62" y1="296.88" x2="125" y2="312.5" fold-step="5"/>
  <line stroke="red" x1="140.62" y1="296.88" x2="171.88" y2="296.88"/>
  <line stroke="red" x1="281.25" y1="174.56" x2="250" y2="250" fold-step="11"/>
  <line stroke="red" x1="281.25" y1="174.56" x2="281.25" y2="250"/>
  <line stroke="blue" x1="281.25" y1="174.56" x2="312.5" y2="250" fold-step="11"/>
  <line stroke="blue" x1="46.88" y1="328.12" x2="31.25" y2="343.75"/>
  <line stroke="blue" x1="46.88" y1="328.13" x2="97.94" y2="328.13"/>
  <line stroke="red" x1="218.75" y1="250" x2="218.75" y2="265.62"/>
  <line stroke="red" x1="218.75" y1="250" x2="250" y2="250"/>
  <line stroke="blue" x1="203.12" y1="265.62" x2="218.75" y2="265.62"/>
  <line stroke="red" x1="203.12" y1="265.62" x2="218.75" y2="281.25" fold-step="5"/>
  <line stroke="red" x1="203.12" y1="265.62" x2="203.12" y2="328.12"/>
  <line stroke="red" x1="125" y1="312.5" x2="97.94" y2="328.13" fold-step="3"/>
  <line stroke="blue" x1="125" y1="312.5" x2="140.62" y2="328.12" fold-step="5"/>
  <line stroke="blue" x1="125" y1="312.5" x2="187.5" y2="312.5"/>
  <line stroke="red" x1="125" y1="312.5" x2="125" y2="343.75"/>
  <line stroke="red" x1="97.94" y1="328.13" x2="88.92" y2="343.75"/>
  <line stroke="red" x1="97.94" y1="328.13" x2="70.87" y2="343.75" fold-step="3"/>
  <line stroke="red" x1="171.88" y1="296.88" x2="187.5" y2="312.5"/>
  <line stroke="red" x1="218.75" y1="265.62" x2="234.38" y2="265.62"/>
  <line stroke="red" x1="218.75" y1="265.62" x2="218.75" y2="281.25"/>
  <line stroke="red" x1="31.25" y1="343.75" x2="70.87" y2="343.75"/>
  <line stroke="red" x1="31.25" y1="343.75" x2="15.63" y2="359.37"/>
  <line stroke="red" x1="70.87" y1="343.75" x2="88.92" y2="343.75"/>
  <line stroke="blue" x1="70.87" y1="343.75" x2="97.94" y2="359.37" fold-step="3"/>
  <line stroke="blue" x1="250" y1="250" x2="234.38" y2="265.62" fold-step="5"/>
  <line stroke="red" x1="250" y1="250" x2="265.62" y2="265.62" fold-step="5"/>
  <line stroke="blue" x1="250" y1="250" x2="281.25" y2="250"/>
  <line stroke="blue" x1="250" y1="250" x2="250" y2="375"/>
  <line stroke="blue" x1="234.38" y1="265.62" x2="234.38" y2="359.38"/>
  <line stroke="blue" x1="234.38" y1="265.62" x2="218.75" y2="281.25" fold-step="5"/>
  <line stroke="blue" x1="88.92" y1="343.75" x2="97.94" y2="359.37"/>
  <line stroke="red" x1="88.92" y1="343.75" x2="125" y2="343.75"/>
  <line stroke="red" x1="218.75" y1="281.25" x2="218.75" y2="343.75"/>
  <line stroke="red" x1="343.75" y1="93.75" x2="343.75" y2="174.56"/>
  <line stroke="blue" x1="343.75" y1="93.75" x2="375" y2="125"/>
  <line stroke="red" x1="343.75" y1="93.75" x2="424.56" y2="93.75"/>
  <line stroke="blue" x1="140.62" y1="328.12" x2="156.25" y2="343.75" fold-step="5"/>
  <line stroke="red" x1="140.62" y1="328.12" x2="140.62" y2="343.75"/>
  <line stroke="blue" x1="140.62" y1="328.12" x2="203.12" y2="328.12"/>
  <line stroke="red" x1="15.63" y1="359.37" x2="0" y2="375"/>
  <line stroke="red" x1="15.63" y1="359.37" x2="97.94" y2="359.37"/>
  <line stroke="blue" x1="187.5" y1="312.5" x2="203.12" y2="328.12"/>
  <line stroke="red" x1="125" y1="343.75" x2="140.62" y2="343.75"/>
  <line stroke="blue" x1="125" y1="343.75" x2="125" y2="375"/>
  <line stroke="red" x1="140.62" y1="343.75" x2="156.25" y2="343.75"/>
  <line stroke="blue" x1="140.62" y1="343.75" x2="140.62" y2="359.38"/>
  <line stroke="blue" x1="97.94" y1="359.37" x2="125" y2="375" fold-step="3"/>
  <line stroke="red" x1="0" y1="375" x2="15.63" y2="390.62"/>
  <line stroke="blue" x1="0" y1="375" x2="125" y2="375"/>
  <line stroke="black" x1="0" y1="375" x2="0" y2="500"/>
  <line stroke="red" x1="265.62" y1="265.62" x2="281.25" y2="281.25" fold-step="5"/>
  <line stroke="red" x1="265.62" y1="265.62" x2="265.62" y2="359.38"/>
  <line stroke="blue" x1="265.63" y1="265.63" x2="281.25" y2="265.63"/>
  <line stroke="red" x1="281.25" y1="250" x2="281.25" y2="265.63"/>
  <line stroke="red" x1="281.25" y1="250" x2="312.5" y2="250"/>
  <line stroke="red" x1="156.25" y1="343.75" x2="140.62" y2="359.38" fold-step="5"/>
  <line stroke="red" x1="156.25" y1="343.75" x2="218.75" y2="343.75"/>
  <line stroke="red" x1="343.75" y1="174.56" x2="312.5" y2="250" fold-step="11"/>
  <line stroke="red" x1="343.75" y1="174.56" x2="343.75" y2="250"/>
  <line stroke="blue" x1="343.75" y1="174.56" x2="375" y2="250" fold-step="11"/>
  <line stroke="blue" x1="203.12" y1="328.12" x2="218.75" y2="343.75"/>
  <line stroke="red" x1="140.62" y1="359.38" x2="125" y2="375" fold-step="5"/>
  <line stroke="red" x1="140.62" y1="359.38" x2="234.38" y2="359.38"/>
  <line stroke="red" x1="281.25" y1="265.63" x2="281.25" y2="281.25"/>
  <line stroke="red" x1="281.25" y1="265.63" x2="296.88" y2="265.63"/>
  <line stroke="red" x1="15.62" y1="390.62" x2="97.94" y2="390.62"/>
  <line stroke="blue" x1="15.62" y1="390.62" x2="15.62" y2="472.94"/>
  <line stroke="red" x1="15.63" y1="390.62" x2="31.25" y2="406.25"/>
  <line stroke="red" x1="125" y1="375" x2="97.94" y2="390.62" fold-step="3"/>
  <line stroke="blue" x1="125" y1="375" x2="140.62" y2="390.62" fold-step="5"/>
  <line stroke="blue" x1="125" y1="375" x2="125" y2="406.25"/>
  <line stroke="red" x1="125" y1="375" x2="250" y2="375"/>
  <line stroke="red" x1="375" y1="125" x2="406.25" y2="156.25"/>
  <line stroke="blue" x1="375" y1="125" x2="375" y2="250"/>
  <line stroke="blue" x1="375" y1="125" x2="500" y2="125"/>
  <line stroke="blue" x1="281.25" y1="281.25" x2="296.88" y2="265.62" fold-step="5"/>
  <line stroke="red" x1="281.25" y1="281.25" x2="281.25" y2="343.75"/>
  <line stroke="blue" x1="296.88" y1="265.62" x2="312.5" y2="250" fold-step="5"/>
  <line stroke="blue" x1="296.88" y1="265.63" x2="296.88" y2="328.13"/>
  <line stroke="red" x1="312.5" y1="250" x2="328.12" y2="265.62" fold-step="5"/>
  <line stroke="blue" x1="312.5" y1="250" x2="343.75" y2="250"/>
  <line stroke="blue" x1="312.5" y1="250" x2="312.5" y2="312.5"/>
  <line stroke="blue" x1="97.94" y1="390.62" x2="88.92" y2="406.25"/>
  <line stroke="red" x1="97.94" y1="390.62" x2="70.87" y2="406.25" fold-step="3"/>
  <line stroke="red" x1="31.25" y1="406.25" x2="70.87" y2="406.25"/>
  <line stroke="blue" x1="31.25" y1="406.25" x2="46.88" y2="421.88"/>
  <line stroke="red" x1="31.25" y1="406.25" x2="31.25" y2="445.87"/>
  <line stroke="red" x1="218.75" y1="343.75" x2="234.38" y2="359.38"/>
  <line stroke="red" x1="70.87" y1="406.25" x2="88.92" y2="406.25"/>
  <line stroke="blue" x1="70.87" y1="406.25" x2="97.94" y2="421.87" fold-step="3"/>
  <line stroke="red" x1="140.62" y1="390.62" x2="140.62" y2="406.25"/>
  <line stroke="blue" x1="140.62" y1="390.62" x2="156.25" y2="406.25" fold-step="5"/>
  <line stroke="blue" x1="140.62" y1="390.62" x2="234.38" y2="390.62"/>
  <line stroke="red" x1="88.92" y1="406.25" x2="97.94" y2="421.87"/>
  <line stroke="red" x1="88.92" y1="406.25" x2="125" y2="406.25"/>
  <line stroke="blue" x1="328.12" y1="265.62" x2="343.75" y2="265.62"/>
  <line stroke="red" x1="328.12" y1="265.62" x2="328.12" y2="296.88"/>
  <line stroke="red" x1="328.12" y1="265.62" x2="343.75" y2="281.25" fold-step="5"/>
  <line stroke="blue" x1="46.88" y1="421.87" x2="97.94" y2="421.87"/>
  <line stroke="red" x1="46.88" y1="421.87" x2="46.88" y2="472.94"/>
  <line stroke="blue" x1="46.88" y1="421.88" x2="62.5" y2="437.5"/>
  <line stroke="red" x1="125" y1="406.25" x2="140.62" y2="406.25"/>
  <line stroke="red" x1="125" y1="406.25" x2="125" y2="437.5"/>
  <line stroke="red" x1="343.75" y1="250" x2="343.75" y2="265.62"/>
  <line stroke="red" x1="343.75" y1="250" x2="375" y2="250"/>
  <line stroke="red" x1="424.56" y1="31.25" x2="468.75" y2="31.25"/>
  <line stroke="blue" x1="424.56" y1="31.25" x2="500" y2="0" fold-step="11"/>
  <line stroke="red" x1="424.56" y1="31.25" x2="500" y2="62.5" fold-step="11"/>
  <line stroke="red" x1="234.38" y1="359.38" x2="250" y2="375"/>
  <line stroke="red" x1="140.62" y1="406.25" x2="156.25" y2="406.25"/>
  <line stroke="blue" x1="140.62" y1="406.25" x2="140.62" y2="421.88"/>
  <line stroke="blue" x1="97.94" y1="421.87" x2="125" y2="437.5" fold-step="3"/>
  <line stroke="red" x1="343.75" y1="265.62" x2="343.75" y2="281.25"/>
  <line stroke="red" x1="343.75" y1="265.62" x2="359.38" y2="265.62"/>
  <line stroke="red" x1="424.56" y1="93.75" x2="468.75" y2="93.75"/>
  <line stroke="blue" x1="424.56" y1="93.75" x2="500" y2="62.5" fold-step="11"/>
  <line stroke="red" x1="424.56" y1="93.75" x2="500" y2="125" fold-step="11"/>
  <line stroke="red" x1="156.25" y1="406.25" x2="140.62" y2="421.88" fold-step="5"/>
  <line stroke="red" x1="156.25" y1="406.25" x2="218.75" y2="406.25"/>
  <line stroke="red" x1="406.25" y1="156.25" x2="406.25" y2="174.56"/>
  <line stroke="red" x1="406.25" y1="156.25" x2="424.56" y2="156.25"/>
  <line stroke="blue" x1="406.25" y1="156.25" x2="437.5" y2="187.5"/>
  <line stroke="blue" x1="62.5" y1="437.5" x2="125" y2="437.5"/>
  <line stroke="red" x1="62.5" y1="437.5" x2="78.12" y2="453.12"/>
  <line stroke="blue" x1="62.5" y1="437.5" x2="62.5" y2="500"/>
  <line stroke="blue" x1="312.5" y1="312.5" x2="296.87" y2="328.13"/>
  <line stroke="red" x1="312.5" y1="312.5" x2="328.12" y2="296.88"/>
  <line stroke="blue" x1="312.5" y1="312.5" x2="375" y2="312.5"/>
  <line stroke="red" x1="406.25" y1="174.56" x2="375" y2="250" fold-step="11"/>
  <line stroke="red" x1="406.25" y1="174.56" x2="406.25" y2="281.25"/>
  <line stroke="blue" x1="406.25" y1="174.56" x2="437.5" y2="250" fold-step="11"/>
  <line stroke="blue" x1="296.87" y1="328.13" x2="281.25" y2="343.75"/>
  <line stroke="red" x1="328.12" y1="296.88" x2="343.75" y2="281.25"/>
  <line stroke="blue" x1="328.12" y1="296.88" x2="359.38" y2="296.88"/>
  <line stroke="red" x1="296.88" y1="328.13" x2="359.37" y2="328.13"/>
  <line stroke="red" x1="281.25" y1="343.75" x2="265.62" y2="359.38"/>
  <line stroke="red" x1="281.25" y1="343.75" x2="343.75" y2="343.75"/>
  <line stroke="blue" x1="343.75" y1="281.25" x2="359.38" y2="265.63"/>
  <line stroke="red" x1="343.75" y1="281.25" x2="359.38" y2="281.25"/>
  <line stroke="blue" x1="343.75" y1="281.25" x2="359.37" y2="296.88" fold-step="5"/>
  <line stroke="red" x1="140.62" y1="421.88" x2="125" y2="437.5" fold-step="5"/>
  <line stroke="red" x1="140.62" y1="421.88" x2="203.12" y2="421.88"/>
  <line stroke="red" x1="265.62" y1="359.38" x2="250" y2="375"/>
  <line stroke="blue" x1="265.62" y1="359.38" x2="359.37" y2="359.38"/>
  <line stroke="blue" x1="359.38" y1="265.62" x2="359.38" y2="281.25"/>
  <line stroke="blue" x1="359.38" y1="265.63" x2="375" y2="250"/>
  <line stroke="red" x1="31.25" y1="445.87" x2="31.25" y2="463.92"/>
  <line stroke="blue" x1="31.25" y1="445.87" x2="15.62" y2="472.94" fold-step="3"/>
  <line stroke="red" x1="31.25" y1="445.87" x2="46.88" y2="472.94" fold-step="3"/>
  <line stroke="red" x1="250" y1="375" x2="234.38" y2="390.62"/>
  <line stroke="red" x1="250" y1="375" x2="265.62" y2="390.62"/>
  <line stroke="blue" x1="250" y1="375" x2="375" y2="375"/>
  <line stroke="blue" x1="250" y1="375" x2="250" y2="500"/>
  <line stroke="blue" x1="375" y1="250" x2="406.25" y2="281.25"/>
  <line stroke="red" x1="424.56" y1="156.25" x2="468.75" y2="156.25"/>
  <line stroke="blue" x1="424.56" y1="156.25" x2="500" y2="125" fold-step="11"/>
  <line stroke="red" x1="424.56" y1="156.25" x2="500" y2="187.5" fold-step="11"/>
  <line stroke="red" x1="125" y1="437.5" x2="97.94" y2="453.12" fold-step="3"/>
  <line stroke="blue" x1="125" y1="437.5" x2="140.62" y2="453.12" fold-step="5"/>
  <line stroke="blue" x1="125" y1="437.5" x2="187.5" y2="437.5"/>
  <line stroke="blue" x1="125" y1="437.5" x2="125" y2="468.75"/>
  <line stroke="red" x1="234.38" y1="390.62" x2="218.75" y2="406.25"/>
  <line stroke="red" x1="234.38" y1="390.62" x2="234.38" y2="484.38"/>
  <line stroke="red" x1="359.38" y1="281.25" x2="359.38" y2="296.88"/>
  <line stroke="red" x1="359.38" y1="281.25" x2="406.25" y2="281.25"/>
  <line stroke="red" x1="78.12" y1="453.12" x2="97.94" y2="453.12"/>
  <line stroke="blue" x1="78.12" y1="453.12" x2="78.12" y2="472.94"/>
  <line stroke="red" x1="78.12" y1="453.12" x2="85.38" y2="460.38"/>
  <line stroke="blue" x1="218.75" y1="406.25" x2="203.12" y2="421.88"/>
  <line stroke="red" x1="218.75" y1="406.25" x2="218.75" y2="468.75"/>
  <line stroke="blue" x1="97.94" y1="453.12" x2="90.69" y2="465.69"/>
  <line stroke="red" x1="97.94" y1="453.12" x2="85.38" y2="460.38" fold-step="3"/>
  <line stroke="red" x1="31.25" y1="463.92" x2="31.25" y2="500"/>
  <line stroke="red" x1="31.25" y1="463.92" x2="15.62" y2="472.94"/>
  <line stroke="blue" x1="31.25" y1="463.92" x2="46.88" y2="472.94"/>
  <line stroke="blue" x1="359.37" y1="296.88" x2="375" y2="312.5" fold-step="5"/>
  <line stroke="red" x1="85.38" y1="460.38" x2="90.69" y2="465.69"/>
  <line stroke="blue" x1="85.38" y1="460.38" x2="78.12" y2="472.94" fold-step="3"/>
  <line stroke="blue" x1="203.12" y1="421.88" x2="203.12" y2="484.38"/>
  <line stroke="blue" x1="203.12" y1="421.88" x2="187.5" y2="437.5"/>
  <line stroke="blue" x1="468.75" y1="31.25" x2="500" y2="0"/>
  <line stroke="red" x1="468.75" y1="31.25" x2="500" y2="62.5"/>
  <line stroke="red" x1="468.75" y1="31.25" x2="531.25" y2="31.25"/>
  <line stroke="red" x1="265.62" y1="390.62" x2="281.25" y2="406.25"/>
  <line stroke="red" x1="265.62" y1="390.62" x2="359.38" y2="390.62"/>
  <line stroke="blue" x1="265.62" y1="390.62" x2="265.62" y2="484.38"/>
  <line stroke="blue" x1="15.62" y1="472.94" x2="0" y2="500" fold-step="3"/>
  <line stroke="red" x1="90.69" y1="465.69" x2="78.12" y2="472.94"/>
  <line stroke="red" x1="90.69" y1="465.69" x2="93.75" y2="468.75"/>
  <line stroke="blue" x1="140.62" y1="453.12" x2="156.25" y2="468.75" fold-step="5"/>
  <line stroke="blue" x1="140.63" y1="453.12" x2="171.88" y2="453.12"/>
  <line stroke="red" x1="140.63" y1="453.12" x2="140.63" y2="468.75"/>
  <line stroke="red" x1="46.88" y1="472.94" x2="62.5" y2="500" fold-step="3"/>
  <line stroke="red" x1="187.5" y1="437.5" x2="171.88" y2="453.12"/>
  <line stroke="blue" x1="187.5" y1="437.5" x2="187.5" y2="500"/>
  <line stroke="red" x1="437.5" y1="187.5" x2="455.81" y2="205.81"/>
  <line stroke="blue" x1="437.5" y1="187.5" x2="437.5" y2="250"/>
  <line stroke="blue" x1="437.5" y1="187.5" x2="500" y2="187.5"/>
  <line stroke="red" x1="93.75" y1="468.75" x2="125" y2="468.75"/>
  <line stroke="red" x1="93.75" y1="468.75" x2="93.75" y2="500"/>
  <line stroke="blue" x1="93.75" y1="468.75" x2="125" y2="500"/>
  <line stroke="blue" x1="468.75" y1="93.75" x2="500" y2="62.5"/>
  <line stroke="red" x1="468.75" y1="93.75" x2="500" y2="125"/>
  <line stroke="red" x1="468.75" y1="93.75" x2="531.25" y2="93.75"/>
  <line stroke="blue" x1="78.12" y1="472.94" x2="62.5" y2="500" fold-step="3"/>
  <line stroke="red" x1="171.88" y1="453.12" x2="156.25" y2="468.75"/>
  <line stroke="red" x1="171.88" y1="453.12" x2="171.88" y2="484.38"/>
  <line stroke="red" x1="125" y1="468.75" x2="140.63" y2="468.75"/>
  <line stroke="red" x1="125" y1="468.75" x2="125" y2="500"/>
  <line stroke="red" x1="343.75" y1="343.75" x2="359.37" y2="328.12" fold-step="5"/>
  <line stroke="red" x1="343.75" y1="343.75" x2="359.37" y2="343.75"/>
  <line stroke="blue" x1="343.75" y1="343.75" x2="359.37" y2="359.38" fold-step="5"/>
  <line stroke="red" x1="359.37" y1="328.12" x2="375" y2="312.5" fold-step="5"/>
  <line stroke="blue" x1="359.37" y1="328.13" x2="359.37" y2="343.75"/>
  <line stroke="blue" x1="375" y1="312.5" x2="406.25" y2="281.25"/>
  <line stroke="red" x1="375" y1="312.5" x2="406.25" y2="312.5"/>
  <line stroke="blue" x1="375" y1="312.5" x2="406.25" y2="343.75"/>
  <line stroke="red" x1="140.63" y1="468.75" x2="156.25" y2="468.75"/>
  <line stroke="blue" x1="140.63" y1="468.75" x2="140.63" y2="484.37"/>
  <line stroke="blue" x1="156.25" y1="468.75" x2="140.63" y2="484.38"/>
  <line stroke="red" x1="156.25" y1="468.75" x2="156.25" y2="484.37"/>
  <line stroke="red" x1="156.25" y1="468.75" x2="171.88" y2="484.38" fold-step="5"/>
  <line stroke="blue" x1="281.25" y1="406.25" x2="296.88" y2="421.88"/>
  <line stroke="red" x1="281.25" y1="406.25" x2="343.75" y2="406.25"/>
  <line stroke="red" x1="281.25" y1="406.25" x2="281.25" y2="468.75"/>
  <line stroke="red" x1="406.25" y1="281.25" x2="406.25" y2="312.5"/>
  <line stroke="red" x1="406.25" y1="281.25" x2="437.5" y2="281.25"/>
  <line stroke="blue" x1="468.75" y1="156.25" x2="500" y2="125"/>
  <line stroke="red" x1="468.75" y1="156.25" x2="500" y2="187.5"/>
  <line stroke="red" x1="468.75" y1="156.25" x2="531.25" y2="156.25"/>
  <line stroke="red" x1="359.37" y1="343.75" x2="359.37" y2="359.38"/>
  <line stroke="red" x1="359.37" y1="343.75" x2="406.25" y2="343.75"/>
  <line stroke="blue" x1="0" y1="500" x2="31.25" y2="500"/>
  <line stroke="red" x1="0" y1="500" x2="15.62" y2="506.47" fold-step="7"/>
  <line stroke="black" x1="0" y1="500" x2="0" y2="625"/>
  <line stroke="blue" x1="500" y1="0" x2="575.44" y2="31.25" fold-step="12"/>
  <line stroke="blue" x1="500" y1="0" x2="531.25" y2="31.25"/>
  <line stroke="black" x1="500" y1="0" x2="750" y2="0"/>
  <line stroke="red" x1="455.81" y1="205.81" x2="437.5" y2="250" fold-step="11"/>
  <line stroke="red" x1="455.81" y1="205.81" x2="468.75" y2="218.75"/>
  <line stroke="blue" x1="455.81" y1="205.81" x2="500" y2="187.5" fold-step="11"/>
  <line stroke="red" x1="31.25" y1="500" x2="62.5" y2="500"/>
  <line stroke="red" x1="31.25" y1="500" x2="15.62" y2="506.47"/>
  <line stroke="blue" x1="31.25" y1="500" x2="46.88" y2="506.47"/>
  <line stroke="red" x1="31.25" y1="500" x2="31.25" y2="522.1"/>
  <line stroke="blue" x1="62.5" y1="500" x2="46.88" y2="506.47" fold-step="7"/>
  <line stroke="blue" x1="62.5" y1="500" x2="93.75" y2="500"/>
  <line stroke="red" x1="62.5" y1="500" x2="78.12" y2="506.47" fold-step="7"/>
  <line stroke="blue" x1="62.5" y1="500" x2="62.5" y2="562.5"/>
  <line stroke="blue" x1="437.5" y1="250" x2="437.5" y2="281.25"/>
  <line stroke="red" x1="500" y1="62.5" x2="575.44" y2="31.25" fold-step="12"/>
  <line stroke="blue" x1="500" y1="62.5" x2="575.44" y2="93.75" fold-step="12"/>
  <line stroke="red" x1="500" y1="62.5" x2="531.25" y2="31.25"/>
  <line stroke="blue" x1="500" y1="62.5" x2="531.25" y2="93.75"/>
  <line stroke="blue" x1="500" y1="62.5" x2="687.5" y2="62.5"/>
  <line stroke="red" x1="140.63" y1="484.37" x2="156.25" y2="484.37"/>
  <line stroke="blue" x1="140.63" y1="484.38" x2="125" y2="500"/>
  <line stroke="red" x1="15.62" y1="506.47" x2="31.25" y2="522.1" fold-step="7"/>
  <line stroke="blue" x1="15.62" y1="506.47" x2="15.62" y2="609.38"/>
  <line stroke="blue" x1="359.37" y1="359.38" x2="375" y2="375" fold-step="5"/>
  <line stroke="blue" x1="46.88" y1="506.47" x2="31.25" y2="522.1" fold-step="7"/>
  <line stroke="red" x1="46.88" y1="506.47" x2="46.88" y2="578.12"/>
  <line stroke="red" x1="93.75" y1="500" x2="78.13" y2="506.47"/>
  <line stroke="red" x1="93.75" y1="500" x2="125" y2="500"/>
  <line stroke="blue" x1="93.75" y1="500" x2="109.38" y2="506.47"/>
  <line stroke="red" x1="93.75" y1="500" x2="93.75" y2="522.1"/>
  <line stroke="blue" x1="156.25" y1="484.37" x2="171.87" y2="484.37"/>
  <line stroke="red" x1="156.25" y1="484.37" x2="156.25" y2="531.25"/>
  <line stroke="blue" x1="78.12" y1="506.47" x2="78.12" y2="546.88"/>
  <line stroke="red" x1="78.13" y1="506.47" x2="93.75" y2="522.1" fold-step="7"/>
  <line stroke="red" x1="406.25" y1="312.5" x2="406.25" y2="343.75"/>
  <line stroke="blue" x1="406.25" y1="312.5" x2="437.5" y2="312.5"/>
  <line stroke="red" x1="171.88" y1="484.38" x2="187.5" y2="500" fold-step="5"/>
  <line stroke="blue" x1="125" y1="500" x2="109.38" y2="506.47" fold-step="7"/>
  <line stroke="blue" x1="125" y1="500" x2="109.37" y2="515.62"/>
  <line stroke="red" x1="125" y1="500" x2="118.53" y2="515.62" fold-step="7"/>
  <line stroke="blue" x1="125" y1="500" x2="156.25" y2="531.25"/>
  <line stroke="red" x1="500" y1="125" x2="575.44" y2="93.75" fold-step="12"/>
  <line stroke="blue" x1="500" y1="125" x2="575.44" y2="156.25" fold-step="12"/>
  <line stroke="red" x1="500" y1="125" x2="531.25" y2="93.75"/>
  <line stroke="blue" x1="500" y1="125" x2="531.25" y2="156.25"/>
  <line stroke="blue" x1="500" y1="125" x2="625" y2="125"/>
  <line stroke="blue" x1="296.88" y1="421.88" x2="359.38" y2="421.88"/>
  <line stroke="red" x1="296.88" y1="421.88" x2="296.88" y2="484.38"/>
  <line stroke="blue" x1="296.88" y1="421.88" x2="312.5" y2="437.5"/>
  <line stroke="blue" x1="218.75" y1="468.75" x2="203.12" y2="484.38" fold-step="5"/>
  <line stroke="red" x1="218.75" y1="468.75" x2="218.75" y2="484.38"/>
  <line stroke="red" x1="218.75" y1="468.75" x2="234.38" y2="484.38" fold-step="5"/>
  <line stroke="blue" x1="468.75" y1="218.75" x2="500" y2="187.5"/>
  <line stroke="red" x1="468.75" y1="218.75" x2="468.75" y2="281.25"/>
  <line stroke="red" x1="468.75" y1="218.75" x2="531.25" y2="218.75"/>
  <line stroke="red" x1="109.38" y1="506.47" x2="109.38" y2="515.62"/>
  <line stroke="blue" x1="109.38" y1="506.47" x2="93.75" y2="522.1" fold-step="7"/>
  <line stroke="blue" x1="437.5" y1="281.25" x2="437.5" y2="312.5"/>
  <line stroke="blue" x1="437.5" y1="281.25" x2="468.75" y2="281.25"/>
  <line stroke="red" x1="31.25" y1="522.1" x2="31.25" y2="593.75"/>
  <line stroke="red" x1="203.12" y1="484.38" x2="218.75" y2="484.38"/>
  <line stroke="blue" x1="203.12" y1="484.38" x2="187.5" y2="500" fold-step="5"/>
  <line stroke="blue" x1="109.37" y1="515.62" x2="93.75" y2="531.25"/>
  <line stroke="blue" x1="109.38" y1="515.62" x2="118.53" y2="515.62"/>
  <line stroke="red" x1="118.53" y1="515.62" x2="134.15" y2="531.25"/>
  <line stroke="red" x1="118.53" y1="515.62" x2="112.06" y2="531.25" fold-step="7"/>
  <line stroke="red" x1="375" y1="375" x2="359.38" y2="390.62" fold-step="5"/>
  <line stroke="blue" x1="375" y1="375" x2="406.25" y2="343.75"/>
  <line stroke="red" x1="375" y1="375" x2="406.25" y2="375"/>
  <line stroke="blue" x1="375" y1="375" x2="406.25" y2="406.25"/>
  <line stroke="red" x1="93.75" y1="522.1" x2="93.75" y2="531.25"/>
  <line stroke="red" x1="359.38" y1="390.62" x2="343.75" y2="406.25" fold-step="5"/>
  <line stroke="blue" x1="359.38" y1="390.62" x2="359.38" y2="406.25"/>
  <line stroke="blue" x1="218.75" y1="484.38" x2="234.38" y2="484.38"/>
  <line stroke="red" x1="218.75" y1="484.38" x2="218.75" y2="562.5"/>
  <line stroke="red" x1="343.75" y1="406.25" x2="359.38" y2="406.25"/>
  <line stroke="blue" x1="343.75" y1="406.25" x2="359.38" y2="421.88" fold-step="5"/>
  <line stroke="red" x1="406.25" y1="343.75" x2="406.25" y2="375"/>
  <line stroke="red" x1="406.25" y1="343.75" x2="437.5" y2="343.75"/>
  <line stroke="red" x1="531.25" y1="31.25" x2="575.44" y2="31.25"/>
  <line stroke="blue" x1="187.5" y1="500" x2="187.5" y2="562.5"/>
  <line stroke="blue" x1="500" y1="187.5" x2="544.19" y2="205.81" fold-step="12"/>
  <line stroke="red" x1="500" y1="187.5" x2="575.44" y2="156.25" fold-step="12"/>
  <line stroke="red" x1="500" y1="187.5" x2="531.25" y2="156.25"/>
  <line stroke="blue" x1="500" y1="187.5" x2="531.25" y2="218.75"/>
  <line stroke="blue" x1="500" y1="187.5" x2="562.5" y2="187.5"/>
  <line stroke="red" x1="312.5" y1="437.5" x2="328.12" y2="453.12"/>
  <line stroke="blue" x1="312.5" y1="437.5" x2="375" y2="437.5"/>
  <line stroke="blue" x1="312.5" y1="437.5" x2="312.5" y2="500"/>
  <line stroke="blue" x1="437.5" y1="312.5" x2="437.5" y2="343.75"/>
  <line stroke="red" x1="437.5" y1="312.5" x2="468.75" y2="312.5"/>
  <line stroke="red" x1="234.38" y1="484.38" x2="250" y2="500" fold-step="5"/>
  <line stroke="red" x1="93.75" y1="531.25" x2="112.06" y2="531.25"/>
  <line stroke="red" x1="93.75" y1="531.25" x2="78.12" y2="546.88"/>
  <line stroke="red" x1="531.25" y1="93.75" x2="575.44" y2="93.75"/>
  <line stroke="red" x1="359.38" y1="406.25" x2="359.38" y2="421.88"/>
  <line stroke="red" x1="359.38" y1="406.25" x2="406.25" y2="406.25"/>
  <line stroke="red" x1="112.06" y1="531.25" x2="134.15" y2="531.25"/>
  <line stroke="blue" x1="112.06" y1="531.25" x2="118.53" y2="546.88" fold-step="7"/>
  <line stroke="blue" x1="281.25" y1="468.75" x2="265.62" y2="484.37" fold-step="5"/>
  <line stroke="red" x1="281.25" y1="468.75" x2="281.25" y2="484.38"/>
  <line stroke="red" x1="281.25" y1="468.75" x2="296.88" y2="484.38" fold-step="5"/>
  <line stroke="red" x1="468.75" y1="281.25" x2="468.75" y2="312.5"/>
  <line stroke="red" x1="468.75" y1="281.25" x2="531.25" y2="281.25"/>
  <line stroke="blue" x1="134.15" y1="531.25" x2="118.53" y2="546.88"/>
  <line stroke="red" x1="134.15" y1="531.25" x2="156.25" y2="531.25"/>
  <line stroke="blue" x1="265.62" y1="484.37" x2="250" y2="500" fold-step="5"/>
  <line stroke="red" x1="78.12" y1="546.88" x2="118.53" y2="546.88"/>
  <line stroke="red" x1="265.62" y1="484.38" x2="281.25" y2="484.38"/>
  <line stroke="red" x1="78.12" y1="546.88" x2="62.5" y2="562.5"/>
  <line stroke="red" x1="406.25" y1="375" x2="406.25" y2="406.25"/>
  <line stroke="blue" x1="406.25" y1="375" x2="437.5" y2="375"/>
  <line stroke="red" x1="156.25" y1="531.25" x2="187.5" y2="562.5"/>
  <line stroke="red" x1="531.25" y1="156.25" x2="575.44" y2="156.25"/>
  <line stroke="blue" x1="359.38" y1="421.88" x2="375" y2="437.5" fold-step="5"/>
  <line stroke="blue" x1="437.5" y1="343.75" x2="437.5" y2="375"/>
  <line stroke="blue" x1="437.5" y1="343.75" x2="468.75" y2="343.75"/>
  <line stroke="blue" x1="250" y1="500" x2="250" y2="562.5"/>
  <line stroke="red" x1="328.12" y1="453.12" x2="359.38" y2="453.12"/>
  <line stroke="red" x1="328.12" y1="453.12" x2="343.75" y2="468.75"/>
  <line stroke="blue" x1="328.12" y1="453.12" x2="328.12" y2="484.38"/>
  <line stroke="blue" x1="118.53" y1="546.88" x2="125" y2="562.5" fold-step="7"/>
  <line stroke="blue" x1="281.25" y1="484.38" x2="296.88" y2="484.38"/>
  <line stroke="red" x1="281.25" y1="484.38" x2="281.25" y2="562.5"/>
  <line stroke="red" x1="468.75" y1="312.5" x2="468.75" y2="343.75"/>
  <line stroke="blue" x1="468.75" y1="312.5" x2="531.25" y2="312.5"/>
  <line stroke="blue" x1="62.5" y1="562.5" x2="125" y2="562.5"/>
  <line stroke="blue" x1="62.5" y1="562.5" x2="46.88" y2="578.12"/>
  <line stroke="red" x1="296.88" y1="484.38" x2="312.5" y2="500" fold-step="5"/>
  <line stroke="blue" x1="406.25" y1="406.25" x2="375" y2="437.5"/>
  <line stroke="red" x1="406.25" y1="406.25" x2="406.25" y2="437.5"/>
  <line stroke="red" x1="406.25" y1="406.25" x2="437.5" y2="406.25"/>
  <line stroke="red" x1="531.25" y1="218.75" x2="544.19" y2="205.81"/>
  <line stroke="red" x1="531.25" y1="218.75" x2="531.25" y2="281.25"/>
  <line stroke="red" x1="125" y1="562.5" x2="118.53" y2="578.12" fold-step="7"/>
  <line stroke="blue" x1="125" y1="562.5" x2="187.5" y2="562.5"/>
  <line stroke="red" x1="375" y1="437.5" x2="359.38" y2="453.12" fold-step="5"/>
  <line stroke="red" x1="375" y1="437.5" x2="406.25" y2="437.5"/>
  <line stroke="blue" x1="375" y1="437.5" x2="406.25" y2="468.75"/>
  <line stroke="blue" x1="437.5" y1="375" x2="437.5" y2="406.25"/>
  <line stroke="red" x1="437.5" y1="375" x2="468.75" y2="375"/>
  <line stroke="red" x1="575.44" y1="31.25" x2="718.75" y2="31.25"/>
  <line stroke="red" x1="359.38" y1="453.12" x2="343.75" y2="468.75" fold-step="5"/>
  <line stroke="blue" x1="359.38" y1="453.12" x2="359.38" y2="468.75"/>
  <line stroke="blue" x1="46.88" y1="578.12" x2="118.53" y2="578.12"/>
  <line stroke="blue" x1="46.88" y1="578.12" x2="31.25" y2="593.75"/>
  <line stroke="blue" x1="343.75" y1="468.75" x2="328.12" y2="484.37" fold-step="5"/>
  <line stroke="red" x1="343.75" y1="468.75" x2="359.38" y2="468.75"/>
  <line stroke="red" x1="343.75" y1="468.75" x2="343.75" y2="484.38"/>
  <line stroke="blue" x1="343.75" y1="468.75" x2="359.38" y2="484.38"/>
  <line stroke="red" x1="468.75" y1="343.75" x2="468.75" y2="375"/>
  <line stroke="red" x1="468.75" y1="343.75" x2="531.25" y2="343.75"/>
  <line stroke="red" x1="544.19" y1="205.81" x2="562.5" y2="187.5"/>
  <line stroke="red" x1="544.19" y1="205.81" x2="562.5" y2="250" fold-step="12"/>
  <line stroke="red" x1="575.44" y1="93.75" x2="656.25" y2="93.75"/>
  <line stroke="blue" x1="328.12" y1="484.37" x2="312.5" y2="500" fold-step="5"/>
  <line stroke="red" x1="328.12" y1="484.38" x2="343.75" y2="484.38"/>
  <line stroke="blue" x1="312.5" y1="500" x2="312.5" y2="562.5"/>
  <line stroke="red" x1="118.53" y1="578.12" x2="112.06" y2="593.75" fold-step="7"/>
  <line stroke="red" x1="118.53" y1="578.12" x2="134.15" y2="593.75"/>
  <line stroke="red" x1="359.38" y1="468.75" x2="359.38" y2="484.38"/>
  <line stroke="red" x1="359.38" y1="468.75" x2="406.25" y2="468.75"/>
  <line stroke="blue" x1="187.5" y1="562.5" x2="218.75" y2="562.5"/>
  <line stroke="red" x1="187.5" y1="562.5" x2="156.25" y2="593.75"/>
  <line stroke="blue" x1="187.5" y1="562.5" x2="187.5" y2="625"/>
  <line stroke="blue" x1="562.5" y1="187.5" x2="593.75" y2="156.25"/>
  <line stroke="blue" x1="562.5" y1="187.5" x2="562.5" y2="250"/>
  <line stroke="blue" x1="343.75" y1="484.38" x2="359.38" y2="484.38"/>
  <line stroke="red" x1="343.75" y1="484.38" x2="343.75" y2="562.5"/>
  <line stroke="red" x1="31.25" y1="593.75" x2="112.06" y2="593.75"/>
  <line stroke="red" x1="31.25" y1="593.75" x2="15.63" y2="609.38"/>
  <line stroke="red" x1="575.44" y1="156.25" x2="593.75" y2="156.25"/>
  <line stroke="blue" x1="406.25" y1="437.5" x2="437.5" y2="437.5"/>
  <line stroke="red" x1="406.25" y1="437.5" x2="406.25" y2="468.75"/>
  <line stroke="blue" x1="437.5" y1="406.25" x2="437.5" y2="437.5"/>
  <line stroke="blue" x1="437.5" y1="406.25" x2="468.75" y2="406.25"/>
  <line stroke="red" x1="468.75" y1="375" x2="468.75" y2="406.25"/>
  <line stroke="blue" x1="468.75" y1="375" x2="531.25" y2="375"/>
  <line stroke="red" x1="531.25" y1="281.25" x2="531.25" y2="312.5"/>
  <line stroke="blue" x1="531.25" y1="281.25" x2="562.5" y2="281.25"/>
  <line stroke="blue" x1="359.38" y1="484.38" x2="375" y2="500"/>
  <line stroke="red" x1="218.75" y1="562.5" x2="250" y2="562.5"/>
  <line stroke="red" x1="218.75" y1="562.5" x2="218.75" y2="625"/>
  <line stroke="red" x1="112.06" y1="593.75" x2="134.15" y2="593.75"/>
  <line stroke="blue" x1="112.06" y1="593.75" x2="118.53" y2="609.37" fold-step="7"/>
  <line stroke="blue" x1="134.15" y1="593.75" x2="118.53" y2="609.38"/>
  <line stroke="red" x1="134.15" y1="593.75" x2="156.25" y2="593.75"/>
  <line stroke="red" x1="15.62" y1="609.38" x2="118.53" y2="609.38"/>
  <line stroke="red" x1="15.63" y1="609.38" x2="0" y2="625"/>
  <line stroke="blue" x1="156.25" y1="593.75" x2="125" y2="625"/>
  <line stroke="red" x1="156.25" y1="593.75" x2="156.25" y2="625"/>
  <line stroke="red" x1="593.75" y1="156.25" x2="593.75" y2="174.56"/>
  <line stroke="red" x1="593.75" y1="156.25" x2="625" y2="125"/>
  <line stroke="blue" x1="250" y1="562.5" x2="281.25" y2="562.5"/>
  <line stroke="blue" x1="250" y1="562.5" x2="250" y2="625"/>
  <line stroke="blue" x1="562.5" y1="250" x2="593.75" y2="174.56" fold-step="12"/>
  <line stroke="blue" x1="562.5" y1="250" x2="562.5" y2="281.25"/>
  <line stroke="red" x1="531.25" y1="312.5" x2="531.25" y2="343.75"/>
  <line stroke="red" x1="531.25" y1="312.5" x2="562.5" y2="312.5"/>
  <line stroke="blue" x1="437.5" y1="437.5" x2="437.5" y2="468.75"/>
  <line stroke="red" x1="437.5" y1="437.5" x2="468.75" y2="437.5"/>
  <line stroke="red" x1="593.75" y1="174.56" x2="593.75" y2="281.25"/>
  <line stroke="red" x1="593.75" y1="174.56" x2="625" y2="250" fold-step="12"/>
  <line stroke="blue" x1="406.25" y1="468.75" x2="375" y2="500"/>
  <line stroke="red" x1="406.25" y1="468.75" x2="437.5" y2="468.75"/>
  <line stroke="red" x1="406.25" y1="468.75" x2="406.25" y2="500"/>
  <line stroke="red" x1="468.75" y1="406.25" x2="468.75" y2="437.5"/>
  <line stroke="red" x1="468.75" y1="406.25" x2="531.25" y2="406.25"/>
  <line stroke="blue" x1="118.53" y1="609.37" x2="125" y2="625" fold-step="7"/>
  <line stroke="blue" x1="0" y1="625" x2="125" y2="625"/>
  <line stroke="red" x1="0" y1="625" x2="15.63" y2="640.63"/>
  <line stroke="black" x1="0" y1="625" x2="0" y2="750"/>
  <line stroke="red" x1="375" y1="500" x2="406.25" y2="500"/>
  <line stroke="blue" x1="375" y1="500" x2="375" y2="562.5"/>
  <line stroke="red" x1="281.25" y1="562.5" x2="312.5" y2="562.5"/>
  <line stroke="red" x1="281.25" y1="562.5" x2="281.25" y2="625"/>
  <line stroke="blue" x1="562.5" y1="281.25" x2="562.5" y2="312.5"/>
  <line stroke="red" x1="562.5" y1="281.25" x2="593.75" y2="281.25"/>
  <line stroke="red" x1="531.25" y1="343.75" x2="531.25" y2="375"/>
  <line stroke="blue" x1="531.25" y1="343.75" x2="562.5" y2="343.75"/>
  <line stroke="red" x1="125" y1="625" x2="156.25" y2="625"/>
  <line stroke="red" x1="125" y1="625" x2="118.53" y2="640.63" fold-step="7"/>
  <line stroke="blue" x1="125" y1="625" x2="156.25" y2="656.25"/>
  <line stroke="blue" x1="625" y1="125" x2="656.25" y2="93.75"/>
  <line stroke="blue" x1="625" y1="125" x2="625" y2="250"/>
  <line stroke="blue" x1="15.63" y1="640.63" x2="118.53" y2="640.63"/>
  <line stroke="red" x1="15.63" y1="640.63" x2="15.63" y2="743.53"/>
  <line stroke="red" x1="15.63" y1="640.63" x2="31.25" y2="656.25"/>
  <line stroke="blue" x1="437.5" y1="468.75" x2="468.75" y2="468.75"/>
  <line stroke="blue" x1="437.5" y1="468.75" x2="437.5" y2="500"/>
  <line stroke="red" x1="468.75" y1="437.5" x2="468.75" y2="468.75"/>
  <line stroke="blue" x1="468.75" y1="437.5" x2="531.25" y2="437.5"/>
  <line stroke="blue" x1="312.5" y1="562.5" x2="343.75" y2="562.5"/>
  <line stroke="blue" x1="312.5" y1="562.5" x2="312.5" y2="625"/>
  <line stroke="blue" x1="562.5" y1="312.5" x2="562.5" y2="343.75"/>
  <line stroke="blue" x1="562.5" y1="312.5" x2="593.75" y2="312.5"/>
  <line stroke="blue" x1="156.25" y1="625" x2="187.5" y2="625"/>
  <line stroke="red" x1="156.25" y1="625" x2="156.25" y2="656.25"/>
  <line stroke="blue" x1="406.25" y1="500" x2="437.5" y2="500"/>
  <line stroke="red" x1="406.25" y1="500" x2="406.25" y2="562.5"/>
  <line stroke="red" x1="531.25" y1="375" x2="531.25" y2="406.25"/>
  <line stroke="red" x1="531.25" y1="375" x2="562.5" y2="375"/>
  <line stroke="red" x1="118.53" y1="640.62" x2="134.15" y2="656.25"/>
  <line stroke="red" x1="118.53" y1="640.63" x2="112.06" y2="656.25" fold-step="7"/>
  <line stroke="red" x1="187.5" y1="625" x2="218.75" y2="625"/>
  <line stroke="blue" x1="187.5" y1="625" x2="187.5" y2="687.5"/>
  <line stroke="red" x1="31.25" y1="656.25" x2="112.06" y2="656.25"/>
  <line stroke="blue" x1="31.25" y1="656.25" x2="46.87" y2="671.87"/>
  <line stroke="red" x1="31.25" y1="656.25" x2="31.25" y2="727.9"/>
  <line stroke="red" x1="593.75" y1="281.25" x2="593.75" y2="312.5"/>
  <line stroke="blue" x1="593.75" y1="281.25" x2="625" y2="250"/>
  <line stroke="blue" x1="593.75" y1="281.25" x2="625" y2="312.5"/>
  <line stroke="red" x1="593.75" y1="281.25" x2="640.62" y2="281.25"/>
  <line stroke="red" x1="343.75" y1="562.5" x2="375" y2="562.5"/>
  <line stroke="red" x1="343.75" y1="562.5" x2="343.75" y2="625"/>
  <line stroke="blue" x1="562.5" y1="343.75" x2="562.5" y2="375"/>
  <line stroke="red" x1="562.5" y1="343.75" x2="593.75" y2="343.75"/>
  <line stroke="blue" x1="218.75" y1="625" x2="250" y2="625"/>
  <line stroke="red" x1="218.75" y1="625" x2="218.75" y2="718.75"/>
  <line stroke="red" x1="468.75" y1="468.75" x2="468.75" y2="500"/>
  <line stroke="red" x1="468.75" y1="468.75" x2="531.25" y2="468.75"/>
  <line stroke="red" x1="656.25" y1="93.75" x2="656.25" y2="174.56"/>
  <line stroke="red" x1="656.25" y1="93.75" x2="687.5" y2="62.5"/>
  <line stroke="red" x1="437.5" y1="500" x2="468.75" y2="500"/>
  <line stroke="blue" x1="437.5" y1="500" x2="437.5" y2="562.5"/>
  <line stroke="red" x1="112.06" y1="656.25" x2="134.15" y2="656.25"/>
  <line stroke="blue" x1="112.06" y1="656.25" x2="118.53" y2="671.87" fold-step="7"/>
  <line stroke="red" x1="531.25" y1="406.25" x2="531.25" y2="437.5"/>
  <line stroke="blue" x1="531.25" y1="406.25" x2="562.5" y2="406.25"/>
  <line stroke="blue" x1="134.15" y1="656.25" x2="118.53" y2="671.87"/>
  <line stroke="red" x1="134.15" y1="656.25" x2="156.25" y2="656.25"/>
  <line stroke="red" x1="593.75" y1="312.5" x2="593.75" y2="343.75"/>
  <line stroke="red" x1="593.75" y1="312.5" x2="625" y2="312.5"/>
  <line stroke="red" x1="250" y1="625" x2="281.25" y2="625"/>
  <line stroke="blue" x1="250" y1="625" x2="250" y2="750"/>
  <line stroke="blue" x1="625" y1="250" x2="656.25" y2="174.56" fold-step="12"/>
  <line stroke="blue" x1="625" y1="250" x2="640.62" y2="265.62"/>
  <line stroke="red" x1="625" y1="250" x2="656.25" y2="250"/>
  <line stroke="blue" x1="46.87" y1="671.87" x2="62.5" y2="687.5"/>
  <line stroke="red" x1="46.87" y1="671.87" x2="118.53" y2="671.87"/>
  <line stroke="blue" x1="46.87" y1="671.87" x2="46.87" y2="743.53"/>
  <line stroke="red" x1="156.25" y1="656.25" x2="187.5" y2="687.5"/>
  <line stroke="blue" x1="375" y1="562.5" x2="406.25" y2="562.5"/>
  <line stroke="blue" x1="375" y1="562.5" x2="375" y2="625"/>
  <line stroke="blue" x1="562.5" y1="375" x2="562.5" y2="406.25"/>
  <line stroke="blue" x1="562.5" y1="375" x2="593.75" y2="375"/>
  <line stroke="red" x1="656.25" y1="174.56" x2="656.25" y2="250"/>
  <line stroke="red" x1="656.25" y1="174.56" x2="687.5" y2="250" fold-step="12"/>
  <line stroke="blue" x1="118.53" y1="671.87" x2="125" y2="687.5" fold-step="7"/>
  <line stroke="blue" x1="281.25" y1="625" x2="312.5" y2="625"/>
  <line stroke="red" x1="281.25" y1="625" x2="281.25" y2="781.25"/>
  <line stroke="blue" x1="468.75" y1="500" x2="531.25" y2="500"/>
  <line stroke="red" x1="468.75" y1="500" x2="468.75" y2="562.5"/>
  <line stroke="blue" x1="593.75" y1="343.75" x2="625" y2="312.5"/>
  <line stroke="red" x1="593.75" y1="343.75" x2="593.75" y2="375"/>
  <line stroke="red" x1="593.75" y1="343.75" x2="640.62" y2="343.75"/>
  <line stroke="blue" x1="593.75" y1="343.75" x2="625" y2="375"/>
  <line stroke="red" x1="531.25" y1="437.5" x2="531.25" y2="468.75"/>
  <line stroke="red" x1="531.25" y1="437.5" x2="562.5" y2="437.5"/>
  <line stroke="blue" x1="62.5" y1="687.5" x2="125" y2="687.5"/>
  <line stroke="red" x1="62.5" y1="687.5" x2="78.13" y2="703.12"/>
  <line stroke="blue" x1="62.5" y1="687.5" x2="62.5" y2="750"/>
  <line stroke="blue" x1="687.5" y1="62.5" x2="718.75" y2="31.25"/>
  <line stroke="blue" x1="687.5" y1="62.5" x2="687.5" y2="250"/>
  <line stroke="red" x1="640.62" y1="265.62" x2="640.62" y2="281.25"/>
  <line stroke="blue" x1="640.62" y1="265.62" x2="656.25" y2="265.62"/>
  <line stroke="blue" x1="640.62" y1="265.62" x2="656.25" y2="281.25"/>
  <line stroke="red" x1="406.25" y1="562.5" x2="437.5" y2="562.5"/>
  <line stroke="red" x1="406.25" y1="562.5" x2="406.25" y2="625"/>
  <line stroke="blue" x1="562.5" y1="406.25" x2="562.5" y2="437.5"/>
  <line stroke="red" x1="562.5" y1="406.25" x2="593.75" y2="406.25"/>
  <line stroke="blue" x1="125" y1="687.5" x2="187.5" y2="687.5"/>
  <line stroke="red" x1="125" y1="687.5" x2="118.53" y2="703.12" fold-step="7"/>
  <line stroke="red" x1="312.5" y1="625" x2="343.75" y2="625"/>
  <line stroke="blue" x1="312.5" y1="625" x2="312.5" y2="812.5"/>
  <line stroke="red" x1="625" y1="312.5" x2="640.62" y2="296.88" fold-step="6"/>
  <line stroke="blue" x1="625" y1="312.5" x2="640.62" y2="328.12" fold-step="6"/>
  <line stroke="blue" x1="625" y1="312.5" x2="687.5" y2="312.5"/>
  <line stroke="blue" x1="640.62" y1="281.25" x2="640.62" y2="296.88"/>
  <line stroke="red" x1="640.62" y1="281.25" x2="656.25" y2="281.25"/>
  <line stroke="red" x1="593.75" y1="375" x2="593.75" y2="406.25"/>
  <line stroke="red" x1="593.75" y1="375" x2="625" y2="375"/>
  <line stroke="red" x1="656.25" y1="250" x2="656.25" y2="265.62"/>
  <line stroke="blue" x1="656.25" y1="250" x2="687.5" y2="250"/>
  <line stroke="red" x1="640.62" y1="296.88" x2="656.25" y2="281.25" fold-step="6"/>
  <line stroke="red" x1="640.62" y1="296.88" x2="671.88" y2="296.88"/>
  <line stroke="blue" x1="78.12" y1="703.12" x2="118.53" y2="703.12"/>
  <line stroke="red" x1="78.12" y1="703.12" x2="78.12" y2="743.53"/>
  <line stroke="red" x1="78.13" y1="703.12" x2="93.75" y2="718.75"/>
  <line stroke="red" x1="656.25" y1="265.62" x2="656.25" y2="281.25"/>
  <line stroke="red" x1="656.25" y1="265.62" x2="671.88" y2="265.62"/>
  <line stroke="red" x1="531.25" y1="468.75" x2="531.25" y2="500"/>
  <line stroke="blue" x1="531.25" y1="468.75" x2="562.5" y2="468.75"/>
  <line stroke="blue" x1="187.5" y1="687.5" x2="218.75" y2="718.75"/>
  <line stroke="blue" x1="437.5" y1="562.5" x2="468.75" y2="562.5"/>
  <line stroke="blue" x1="437.5" y1="562.5" x2="437.5" y2="625"/>
  <line stroke="blue" x1="562.5" y1="437.5" x2="562.5" y2="468.75"/>
  <line stroke="blue" x1="562.5" y1="437.5" x2="593.75" y2="437.5"/>
  <line stroke="red" x1="118.53" y1="703.12" x2="112.06" y2="718.75" fold-step="7"/>
  <line stroke="red" x1="118.53" y1="703.13" x2="134.15" y2="718.75"/>
  <line stroke="blue" x1="343.75" y1="625" x2="375" y2="625"/>
  <line stroke="red" x1="343.75" y1="625" x2="343.75" y2="843.75"/>
  <line stroke="blue" x1="656.25" y1="281.25" x2="671.88" y2="265.63" fold-step="6"/>
  <line stroke="red" x1="656.25" y1="281.25" x2="671.88" y2="296.88"/>
  <line stroke="blue" x1="593.75" y1="406.25" x2="625" y2="375"/>
  <line stroke="red" x1="593.75" y1="406.25" x2="593.75" y2="437.5"/>
  <line stroke="red" x1="593.75" y1="406.25" x2="640.62" y2="406.25"/>
  <line stroke="blue" x1="593.75" y1="406.25" x2="625" y2="437.5"/>
  <line stroke="red" x1="718.75" y1="31.25" x2="718.75" y2="174.56"/>
  <line stroke="red" x1="718.75" y1="31.25" x2="750" y2="0"/>
  <line stroke="red" x1="640.62" y1="328.12" x2="640.62" y2="343.75"/>
  <line stroke="blue" x1="640.62" y1="328.12" x2="703.12" y2="328.12"/>
  <line stroke="blue" x1="640.62" y1="328.12" x2="656.25" y2="343.75" fold-step="6"/>
  <line stroke="blue" x1="671.88" y1="265.62" x2="671.88" y2="296.88"/>
  <line stroke="blue" x1="671.88" y1="265.63" x2="687.5" y2="250" fold-step="6"/>
  <line stroke="red" x1="93.75" y1="718.75" x2="112.06" y2="718.75"/>
  <line stroke="red" x1="93.75" y1="718.75" x2="93.75" y2="727.9"/>
  <line stroke="blue" x1="93.75" y1="718.75" x2="109.38" y2="734.38"/>
  <line stroke="blue" x1="640.62" y1="343.75" x2="640.62" y2="359.38"/>
  <line stroke="red" x1="640.62" y1="343.75" x2="656.25" y2="343.75"/>
  <line stroke="red" x1="112.06" y1="718.75" x2="134.15" y2="718.75"/>
  <line stroke="blue" x1="112.06" y1="718.75" x2="118.53" y2="734.38" fold-step="7"/>
  <line stroke="blue" x1="31.25" y1="727.9" x2="15.63" y2="743.53" fold-step="7"/>
  <line stroke="red" x1="31.25" y1="727.9" x2="46.87" y2="743.53" fold-step="7"/>
  <line stroke="red" x1="31.25" y1="727.9" x2="31.25" y2="750"/>
  <line stroke="red" x1="375" y1="625" x2="406.25" y2="625"/>
  <line stroke="blue" x1="375" y1="625" x2="375" y2="875"/>
  <line stroke="red" x1="625" y1="375" x2="640.62" y2="359.38" fold-step="6"/>
  <line stroke="blue" x1="625" y1="375" x2="640.62" y2="390.62" fold-step="6"/>
  <line stroke="blue" x1="625" y1="375" x2="750" y2="375"/>
  <line stroke="red" x1="531.25" y1="500" x2="562.5" y2="500"/>
  <line stroke="red" x1="531.25" y1="500" x2="531.25" y2="562.5"/>
  <line stroke="blue" x1="134.15" y1="718.75" x2="118.53" y2="734.38"/>
  <line stroke="red" x1="134.15" y1="718.75" x2="218.75" y2="718.75"/>
  <line stroke="blue" x1="687.5" y1="250" x2="718.75" y2="174.56" fold-step="12"/>
  <line stroke="red" x1="687.5" y1="250" x2="703.13" y2="265.63" fold-step="6"/>
  <line stroke="blue" x1="687.5" y1="250" x2="687.5" y2="312.5"/>
  <line stroke="red" x1="687.5" y1="250" x2="718.75" y2="250"/>
  <line stroke="red" x1="468.75" y1="562.5" x2="531.25" y2="562.5"/>
  <line stroke="red" x1="468.75" y1="562.5" x2="468.75" y2="625"/>
  <line stroke="blue" x1="562.5" y1="468.75" x2="562.5" y2="500"/>
  <line stroke="red" x1="562.5" y1="468.75" x2="593.75" y2="468.75"/>
  <line stroke="red" x1="93.75" y1="727.9" x2="109.37" y2="743.53" fold-step="7"/>
  <line stroke="blue" x1="93.75" y1="727.9" x2="78.12" y2="743.53" fold-step="7"/>
  <line stroke="red" x1="93.75" y1="727.9" x2="93.75" y2="750"/>
  <line stroke="red" x1="640.62" y1="359.38" x2="656.25" y2="343.75" fold-step="6"/>
  <line stroke="red" x1="640.62" y1="359.38" x2="734.38" y2="359.38"/>
  <line stroke="red" x1="671.88" y1="296.88" x2="687.5" y2="312.5"/>
  <line stroke="red" x1="593.75" y1="437.5" x2="593.75" y2="468.75"/>
  <line stroke="red" x1="593.75" y1="437.5" x2="625" y2="437.5"/>
  <line stroke="red" x1="718.75" y1="174.56" x2="718.75" y2="250"/>
  <line stroke="red" x1="718.75" y1="174.56" x2="750" y2="250" fold-step="12"/>
  <line stroke="red" x1="656.25" y1="343.75" x2="718.75" y2="343.75"/>
  <line stroke="red" x1="109.38" y1="734.38" x2="118.53" y2="734.38"/>
  <line stroke="blue" x1="109.38" y1="734.38" x2="109.37" y2="743.53"/>
  <line stroke="blue" x1="109.38" y1="734.38" x2="125" y2="750"/>
  <line stroke="blue" x1="15.63" y1="743.53" x2="0" y2="750" fold-step="7"/>
  <line stroke="blue" x1="15.63" y1="743.53" x2="31.25" y2="750"/>
  <line stroke="blue" x1="118.53" y1="734.38" x2="125" y2="750" fold-step="7"/>
  <line stroke="red" x1="46.87" y1="743.53" x2="31.25" y2="750"/>
  <line stroke="red" x1="46.87" y1="743.53" x2="62.5" y2="750" fold-step="7"/>
  <line stroke="blue" x1="406.25" y1="625" x2="437.5" y2="625"/>
  <line stroke="red" x1="406.25" y1="625" x2="406.25" y2="906.25"/>
  <line stroke="blue" x1="78.12" y1="743.53" x2="62.5" y2="750" fold-step="7"/>
  <line stroke="blue" x1="78.12" y1="743.53" x2="93.75" y2="750"/>
  <line stroke="blue" x1="0" y1="750" x2="31.25" y2="750"/>
  <line stroke="red" x1="0" y1="750" x2="15.63" y2="754.19" fold-step="9"/>
  <line stroke="black" x1="0" y1="750" x2="0" y2="875"/>
  <line stroke="red" x1="750" y1="0" x2="781.25" y2="31.25"/>
  <line stroke="blue" x1="750" y1="0" x2="750" y2="250"/>
  <line stroke="black" x1="750" y1="0" x2="875" y2="0"/>
  <line stroke="red" x1="640.62" y1="390.62" x2="640.62" y2="406.25"/>
  <line stroke="blue" x1="640.62" y1="390.62" x2="656.25" y2="406.25" fold-step="6"/>
  <line stroke="blue" x1="640.62" y1="390.62" x2="734.38" y2="390.62"/>
  <line stroke="red" x1="31.25" y1="750" x2="62.5" y2="750"/>
  <line stroke="red" x1="31.25" y1="750" x2="15.63" y2="754.19"/>
  <line stroke="blue" x1="31.25" y1="750" x2="46.87" y2="754.19"/>
  <line stroke="red" x1="31.25" y1="750" x2="31.25" y2="781.25"/>
  <line stroke="red" x1="218.75" y1="718.75" x2="250" y2="750"/>
  <line stroke="red" x1="109.37" y1="743.53" x2="125" y2="750" fold-step="7"/>
  <line stroke="red" x1="109.37" y1="743.53" x2="93.75" y2="750"/>
  <line stroke="blue" x1="703.12" y1="265.63" x2="718.75" y2="265.62"/>
  <line stroke="red" x1="703.12" y1="265.63" x2="703.12" y2="328.12"/>
  <line stroke="red" x1="703.13" y1="265.63" x2="718.75" y2="281.25" fold-step="6"/>
  <line stroke="blue" x1="62.5" y1="750" x2="46.88" y2="754.19" fold-step="9"/>
  <line stroke="blue" x1="62.5" y1="750" x2="93.75" y2="750"/>
  <line stroke="red" x1="62.5" y1="750" x2="78.12" y2="754.19" fold-step="9"/>
  <line stroke="blue" x1="62.5" y1="750" x2="62.5" y2="812.5"/>
  <line stroke="blue" x1="562.5" y1="500" x2="593.75" y2="500"/>
  <line stroke="blue" x1="562.5" y1="500" x2="562.5" y2="562.5"/>
  <line stroke="red" x1="15.63" y1="754.19" x2="31.25" y2="781.25" fold-step="9"/>
  <line stroke="blue" x1="15.63" y1="754.19" x2="15.63" y2="859.38"/>
  <line stroke="blue" x1="687.5" y1="312.5" x2="703.12" y2="328.12"/>
  <line stroke="red" x1="46.88" y1="754.19" x2="46.88" y2="828.12"/>
  <line stroke="blue" x1="46.87" y1="754.19" x2="31.25" y2="781.25" fold-step="9"/>
  <line stroke="red" x1="93.75" y1="750" x2="78.12" y2="754.19"/>
  <line stroke="red" x1="93.75" y1="750" x2="125" y2="750"/>
  <line stroke="blue" x1="93.75" y1="750" x2="109.37" y2="754.19"/>
  <line stroke="red" x1="93.75" y1="750" x2="93.75" y2="781.25"/>
  <line stroke="blue" x1="593.75" y1="468.75" x2="625" y2="437.5"/>
  <line stroke="red" x1="593.75" y1="468.75" x2="593.75" y2="500"/>
  <line stroke="red" x1="593.75" y1="468.75" x2="640.62" y2="468.75"/>
  <line stroke="blue" x1="593.75" y1="468.75" x2="625" y2="500"/>
  <line stroke="red" x1="78.12" y1="754.19" x2="93.75" y2="781.25" fold-step="9"/>
  <line stroke="blue" x1="78.12" y1="754.19" x2="78.12" y2="796.88"/>
  <line stroke="blue" x1="640.62" y1="406.25" x2="640.62" y2="421.88"/>
  <line stroke="red" x1="640.62" y1="406.25" x2="656.25" y2="406.25"/>
  <line stroke="blue" x1="125" y1="750" x2="109.37" y2="754.19" fold-step="9"/>
  <line stroke="blue" x1="125" y1="750" x2="109.37" y2="765.62"/>
  <line stroke="red" x1="125" y1="750" x2="120.81" y2="765.62" fold-step="9"/>
  <line stroke="blue" x1="125" y1="750" x2="250" y2="750"/>
  <line stroke="red" x1="718.75" y1="250" x2="718.75" y2="265.62"/>
  <line stroke="blue" x1="718.75" y1="250" x2="750" y2="250"/>
  <line stroke="red" x1="109.37" y1="754.19" x2="109.38" y2="765.62"/>
  <line stroke="blue" x1="109.37" y1="754.19" x2="93.75" y2="781.25" fold-step="9"/>
  <line stroke="red" x1="437.5" y1="625" x2="468.75" y2="625"/>
  <line stroke="blue" x1="437.5" y1="625" x2="437.5" y2="937.5"/>
  <line stroke="red" x1="625" y1="437.5" x2="640.62" y2="421.87" fold-step="6"/>
  <line stroke="blue" x1="625" y1="437.5" x2="640.62" y2="453.12" fold-step="6"/>
  <line stroke="blue" x1="625" y1="437.5" x2="687.5" y2="437.5"/>
  <line stroke="red" x1="718.75" y1="265.62" x2="718.75" y2="281.25"/>
  <line stroke="red" x1="718.75" y1="265.62" x2="734.38" y2="265.62"/>
  <line stroke="red" x1="640.62" y1="421.87" x2="656.25" y2="406.25" fold-step="6"/>
  <line stroke="red" x1="640.62" y1="421.88" x2="703.12" y2="421.88"/>
  <line stroke="red" x1="656.25" y1="406.25" x2="718.75" y2="406.25"/>
  <line stroke="blue" x1="718.75" y1="281.25" x2="734.38" y2="265.63" fold-step="6"/>
  <line stroke="red" x1="718.75" y1="281.25" x2="718.75" y2="343.75"/>
  <line stroke="blue" x1="109.37" y1="765.62" x2="93.75" y2="781.25"/>
  <line stroke="blue" x1="109.38" y1="765.62" x2="120.81" y2="765.62"/>
  <line stroke="blue" x1="531.25" y1="562.5" x2="562.5" y2="562.5"/>
  <line stroke="red" x1="531.25" y1="562.5" x2="531.25" y2="625"/>
  <line stroke="red" x1="120.81" y1="765.62" x2="147.88" y2="781.25"/>
  <line stroke="red" x1="120.81" y1="765.62" x2="116.63" y2="781.25" fold-step="9"/>
  <line stroke="blue" x1="703.12" y1="328.12" x2="718.75" y2="343.75"/>
  <line stroke="red" x1="593.75" y1="500" x2="625" y2="500"/>
  <line stroke="red" x1="593.75" y1="500" x2="593.75" y2="562.5"/>
  <line stroke="blue" x1="734.38" y1="265.62" x2="734.38" y2="359.38"/>
  <line stroke="blue" x1="734.38" y1="265.63" x2="750" y2="250" fold-step="6"/>
  <line stroke="blue" x1="468.75" y1="625" x2="531.25" y2="625"/>
  <line stroke="red" x1="468.75" y1="625" x2="468.75" y2="968.75"/>
  <line stroke="red" x1="31.25" y1="781.25" x2="31.25" y2="843.75"/>
  <line stroke="red" x1="781.25" y1="31.25" x2="781.25" y2="174.56"/>
  <line stroke="blue" x1="781.25" y1="31.25" x2="812.5" y2="62.5"/>
  <line stroke="red" x1="781.25" y1="31.25" x2="875" y2="31.25"/>
  <line stroke="red" x1="640.62" y1="453.12" x2="640.62" y2="468.75"/>
  <line stroke="blue" x1="640.62" y1="453.12" x2="656.25" y2="468.75" fold-step="6"/>
  <line stroke="blue" x1="640.62" y1="453.12" x2="671.88" y2="453.12"/>
  <line stroke="red" x1="93.75" y1="781.25" x2="116.63" y2="781.25"/>
  <line stroke="red" x1="93.75" y1="781.25" x2="78.12" y2="796.87"/>
  <line stroke="red" x1="116.63" y1="781.25" x2="147.88" y2="781.25"/>
  <line stroke="blue" x1="116.63" y1="781.25" x2="120.81" y2="796.88" fold-step="9"/>
  <line stroke="blue" x1="250" y1="750" x2="281.25" y2="781.25"/>
  <line stroke="blue" x1="750" y1="250" x2="781.25" y2="174.56" fold-step="12"/>
  <line stroke="red" x1="750" y1="250" x2="765.62" y2="265.62" fold-step="6"/>
  <line stroke="red" x1="750" y1="250" x2="781.25" y2="250"/>
  <line stroke="blue" x1="750" y1="250" x2="750" y2="375"/>
  <line stroke="blue" x1="640.62" y1="468.75" x2="640.62" y2="484.38"/>
  <line stroke="red" x1="640.62" y1="468.75" x2="656.25" y2="468.75"/>
  <line stroke="blue" x1="147.88" y1="781.25" x2="120.81" y2="796.88"/>
  <line stroke="red" x1="147.88" y1="781.25" x2="281.25" y2="781.25"/>
  <line stroke="red" x1="562.5" y1="562.5" x2="593.75" y2="562.5"/>
  <line stroke="blue" x1="562.5" y1="562.5" x2="562.5" y2="625"/>
  <line stroke="red" x1="718.75" y1="343.75" x2="734.38" y2="359.38"/>
  <line stroke="blue" x1="625" y1="500" x2="640.62" y2="484.38"/>
  <line stroke="blue" x1="625" y1="500" x2="625" y2="562.5"/>
  <line stroke="red" x1="781.25" y1="174.56" x2="781.25" y2="250"/>
  <line stroke="red" x1="781.25" y1="174.56" x2="812.5" y2="250" fold-step="12"/>
  <line stroke="red" x1="78.12" y1="796.87" x2="62.5" y2="812.5"/>
  <line stroke="red" x1="78.12" y1="796.88" x2="120.81" y2="796.88"/>
  <line stroke="red" x1="640.62" y1="484.38" x2="656.25" y2="484.38"/>
  <line stroke="blue" x1="640.62" y1="484.38" x2="656.25" y2="468.75"/>
  <line stroke="blue" x1="120.81" y1="796.88" x2="125" y2="812.5" fold-step="9"/>
  <line stroke="red" x1="656.25" y1="468.75" x2="671.88" y2="453.12"/>
  <line stroke="red" x1="656.25" y1="468.75" x2="656.25" y2="484.38"/>
  <line stroke="red" x1="656.25" y1="468.75" x2="671.88" y2="484.38" fold-step="6"/>
  <line stroke="red" x1="671.88" y1="453.12" x2="687.5" y2="437.5"/>
  <line stroke="red" x1="671.88" y1="453.12" x2="671.88" y2="484.38"/>
  <line stroke="red" x1="765.62" y1="265.62" x2="781.25" y2="281.25" fold-step="6"/>
  <line stroke="red" x1="765.62" y1="265.62" x2="765.62" y2="359.38"/>
  <line stroke="blue" x1="765.62" y1="265.63" x2="781.25" y2="265.62"/>
  <line stroke="blue" x1="62.5" y1="812.5" x2="125" y2="812.5"/>
  <line stroke="blue" x1="62.5" y1="812.5" x2="46.88" y2="828.12"/>
  <line stroke="blue" x1="687.5" y1="437.5" x2="703.12" y2="421.88"/>
  <line stroke="blue" x1="687.5" y1="437.5" x2="687.5" y2="500"/>
  <line stroke="red" x1="812.5" y1="62.5" x2="843.75" y2="93.75"/>
  <line stroke="blue" x1="812.5" y1="62.5" x2="812.5" y2="250"/>
  <line stroke="blue" x1="812.5" y1="62.5" x2="875" y2="62.5"/>
  <line stroke="blue" x1="656.25" y1="484.38" x2="671.88" y2="484.38"/>
  <line stroke="red" x1="656.25" y1="484.38" x2="656.25" y2="562.5"/>
  <line stroke="red" x1="734.38" y1="359.38" x2="750" y2="375"/>
  <line stroke="blue" x1="593.75" y1="562.5" x2="625" y2="562.5"/>
  <line stroke="red" x1="593.75" y1="562.5" x2="593.75" y2="625"/>
  <line stroke="blue" x1="703.12" y1="421.88" x2="703.12" y2="484.38"/>
  <line stroke="blue" x1="703.12" y1="421.88" x2="718.75" y2="406.25"/>
  <line stroke="red" x1="531.25" y1="625" x2="562.5" y2="625"/>
  <line stroke="red" x1="531.25" y1="625" x2="531.25" y2="968.75"/>
  <line stroke="red" x1="781.25" y1="250" x2="781.25" y2="265.62"/>
  <line stroke="blue" x1="781.25" y1="250" x2="812.5" y2="250"/>
  <line stroke="red" x1="125" y1="812.5" x2="120.81" y2="828.12" fold-step="9"/>
  <line stroke="blue" x1="125" y1="812.5" x2="312.5" y2="812.5"/>
  <line stroke="red" x1="781.25" y1="265.62" x2="781.25" y2="281.25"/>
  <line stroke="red" x1="781.25" y1="265.62" x2="796.88" y2="265.62"/>
  <line stroke="red" x1="718.75" y1="406.25" x2="734.38" y2="390.62"/>
  <line stroke="red" x1="718.75" y1="406.25" x2="718.75" y2="468.75"/>
  <line stroke="red" x1="671.88" y1="484.38" x2="687.5" y2="500" fold-step="6"/>
  <line stroke="blue" x1="46.88" y1="828.12" x2="120.81" y2="828.12"/>
  <line stroke="blue" x1="46.88" y1="828.12" x2="31.25" y2="843.75"/>
  <line stroke="red" x1="281.25" y1="781.25" x2="312.5" y2="812.5"/>
  <line stroke="blue" x1="781.25" y1="281.25" x2="796.88" y2="265.63" fold-step="6"/>
  <line stroke="red" x1="781.25" y1="281.25" x2="781.25" y2="343.75"/>
  <line stroke="red" x1="734.38" y1="390.62" x2="750" y2="375"/>
  <line stroke="red" x1="734.38" y1="390.62" x2="734.38" y2="484.38"/>
  <line stroke="red" x1="120.81" y1="828.12" x2="147.88" y2="843.75"/>
  <line stroke="red" x1="120.81" y1="828.12" x2="116.63" y2="843.75" fold-step="9"/>
  <line stroke="red" x1="750" y1="375" x2="765.62" y2="359.38"/>
  <line stroke="red" x1="750" y1="375" x2="765.62" y2="390.62"/>
  <line stroke="blue" x1="750" y1="375" x2="750" y2="500"/>
  <line stroke="red" x1="750" y1="375" x2="875" y2="375"/>
  <line stroke="blue" x1="796.88" y1="265.62" x2="796.88" y2="328.12"/>
  <line stroke="blue" x1="796.88" y1="265.63" x2="812.5" y2="250" fold-step="6"/>
  <line stroke="blue" x1="562.5" y1="625" x2="593.75" y2="625"/>
  <line stroke="blue" x1="562.5" y1="625" x2="562.5" y2="937.5"/>
  <line stroke="red" x1="625" y1="562.5" x2="656.25" y2="562.5"/>
  <line stroke="blue" x1="625" y1="562.5" x2="625" y2="625"/>
  <line stroke="red" x1="31.25" y1="843.75" x2="116.63" y2="843.75"/>
  <line stroke="red" x1="31.25" y1="843.75" x2="15.62" y2="859.38"/>
  <line stroke="red" x1="765.62" y1="359.38" x2="781.25" y2="343.75"/>
  <line stroke="blue" x1="765.62" y1="359.38" x2="859.38" y2="359.38"/>
  <line stroke="red" x1="843.75" y1="93.75" x2="843.75" y2="174.56"/>
  <line stroke="red" x1="843.75" y1="93.75" x2="875" y2="93.75"/>
  <line stroke="blue" x1="843.75" y1="93.75" x2="875" y2="125"/>
  <line stroke="blue" x1="687.5" y1="500" x2="703.12" y2="484.37" fold-step="6"/>
  <line stroke="blue" x1="687.5" y1="500" x2="687.5" y2="562.5"/>
  <line stroke="blue" x1="812.5" y1="250" x2="843.75" y2="174.56" fold-step="12"/>
  <line stroke="red" x1="812.5" y1="250" x2="828.12" y2="265.62" fold-step="6"/>
  <line stroke="blue" x1="812.5" y1="250" x2="812.5" y2="312.5"/>
  <line stroke="red" x1="812.5" y1="250" x2="843.75" y2="250"/>
  <line stroke="red" x1="116.63" y1="843.75" x2="147.88" y2="843.75"/>
  <line stroke="blue" x1="116.63" y1="843.75" x2="120.81" y2="859.38" fold-step="9"/>
  <line stroke="blue" x1="781.25" y1="343.75" x2="796.88" y2="328.12"/>
  <line stroke="red" x1="781.25" y1="343.75" x2="843.75" y2="343.75"/>
  <line stroke="blue" x1="703.12" y1="484.37" x2="718.75" y2="468.75" fold-step="6"/>
  <line stroke="red" x1="703.12" y1="484.38" x2="718.75" y2="484.38"/>
  <line stroke="blue" x1="147.88" y1="843.75" x2="120.81" y2="859.38"/>
  <line stroke="red" x1="147.88" y1="843.75" x2="343.75" y2="843.75"/>
  <line stroke="red" x1="718.75" y1="468.75" x2="718.75" y2="484.38"/>
  <line stroke="red" x1="718.75" y1="468.75" x2="734.38" y2="484.38" fold-step="6"/>
  <line stroke="red" x1="765.62" y1="390.62" x2="781.25" y2="406.25"/>
  <line stroke="blue" x1="765.62" y1="390.62" x2="765.62" y2="484.37"/>
  <line stroke="red" x1="765.62" y1="390.62" x2="859.38" y2="390.62"/>
  <line stroke="red" x1="15.62" y1="859.38" x2="0" y2="875"/>
  <line stroke="red" x1="15.63" y1="859.38" x2="120.81" y2="859.38"/>
  <line stroke="red" x1="843.75" y1="174.56" x2="843.75" y2="250"/>
  <line stroke="red" x1="843.75" y1="174.56" x2="875" y2="250" fold-step="12"/>
  <line stroke="red" x1="796.88" y1="328.12" x2="859.38" y2="328.12"/>
  <line stroke="blue" x1="796.88" y1="328.12" x2="812.5" y2="312.5"/>
  <line stroke="red" x1="593.75" y1="625" x2="625" y2="625"/>
  <line stroke="red" x1="593.75" y1="625" x2="593.75" y2="906.25"/>
  <line stroke="blue" x1="656.25" y1="562.5" x2="687.5" y2="562.5"/>
  <line stroke="red" x1="656.25" y1="562.5" x2="656.25" y2="625"/>
  <line stroke="blue" x1="718.75" y1="484.38" x2="734.38" y2="484.38"/>
  <line stroke="red" x1="718.75" y1="484.38" x2="718.75" y2="562.5"/>
  <line stroke="blue" x1="120.81" y1="859.38" x2="125" y2="875" fold-step="9"/>
  <line stroke="red" x1="828.12" y1="265.62" x2="828.12" y2="296.88"/>
  <line stroke="red" x1="828.12" y1="265.62" x2="843.75" y2="281.25" fold-step="6"/>
  <line stroke="blue" x1="828.12" y1="265.63" x2="843.75" y2="265.62"/>
  <line stroke="blue" x1="312.5" y1="812.5" x2="343.75" y2="843.75"/>
  <line stroke="red" x1="812.5" y1="312.5" x2="828.12" y2="296.88"/>
  <line stroke="blue" x1="812.5" y1="312.5" x2="875" y2="312.5"/>
  <line stroke="blue" x1="0" y1="875" x2="125" y2="875"/>
  <line stroke="red" x1="0" y1="875" x2="15.63" y2="890.63"/>
  <line stroke="black" x1="0" y1="875" x2="0" y2="1000"/>
  <line stroke="blue" x1="875" y1="0" x2="875" y2="31.25"/>
  <line stroke="blue" x1="875" y1="0" x2="890.63" y2="15.63" fold-step="2"/>
  <line stroke="black" x1="875" y1="0" x2="1000" y2="0"/>
  <line stroke="red" x1="875" y1="31.25" x2="875" y2="62.5"/>
  <line stroke="red" x1="875" y1="31.25" x2="890.63" y2="31.25"/>
  <line stroke="blue" x1="875" y1="62.5" x2="875" y2="93.75"/>
  <line stroke="red" x1="875" y1="62.5" x2="890.63" y2="46.87" fold-step="2"/>
  <line stroke="blue" x1="875" y1="62.5" x2="890.62" y2="78.12" fold-step="2"/>
  <line stroke="blue" x1="875" y1="62.5" x2="937.5" y2="62.5"/>
  <line stroke="red" x1="734.38" y1="484.38" x2="750" y2="500" fold-step="6"/>
  <line stroke="red" x1="828.12" y1="296.88" x2="843.75" y2="281.25"/>
  <line stroke="blue" x1="828.12" y1="296.88" x2="859.38" y2="296.88"/>
  <line stroke="red" x1="843.75" y1="250" x2="843.75" y2="265.62"/>
  <line stroke="blue" x1="843.75" y1="250" x2="875" y2="250"/>
  <line stroke="red" x1="875" y1="93.75" x2="875" y2="125"/>
  <line stroke="red" x1="875" y1="93.75" x2="890.62" y2="93.75"/>
  <line stroke="blue" x1="781.25" y1="406.25" x2="796.87" y2="421.87"/>
  <line stroke="red" x1="781.25" y1="406.25" x2="781.25" y2="468.75"/>
  <line stroke="red" x1="781.25" y1="406.25" x2="843.75" y2="406.25"/>
  <line stroke="red" x1="125" y1="875" x2="120.81" y2="890.63" fold-step="9"/>
  <line stroke="blue" x1="125" y1="875" x2="375" y2="875"/>
  <line stroke="blue" x1="625" y1="625" x2="656.25" y2="625"/>
  <line stroke="blue" x1="625" y1="625" x2="625" y2="875"/>
  <line stroke="blue" x1="875" y1="125" x2="890.62" y2="109.37"/>
  <line stroke="blue" x1="875" y1="125" x2="875" y2="250"/>
  <line stroke="red" x1="843.75" y1="265.62" x2="843.75" y2="281.25"/>
  <line stroke="red" x1="843.75" y1="265.62" x2="859.38" y2="265.62"/>
  <line stroke="red" x1="687.5" y1="562.5" x2="718.75" y2="562.5"/>
  <line stroke="blue" x1="687.5" y1="562.5" x2="687.5" y2="625"/>
  <line stroke="blue" x1="843.75" y1="281.25" x2="859.38" y2="265.62"/>
  <line stroke="red" x1="843.75" y1="281.25" x2="859.38" y2="281.25"/>
  <line stroke="blue" x1="843.75" y1="281.25" x2="859.38" y2="296.88" fold-step="6"/>
  <line stroke="blue" x1="890.63" y1="15.63" x2="890.63" y2="31.25"/>
  <line stroke="blue" x1="890.63" y1="15.63" x2="906.25" y2="31.25" fold-step="2"/>
  <line stroke="red" x1="890.63" y1="15.63" x2="984.37" y2="15.63"/>
  <line stroke="red" x1="15.63" y1="890.63" x2="120.81" y2="890.63"/>
  <line stroke="blue" x1="15.63" y1="890.63" x2="15.63" y2="995.81"/>
  <line stroke="red" x1="15.63" y1="890.63" x2="31.25" y2="906.25"/>
  <line stroke="red" x1="890.63" y1="31.25" x2="890.63" y2="46.87"/>
  <line stroke="red" x1="890.63" y1="31.25" x2="906.25" y2="31.25"/>
  <line stroke="blue" x1="890.63" y1="46.87" x2="953.12" y2="46.88"/>
  <line stroke="red" x1="890.63" y1="46.87" x2="906.25" y2="31.25" fold-step="2"/>
  <line stroke="blue" x1="890.62" y1="78.12" x2="906.25" y2="93.75" fold-step="2"/>
  <line stroke="red" x1="890.62" y1="78.12" x2="890.62" y2="93.75"/>
  <line stroke="blue" x1="890.62" y1="78.12" x2="921.88" y2="78.12"/>
  <line stroke="blue" x1="890.62" y1="93.75" x2="890.62" y2="109.37"/>
  <line stroke="red" x1="890.62" y1="93.75" x2="906.25" y2="93.75"/>
  <line stroke="blue" x1="890.62" y1="109.37" x2="906.25" y2="93.75"/>
  <line stroke="red" x1="890.62" y1="109.37" x2="906.25" y2="109.37"/>
  <line stroke="red" x1="120.81" y1="890.63" x2="116.63" y2="906.25" fold-step="9"/>
  <line stroke="blue" x1="120.81" y1="890.63" x2="147.88" y2="906.25"/>
  <line stroke="blue" x1="859.38" y1="265.62" x2="875" y2="250"/>
  <line stroke="blue" x1="859.38" y1="265.62" x2="859.38" y2="281.25"/>
  <line stroke="blue" x1="750" y1="500" x2="765.62" y2="484.38" fold-step="6"/>
  <line stroke="blue" x1="750" y1="500" x2="750" y2="562.5"/>
  <line stroke="blue" x1="796.87" y1="421.87" x2="812.5" y2="437.5"/>
  <line stroke="red" x1="796.87" y1="421.87" x2="796.87" y2="484.37"/>
  <line stroke="blue" x1="796.87" y1="421.87" x2="859.37" y2="421.87"/>
  <line stroke="red" x1="859.38" y1="281.25" x2="859.38" y2="296.88"/>
  <line stroke="red" x1="859.38" y1="281.25" x2="875" y2="281.25"/>
  <line stroke="red" x1="765.62" y1="484.37" x2="781.25" y2="484.37"/>
  <line stroke="blue" x1="765.62" y1="484.38" x2="781.25" y2="468.75" fold-step="6"/>
  <line stroke="red" x1="656.25" y1="625" x2="687.5" y2="625"/>
  <line stroke="red" x1="656.25" y1="625" x2="656.25" y2="843.75"/>
  <line stroke="red" x1="31.25" y1="906.25" x2="116.63" y2="906.25"/>
  <line stroke="blue" x1="31.25" y1="906.25" x2="46.87" y2="921.87"/>
  <line stroke="red" x1="31.25" y1="906.25" x2="31.25" y2="991.63"/>
  <line stroke="red" x1="906.25" y1="31.25" x2="968.75" y2="31.25"/>
  <line stroke="blue" x1="859.38" y1="296.88" x2="875" y2="312.5" fold-step="6"/>
  <line stroke="red" x1="875" y1="250" x2="875" y2="281.25"/>
  <line stroke="blue" x1="875" y1="250" x2="906.25" y2="174.56" fold-step="12"/>
  <line stroke="red" x1="875" y1="250" x2="906.25" y2="250"/>
  <line stroke="blue" x1="875" y1="250" x2="906.25" y2="281.25"/>
  <line stroke="red" x1="343.75" y1="843.75" x2="375" y2="875"/>
  <line stroke="red" x1="781.25" y1="468.75" x2="781.25" y2="484.37"/>
  <line stroke="red" x1="781.25" y1="468.75" x2="796.87" y2="484.37" fold-step="6"/>
  <line stroke="red" x1="843.75" y1="343.75" x2="859.38" y2="328.12" fold-step="6"/>
  <line stroke="red" x1="843.75" y1="343.75" x2="859.38" y2="343.75"/>
  <line stroke="blue" x1="843.75" y1="343.75" x2="859.38" y2="359.38" fold-step="6"/>
  <line stroke="red" x1="906.25" y1="93.75" x2="906.25" y2="109.37"/>
  <line stroke="red" x1="906.25" y1="93.75" x2="921.88" y2="78.12"/>
  <line stroke="red" x1="906.25" y1="93.75" x2="921.88" y2="109.38" fold-step="2"/>
  <line stroke="blue" x1="718.75" y1="562.5" x2="750" y2="562.5"/>
  <line stroke="red" x1="718.75" y1="562.5" x2="718.75" y2="625"/>
  <line stroke="red" x1="906.25" y1="109.37" x2="906.25" y2="174.56"/>
  <line stroke="blue" x1="906.25" y1="109.37" x2="921.88" y2="109.37"/>
  <line stroke="red" x1="116.63" y1="906.25" x2="147.88" y2="906.25"/>
  <line stroke="blue" x1="116.63" y1="906.25" x2="120.81" y2="921.87" fold-step="9"/>
  <line stroke="red" x1="147.88" y1="906.25" x2="120.81" y2="921.87"/>
  <line stroke="red" x1="147.88" y1="906.25" x2="406.25" y2="906.25"/>
  <line stroke="blue" x1="875" y1="281.25" x2="875" y2="312.5"/>
  <line stroke="red" x1="875" y1="281.25" x2="906.25" y2="281.25"/>
  <line stroke="blue" x1="781.25" y1="484.37" x2="796.87" y2="484.37"/>
  <line stroke="red" x1="781.25" y1="484.37" x2="781.25" y2="562.5"/>
  <line stroke="red" x1="859.38" y1="328.12" x2="875" y2="312.5" fold-step="6"/>
  <line stroke="blue" x1="859.38" y1="328.12" x2="859.38" y2="343.75"/>
  <line stroke="red" x1="812.5" y1="437.5" x2="828.12" y2="453.12"/>
  <line stroke="blue" x1="812.5" y1="437.5" x2="812.5" y2="500"/>
  <line stroke="blue" x1="812.5" y1="437.5" x2="875" y2="437.5"/>
  <line stroke="red" x1="906.25" y1="174.56" x2="906.25" y2="250"/>
  <line stroke="red" x1="906.25" y1="174.56" x2="937.5" y2="250" fold-step="12"/>
  <line stroke="blue" x1="46.87" y1="921.87" x2="120.81" y2="921.87"/>
  <line stroke="red" x1="46.87" y1="921.87" x2="46.87" y2="995.81"/>
  <line stroke="blue" x1="46.87" y1="921.87" x2="62.5" y2="937.5"/>
  <line stroke="red" x1="921.88" y1="78.12" x2="921.88" y2="109.38"/>
  <line stroke="red" x1="921.88" y1="78.12" x2="937.5" y2="62.5"/>
  <line stroke="red" x1="859.38" y1="343.75" x2="859.38" y2="359.38"/>
  <line stroke="red" x1="859.38" y1="343.75" x2="875" y2="343.75"/>
  <line stroke="red" x1="921.88" y1="109.38" x2="937.5" y2="125" fold-step="2"/>
  <line stroke="blue" x1="687.5" y1="625" x2="718.75" y2="625"/>
  <line stroke="blue" x1="687.5" y1="625" x2="687.5" y2="812.5"/>
  <line stroke="red" x1="875" y1="312.5" x2="875" y2="343.75"/>
  <line stroke="blue" x1="875" y1="312.5" x2="937.5" y2="312.5"/>
  <line stroke="red" x1="875" y1="312.5" x2="902.06" y2="296.88" fold-step="4"/>
  <line stroke="blue" x1="875" y1="312.5" x2="902.06" y2="328.13" fold-step="4"/>
  <line stroke="blue" x1="120.81" y1="921.87" x2="125" y2="937.5" fold-step="9"/>
  <line stroke="blue" x1="859.38" y1="359.38" x2="875" y2="375" fold-step="6"/>
  <line stroke="red" x1="796.87" y1="484.37" x2="812.5" y2="500" fold-step="6"/>
  <line stroke="red" x1="843.75" y1="406.25" x2="859.38" y2="390.62" fold-step="6"/>
  <line stroke="red" x1="843.75" y1="406.25" x2="859.37" y2="406.25"/>
  <line stroke="blue" x1="843.75" y1="406.25" x2="859.38" y2="421.87" fold-step="6"/>
  <line stroke="red" x1="750" y1="562.5" x2="781.25" y2="562.5"/>
  <line stroke="blue" x1="750" y1="562.5" x2="750" y2="625"/>
  <line stroke="blue" x1="62.5" y1="937.5" x2="125" y2="937.5"/>
  <line stroke="red" x1="62.5" y1="937.5" x2="78.12" y2="953.12"/>
  <line stroke="blue" x1="62.5" y1="937.5" x2="62.5" y2="1000"/>
  <line stroke="blue" x1="937.5" y1="62.5" x2="937.5" y2="125"/>
  <line stroke="blue" x1="937.5" y1="62.5" x2="953.12" y2="46.88"/>
  <line stroke="blue" x1="875" y1="343.75" x2="875" y2="375"/>
  <line stroke="red" x1="875" y1="343.75" x2="911.08" y2="343.75"/>
  <line stroke="red" x1="906.25" y1="250" x2="906.25" y2="281.25"/>
  <line stroke="blue" x1="906.25" y1="250" x2="937.5" y2="250"/>
  <line stroke="blue" x1="859.37" y1="390.63" x2="859.37" y2="406.25"/>
  <line stroke="blue" x1="828.12" y1="453.12" x2="828.12" y2="484.37"/>
  <line stroke="red" x1="828.12" y1="453.12" x2="843.75" y2="468.75"/>
  <line stroke="red" x1="828.12" y1="453.12" x2="859.38" y2="453.12"/>
  <line stroke="red" x1="859.38" y1="390.62" x2="875" y2="375" fold-step="6"/>
  <line stroke="red" x1="125" y1="937.5" x2="120.81" y2="953.12" fold-step="9"/>
  <line stroke="blue" x1="125" y1="937.5" x2="437.5" y2="937.5"/>
  <line stroke="blue" x1="937.5" y1="125" x2="953.12" y2="109.38" fold-step="2"/>
  <line stroke="blue" x1="937.5" y1="125" x2="937.5" y2="250"/>
  <line stroke="red" x1="906.25" y1="281.25" x2="909.31" y2="284.31"/>
  <line stroke="red" x1="902.06" y1="296.88" x2="914.62" y2="289.62" fold-step="4"/>
  <line stroke="blue" x1="902.06" y1="296.88" x2="909.31" y2="284.31"/>
  <line stroke="red" x1="902.06" y1="296.88" x2="921.88" y2="296.88"/>
  <line stroke="red" x1="859.37" y1="406.25" x2="859.37" y2="421.87"/>
  <line stroke="red" x1="859.37" y1="406.25" x2="875" y2="406.25"/>
  <line stroke="blue" x1="375" y1="875" x2="406.25" y2="906.25"/>
  <line stroke="blue" x1="875" y1="375" x2="875" y2="406.25"/>
  <line stroke="blue" x1="875" y1="375" x2="1000" y2="375"/>
  <line stroke="red" x1="875" y1="375" x2="902.06" y2="359.37" fold-step="4"/>
  <line stroke="blue" x1="875" y1="375" x2="902.06" y2="390.63" fold-step="4"/>
  <line stroke="red" x1="718.75" y1="625" x2="750" y2="625"/>
  <line stroke="red" x1="718.75" y1="625" x2="718.75" y2="781.25"/>
  <line stroke="red" x1="909.31" y1="284.31" x2="914.62" y2="289.62"/>
  <line stroke="red" x1="909.31" y1="284.31" x2="921.88" y2="277.06"/>
  <line stroke="blue" x1="812.5" y1="500" x2="828.12" y2="484.38" fold-step="6"/>
  <line stroke="blue" x1="812.5" y1="500" x2="812.5" y2="562.5"/>
  <line stroke="red" x1="953.12" y1="46.88" x2="953.12" y2="109.38"/>
  <line stroke="blue" x1="953.12" y1="46.88" x2="968.75" y2="31.25"/>
  <line stroke="red" x1="78.12" y1="953.12" x2="93.75" y2="968.75"/>
  <line stroke="red" x1="78.12" y1="953.12" x2="120.81" y2="953.12"/>
  <line stroke="blue" x1="78.12" y1="953.12" x2="78.12" y2="995.81"/>
  <line stroke="blue" x1="859.38" y1="421.87" x2="875" y2="437.5" fold-step="6"/>
  <line stroke="blue" x1="953.13" y1="109.37" x2="968.75" y2="109.38"/>
  <line stroke="red" x1="828.12" y1="484.37" x2="843.75" y2="484.37"/>
  <line stroke="blue" x1="953.12" y1="109.38" x2="968.75" y2="93.75" fold-step="2"/>
  <line stroke="blue" x1="828.12" y1="484.38" x2="843.75" y2="468.75" fold-step="6"/>
  <line stroke="blue" x1="914.62" y1="289.62" x2="921.88" y2="277.06" fold-step="4"/>
  <line stroke="red" x1="914.62" y1="289.62" x2="921.88" y2="296.88"/>
  <line stroke="red" x1="902.06" y1="328.13" x2="911.08" y2="343.75"/>
  <line stroke="blue" x1="902.06" y1="328.13" x2="929.13" y2="343.75" fold-step="4"/>
  <line stroke="blue" x1="902.06" y1="328.13" x2="953.13" y2="328.13"/>
  <line stroke="red" x1="120.81" y1="953.12" x2="116.63" y2="968.75" fold-step="9"/>
  <line stroke="blue" x1="120.81" y1="953.12" x2="147.88" y2="968.75"/>
  <line stroke="blue" x1="921.88" y1="277.06" x2="937.5" y2="250" fold-step="4"/>
  <line stroke="blue" x1="921.88" y1="277.06" x2="921.88" y2="296.88"/>
  <line stroke="blue" x1="781.25" y1="562.5" x2="812.5" y2="562.5"/>
  <line stroke="red" x1="781.25" y1="562.5" x2="781.25" y2="625"/>
  <line stroke="red" x1="875" y1="406.25" x2="875" y2="437.5"/>
  <line stroke="red" x1="875" y1="406.25" x2="911.08" y2="406.25"/>
  <line stroke="red" x1="843.75" y1="468.75" x2="859.38" y2="453.12" fold-step="6"/>
  <line stroke="red" x1="843.75" y1="468.75" x2="843.75" y2="484.37"/>
  <line stroke="red" x1="843.75" y1="468.75" x2="859.38" y2="468.75"/>
  <line stroke="blue" x1="843.75" y1="468.75" x2="859.38" y2="484.37"/>
  <line stroke="red" x1="921.88" y1="296.88" x2="937.5" y2="312.5"/>
  <line stroke="red" x1="968.75" y1="31.25" x2="968.75" y2="93.75"/>
  <line stroke="red" x1="968.75" y1="31.25" x2="984.37" y2="15.63"/>
  <line stroke="blue" x1="937.5" y1="250" x2="968.75" y2="174.56" fold-step="12"/>
  <line stroke="blue" x1="937.5" y1="250" x2="937.5" y2="312.5"/>
  <line stroke="red" x1="937.5" y1="250" x2="953.12" y2="277.06" fold-step="4"/>
  <line stroke="red" x1="937.5" y1="250" x2="968.75" y2="250"/>
  <line stroke="red" x1="902.06" y1="359.37" x2="929.13" y2="343.75" fold-step="4"/>
  <line stroke="red" x1="902.06" y1="359.37" x2="984.37" y2="359.37"/>
  <line stroke="blue" x1="902.06" y1="359.37" x2="911.08" y2="343.75"/>
  <line stroke="red" x1="859.38" y1="453.12" x2="875" y2="437.5" fold-step="6"/>
  <line stroke="blue" x1="859.38" y1="453.12" x2="859.38" y2="468.75"/>
  <line stroke="blue" x1="843.75" y1="484.37" x2="859.38" y2="484.37"/>
  <line stroke="red" x1="843.75" y1="484.37" x2="843.75" y2="531.25"/>
  <line stroke="red" x1="93.75" y1="968.75" x2="116.63" y2="968.75"/>
  <line stroke="blue" x1="93.75" y1="968.75" x2="109.38" y2="984.38"/>
  <line stroke="red" x1="93.75" y1="968.75" x2="93.75" y2="991.63"/>
  <line stroke="red" x1="968.75" y1="93.75" x2="968.75" y2="109.38"/>
  <line stroke="red" x1="968.75" y1="93.75" x2="984.37" y2="109.37" fold-step="2"/>
  <line stroke="red" x1="911.08" y1="343.75" x2="929.13" y2="343.75"/>
  <line stroke="red" x1="968.75" y1="109.38" x2="968.75" y2="174.56"/>
  <line stroke="red" x1="968.75" y1="109.38" x2="984.37" y2="109.37"/>
  <line stroke="red" x1="116.63" y1="968.75" x2="147.88" y2="968.75"/>
  <line stroke="blue" x1="116.63" y1="968.75" x2="120.81" y2="984.38" fold-step="9"/>
  <line stroke="blue" x1="750" y1="625" x2="781.25" y2="625"/>
  <line stroke="blue" x1="750" y1="625" x2="750" y2="750"/>
  <line stroke="blue" x1="875" y1="437.5" x2="875" y2="468.75"/>
  <line stroke="blue" x1="875" y1="437.5" x2="937.5" y2="437.5"/>
  <line stroke="red" x1="875" y1="437.5" x2="902.06" y2="421.87" fold-step="4"/>
  <line stroke="blue" x1="875" y1="437.5" x2="902.06" y2="453.13" fold-step="4"/>
  <line stroke="red" x1="859.38" y1="468.75" x2="859.38" y2="484.37"/>
  <line stroke="red" x1="859.38" y1="468.75" x2="875" y2="468.75"/>
  <line stroke="red" x1="147.88" y1="968.75" x2="120.81" y2="984.38"/>
  <line stroke="red" x1="147.88" y1="968.75" x2="468.75" y2="968.75"/>
  <line stroke="blue" x1="902.06" y1="390.63" x2="911.08" y2="406.25"/>
  <line stroke="red" x1="902.06" y1="390.63" x2="984.37" y2="390.63"/>
  <line stroke="blue" x1="902.06" y1="390.63" x2="929.13" y2="406.25" fold-step="4"/>
  <line stroke="red" x1="968.75" y1="174.56" x2="968.75" y2="250"/>
  <line stroke="red" x1="968.75" y1="174.56" x2="1000" y2="250" fold-step="12"/>
  <line stroke="red" x1="984.37" y1="15.63" x2="1000" y2="0"/>
  <line stroke="blue" x1="984.37" y1="15.63" x2="984.37" y2="109.37"/>
  <line stroke="blue" x1="859.38" y1="484.37" x2="875" y2="500"/>
  <line stroke="red" x1="812.5" y1="562.5" x2="843.75" y2="531.25"/>
  <line stroke="blue" x1="812.5" y1="562.5" x2="812.5" y2="625"/>
  <line stroke="red" x1="812.5" y1="562.5" x2="843.75" y2="593.75"/>
  <line stroke="blue" x1="812.5" y1="562.5" x2="875" y2="562.5"/>
  <line stroke="blue" x1="937.5" y1="312.5" x2="953.13" y2="328.13"/>
  <line stroke="red" x1="984.37" y1="109.37" x2="1000" y2="125" fold-step="2"/>
  <line stroke="blue" x1="109.38" y1="984.38" x2="125" y2="1000"/>
  <line stroke="blue" x1="109.38" y1="984.38" x2="120.81" y2="984.38"/>
  <line stroke="red" x1="109.38" y1="984.38" x2="109.38" y2="995.81"/>
  <line stroke="red" x1="929.13" y1="343.75" x2="968.75" y2="343.75"/>
  <line stroke="blue" x1="120.81" y1="984.38" x2="125" y2="1000" fold-step="9"/>
  <line stroke="blue" x1="31.25" y1="991.63" x2="15.63" y2="995.81" fold-step="9"/>
  <line stroke="red" x1="31.25" y1="991.63" x2="46.87" y2="995.81" fold-step="9"/>
  <line stroke="red" x1="31.25" y1="991.63" x2="31.25" y2="1000"/>
  <line stroke="red" x1="953.12" y1="277.06" x2="968.75" y2="304.13" fold-step="4"/>
  <line stroke="red" x1="953.13" y1="277.06" x2="953.13" y2="328.13"/>
  <line stroke="blue" x1="953.13" y1="277.06" x2="968.75" y2="286.08"/>
  <line stroke="red" x1="875" y1="468.75" x2="875" y2="500"/>
  <line stroke="red" x1="875" y1="468.75" x2="906.25" y2="468.75"/>
  <line stroke="red" x1="406.25" y1="906.25" x2="437.5" y2="937.5"/>
  <line stroke="red" x1="902.06" y1="421.87" x2="911.08" y2="406.25"/>
  <line stroke="blue" x1="902.06" y1="421.87" x2="953.13" y2="421.87"/>
  <line stroke="red" x1="902.06" y1="421.87" x2="929.13" y2="406.25" fold-step="4"/>
  <line stroke="blue" x1="15.63" y1="995.81" x2="0" y2="1000" fold-step="9"/>
  <line stroke="red" x1="15.63" y1="995.81" x2="18.04" y2="1000"/>
  <line stroke="blue" x1="93.75" y1="991.63" x2="78.12" y2="995.81" fold-step="9"/>
  <line stroke="red" x1="93.75" y1="991.63" x2="109.38" y2="995.81" fold-step="9"/>
  <line stroke="red" x1="93.75" y1="991.63" x2="93.75" y2="1000"/>
  <line stroke="blue" x1="46.87" y1="995.81" x2="44.46" y2="1000"/>
  <line stroke="red" x1="46.87" y1="995.81" x2="62.5" y2="1000" fold-step="9"/>
  <line stroke="blue" x1="843.75" y1="531.25" x2="875" y2="500"/>
  <line stroke="red" x1="843.75" y1="531.25" x2="865.85" y2="531.25"/>
  <line stroke="red" x1="911.08" y1="406.25" x2="929.13" y2="406.25"/>
  <line stroke="red" x1="78.12" y1="995.81" x2="80.54" y2="1000"/>
  <line stroke="blue" x1="78.12" y1="995.81" x2="62.5" y2="1000" fold-step="9"/>
  <line stroke="black" x1="0" y1="1000" x2="18.04" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="125"/>
  <line stroke="black" x1="18.04" y1="1000" x2="31.25" y2="1000"/>
  <line stroke="black" x1="31.25" y1="1000" x2="44.46" y2="1000"/>
  <line stroke="red" x1="781.25" y1="625" x2="812.5" y2="625"/>
  <line stroke="red" x1="781.25" y1="625" x2="781.25" y2="718.75"/>
  <line stroke="red" x1="968.75" y1="250" x2="968.75" y2="286.08"/>
  <line stroke="blue" x1="968.75" y1="250" x2="1000" y2="250"/>
  <line stroke="black" x1="44.46" y1="1000" x2="62.5" y2="1000"/>
  <line stroke="blue" x1="109.38" y1="995.81" x2="106.96" y2="1000"/>
  <line stroke="red" x1="109.38" y1="995.81" x2="125" y2="1000" fold-step="9"/>
  <line stroke="black" x1="62.5" y1="1000" x2="80.54" y2="1000"/>
  <line stroke="black" x1="80.54" y1="1000" x2="93.75" y2="1000"/>
  <line stroke="black" x1="93.75" y1="1000" x2="106.96" y2="1000"/>
  <line stroke="black" x1="106.96" y1="1000" x2="125" y2="1000"/>
  <line stroke="black" x1="125" y1="1000" x2="500" y2="1000"/>
  <line stroke="blue" x1="875" y1="500" x2="906.25" y2="468.75"/>
  <line stroke="blue" x1="875" y1="500" x2="881.47" y2="515.62" fold-step="8"/>
  <line stroke="red" x1="875" y1="500" x2="890.62" y2="506.47" fold-step="8"/>
  <line stroke="blue" x1="875" y1="500" x2="890.62" y2="515.62"/>
  <line stroke="red" x1="875" y1="500" x2="906.25" y2="500"/>
  <line stroke="black" x1="1000" y1="125" x2="1000" y2="250"/>
  <line stroke="blue" x1="953.13" y1="328.13" x2="968.75" y2="343.75"/>
  <line stroke="red" x1="902.06" y1="453.13" x2="909.31" y2="465.69"/>
  <line stroke="blue" x1="902.06" y1="453.13" x2="921.87" y2="453.13"/>
  <line stroke="blue" x1="902.06" y1="453.13" x2="914.62" y2="460.38" fold-step="4"/>
  <line stroke="red" x1="968.75" y1="286.08" x2="968.75" y2="304.13"/>
  <line stroke="red" x1="968.75" y1="286.08" x2="984.37" y2="277.06"/>
  <line stroke="red" x1="929.13" y1="406.25" x2="968.75" y2="406.25"/>
  <line stroke="blue" x1="968.75" y1="304.13" x2="984.37" y2="277.06" fold-step="4"/>
  <line stroke="red" x1="968.75" y1="304.13" x2="968.75" y2="343.75"/>
  <line stroke="red" x1="865.85" y1="531.25" x2="887.94" y2="531.25"/>
  <line stroke="blue" x1="865.85" y1="531.25" x2="881.47" y2="515.62"/>
  <line stroke="red" x1="865.85" y1="531.25" x2="881.47" y2="546.88"/>
  <line stroke="red" x1="906.25" y1="468.75" x2="909.31" y2="465.69"/>
  <line stroke="red" x1="906.25" y1="468.75" x2="906.25" y2="500"/>
  <line stroke="red" x1="881.47" y1="515.62" x2="890.62" y2="515.62"/>
  <line stroke="blue" x1="881.47" y1="515.62" x2="887.94" y2="531.25" fold-step="8"/>
  <line stroke="blue" x1="909.31" y1="465.69" x2="921.88" y2="472.94"/>
  <line stroke="red" x1="909.31" y1="465.69" x2="914.62" y2="460.38"/>
  <line stroke="blue" x1="984.37" y1="277.06" x2="1000" y2="250" fold-step="4"/>
  <line stroke="blue" x1="984.37" y1="277.06" x2="984.37" y2="359.37"/>
  <line stroke="red" x1="914.62" y1="460.38" x2="921.87" y2="453.13"/>
  <line stroke="red" x1="914.62" y1="460.38" x2="921.87" y2="472.94" fold-step="4"/>
  <line stroke="blue" x1="890.62" y1="506.47" x2="890.62" y2="515.62"/>
  <line stroke="red" x1="890.62" y1="506.47" x2="906.25" y2="500"/>
  <line stroke="red" x1="890.62" y1="506.47" x2="906.25" y2="522.1" fold-step="8"/>
  <line stroke="blue" x1="812.5" y1="625" x2="843.75" y2="625"/>
  <line stroke="blue" x1="812.5" y1="625" x2="812.5" y2="687.5"/>
  <line stroke="red" x1="921.87" y1="453.13" x2="921.87" y2="472.94"/>
  <line stroke="red" x1="921.87" y1="453.13" x2="937.5" y2="437.5"/>
  <line stroke="red" x1="968.75" y1="343.75" x2="984.37" y2="359.37"/>
  <line stroke="blue" x1="890.62" y1="515.62" x2="906.25" y2="531.25"/>
  <line stroke="black" x1="1000" y1="250" x2="1000" y2="375"/>
  <line stroke="red" x1="843.75" y1="593.75" x2="865.85" y2="593.75"/>
  <line stroke="red" x1="843.75" y1="593.75" x2="843.75" y2="625"/>
  <line stroke="blue" x1="843.75" y1="593.75" x2="875" y2="625"/>
  <line stroke="blue" x1="437.5" y1="937.5" x2="468.75" y2="968.75"/>
  <line stroke="blue" x1="937.5" y1="437.5" x2="953.13" y2="421.87"/>
  <line stroke="blue" x1="937.5" y1="437.5" x2="937.5" y2="500"/>
  <line stroke="red" x1="887.94" y1="531.25" x2="881.47" y2="546.88" fold-step="8"/>
  <line stroke="red" x1="887.94" y1="531.25" x2="906.25" y2="531.25"/>
  <line stroke="red" x1="906.25" y1="500" x2="906.25" y2="522.1"/>
  <line stroke="blue" x1="906.25" y1="500" x2="921.88" y2="506.47"/>
  <line stroke="blue" x1="906.25" y1="500" x2="937.5" y2="500"/>
  <line stroke="red" x1="921.87" y1="472.94" x2="937.5" y2="500" fold-step="4"/>
  <line stroke="blue" x1="881.47" y1="546.88" x2="921.88" y2="546.88"/>
  <line stroke="red" x1="881.47" y1="546.88" x2="875" y2="562.5" fold-step="8"/>
  <line stroke="blue" x1="875" y1="562.5" x2="881.47" y2="578.12" fold-step="8"/>
  <line stroke="blue" x1="875" y1="562.5" x2="937.5" y2="562.5"/>
  <line stroke="blue" x1="953.13" y1="421.87" x2="968.75" y2="406.25"/>
  <line stroke="red" x1="953.13" y1="421.87" x2="953.13" y2="472.94"/>
  <line stroke="red" x1="906.25" y1="522.1" x2="906.25" y2="531.25"/>
  <line stroke="blue" x1="906.25" y1="522.1" x2="921.88" y2="506.47" fold-step="8"/>
  <line stroke="red" x1="984.37" y1="359.37" x2="1000" y2="375"/>
  <line stroke="red" x1="865.85" y1="593.75" x2="887.94" y2="593.75"/>
  <line stroke="blue" x1="865.85" y1="593.75" x2="881.47" y2="578.12"/>
  <line stroke="red" x1="865.85" y1="593.75" x2="881.47" y2="609.38"/>
  <line stroke="red" x1="843.75" y1="625" x2="843.75" y2="656.25"/>
  <line stroke="red" x1="843.75" y1="625" x2="875" y2="625"/>
  <line stroke="red" x1="906.25" y1="531.25" x2="921.88" y2="546.88"/>
  <line stroke="red" x1="968.75" y1="406.25" x2="984.37" y2="390.63"/>
  <line stroke="red" x1="968.75" y1="406.25" x2="968.75" y2="445.87"/>
  <line stroke="blue" x1="921.88" y1="506.47" x2="937.5" y2="500" fold-step="8"/>
  <line stroke="red" x1="921.88" y1="506.47" x2="921.88" y2="546.88"/>
  <line stroke="red" x1="881.47" y1="578.12" x2="953.13" y2="578.12"/>
  <line stroke="blue" x1="881.47" y1="578.12" x2="887.94" y2="593.75" fold-step="8"/>
  <line stroke="red" x1="984.37" y1="390.63" x2="1000" y2="375"/>
  <line stroke="blue" x1="984.37" y1="390.63" x2="984.37" y2="472.94"/>
  <line stroke="blue" x1="750" y1="750" x2="718.75" y2="781.25"/>
  <line stroke="red" x1="750" y1="750" x2="781.25" y2="718.75"/>
  <line stroke="blue" x1="750" y1="750" x2="875" y2="750"/>
  <line stroke="red" x1="718.75" y1="781.25" x2="687.5" y2="812.5"/>
  <line stroke="red" x1="718.75" y1="781.25" x2="852.12" y2="781.25"/>
  <line stroke="blue" x1="781.25" y1="718.75" x2="812.5" y2="687.5"/>
  <line stroke="red" x1="781.25" y1="718.75" x2="865.85" y2="718.75"/>
  <line stroke="blue" x1="937.5" y1="500" x2="953.13" y2="472.94" fold-step="4"/>
  <line stroke="red" x1="937.5" y1="500" x2="953.13" y2="506.47" fold-step="8"/>
  <line stroke="red" x1="937.5" y1="500" x2="968.75" y2="500"/>
  <line stroke="blue" x1="937.5" y1="500" x2="937.5" y2="562.5"/>
  <line stroke="blue" x1="953.13" y1="472.94" x2="968.75" y2="445.87" fold-step="4"/>
  <line stroke="blue" x1="953.13" y1="472.94" x2="968.75" y2="463.92"/>
  <line stroke="blue" x1="687.5" y1="812.5" x2="656.25" y2="843.75"/>
  <line stroke="blue" x1="687.5" y1="812.5" x2="875" y2="812.5"/>
  <line stroke="red" x1="812.5" y1="687.5" x2="843.75" y2="656.25"/>
  <line stroke="blue" x1="812.5" y1="687.5" x2="875" y2="687.5"/>
  <line stroke="red" x1="968.75" y1="445.87" x2="968.75" y2="463.92"/>
  <line stroke="red" x1="968.75" y1="445.87" x2="984.37" y2="472.94" fold-step="4"/>
  <line stroke="black" x1="1000" y1="375" x2="1000" y2="500"/>
  <line stroke="red" x1="887.94" y1="593.75" x2="881.47" y2="609.37" fold-step="8"/>
  <line stroke="red" x1="887.94" y1="593.75" x2="968.75" y2="593.75"/>
  <line stroke="red" x1="656.25" y1="843.75" x2="625" y2="875"/>
  <line stroke="red" x1="656.25" y1="843.75" x2="852.12" y2="843.75"/>
  <line stroke="blue" x1="843.75" y1="656.25" x2="875" y2="625"/>
  <line stroke="red" x1="843.75" y1="656.25" x2="865.85" y2="656.25"/>
  <line stroke="red" x1="881.47" y1="609.37" x2="875" y2="625" fold-step="8"/>
  <line stroke="blue" x1="881.47" y1="609.37" x2="984.37" y2="609.37"/>
  <line stroke="red" x1="921.88" y1="546.88" x2="937.5" y2="562.5"/>
  <line stroke="red" x1="968.75" y1="463.92" x2="968.75" y2="500"/>
  <line stroke="red" x1="968.75" y1="463.92" x2="984.37" y2="472.94"/>
  <line stroke="blue" x1="625" y1="875" x2="593.75" y2="906.25"/>
  <line stroke="blue" x1="625" y1="875" x2="875" y2="875"/>
  <line stroke="blue" x1="875" y1="625" x2="881.47" y2="640.63" fold-step="8"/>
  <line stroke="blue" x1="875" y1="625" x2="1000" y2="625"/>
  <line stroke="red" x1="468.75" y1="968.75" x2="500" y2="1000"/>
  <line stroke="red" x1="953.13" y1="506.47" x2="968.75" y2="500"/>
  <line stroke="red" x1="953.13" y1="506.47" x2="968.75" y2="522.1" fold-step="8"/>
  <line stroke="blue" x1="953.13" y1="506.47" x2="953.13" y2="578.12"/>
  <line stroke="red" x1="593.75" y1="906.25" x2="562.5" y2="937.5"/>
  <line stroke="red" x1="593.75" y1="906.25" x2="852.12" y2="906.25"/>
  <line stroke="blue" x1="865.85" y1="656.25" x2="881.47" y2="640.62"/>
  <line stroke="red" x1="865.85" y1="656.25" x2="881.47" y2="671.88"/>
  <line stroke="red" x1="865.85" y1="656.25" x2="887.94" y2="656.25"/>
  <line stroke="red" x1="881.47" y1="640.62" x2="984.38" y2="640.62"/>
  <line stroke="blue" x1="881.47" y1="640.63" x2="887.94" y2="656.25" fold-step="8"/>
  <line stroke="red" x1="968.75" y1="500" x2="968.75" y2="522.1"/>
  <line stroke="blue" x1="968.75" y1="500" x2="984.37" y2="506.47"/>
  <line stroke="blue" x1="968.75" y1="500" x2="1000" y2="500"/>
  <line stroke="red" x1="984.37" y1="472.94" x2="1000" y2="500" fold-step="4"/>
  <line stroke="blue" x1="562.5" y1="937.5" x2="531.25" y2="968.75"/>
  <line stroke="blue" x1="562.5" y1="937.5" x2="875" y2="937.5"/>
  <line stroke="blue" x1="937.5" y1="562.5" x2="953.13" y2="578.13"/>
  <line stroke="blue" x1="968.75" y1="522.1" x2="984.37" y2="506.47" fold-step="8"/>
  <line stroke="red" x1="968.75" y1="522.1" x2="968.75" y2="593.75"/>
  <line stroke="red" x1="887.94" y1="656.25" x2="881.47" y2="671.88" fold-step="8"/>
  <line stroke="red" x1="887.94" y1="656.25" x2="968.75" y2="656.25"/>
  <line stroke="red" x1="531.25" y1="968.75" x2="500" y2="1000"/>
  <line stroke="red" x1="531.25" y1="968.75" x2="852.12" y2="968.75"/>
  <line stroke="red" x1="984.37" y1="506.47" x2="984.37" y2="609.37"/>
  <line stroke="blue" x1="984.37" y1="506.47" x2="1000" y2="500" fold-step="8"/>
  <line stroke="red" x1="881.47" y1="671.88" x2="875" y2="687.5" fold-step="8"/>
  <line stroke="blue" x1="881.47" y1="671.88" x2="953.12" y2="671.88"/>
  <line stroke="blue" x1="875" y1="687.5" x2="881.47" y2="703.13" fold-step="8"/>
  <line stroke="blue" x1="875" y1="687.5" x2="937.5" y2="687.5"/>
  <line stroke="blue" x1="953.13" y1="578.13" x2="968.75" y2="593.75"/>
  <line stroke="black" x1="500" y1="1000" x2="875" y2="1000"/>
  <line stroke="black" x1="1000" y1="500" x2="1000" y2="625"/>
  <line stroke="blue" x1="865.85" y1="718.75" x2="881.47" y2="703.13"/>
  <line stroke="red" x1="865.85" y1="718.75" x2="887.94" y2="718.75"/>
  <line stroke="red" x1="865.85" y1="718.75" x2="881.47" y2="734.37"/>
  <line stroke="blue" x1="881.47" y1="703.13" x2="887.94" y2="718.75" fold-step="8"/>
  <line stroke="red" x1="881.47" y1="703.13" x2="921.87" y2="703.13"/>
  <line stroke="red" x1="968.75" y1="593.75" x2="984.37" y2="609.37"/>
  <line stroke="red" x1="887.94" y1="718.75" x2="881.47" y2="734.38" fold-step="8"/>
  <line stroke="red" x1="887.94" y1="718.75" x2="906.25" y2="718.75"/>
  <line stroke="blue" x1="881.47" y1="734.37" x2="890.63" y2="734.37"/>
  <line stroke="red" x1="881.47" y1="734.38" x2="875" y2="750" fold-step="8"/>
  <line stroke="blue" x1="875" y1="750" x2="890.63" y2="734.37"/>
  <line stroke="blue" x1="875" y1="750" x2="890.63" y2="743.53" fold-step="8"/>
  <line stroke="blue" x1="875" y1="750" x2="879.19" y2="765.62" fold-step="10"/>
  <line stroke="red" x1="875" y1="750" x2="890.63" y2="754.19" fold-step="10"/>
  <line stroke="blue" x1="875" y1="750" x2="890.62" y2="765.62"/>
  <line stroke="red" x1="875" y1="750" x2="906.25" y2="750"/>
  <line stroke="blue" x1="890.63" y1="734.37" x2="906.25" y2="718.75"/>
  <line stroke="red" x1="890.63" y1="734.37" x2="890.63" y2="743.53"/>
  <line stroke="red" x1="852.12" y1="781.25" x2="883.37" y2="781.25"/>
  <line stroke="blue" x1="852.12" y1="781.25" x2="879.19" y2="765.62"/>
  <line stroke="red" x1="852.12" y1="781.25" x2="879.19" y2="796.88"/>
  <line stroke="red" x1="906.25" y1="718.75" x2="921.87" y2="703.13"/>
  <line stroke="red" x1="906.25" y1="718.75" x2="906.25" y2="727.9"/>
  <line stroke="red" x1="984.37" y1="609.37" x2="1000" y2="625"/>
  <line stroke="blue" x1="921.87" y1="703.13" x2="921.87" y2="743.53"/>
  <line stroke="red" x1="921.87" y1="703.13" x2="937.5" y2="687.5"/>
  <line stroke="blue" x1="890.63" y1="743.53" x2="906.25" y2="727.9" fold-step="8"/>
  <line stroke="blue" x1="890.63" y1="743.53" x2="906.25" y2="750"/>
  <line stroke="red" x1="906.25" y1="727.9" x2="906.25" y2="750"/>
  <line stroke="red" x1="906.25" y1="727.9" x2="921.87" y2="743.53" fold-step="8"/>
  <line stroke="blue" x1="937.5" y1="687.5" x2="953.12" y2="671.88"/>
  <line stroke="blue" x1="937.5" y1="687.5" x2="937.5" y2="750"/>
  <line stroke="red" x1="879.19" y1="765.62" x2="890.62" y2="765.62"/>
  <line stroke="blue" x1="879.19" y1="765.62" x2="883.37" y2="781.25" fold-step="10"/>
  <line stroke="blue" x1="953.12" y1="671.88" x2="968.75" y2="656.25"/>
  <line stroke="red" x1="953.12" y1="671.88" x2="953.12" y2="743.53"/>
  <line stroke="blue" x1="890.62" y1="754.19" x2="890.62" y2="765.62"/>
  <line stroke="red" x1="890.62" y1="754.19" x2="906.25" y2="750"/>
  <line stroke="red" x1="890.62" y1="754.19" x2="906.25" y2="781.25" fold-step="10"/>
  <line stroke="red" x1="968.75" y1="656.25" x2="984.38" y2="640.62"/>
  <line stroke="red" x1="968.75" y1="656.25" x2="968.75" y2="727.9"/>
  <line stroke="blue" x1="890.62" y1="765.62" x2="906.25" y2="781.25"/>
  <line stroke="red" x1="984.38" y1="640.62" x2="1000" y2="625"/>
  <line stroke="blue" x1="984.38" y1="640.62" x2="984.38" y2="743.53"/>
  <line stroke="red" x1="906.25" y1="750" x2="921.87" y2="743.53"/>
  <line stroke="blue" x1="906.25" y1="750" x2="921.87" y2="754.19"/>
  <line stroke="red" x1="906.25" y1="750" x2="906.25" y2="781.25"/>
  <line stroke="blue" x1="906.25" y1="750" x2="937.5" y2="750"/>
  <line stroke="black" x1="1000" y1="625" x2="1000" y2="750"/>
  <line stroke="red" x1="883.37" y1="781.25" x2="879.19" y2="796.87" fold-step="10"/>
  <line stroke="red" x1="883.37" y1="781.25" x2="906.25" y2="781.25"/>
  <line stroke="red" x1="921.87" y1="743.53" x2="937.5" y2="750" fold-step="8"/>
  <line stroke="red" x1="879.19" y1="796.87" x2="875" y2="812.5" fold-step="10"/>
  <line stroke="blue" x1="879.19" y1="796.87" x2="921.87" y2="796.87"/>
  <line stroke="blue" x1="921.87" y1="754.19" x2="937.5" y2="750" fold-step="10"/>
  <line stroke="blue" x1="921.87" y1="754.19" x2="906.25" y2="781.25" fold-step="10"/>
  <line stroke="red" x1="921.87" y1="754.19" x2="921.87" y2="796.87"/>
  <line stroke="blue" x1="875" y1="812.5" x2="879.19" y2="828.13" fold-step="10"/>
  <line stroke="blue" x1="875" y1="812.5" x2="937.5" y2="812.5"/>
  <line stroke="red" x1="906.25" y1="781.25" x2="921.87" y2="796.87"/>
  <line stroke="red" x1="852.12" y1="843.75" x2="883.37" y2="843.75"/>
  <line stroke="blue" x1="852.12" y1="843.75" x2="879.19" y2="828.13"/>
  <line stroke="red" x1="852.12" y1="843.75" x2="879.19" y2="859.37"/>
  <line stroke="blue" x1="937.5" y1="750" x2="953.12" y2="743.53" fold-step="8"/>
  <line stroke="red" x1="937.5" y1="750" x2="953.13" y2="754.19" fold-step="10"/>
  <line stroke="red" x1="937.5" y1="750" x2="968.75" y2="750"/>
  <line stroke="blue" x1="937.5" y1="750" x2="937.5" y2="812.5"/>
  <line stroke="blue" x1="879.19" y1="828.13" x2="883.37" y2="843.75" fold-step="10"/>
  <line stroke="red" x1="879.19" y1="828.13" x2="953.13" y2="828.13"/>
  <line stroke="blue" x1="953.12" y1="743.53" x2="968.75" y2="727.9" fold-step="8"/>
  <line stroke="blue" x1="953.12" y1="743.53" x2="968.75" y2="750"/>
  <line stroke="red" x1="968.75" y1="727.9" x2="968.75" y2="750"/>
  <line stroke="red" x1="968.75" y1="727.9" x2="984.38" y2="743.53" fold-step="8"/>
  <line stroke="red" x1="953.13" y1="754.19" x2="968.75" y2="750"/>
  <line stroke="red" x1="953.13" y1="754.19" x2="968.75" y2="781.25" fold-step="10"/>
  <line stroke="blue" x1="953.13" y1="754.19" x2="953.13" y2="828.13"/>
  <line stroke="red" x1="921.87" y1="796.87" x2="937.5" y2="812.5"/>
  <line stroke="red" x1="883.37" y1="843.75" x2="879.19" y2="859.37" fold-step="10"/>
  <line stroke="red" x1="883.37" y1="843.75" x2="968.75" y2="843.75"/>
  <line stroke="red" x1="968.75" y1="750" x2="984.38" y2="743.53"/>
  <line stroke="blue" x1="968.75" y1="750" x2="984.37" y2="754.19"/>
  <line stroke="red" x1="968.75" y1="750" x2="968.75" y2="781.25"/>
  <line stroke="blue" x1="968.75" y1="750" x2="1000" y2="750"/>
  <line stroke="red" x1="879.19" y1="859.37" x2="875" y2="875" fold-step="10"/>
  <line stroke="blue" x1="879.19" y1="859.37" x2="984.37" y2="859.37"/>
  <line stroke="red" x1="984.38" y1="743.53" x2="1000" y2="750" fold-step="8"/>
  <line stroke="blue" x1="875" y1="875" x2="879.19" y2="890.63" fold-step="10"/>
  <line stroke="blue" x1="875" y1="875" x2="1000" y2="875"/>
  <line stroke="blue" x1="984.37" y1="754.19" x2="968.75" y2="781.25" fold-step="10"/>
  <line stroke="blue" x1="984.37" y1="754.19" x2="1000" y2="750" fold-step="10"/>
  <line stroke="red" x1="984.37" y1="754.19" x2="984.37" y2="859.37"/>
  <line stroke="blue" x1="937.5" y1="812.5" x2="953.13" y2="828.13"/>
  <line stroke="red" x1="852.12" y1="906.25" x2="883.37" y2="906.25"/>
  <line stroke="blue" x1="852.12" y1="906.25" x2="879.19" y2="890.63"/>
  <line stroke="red" x1="852.12" y1="906.25" x2="879.19" y2="921.87"/>
  <line stroke="red" x1="968.75" y1="781.25" x2="968.75" y2="843.75"/>
  <line stroke="black" x1="1000" y1="750" x2="1000" y2="875"/>
  <line stroke="red" x1="879.19" y1="890.63" x2="984.37" y2="890.63"/>
  <line stroke="blue" x1="879.19" y1="890.63" x2="883.37" y2="906.25" fold-step="10"/>
  <line stroke="blue" x1="953.13" y1="828.13" x2="968.75" y2="843.75"/>
  <line stroke="red" x1="883.37" y1="906.25" x2="879.19" y2="921.87" fold-step="10"/>
  <line stroke="red" x1="883.37" y1="906.25" x2="968.75" y2="906.25"/>
  <line stroke="blue" x1="879.19" y1="921.87" x2="953.13" y2="921.87"/>
  <line stroke="red" x1="879.19" y1="921.87" x2="875" y2="937.5" fold-step="10"/>
  <line stroke="blue" x1="875" y1="937.5" x2="879.19" y2="953.12" fold-step="10"/>
  <line stroke="blue" x1="875" y1="937.5" x2="937.5" y2="937.5"/>
  <line stroke="red" x1="968.75" y1="843.75" x2="984.37" y2="859.37"/>
  <line stroke="red" x1="852.12" y1="968.75" x2="879.19" y2="953.12"/>
  <line stroke="blue" x1="852.12" y1="968.75" x2="879.19" y2="984.38"/>
  <line stroke="red" x1="852.12" y1="968.75" x2="883.37" y2="968.75"/>
  <line stroke="blue" x1="879.19" y1="953.12" x2="883.37" y2="968.75" fold-step="10"/>
  <line stroke="blue" x1="879.19" y1="953.12" x2="921.88" y2="953.12"/>
  <line stroke="red" x1="984.37" y1="859.37" x2="1000" y2="875"/>
  <line stroke="red" x1="883.37" y1="968.75" x2="879.19" y2="984.38" fold-step="10"/>
  <line stroke="red" x1="883.37" y1="968.75" x2="906.25" y2="968.75"/>
  <line stroke="red" x1="879.19" y1="984.38" x2="875" y2="1000" fold-step="10"/>
  <line stroke="red" x1="879.19" y1="984.38" x2="890.62" y2="984.38"/>
  <line stroke="red" x1="937.5" y1="937.5" x2="921.88" y2="953.12"/>
  <line stroke="blue" x1="937.5" y1="937.5" x2="953.13" y2="921.87"/>
  <line stroke="blue" x1="937.5" y1="937.5" x2="937.5" y2="1000"/>
  <line stroke="red" x1="921.88" y1="953.12" x2="906.25" y2="968.75"/>
  <line stroke="red" x1="921.88" y1="953.12" x2="921.88" y2="995.81"/>
  <line stroke="red" x1="953.13" y1="921.87" x2="953.13" y2="995.81"/>
  <line stroke="blue" x1="953.13" y1="921.87" x2="968.75" y2="906.25"/>
  <line stroke="blue" x1="906.25" y1="968.75" x2="890.62" y2="984.38"/>
  <line stroke="red" x1="906.25" y1="968.75" x2="906.25" y2="991.63"/>
  <line stroke="red" x1="968.75" y1="906.25" x2="984.37" y2="890.63"/>
  <line stroke="red" x1="968.75" y1="906.25" x2="968.75" y2="991.63"/>
  <line stroke="blue" x1="890.62" y1="984.38" x2="875" y2="1000"/>
  <line stroke="blue" x1="984.37" y1="890.63" x2="984.37" y2="995.81"/>
  <line stroke="red" x1="984.37" y1="890.63" x2="1000" y2="875"/>
  <line stroke="blue" x1="890.62" y1="984.38" x2="890.62" y2="995.81"/>
  <line stroke="blue" x1="875" y1="1000" x2="890.62" y2="995.81" fold-step="10"/>
  <line stroke="black" x1="875" y1="1000" x2="893.04" y2="1000"/>
  <line stroke="black" x1="1000" y1="875" x2="1000" y2="1000"/>
  <line stroke="red" x1="890.62" y1="995.81" x2="893.04" y2="1000"/>
  <line stroke="blue" x1="890.62" y1="995.81" x2="906.25" y2="991.63" fold-step="10"/>
  <line stroke="black" x1="893.04" y1="1000" x2="906.25" y2="1000"/>
  <line stroke="red" x1="906.25" y1="991.63" x2="906.25" y2="1000"/>
  <line stroke="red" x1="906.25" y1="991.63" x2="921.88" y2="995.81" fold-step="10"/>
  <line stroke="black" x1="906.25" y1="1000" x2="919.46" y2="1000"/>
  <line stroke="red" x1="921.88" y1="995.81" x2="937.5" y2="1000" fold-step="10"/>
  <line stroke="blue" x1="921.88" y1="995.81" x2="919.46" y2="1000"/>
  <line stroke="black" x1="919.46" y1="1000" x2="937.5" y2="1000"/>
  <line stroke="blue" x1="937.5" y1="1000" x2="953.12" y2="995.81" fold-step="10"/>
  <line stroke="black" x1="937.5" y1="1000" x2="955.54" y2="1000"/>
  <line stroke="blue" x1="953.12" y1="995.81" x2="968.75" y2="991.63" fold-step="10"/>
  <line stroke="blue" x1="953.13" y1="995.81" x2="955.54" y2="1000"/>
  <line stroke="black" x1="955.54" y1="1000" x2="968.75" y2="1000"/>
  <line stroke="red" x1="968.75" y1="991.63" x2="968.75" y2="1000"/>
  <line stroke="red" x1="968.75" y1="991.63" x2="984.37" y2="995.81" fold-step="10"/>
  <line stroke="black" x1="968.75" y1="1000" x2="981.96" y2="1000"/>
  <line stroke="red" x1="984.37" y1="995.81" x2="981.96" y2="1000"/>
  <line stroke="red" x1="984.37" y1="995.81" x2="1000" y2="1000" fold-step="10"/>
  <line stroke="black" x1="981.96" y1="1000" x2="1000" y2="1000"/>
</svg>""",
)

SCORPION_INSCRIBED_TEST_EXAMPLE = BoxPleatingShapingExample(
    base_example=SCORPION,
    central_crease_removed=True,
    rectangles=[
        shaping_elements.ShapingFlap.from_legacy_rectangle_direct(
            rect, SCORPION.grid_size
        )
        for rect in SCORPION_SHAPING_UNCLIPPED.rectangles
    ],
    rivers=SCORPION_SHAPING_UNCLIPPED.rivers,
    inscribed_rectangle_shaping_params=[
        inscribed_rectangle_shaping.InscribedShapingParamsForElement(
            element_name="tail",
            inscribed_params=[
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=1,
                    fold_order=_FOLD_ORDERS["MV"],
                    fold_percent=0.2,
                ),
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=4,
                    fold_order=_FOLD_ORDERS["MV"],
                    fold_percent=0.2,
                ),
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=7,
                    fold_order=_FOLD_ORDERS["MV"],
                    fold_percent=0.2,
                ),
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=10,
                    fold_order=_FOLD_ORDERS["MV"],
                    fold_percent=0.2,
                ),
            ],
        ),
        inscribed_rectangle_shaping.InscribedShapingParamsForElement(
            element_name="limb6_left",
            inscribed_params=[
                inscribed_rectangle_shaping.InscribedRiverParams(
                    depth=1,
                    fold_order=_FOLD_ORDERS["MV"],
                    fold_percent=0.3,
                    bank_index="SECOND",
                ),
            ],
        ),
        inscribed_rectangle_shaping.InscribedShapingParamsForElement(
            element_name="limb6_right",
            inscribed_params=[
                inscribed_rectangle_shaping.InscribedRiverParams(
                    depth=1,
                    fold_order=_FOLD_ORDERS["VM"],
                    fold_percent=0.3,
                    bank_index="SECOND",
                ),
            ],
        ),
    ],
    inscribed_rectangle_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <rect stroke="black" x="0" y="0" width="1000" height="1000"/>
  <line stroke="red" x1="0" y1="0" x2="31" y2="31"/>
  <line stroke="blue" x1="31" y1="31" x2="62" y2="62"/>
  <line stroke="red" x1="31" y1="31" x2="125" y2="31"/>
  <line stroke="red" x1="31" y1="31" x2="31" y2="219"/>
  <line stroke="red" x1="62" y1="62" x2="94" y2="94"/>
  <line stroke="blue" x1="62" y1="62" x2="125" y2="62"/>
  <line stroke="blue" x1="62" y1="62" x2="62" y2="219"/>
  <line stroke="blue" x1="125" y1="0" x2="125" y2="31"/>
  <line stroke="red" x1="125" y1="31" x2="125" y2="62"/>
  <line stroke="red" x1="125" y1="31" x2="219" y2="31"/>
  <line stroke="red" x1="94" y1="94" x2="125" y2="94"/>
  <line stroke="blue" x1="94" y1="94" x2="125" y2="125"/>
  <line stroke="red" x1="94" y1="94" x2="94" y2="219"/>
  <line stroke="blue" x1="125" y1="62" x2="125" y2="94"/>
  <line stroke="blue" x1="125" y1="62" x2="188" y2="62"/>
  <line stroke="red" x1="125" y1="94" x2="125" y2="125"/>
  <line stroke="red" x1="125" y1="94" x2="156" y2="94"/>
  <line stroke="blue" x1="125" y1="125" x2="156" y2="94"/>
  <line stroke="blue" x1="125" y1="125" x2="125" y2="219"/>
  <line stroke="red" x1="156" y1="94" x2="188" y2="62"/>
  <line stroke="red" x1="156" y1="94" x2="156" y2="219"/>
  <line stroke="blue" x1="188" y1="62" x2="219" y2="31"/>
  <line stroke="blue" x1="188" y1="62" x2="188" y2="219"/>
  <line stroke="red" x1="0" y1="219" x2="31" y2="219" opacity="0.3"/>
  <line stroke="blue" x1="31" y1="219" x2="62" y2="219" opacity="0.3"/>
  <line stroke="red" x1="31" y1="219" x2="31" y2="250"/>
  <line stroke="red" x1="219" y1="31" x2="250" y2="0"/>
  <line stroke="red" x1="219" y1="31" x2="219" y2="219"/>
  <line stroke="red" x1="62" y1="219" x2="94" y2="219" opacity="0.3"/>
  <line stroke="blue" x1="62" y1="219" x2="62" y2="250"/>
  <line stroke="blue" x1="94" y1="219" x2="125" y2="219" opacity="0.3"/>
  <line stroke="red" x1="94" y1="219" x2="94" y2="250"/>
  <line stroke="blue" x1="0" y1="250" x2="31" y2="250"/>
  <line stroke="red" x1="250" y1="0" x2="281" y2="31"/>
  <line stroke="blue" x1="250" y1="0" x2="250" y2="219"/>
  <line stroke="red" x1="31" y1="250" x2="62" y2="250"/>
  <line stroke="red" x1="31" y1="250" x2="31" y2="344"/>
  <line stroke="red" x1="125" y1="219" x2="156" y2="219" opacity="0.3"/>
  <line stroke="blue" x1="125" y1="219" x2="125" y2="250"/>
  <line stroke="blue" x1="62" y1="250" x2="94" y2="250"/>
  <line stroke="blue" x1="62" y1="250" x2="62" y2="312"/>
  <line stroke="red" x1="94" y1="250" x2="125" y2="250"/>
  <line stroke="red" x1="94" y1="250" x2="94" y2="281"/>
  <line stroke="blue" x1="156" y1="219" x2="188" y2="219" opacity="0.3"/>
  <line stroke="red" x1="156" y1="219" x2="156" y2="250"/>
  <line stroke="blue" x1="125" y1="250" x2="156" y2="250"/>
  <line stroke="blue" x1="125" y1="250" x2="94" y2="281"/>
  <line stroke="red" x1="125" y1="250" x2="125" y2="281"/>
  <line stroke="blue" x1="125" y1="250" x2="156" y2="281"/>
  <line stroke="blue" x1="281" y1="31" x2="312" y2="62"/>
  <line stroke="red" x1="281" y1="31" x2="281" y2="219"/>
  <line stroke="red" x1="281" y1="31" x2="469" y2="31"/>
  <line stroke="red" x1="188" y1="219" x2="219" y2="219" opacity="0.3"/>
  <line stroke="blue" x1="188" y1="219" x2="188" y2="250"/>
  <line stroke="red" x1="156" y1="250" x2="188" y2="250"/>
  <line stroke="red" x1="156" y1="250" x2="156" y2="281"/>
  <line stroke="red" x1="94" y1="281" x2="125" y2="281"/>
  <line stroke="red" x1="94" y1="281" x2="62" y2="312"/>
  <line stroke="red" x1="125" y1="281" x2="156" y2="281"/>
  <line stroke="blue" x1="125" y1="281" x2="125" y2="312"/>
  <line stroke="red" x1="219" y1="219" x2="219" y2="250"/>
  <line stroke="blue" x1="219" y1="219" x2="250" y2="219" opacity="0.3"/>
  <line stroke="blue" x1="188" y1="250" x2="219" y2="250"/>
  <line stroke="blue" x1="188" y1="250" x2="188" y2="312"/>
  <line stroke="blue" x1="62" y1="312" x2="125" y2="312"/>
  <line stroke="blue" x1="62" y1="312" x2="31" y2="344"/>
  <line stroke="red" x1="312" y1="62" x2="344" y2="94"/>
  <line stroke="blue" x1="312" y1="62" x2="312" y2="219"/>
  <line stroke="blue" x1="312" y1="62" x2="469" y2="62"/>
  <line stroke="red" x1="156" y1="281" x2="188" y2="312"/>
  <line stroke="red" x1="219" y1="250" x2="250" y2="250"/>
  <line stroke="red" x1="219" y1="250" x2="219" y2="344"/>
  <line stroke="blue" x1="250" y1="219" x2="250" y2="250"/>
  <line stroke="red" x1="250" y1="219" x2="281" y2="219" opacity="0.3"/>
  <line stroke="blue" x1="125" y1="312" x2="188" y2="312"/>
  <line stroke="red" x1="125" y1="312" x2="125" y2="344"/>
  <line stroke="red" x1="31" y1="344" x2="125" y2="344"/>
  <line stroke="red" x1="31" y1="344" x2="0" y2="375"/>
  <line stroke="blue" x1="250" y1="250" x2="281" y2="250"/>
  <line stroke="blue" x1="250" y1="250" x2="250" y2="375"/>
  <line stroke="red" x1="281" y1="219" x2="281" y2="250"/>
  <line stroke="blue" x1="281" y1="219" x2="312" y2="219" opacity="0.3"/>
  <line stroke="blue" x1="344" y1="94" x2="375" y2="125"/>
  <line stroke="red" x1="344" y1="94" x2="344" y2="219"/>
  <line stroke="red" x1="344" y1="94" x2="469" y2="94"/>
  <line stroke="blue" x1="188" y1="312" x2="219" y2="344"/>
  <line stroke="blue" x1="125" y1="344" x2="125" y2="375"/>
  <line stroke="red" x1="125" y1="344" x2="219" y2="344"/>
  <line stroke="blue" x1="0" y1="375" x2="125" y2="375"/>
  <line stroke="red" x1="0" y1="375" x2="31" y2="406"/>
  <line stroke="red" x1="281" y1="250" x2="312" y2="250"/>
  <line stroke="red" x1="281" y1="250" x2="281" y2="344"/>
  <line stroke="blue" x1="312" y1="219" x2="312" y2="250"/>
  <line stroke="red" x1="312" y1="219" x2="344" y2="219" opacity="0.3"/>
  <line stroke="blue" x1="125" y1="375" x2="125" y2="406"/>
  <line stroke="red" x1="125" y1="375" x2="250" y2="375"/>
  <line stroke="blue" x1="375" y1="125" x2="375" y2="219"/>
  <line stroke="red" x1="375" y1="125" x2="406" y2="156"/>
  <line stroke="blue" x1="375" y1="125" x2="469" y2="125"/>
  <line stroke="blue" x1="312" y1="250" x2="344" y2="250"/>
  <line stroke="blue" x1="312" y1="250" x2="312" y2="312"/>
  <line stroke="red" x1="31" y1="406" x2="125" y2="406"/>
  <line stroke="blue" x1="31" y1="406" x2="62" y2="438"/>
  <line stroke="red" x1="31" y1="406" x2="31" y2="500"/>
  <line stroke="red" x1="219" y1="344" x2="250" y2="375"/>
  <line stroke="red" x1="344" y1="219" x2="344" y2="250"/>
  <line stroke="blue" x1="344" y1="219" x2="375" y2="219" opacity="0.3"/>
  <line stroke="red" x1="125" y1="406" x2="125" y2="438"/>
  <line stroke="red" x1="125" y1="406" x2="219" y2="406"/>
  <line stroke="red" x1="344" y1="250" x2="344" y2="281"/>
  <line stroke="red" x1="344" y1="250" x2="375" y2="250"/>
  <line stroke="blue" x1="375" y1="219" x2="375" y2="250"/>
  <line stroke="red" x1="375" y1="219" x2="406" y2="219" opacity="0.3"/>
  <line stroke="red" x1="406" y1="156" x2="406" y2="219"/>
  <line stroke="blue" x1="406" y1="156" x2="438" y2="188"/>
  <line stroke="red" x1="406" y1="156" x2="469" y2="156"/>
  <line stroke="blue" x1="62" y1="438" x2="125" y2="438"/>
  <line stroke="red" x1="62" y1="438" x2="94" y2="469"/>
  <line stroke="blue" x1="62" y1="438" x2="62" y2="500"/>
  <line stroke="blue" x1="312" y1="312" x2="281" y2="344"/>
  <line stroke="red" x1="312" y1="312" x2="344" y2="281"/>
  <line stroke="blue" x1="312" y1="312" x2="375" y2="312"/>
  <line stroke="red" x1="281" y1="344" x2="250" y2="375"/>
  <line stroke="red" x1="281" y1="344" x2="406" y2="344"/>
  <line stroke="blue" x1="344" y1="281" x2="375" y2="250"/>
  <line stroke="red" x1="344" y1="281" x2="406" y2="281"/>
  <line stroke="red" x1="250" y1="375" x2="219" y2="406"/>
  <line stroke="red" x1="250" y1="375" x2="281" y2="406"/>
  <line stroke="blue" x1="250" y1="375" x2="375" y2="375"/>
  <line stroke="blue" x1="250" y1="375" x2="250" y2="562"/>
  <line stroke="blue" x1="375" y1="250" x2="406" y2="281"/>
  <line stroke="blue" x1="125" y1="438" x2="188" y2="438"/>
  <line stroke="blue" x1="125" y1="438" x2="125" y2="469"/>
  <line stroke="blue" x1="219" y1="406" x2="188" y2="438"/>
  <line stroke="red" x1="219" y1="406" x2="219" y2="562"/>
  <line stroke="blue" x1="406" y1="219" x2="438" y2="219" opacity="0.3"/>
  <line stroke="red" x1="406" y1="219" x2="406" y2="281"/>
  <line stroke="blue" x1="469" y1="0" x2="469" y2="31" opacity="0.3"/>
  <line stroke="red" x1="469" y1="31" x2="469" y2="62" opacity="0.3"/>
  <line stroke="blue" x1="469" y1="31" x2="500" y2="0"/>
  <line stroke="red" x1="469" y1="31" x2="500" y2="62"/>
  <line stroke="red" x1="469" y1="31" x2="531" y2="31"/>
  <line stroke="blue" x1="469" y1="62" x2="469" y2="94" opacity="0.3"/>
  <line stroke="blue" x1="469" y1="62" x2="500" y2="62"/>
  <line stroke="red" x1="188" y1="438" x2="156" y2="469"/>
  <line stroke="blue" x1="188" y1="438" x2="188" y2="562"/>
  <line stroke="blue" x1="438" y1="188" x2="438" y2="219"/>
  <line stroke="blue" x1="438" y1="188" x2="469" y2="188"/>
  <line stroke="red" x1="438" y1="188" x2="469" y2="219"/>
  <line stroke="red" x1="94" y1="469" x2="125" y2="469"/>
  <line stroke="red" x1="94" y1="469" x2="94" y2="500"/>
  <line stroke="blue" x1="94" y1="469" x2="125" y2="500"/>
  <line stroke="red" x1="469" y1="94" x2="469" y2="125" opacity="0.3"/>
  <line stroke="blue" x1="469" y1="94" x2="500" y2="62"/>
  <line stroke="red" x1="469" y1="94" x2="500" y2="125"/>
  <line stroke="red" x1="469" y1="94" x2="531" y2="94"/>
  <line stroke="red" x1="125" y1="469" x2="156" y2="469"/>
  <line stroke="red" x1="125" y1="469" x2="125" y2="500"/>
  <line stroke="blue" x1="469" y1="125" x2="469" y2="156" opacity="0.3"/>
  <line stroke="blue" x1="469" y1="125" x2="500" y2="125"/>
  <line stroke="blue" x1="375" y1="312" x2="406" y2="281"/>
  <line stroke="red" x1="375" y1="312" x2="406" y2="312"/>
  <line stroke="blue" x1="375" y1="312" x2="406" y2="344"/>
  <line stroke="red" x1="438" y1="219" x2="469" y2="219" opacity="0.3"/>
  <line stroke="blue" x1="438" y1="219" x2="438" y2="281"/>
  <line stroke="blue" x1="156" y1="469" x2="125" y2="500"/>
  <line stroke="red" x1="156" y1="469" x2="156" y2="531"/>
  <line stroke="blue" x1="281" y1="406" x2="312" y2="438"/>
  <line stroke="red" x1="281" y1="406" x2="406" y2="406"/>
  <line stroke="red" x1="281" y1="406" x2="281" y2="562"/>
  <line stroke="red" x1="406" y1="281" x2="406" y2="312"/>
  <line stroke="red" x1="406" y1="281" x2="438" y2="281"/>
  <line stroke="red" x1="469" y1="156" x2="469" y2="188" opacity="0.3"/>
  <line stroke="blue" x1="469" y1="156" x2="500" y2="125"/>
  <line stroke="red" x1="469" y1="156" x2="500" y2="188"/>
  <line stroke="red" x1="469" y1="156" x2="531" y2="156"/>
  <line stroke="blue" x1="0" y1="500" x2="31" y2="500"/>
  <line stroke="blue" x1="500" y1="0" x2="531" y2="31"/>
  <line stroke="red" x1="31" y1="500" x2="62" y2="500"/>
  <line stroke="red" x1="31" y1="500" x2="31" y2="594"/>
  <line stroke="blue" x1="62" y1="500" x2="94" y2="500"/>
  <line stroke="blue" x1="62" y1="500" x2="62" y2="562"/>
  <line stroke="red" x1="500" y1="62" x2="531" y2="31"/>
  <line stroke="blue" x1="500" y1="62" x2="531" y2="62"/>
  <line stroke="blue" x1="500" y1="62" x2="531" y2="94"/>
  <line stroke="blue" x1="469" y1="188" x2="469" y2="219" opacity="0.3"/>
  <line stroke="blue" x1="469" y1="188" x2="500" y2="188"/>
  <line stroke="red" x1="94" y1="500" x2="125" y2="500"/>
  <line stroke="red" x1="94" y1="500" x2="94" y2="531"/>
  <line stroke="red" x1="406" y1="312" x2="406" y2="344"/>
  <line stroke="blue" x1="406" y1="312" x2="438" y2="312"/>
  <line stroke="blue" x1="125" y1="500" x2="94" y2="531"/>
  <line stroke="blue" x1="125" y1="500" x2="156" y2="531"/>
  <line stroke="red" x1="500" y1="125" x2="531" y2="94"/>
  <line stroke="blue" x1="500" y1="125" x2="531" y2="125"/>
  <line stroke="blue" x1="500" y1="125" x2="531" y2="156"/>
  <line stroke="blue" x1="469" y1="219" x2="500" y2="188"/>
  <line stroke="red" x1="469" y1="219" x2="469" y2="281"/>
  <line stroke="red" x1="469" y1="219" x2="531" y2="219"/>
  <line stroke="blue" x1="438" y1="281" x2="438" y2="312"/>
  <line stroke="blue" x1="438" y1="281" x2="469" y2="281"/>
  <line stroke="blue" x1="375" y1="375" x2="406" y2="344"/>
  <line stroke="red" x1="375" y1="375" x2="406" y2="375"/>
  <line stroke="blue" x1="375" y1="375" x2="406" y2="406"/>
  <line stroke="red" x1="531" y1="0" x2="531" y2="31" opacity="0.3"/>
  <line stroke="red" x1="406" y1="344" x2="406" y2="375"/>
  <line stroke="red" x1="406" y1="344" x2="438" y2="344"/>
  <line stroke="blue" x1="531" y1="31" x2="531" y2="62" opacity="0.3"/>
  <line stroke="red" x1="531" y1="31" x2="719" y2="31"/>
  <line stroke="red" x1="500" y1="188" x2="531" y2="156"/>
  <line stroke="blue" x1="500" y1="188" x2="531" y2="188"/>
  <line stroke="blue" x1="500" y1="188" x2="531" y2="219"/>
  <line stroke="red" x1="531" y1="62" x2="531" y2="94" opacity="0.3"/>
  <line stroke="blue" x1="531" y1="62" x2="688" y2="62"/>
  <line stroke="blue" x1="312" y1="438" x2="375" y2="438"/>
  <line stroke="red" x1="312" y1="438" x2="344" y2="469"/>
  <line stroke="blue" x1="312" y1="438" x2="312" y2="562"/>
  <line stroke="blue" x1="438" y1="312" x2="438" y2="344"/>
  <line stroke="red" x1="438" y1="312" x2="469" y2="312"/>
  <line stroke="red" x1="94" y1="531" x2="156" y2="531"/>
  <line stroke="red" x1="94" y1="531" x2="62" y2="562"/>
  <line stroke="blue" x1="531" y1="94" x2="531" y2="125" opacity="0.3"/>
  <line stroke="red" x1="531" y1="94" x2="656" y2="94"/>
  <line stroke="red" x1="531" y1="125" x2="531" y2="156" opacity="0.3"/>
  <line stroke="blue" x1="531" y1="125" x2="625" y2="125"/>
  <line stroke="red" x1="469" y1="281" x2="469" y2="312"/>
  <line stroke="red" x1="469" y1="281" x2="531" y2="281"/>
  <line stroke="red" x1="406" y1="375" x2="406" y2="406"/>
  <line stroke="blue" x1="406" y1="375" x2="438" y2="375"/>
  <line stroke="red" x1="156" y1="531" x2="188" y2="562"/>
  <line stroke="blue" x1="531" y1="156" x2="531" y2="188" opacity="0.3"/>
  <line stroke="red" x1="531" y1="156" x2="594" y2="156"/>
  <line stroke="blue" x1="438" y1="344" x2="438" y2="375"/>
  <line stroke="blue" x1="438" y1="344" x2="469" y2="344"/>
  <line stroke="red" x1="469" y1="312" x2="469" y2="344"/>
  <line stroke="blue" x1="469" y1="312" x2="531" y2="312"/>
  <line stroke="red" x1="531" y1="188" x2="531" y2="219" opacity="0.3"/>
  <line stroke="blue" x1="531" y1="188" x2="562" y2="188"/>
  <line stroke="blue" x1="62" y1="562" x2="188" y2="562"/>
  <line stroke="blue" x1="62" y1="562" x2="31" y2="594"/>
  <line stroke="blue" x1="406" y1="406" x2="375" y2="438"/>
  <line stroke="red" x1="406" y1="406" x2="406" y2="438"/>
  <line stroke="red" x1="406" y1="406" x2="438" y2="406"/>
  <line stroke="red" x1="531" y1="219" x2="562" y2="188"/>
  <line stroke="red" x1="531" y1="219" x2="531" y2="281"/>
  <line stroke="blue" x1="531" y1="219" x2="562" y2="219" opacity="0.3"/>
  <line stroke="red" x1="375" y1="438" x2="406" y2="438"/>
  <line stroke="blue" x1="375" y1="438" x2="406" y2="469"/>
  <line stroke="blue" x1="438" y1="375" x2="438" y2="406"/>
  <line stroke="red" x1="438" y1="375" x2="469" y2="375"/>
  <line stroke="red" x1="344" y1="469" x2="406" y2="469"/>
  <line stroke="blue" x1="344" y1="469" x2="375" y2="500"/>
  <line stroke="red" x1="344" y1="469" x2="344" y2="562"/>
  <line stroke="red" x1="469" y1="344" x2="469" y2="375"/>
  <line stroke="red" x1="469" y1="344" x2="531" y2="344"/>
  <line stroke="blue" x1="188" y1="562" x2="219" y2="562"/>
  <line stroke="red" x1="188" y1="562" x2="156" y2="594"/>
  <line stroke="blue" x1="188" y1="562" x2="188" y2="625"/>
  <line stroke="blue" x1="562" y1="188" x2="562" y2="219"/>
  <line stroke="blue" x1="562" y1="188" x2="594" y2="156"/>
  <line stroke="red" x1="31" y1="594" x2="156" y2="594"/>
  <line stroke="red" x1="31" y1="594" x2="0" y2="625"/>
  <line stroke="blue" x1="406" y1="438" x2="438" y2="438"/>
  <line stroke="red" x1="406" y1="438" x2="406" y2="469"/>
  <line stroke="blue" x1="438" y1="406" x2="438" y2="438"/>
  <line stroke="blue" x1="438" y1="406" x2="469" y2="406"/>
  <line stroke="red" x1="469" y1="375" x2="469" y2="406"/>
  <line stroke="blue" x1="469" y1="375" x2="531" y2="375"/>
  <line stroke="red" x1="531" y1="281" x2="531" y2="312"/>
  <line stroke="blue" x1="531" y1="281" x2="562" y2="281"/>
  <line stroke="red" x1="219" y1="562" x2="250" y2="562"/>
  <line stroke="red" x1="219" y1="562" x2="219" y2="625"/>
  <line stroke="blue" x1="562" y1="219" x2="562" y2="281"/>
  <line stroke="red" x1="562" y1="219" x2="594" y2="219" opacity="0.3"/>
  <line stroke="blue" x1="156" y1="594" x2="125" y2="625"/>
  <line stroke="red" x1="156" y1="594" x2="156" y2="625"/>
  <line stroke="red" x1="594" y1="156" x2="594" y2="219"/>
  <line stroke="red" x1="594" y1="156" x2="625" y2="125"/>
  <line stroke="blue" x1="250" y1="562" x2="281" y2="562"/>
  <line stroke="blue" x1="250" y1="562" x2="250" y2="625"/>
  <line stroke="red" x1="531" y1="312" x2="531" y2="344"/>
  <line stroke="red" x1="531" y1="312" x2="562" y2="312"/>
  <line stroke="blue" x1="438" y1="438" x2="438" y2="469"/>
  <line stroke="red" x1="438" y1="438" x2="469" y2="438"/>
  <line stroke="blue" x1="406" y1="469" x2="375" y2="500"/>
  <line stroke="red" x1="406" y1="469" x2="438" y2="469"/>
  <line stroke="red" x1="406" y1="469" x2="406" y2="500"/>
  <line stroke="red" x1="469" y1="406" x2="469" y2="438"/>
  <line stroke="red" x1="469" y1="406" x2="531" y2="406"/>
  <line stroke="blue" x1="0" y1="625" x2="125" y2="625"/>
  <line stroke="red" x1="0" y1="625" x2="31" y2="656"/>
  <line stroke="red" x1="375" y1="500" x2="406" y2="500"/>
  <line stroke="blue" x1="375" y1="500" x2="375" y2="562"/>
  <line stroke="red" x1="281" y1="562" x2="312" y2="562"/>
  <line stroke="red" x1="281" y1="562" x2="281" y2="625"/>
  <line stroke="blue" x1="562" y1="281" x2="562" y2="312"/>
  <line stroke="red" x1="562" y1="281" x2="594" y2="281"/>
  <line stroke="red" x1="531" y1="344" x2="531" y2="375"/>
  <line stroke="blue" x1="531" y1="344" x2="562" y2="344"/>
  <line stroke="red" x1="594" y1="219" x2="594" y2="281"/>
  <line stroke="blue" x1="594" y1="219" x2="625" y2="219" opacity="0.3"/>
  <line stroke="red" x1="125" y1="625" x2="156" y2="625"/>
  <line stroke="blue" x1="125" y1="625" x2="156" y2="656"/>
  <line stroke="blue" x1="625" y1="125" x2="625" y2="219"/>
  <line stroke="blue" x1="625" y1="125" x2="656" y2="94"/>
  <line stroke="blue" x1="438" y1="469" x2="469" y2="469"/>
  <line stroke="blue" x1="438" y1="469" x2="438" y2="500"/>
  <line stroke="red" x1="469" y1="438" x2="469" y2="469"/>
  <line stroke="blue" x1="469" y1="438" x2="531" y2="438"/>
  <line stroke="blue" x1="312" y1="562" x2="344" y2="562"/>
  <line stroke="blue" x1="312" y1="562" x2="312" y2="625"/>
  <line stroke="blue" x1="562" y1="312" x2="562" y2="344"/>
  <line stroke="blue" x1="562" y1="312" x2="594" y2="312"/>
  <line stroke="blue" x1="156" y1="625" x2="188" y2="625"/>
  <line stroke="red" x1="156" y1="625" x2="156" y2="656"/>
  <line stroke="blue" x1="406" y1="500" x2="438" y2="500"/>
  <line stroke="red" x1="406" y1="500" x2="406" y2="562"/>
  <line stroke="red" x1="531" y1="375" x2="531" y2="406"/>
  <line stroke="red" x1="531" y1="375" x2="562" y2="375"/>
  <line stroke="red" x1="188" y1="625" x2="219" y2="625"/>
  <line stroke="blue" x1="188" y1="625" x2="188" y2="656"/>
  <line stroke="red" x1="31" y1="656" x2="156" y2="656"/>
  <line stroke="blue" x1="31" y1="656" x2="62" y2="688"/>
  <line stroke="red" x1="31" y1="656" x2="31" y2="750"/>
  <line stroke="red" x1="594" y1="281" x2="594" y2="312"/>
  <line stroke="blue" x1="594" y1="281" x2="625" y2="250"/>
  <line stroke="blue" x1="594" y1="281" x2="625" y2="312"/>
  <line stroke="red" x1="594" y1="281" x2="656" y2="281"/>
  <line stroke="red" x1="344" y1="562" x2="375" y2="562"/>
  <line stroke="red" x1="344" y1="562" x2="344" y2="625"/>
  <line stroke="blue" x1="562" y1="344" x2="562" y2="375"/>
  <line stroke="red" x1="562" y1="344" x2="594" y2="344"/>
  <line stroke="blue" x1="219" y1="625" x2="250" y2="625"/>
  <line stroke="red" x1="219" y1="625" x2="219" y2="656"/>
  <line stroke="blue" x1="625" y1="219" x2="625" y2="250"/>
  <line stroke="red" x1="625" y1="219" x2="656" y2="219" opacity="0.3"/>
  <line stroke="red" x1="469" y1="469" x2="469" y2="500"/>
  <line stroke="red" x1="469" y1="469" x2="531" y2="469"/>
  <line stroke="red" x1="656" y1="94" x2="688" y2="62"/>
  <line stroke="red" x1="656" y1="94" x2="656" y2="219"/>
  <line stroke="red" x1="438" y1="500" x2="469" y2="500"/>
  <line stroke="blue" x1="438" y1="500" x2="438" y2="562"/>
  <line stroke="red" x1="531" y1="406" x2="531" y2="438"/>
  <line stroke="blue" x1="531" y1="406" x2="562" y2="406"/>
  <line stroke="red" x1="594" y1="312" x2="594" y2="344"/>
  <line stroke="red" x1="594" y1="312" x2="625" y2="312"/>
  <line stroke="red" x1="250" y1="625" x2="281" y2="625"/>
  <line stroke="blue" x1="250" y1="625" x2="250" y2="656"/>
  <line stroke="red" x1="625" y1="250" x2="656" y2="250"/>
  <line stroke="blue" x1="625" y1="250" x2="656" y2="281"/>
  <line stroke="blue" x1="156" y1="656" x2="188" y2="656" opacity="0.2"/>
  <line stroke="red" x1="156" y1="656" x2="156" y2="688" opacity="0.2"/>
  <line stroke="red" x1="156" y1="656" x2="188" y2="688"/>
  <line stroke="blue" x1="375" y1="562" x2="406" y2="562"/>
  <line stroke="blue" x1="375" y1="562" x2="375" y2="625"/>
  <line stroke="blue" x1="562" y1="375" x2="562" y2="406"/>
  <line stroke="blue" x1="562" y1="375" x2="594" y2="375"/>
  <line stroke="red" x1="188" y1="656" x2="219" y2="656" opacity="0.2"/>
  <line stroke="blue" x1="188" y1="656" x2="188" y2="688"/>
  <line stroke="blue" x1="281" y1="625" x2="312" y2="625"/>
  <line stroke="red" x1="281" y1="625" x2="281" y2="656"/>
  <line stroke="blue" x1="469" y1="500" x2="531" y2="500"/>
  <line stroke="red" x1="469" y1="500" x2="469" y2="562"/>
  <line stroke="blue" x1="594" y1="344" x2="625" y2="312"/>
  <line stroke="red" x1="594" y1="344" x2="594" y2="375"/>
  <line stroke="blue" x1="594" y1="344" x2="625" y2="375"/>
  <line stroke="red" x1="594" y1="344" x2="719" y2="344"/>
  <line stroke="red" x1="531" y1="438" x2="531" y2="469"/>
  <line stroke="red" x1="531" y1="438" x2="562" y2="438"/>
  <line stroke="blue" x1="62" y1="688" x2="156" y2="688"/>
  <line stroke="red" x1="62" y1="688" x2="94" y2="719"/>
  <line stroke="blue" x1="62" y1="688" x2="62" y2="750"/>
  <line stroke="blue" x1="688" y1="62" x2="719" y2="31"/>
  <line stroke="blue" x1="688" y1="62" x2="688" y2="219"/>
  <line stroke="blue" x1="219" y1="656" x2="250" y2="656" opacity="0.2"/>
  <line stroke="red" x1="219" y1="656" x2="219" y2="719"/>
  <line stroke="red" x1="656" y1="219" x2="656" y2="250"/>
  <line stroke="blue" x1="656" y1="219" x2="688" y2="219" opacity="0.3"/>
  <line stroke="red" x1="406" y1="562" x2="438" y2="562"/>
  <line stroke="red" x1="406" y1="562" x2="406" y2="625"/>
  <line stroke="blue" x1="562" y1="406" x2="562" y2="438"/>
  <line stroke="red" x1="562" y1="406" x2="594" y2="406"/>
  <line stroke="red" x1="312" y1="625" x2="344" y2="625"/>
  <line stroke="blue" x1="312" y1="625" x2="312" y2="656"/>
  <line stroke="blue" x1="625" y1="312" x2="688" y2="312"/>
  <line stroke="red" x1="250" y1="656" x2="281" y2="656" opacity="0.2"/>
  <line stroke="blue" x1="250" y1="656" x2="250" y2="750"/>
  <line stroke="red" x1="594" y1="375" x2="594" y2="406"/>
  <line stroke="red" x1="594" y1="375" x2="625" y2="375"/>
  <line stroke="red" x1="656" y1="250" x2="656" y2="281"/>
  <line stroke="blue" x1="656" y1="250" x2="688" y2="250"/>
  <line stroke="blue" x1="156" y1="688" x2="188" y2="688"/>
  <line stroke="blue" x1="156" y1="688" x2="156" y2="719" opacity="0.2"/>
  <line stroke="red" x1="531" y1="469" x2="531" y2="500"/>
  <line stroke="blue" x1="531" y1="469" x2="562" y2="469"/>
  <line stroke="blue" x1="188" y1="688" x2="219" y2="719"/>
  <line stroke="blue" x1="438" y1="562" x2="469" y2="562"/>
  <line stroke="blue" x1="438" y1="562" x2="438" y2="625"/>
  <line stroke="blue" x1="562" y1="438" x2="562" y2="469"/>
  <line stroke="blue" x1="562" y1="438" x2="594" y2="438"/>
  <line stroke="blue" x1="344" y1="625" x2="375" y2="625"/>
  <line stroke="red" x1="344" y1="625" x2="344" y2="656"/>
  <line stroke="blue" x1="281" y1="656" x2="312" y2="656" opacity="0.2"/>
  <line stroke="red" x1="281" y1="656" x2="281" y2="750"/>
  <line stroke="red" x1="656" y1="281" x2="688" y2="312"/>
  <line stroke="blue" x1="594" y1="406" x2="625" y2="375"/>
  <line stroke="red" x1="594" y1="406" x2="594" y2="438"/>
  <line stroke="blue" x1="594" y1="406" x2="625" y2="438"/>
  <line stroke="red" x1="594" y1="406" x2="719" y2="406"/>
  <line stroke="red" x1="719" y1="31" x2="750" y2="0"/>
  <line stroke="red" x1="719" y1="31" x2="719" y2="219"/>
  <line stroke="blue" x1="688" y1="219" x2="688" y2="250"/>
  <line stroke="red" x1="688" y1="219" x2="719" y2="219" opacity="0.3"/>
  <line stroke="red" x1="94" y1="719" x2="156" y2="719"/>
  <line stroke="red" x1="94" y1="719" x2="94" y2="750"/>
  <line stroke="blue" x1="94" y1="719" x2="125" y2="750"/>
  <line stroke="red" x1="312" y1="656" x2="344" y2="656" opacity="0.2"/>
  <line stroke="blue" x1="312" y1="656" x2="312" y2="750"/>
  <line stroke="red" x1="375" y1="625" x2="406" y2="625"/>
  <line stroke="blue" x1="375" y1="625" x2="375" y2="656"/>
  <line stroke="blue" x1="625" y1="375" x2="750" y2="375"/>
  <line stroke="red" x1="531" y1="500" x2="562" y2="500"/>
  <line stroke="red" x1="531" y1="500" x2="531" y2="562"/>
  <line stroke="blue" x1="688" y1="250" x2="688" y2="312"/>
  <line stroke="red" x1="688" y1="250" x2="719" y2="250"/>
  <line stroke="red" x1="469" y1="562" x2="531" y2="562"/>
  <line stroke="red" x1="469" y1="562" x2="469" y2="625"/>
  <line stroke="blue" x1="562" y1="469" x2="562" y2="500"/>
  <line stroke="red" x1="562" y1="469" x2="594" y2="469"/>
  <line stroke="red" x1="156" y1="719" x2="219" y2="719"/>
  <line stroke="red" x1="156" y1="719" x2="156" y2="750" opacity="0.2"/>
  <line stroke="red" x1="594" y1="438" x2="594" y2="469"/>
  <line stroke="red" x1="594" y1="438" x2="625" y2="438"/>
  <line stroke="blue" x1="344" y1="656" x2="375" y2="656" opacity="0.2"/>
  <line stroke="red" x1="344" y1="656" x2="344" y2="750"/>
  <line stroke="blue" x1="406" y1="625" x2="438" y2="625"/>
  <line stroke="red" x1="406" y1="625" x2="406" y2="656"/>
  <line stroke="blue" x1="0" y1="750" x2="31" y2="750"/>
  <line stroke="blue" x1="750" y1="0" x2="750" y2="219"/>
  <line stroke="red" x1="750" y1="0" x2="781" y2="31"/>
  <line stroke="red" x1="31" y1="750" x2="62" y2="750"/>
  <line stroke="red" x1="31" y1="750" x2="31" y2="844"/>
  <line stroke="red" x1="219" y1="719" x2="250" y2="750"/>
  <line stroke="red" x1="719" y1="219" x2="719" y2="250"/>
  <line stroke="blue" x1="719" y1="219" x2="750" y2="219" opacity="0.3"/>
  <line stroke="blue" x1="62" y1="750" x2="94" y2="750"/>
  <line stroke="blue" x1="62" y1="750" x2="62" y2="812"/>
  <line stroke="blue" x1="562" y1="500" x2="594" y2="500"/>
  <line stroke="blue" x1="562" y1="500" x2="562" y2="562"/>
  <line stroke="blue" x1="688" y1="312" x2="719" y2="344"/>
  <line stroke="red" x1="94" y1="750" x2="125" y2="750"/>
  <line stroke="red" x1="94" y1="750" x2="94" y2="781"/>
  <line stroke="red" x1="375" y1="656" x2="406" y2="656" opacity="0.2"/>
  <line stroke="blue" x1="375" y1="656" x2="375" y2="750"/>
  <line stroke="blue" x1="594" y1="469" x2="625" y2="438"/>
  <line stroke="red" x1="594" y1="469" x2="594" y2="500"/>
  <line stroke="blue" x1="594" y1="469" x2="625" y2="500"/>
  <line stroke="red" x1="594" y1="469" x2="656" y2="469"/>
  <line stroke="blue" x1="125" y1="750" x2="156" y2="750"/>
  <line stroke="blue" x1="125" y1="750" x2="94" y2="781"/>
  <line stroke="blue" x1="719" y1="250" x2="750" y2="250"/>
  <line stroke="red" x1="719" y1="250" x2="719" y2="344"/>
  <line stroke="red" x1="438" y1="625" x2="469" y2="625"/>
  <line stroke="blue" x1="438" y1="625" x2="438" y2="656"/>
  <line stroke="blue" x1="625" y1="438" x2="688" y2="438"/>
  <line stroke="blue" x1="156" y1="750" x2="250" y2="750"/>
  <line stroke="blue" x1="156" y1="750" x2="156" y2="781" opacity="0.2"/>
  <line stroke="blue" x1="406" y1="656" x2="438" y2="656" opacity="0.2"/>
  <line stroke="red" x1="406" y1="656" x2="406" y2="750"/>
  <line stroke="blue" x1="531" y1="562" x2="562" y2="562"/>
  <line stroke="red" x1="531" y1="562" x2="531" y2="625"/>
  <line stroke="red" x1="594" y1="500" x2="625" y2="500"/>
  <line stroke="red" x1="594" y1="500" x2="594" y2="562"/>
  <line stroke="red" x1="469" y1="625" x2="469" y2="656"/>
  <line stroke="blue" x1="469" y1="625" x2="531" y2="625"/>
  <line stroke="blue" x1="750" y1="219" x2="750" y2="250"/>
  <line stroke="red" x1="750" y1="219" x2="781" y2="219" opacity="0.3"/>
  <line stroke="red" x1="781" y1="31" x2="781" y2="219"/>
  <line stroke="blue" x1="781" y1="31" x2="812" y2="62"/>
  <line stroke="red" x1="781" y1="31" x2="875" y2="31"/>
  <line stroke="red" x1="94" y1="781" x2="156" y2="781"/>
  <line stroke="red" x1="94" y1="781" x2="62" y2="812"/>
  <line stroke="red" x1="438" y1="656" x2="469" y2="656" opacity="0.2"/>
  <line stroke="blue" x1="438" y1="656" x2="438" y2="750"/>
  <line stroke="red" x1="250" y1="750" x2="281" y2="750" opacity="0.2"/>
  <line stroke="blue" x1="250" y1="750" x2="250" y2="781" opacity="0.2"/>
  <line stroke="blue" x1="250" y1="750" x2="281" y2="781"/>
  <line stroke="red" x1="750" y1="250" x2="781" y2="250"/>
  <line stroke="blue" x1="750" y1="250" x2="750" y2="375"/>
  <line stroke="red" x1="562" y1="562" x2="594" y2="562"/>
  <line stroke="blue" x1="562" y1="562" x2="562" y2="625"/>
  <line stroke="red" x1="156" y1="781" x2="250" y2="781"/>
  <line stroke="red" x1="156" y1="781" x2="156" y2="812" opacity="0.2"/>
  <line stroke="red" x1="719" y1="344" x2="750" y2="375"/>
  <line stroke="blue" x1="625" y1="500" x2="656" y2="469"/>
  <line stroke="blue" x1="625" y1="500" x2="625" y2="562"/>
  <line stroke="blue" x1="281" y1="750" x2="312" y2="750" opacity="0.2"/>
  <line stroke="red" x1="281" y1="750" x2="281" y2="781"/>
  <line stroke="blue" x1="469" y1="656" x2="531" y2="656" opacity="0.2"/>
  <line stroke="red" x1="469" y1="656" x2="469" y2="750"/>
  <line stroke="red" x1="656" y1="469" x2="688" y2="438"/>
  <line stroke="red" x1="656" y1="469" x2="656" y2="562"/>
  <line stroke="red" x1="781" y1="219" x2="781" y2="250"/>
  <line stroke="blue" x1="781" y1="219" x2="812" y2="219" opacity="0.3"/>
  <line stroke="red" x1="312" y1="750" x2="344" y2="750" opacity="0.2"/>
  <line stroke="blue" x1="312" y1="750" x2="312" y2="812"/>
  <line stroke="blue" x1="62" y1="812" x2="156" y2="812"/>
  <line stroke="blue" x1="62" y1="812" x2="31" y2="844"/>
  <line stroke="blue" x1="688" y1="438" x2="719" y2="406"/>
  <line stroke="blue" x1="688" y1="438" x2="688" y2="562"/>
  <line stroke="blue" x1="812" y1="62" x2="812" y2="219"/>
  <line stroke="red" x1="812" y1="62" x2="844" y2="94"/>
  <line stroke="blue" x1="812" y1="62" x2="875" y2="62"/>
  <line stroke="blue" x1="594" y1="562" x2="625" y2="562"/>
  <line stroke="red" x1="594" y1="562" x2="594" y2="625"/>
  <line stroke="red" x1="250" y1="781" x2="281" y2="781"/>
  <line stroke="red" x1="250" y1="781" x2="250" y2="812" opacity="0.2"/>
  <line stroke="red" x1="531" y1="625" x2="562" y2="625"/>
  <line stroke="red" x1="531" y1="625" x2="531" y2="656"/>
  <line stroke="blue" x1="781" y1="250" x2="812" y2="250"/>
  <line stroke="red" x1="781" y1="250" x2="781" y2="344"/>
  <line stroke="blue" x1="344" y1="750" x2="375" y2="750" opacity="0.2"/>
  <line stroke="red" x1="344" y1="750" x2="344" y2="844"/>
  <line stroke="red" x1="719" y1="406" x2="750" y2="375"/>
  <line stroke="red" x1="719" y1="406" x2="719" y2="562"/>
  <line stroke="blue" x1="156" y1="812" x2="250" y2="812"/>
  <line stroke="blue" x1="156" y1="812" x2="156" y2="844" opacity="0.2"/>
  <line stroke="red" x1="281" y1="781" x2="312" y2="812"/>
  <line stroke="red" x1="375" y1="750" x2="406" y2="750" opacity="0.2"/>
  <line stroke="blue" x1="375" y1="750" x2="375" y2="844"/>
  <line stroke="red" x1="750" y1="375" x2="781" y2="344"/>
  <line stroke="red" x1="750" y1="375" x2="781" y2="406"/>
  <line stroke="blue" x1="750" y1="375" x2="750" y2="562"/>
  <line stroke="red" x1="750" y1="375" x2="875" y2="375"/>
  <line stroke="blue" x1="562" y1="625" x2="594" y2="625"/>
  <line stroke="blue" x1="562" y1="625" x2="562" y2="656"/>
  <line stroke="red" x1="625" y1="562" x2="656" y2="562"/>
  <line stroke="blue" x1="625" y1="562" x2="625" y2="625"/>
  <line stroke="blue" x1="812" y1="219" x2="812" y2="250"/>
  <line stroke="red" x1="812" y1="219" x2="844" y2="219" opacity="0.3"/>
  <line stroke="red" x1="31" y1="844" x2="156" y2="844"/>
  <line stroke="red" x1="31" y1="844" x2="0" y2="875"/>
  <line stroke="red" x1="531" y1="656" x2="562" y2="656" opacity="0.2"/>
  <line stroke="red" x1="531" y1="656" x2="531" y2="750"/>
  <line stroke="red" x1="844" y1="94" x2="844" y2="219"/>
  <line stroke="red" x1="844" y1="94" x2="875" y2="94"/>
  <line stroke="blue" x1="844" y1="94" x2="875" y2="125"/>
  <line stroke="blue" x1="250" y1="812" x2="312" y2="812"/>
  <line stroke="blue" x1="250" y1="812" x2="250" y2="844" opacity="0.2"/>
  <line stroke="blue" x1="812" y1="250" x2="812" y2="312"/>
  <line stroke="red" x1="812" y1="250" x2="844" y2="250"/>
  <line stroke="blue" x1="406" y1="750" x2="438" y2="750" opacity="0.2"/>
  <line stroke="red" x1="406" y1="750" x2="406" y2="844"/>
  <line stroke="blue" x1="781" y1="344" x2="812" y2="312"/>
  <line stroke="red" x1="781" y1="344" x2="875" y2="344"/>
  <line stroke="red" x1="156" y1="844" x2="250" y2="844"/>
  <line stroke="red" x1="156" y1="844" x2="156" y2="875" opacity="0.2"/>
  <line stroke="red" x1="594" y1="625" x2="625" y2="625"/>
  <line stroke="red" x1="594" y1="625" x2="594" y2="656"/>
  <line stroke="blue" x1="562" y1="656" x2="594" y2="656" opacity="0.2"/>
  <line stroke="blue" x1="562" y1="656" x2="562" y2="750"/>
  <line stroke="blue" x1="656" y1="562" x2="688" y2="562"/>
  <line stroke="red" x1="656" y1="562" x2="656" y2="625"/>
  <line stroke="red" x1="438" y1="750" x2="469" y2="750" opacity="0.2"/>
  <line stroke="blue" x1="438" y1="750" x2="438" y2="844"/>
  <line stroke="blue" x1="312" y1="812" x2="344" y2="844"/>
  <line stroke="red" x1="812" y1="312" x2="844" y2="281"/>
  <line stroke="blue" x1="812" y1="312" x2="875" y2="312"/>
  <line stroke="red" x1="844" y1="219" x2="844" y2="250"/>
  <line stroke="blue" x1="844" y1="219" x2="875" y2="219" opacity="0.3"/>
  <line stroke="blue" x1="0" y1="875" x2="156" y2="875"/>
  <line stroke="red" x1="0" y1="875" x2="31" y2="906"/>
  <line stroke="blue" x1="875" y1="0" x2="875" y2="31"/>
  <line stroke="red" x1="875" y1="31" x2="875" y2="62"/>
  <line stroke="red" x1="875" y1="31" x2="969" y2="31"/>
  <line stroke="blue" x1="875" y1="62" x2="875" y2="94"/>
  <line stroke="blue" x1="875" y1="62" x2="938" y2="62"/>
  <line stroke="red" x1="250" y1="844" x2="250" y2="875" opacity="0.2"/>
  <line stroke="red" x1="250" y1="844" x2="344" y2="844"/>
  <line stroke="red" x1="844" y1="250" x2="844" y2="281"/>
  <line stroke="blue" x1="844" y1="250" x2="875" y2="250"/>
  <line stroke="red" x1="875" y1="94" x2="875" y2="125"/>
  <line stroke="red" x1="875" y1="94" x2="906" y2="94"/>
  <line stroke="blue" x1="781" y1="406" x2="812" y2="438"/>
  <line stroke="red" x1="781" y1="406" x2="781" y2="562"/>
  <line stroke="red" x1="781" y1="406" x2="875" y2="406"/>
  <line stroke="blue" x1="625" y1="625" x2="625" y2="656"/>
  <line stroke="blue" x1="625" y1="625" x2="656" y2="625"/>
  <line stroke="blue" x1="875" y1="125" x2="875" y2="219"/>
  <line stroke="blue" x1="875" y1="125" x2="906" y2="94"/>
  <line stroke="blue" x1="469" y1="750" x2="531" y2="750" opacity="0.2"/>
  <line stroke="red" x1="469" y1="750" x2="469" y2="844"/>
  <line stroke="red" x1="594" y1="656" x2="625" y2="656" opacity="0.2"/>
  <line stroke="red" x1="594" y1="656" x2="594" y2="750"/>
  <line stroke="red" x1="688" y1="562" x2="719" y2="562"/>
  <line stroke="blue" x1="688" y1="562" x2="688" y2="625"/>
  <line stroke="blue" x1="156" y1="875" x2="250" y2="875"/>
  <line stroke="blue" x1="156" y1="875" x2="156" y2="906" opacity="0.2"/>
  <line stroke="blue" x1="844" y1="281" x2="875" y2="250"/>
  <line stroke="red" x1="844" y1="281" x2="875" y2="281"/>
  <line stroke="blue" x1="875" y1="219" x2="875" y2="250"/>
  <line stroke="red" x1="875" y1="219" x2="906" y2="219" opacity="0.3"/>
  <line stroke="blue" x1="625" y1="656" x2="656" y2="656" opacity="0.2"/>
  <line stroke="blue" x1="625" y1="656" x2="625" y2="750"/>
  <line stroke="red" x1="656" y1="625" x2="656" y2="656"/>
  <line stroke="red" x1="656" y1="625" x2="688" y2="625"/>
  <line stroke="red" x1="31" y1="906" x2="156" y2="906"/>
  <line stroke="blue" x1="31" y1="906" x2="62" y2="938"/>
  <line stroke="red" x1="31" y1="906" x2="31" y2="1000"/>
  <line stroke="blue" x1="250" y1="875" x2="250" y2="906" opacity="0.2"/>
  <line stroke="blue" x1="250" y1="875" x2="344" y2="875"/>
  <line stroke="red" x1="875" y1="250" x2="875" y2="281"/>
  <line stroke="red" x1="875" y1="250" x2="906" y2="250"/>
  <line stroke="blue" x1="875" y1="250" x2="906" y2="281"/>
  <line stroke="blue" x1="344" y1="844" x2="375" y2="844" opacity="0.2"/>
  <line stroke="red" x1="344" y1="844" x2="344" y2="875" opacity="0.2"/>
  <line stroke="red" x1="344" y1="844" x2="375" y2="875"/>
  <line stroke="red" x1="906" y1="94" x2="906" y2="219"/>
  <line stroke="red" x1="906" y1="94" x2="938" y2="62"/>
  <line stroke="blue" x1="719" y1="562" x2="750" y2="562"/>
  <line stroke="red" x1="719" y1="562" x2="719" y2="625"/>
  <line stroke="red" x1="531" y1="750" x2="562" y2="750" opacity="0.2"/>
  <line stroke="red" x1="531" y1="750" x2="531" y2="844"/>
  <line stroke="blue" x1="875" y1="281" x2="875" y2="312"/>
  <line stroke="red" x1="875" y1="281" x2="906" y2="281"/>
  <line stroke="red" x1="156" y1="906" x2="250" y2="906"/>
  <line stroke="red" x1="156" y1="906" x2="156" y2="938" opacity="0.2"/>
  <line stroke="red" x1="812" y1="438" x2="844" y2="469"/>
  <line stroke="blue" x1="812" y1="438" x2="875" y2="438"/>
  <line stroke="blue" x1="812" y1="438" x2="812" y2="562"/>
  <line stroke="red" x1="375" y1="844" x2="406" y2="844" opacity="0.2"/>
  <line stroke="blue" x1="375" y1="844" x2="375" y2="875"/>
  <line stroke="red" x1="656" y1="656" x2="688" y2="656" opacity="0.2"/>
  <line stroke="red" x1="656" y1="656" x2="656" y2="750"/>
  <line stroke="blue" x1="688" y1="625" x2="688" y2="656"/>
  <line stroke="blue" x1="688" y1="625" x2="719" y2="625"/>
  <line stroke="red" x1="875" y1="312" x2="875" y2="344"/>
  <line stroke="blue" x1="875" y1="312" x2="938" y2="312"/>
  <line stroke="red" x1="906" y1="219" x2="906" y2="250"/>
  <line stroke="blue" x1="906" y1="219" x2="938" y2="219" opacity="0.3"/>
  <line stroke="blue" x1="406" y1="844" x2="438" y2="844" opacity="0.2"/>
  <line stroke="red" x1="406" y1="844" x2="406" y2="906"/>
  <line stroke="blue" x1="562" y1="750" x2="594" y2="750" opacity="0.2"/>
  <line stroke="blue" x1="562" y1="750" x2="562" y2="844"/>
  <line stroke="red" x1="750" y1="562" x2="781" y2="562"/>
  <line stroke="blue" x1="750" y1="562" x2="750" y2="625"/>
  <line stroke="blue" x1="62" y1="938" x2="156" y2="938"/>
  <line stroke="red" x1="62" y1="938" x2="94" y2="969"/>
  <line stroke="blue" x1="62" y1="938" x2="62" y2="1000"/>
  <line stroke="blue" x1="938" y1="62" x2="938" y2="219"/>
  <line stroke="blue" x1="938" y1="62" x2="969" y2="31"/>
  <line stroke="red" x1="250" y1="906" x2="344" y2="906"/>
  <line stroke="red" x1="250" y1="906" x2="250" y2="938" opacity="0.2"/>
  <line stroke="blue" x1="344" y1="875" x2="375" y2="875"/>
  <line stroke="blue" x1="344" y1="875" x2="344" y2="906" opacity="0.2"/>
  <line stroke="blue" x1="875" y1="344" x2="875" y2="375"/>
  <line stroke="red" x1="875" y1="344" x2="969" y2="344"/>
  <line stroke="red" x1="906" y1="250" x2="906" y2="281"/>
  <line stroke="blue" x1="906" y1="250" x2="938" y2="250"/>
  <line stroke="red" x1="906" y1="281" x2="938" y2="312"/>
  <line stroke="blue" x1="156" y1="938" x2="250" y2="938"/>
  <line stroke="blue" x1="156" y1="938" x2="156" y2="969" opacity="0.2"/>
  <line stroke="red" x1="438" y1="844" x2="469" y2="844" opacity="0.2"/>
  <line stroke="blue" x1="438" y1="844" x2="438" y2="938"/>
  <line stroke="blue" x1="688" y1="656" x2="719" y2="656" opacity="0.2"/>
  <line stroke="blue" x1="688" y1="656" x2="688" y2="750"/>
  <line stroke="blue" x1="375" y1="875" x2="406" y2="906"/>
  <line stroke="blue" x1="875" y1="375" x2="875" y2="406"/>
  <line stroke="blue" x1="875" y1="375" x2="1000" y2="375"/>
  <line stroke="red" x1="719" y1="625" x2="719" y2="656"/>
  <line stroke="red" x1="719" y1="625" x2="750" y2="625"/>
  <line stroke="red" x1="594" y1="750" x2="625" y2="750" opacity="0.2"/>
  <line stroke="red" x1="594" y1="750" x2="594" y2="844"/>
  <line stroke="blue" x1="781" y1="562" x2="812" y2="562"/>
  <line stroke="red" x1="781" y1="562" x2="781" y2="625"/>
  <line stroke="blue" x1="938" y1="219" x2="938" y2="250"/>
  <line stroke="red" x1="938" y1="219" x2="969" y2="219" opacity="0.3"/>
  <line stroke="red" x1="875" y1="406" x2="875" y2="438"/>
  <line stroke="red" x1="875" y1="406" x2="969" y2="406"/>
  <line stroke="blue" x1="469" y1="844" x2="531" y2="844" opacity="0.2"/>
  <line stroke="red" x1="469" y1="844" x2="469" y2="938"/>
  <line stroke="red" x1="844" y1="469" x2="875" y2="469"/>
  <line stroke="red" x1="844" y1="469" x2="844" y2="531"/>
  <line stroke="blue" x1="844" y1="469" x2="875" y2="500"/>
  <line stroke="red" x1="344" y1="906" x2="406" y2="906"/>
  <line stroke="red" x1="344" y1="906" x2="344" y2="938" opacity="0.2"/>
  <line stroke="red" x1="969" y1="31" x2="969" y2="219"/>
  <line stroke="red" x1="969" y1="31" x2="1000" y2="0"/>
  <line stroke="blue" x1="250" y1="938" x2="344" y2="938"/>
  <line stroke="blue" x1="250" y1="938" x2="250" y2="969" opacity="0.2"/>
  <line stroke="blue" x1="938" y1="250" x2="938" y2="312"/>
  <line stroke="red" x1="938" y1="250" x2="969" y2="250"/>
  <line stroke="red" x1="94" y1="969" x2="156" y2="969"/>
  <line stroke="red" x1="94" y1="969" x2="94" y2="1000"/>
  <line stroke="blue" x1="94" y1="969" x2="125" y2="1000"/>
  <line stroke="red" x1="719" y1="656" x2="750" y2="656" opacity="0.2"/>
  <line stroke="red" x1="719" y1="656" x2="719" y2="750"/>
  <line stroke="blue" x1="625" y1="750" x2="656" y2="750" opacity="0.2"/>
  <line stroke="blue" x1="625" y1="750" x2="625" y2="844"/>
  <line stroke="blue" x1="750" y1="625" x2="750" y2="656"/>
  <line stroke="blue" x1="750" y1="625" x2="781" y2="625"/>
  <line stroke="blue" x1="875" y1="438" x2="875" y2="469"/>
  <line stroke="blue" x1="875" y1="438" x2="938" y2="438"/>
  <line stroke="red" x1="156" y1="969" x2="250" y2="969"/>
  <line stroke="red" x1="156" y1="969" x2="156" y2="1000" opacity="0.2"/>
  <line stroke="red" x1="812" y1="562" x2="844" y2="531"/>
  <line stroke="blue" x1="812" y1="562" x2="812" y2="625"/>
  <line stroke="red" x1="812" y1="562" x2="844" y2="594"/>
  <line stroke="blue" x1="812" y1="562" x2="938" y2="562"/>
  <line stroke="blue" x1="938" y1="312" x2="969" y2="344"/>
  <line stroke="red" x1="875" y1="469" x2="875" y2="500"/>
  <line stroke="red" x1="875" y1="469" x2="906" y2="469"/>
  <line stroke="red" x1="406" y1="906" x2="438" y2="938"/>
  <line stroke="red" x1="969" y1="219" x2="969" y2="250"/>
  <line stroke="blue" x1="969" y1="219" x2="1000" y2="219" opacity="0.3"/>
  <line stroke="red" x1="656" y1="750" x2="688" y2="750" opacity="0.2"/>
  <line stroke="red" x1="656" y1="750" x2="656" y2="844"/>
  <line stroke="blue" x1="750" y1="656" x2="781" y2="656" opacity="0.2"/>
  <line stroke="blue" x1="750" y1="656" x2="750" y2="750"/>
  <line stroke="red" x1="531" y1="844" x2="562" y2="844" opacity="0.2"/>
  <line stroke="red" x1="531" y1="844" x2="531" y2="938"/>
  <line stroke="blue" x1="844" y1="531" x2="875" y2="500"/>
  <line stroke="red" x1="844" y1="531" x2="906" y2="531"/>
  <line stroke="blue" x1="344" y1="938" x2="344" y2="969" opacity="0.2"/>
  <line stroke="blue" x1="344" y1="938" x2="438" y2="938"/>
  <line stroke="red" x1="250" y1="969" x2="344" y2="969"/>
  <line stroke="red" x1="250" y1="969" x2="250" y2="1000" opacity="0.2"/>
  <line stroke="red" x1="781" y1="625" x2="781" y2="656"/>
  <line stroke="red" x1="781" y1="625" x2="812" y2="625"/>
  <line stroke="red" x1="969" y1="250" x2="969" y2="344"/>
  <line stroke="blue" x1="969" y1="250" x2="1000" y2="250"/>
  <line stroke="blue" x1="875" y1="500" x2="906" y2="469"/>
  <line stroke="red" x1="875" y1="500" x2="906" y2="500"/>
  <line stroke="blue" x1="875" y1="500" x2="906" y2="531"/>
  <line stroke="blue" x1="562" y1="844" x2="594" y2="844" opacity="0.2"/>
  <line stroke="blue" x1="562" y1="844" x2="562" y2="938"/>
  <line stroke="blue" x1="688" y1="750" x2="719" y2="750" opacity="0.2"/>
  <line stroke="blue" x1="688" y1="750" x2="688" y2="812"/>
  <line stroke="red" x1="781" y1="656" x2="812" y2="656" opacity="0.2"/>
  <line stroke="red" x1="781" y1="656" x2="781" y2="719"/>
  <line stroke="red" x1="906" y1="469" x2="938" y2="438"/>
  <line stroke="red" x1="906" y1="469" x2="906" y2="500"/>
  <line stroke="blue" x1="812" y1="625" x2="812" y2="656"/>
  <line stroke="blue" x1="812" y1="625" x2="844" y2="625"/>
  <line stroke="red" x1="344" y1="969" x2="344" y2="1000" opacity="0.2"/>
  <line stroke="red" x1="344" y1="969" x2="438" y2="969"/>
  <line stroke="red" x1="969" y1="344" x2="1000" y2="375"/>
  <line stroke="red" x1="594" y1="844" x2="625" y2="844" opacity="0.2"/>
  <line stroke="red" x1="594" y1="844" x2="594" y2="906"/>
  <line stroke="red" x1="844" y1="594" x2="844" y2="625"/>
  <line stroke="blue" x1="844" y1="594" x2="875" y2="625"/>
  <line stroke="red" x1="844" y1="594" x2="969" y2="594"/>
  <line stroke="red" x1="438" y1="938" x2="469" y2="938" opacity="0.2"/>
  <line stroke="blue" x1="438" y1="938" x2="438" y2="969" opacity="0.2"/>
  <line stroke="blue" x1="438" y1="938" x2="469" y2="969"/>
  <line stroke="blue" x1="938" y1="438" x2="969" y2="406"/>
  <line stroke="blue" x1="938" y1="438" x2="938" y2="500"/>
  <line stroke="red" x1="906" y1="500" x2="906" y2="531"/>
  <line stroke="blue" x1="906" y1="500" x2="938" y2="500"/>
  <line stroke="red" x1="719" y1="750" x2="750" y2="750" opacity="0.2"/>
  <line stroke="red" x1="719" y1="750" x2="719" y2="781"/>
  <line stroke="blue" x1="812" y1="656" x2="812" y2="688"/>
  <line stroke="blue" x1="812" y1="656" x2="844" y2="656" opacity="0.2"/>
  <line stroke="red" x1="469" y1="938" x2="469" y2="969"/>
  <line stroke="blue" x1="469" y1="938" x2="531" y2="938" opacity="0.2"/>
  <line stroke="blue" x1="625" y1="844" x2="656" y2="844" opacity="0.2"/>
  <line stroke="blue" x1="625" y1="844" x2="625" y2="875"/>
  <line stroke="red" x1="844" y1="625" x2="844" y2="656"/>
  <line stroke="red" x1="844" y1="625" x2="875" y2="625"/>
  <line stroke="red" x1="906" y1="531" x2="938" y2="562"/>
  <line stroke="red" x1="969" y1="406" x2="1000" y2="375"/>
  <line stroke="red" x1="969" y1="406" x2="969" y2="500"/>
  <line stroke="blue" x1="750" y1="750" x2="719" y2="781"/>
  <line stroke="red" x1="750" y1="750" x2="781" y2="719"/>
  <line stroke="blue" x1="750" y1="750" x2="750" y2="781" opacity="0.2"/>
  <line stroke="blue" x1="750" y1="750" x2="844" y2="750"/>
  <line stroke="red" x1="719" y1="781" x2="688" y2="812"/>
  <line stroke="red" x1="719" y1="781" x2="750" y2="781"/>
  <line stroke="blue" x1="781" y1="719" x2="812" y2="688"/>
  <line stroke="red" x1="781" y1="719" x2="844" y2="719"/>
  <line stroke="red" x1="938" y1="500" x2="969" y2="500"/>
  <line stroke="blue" x1="938" y1="500" x2="938" y2="562"/>
  <line stroke="red" x1="438" y1="969" x2="469" y2="969"/>
  <line stroke="red" x1="438" y1="969" x2="438" y2="1000" opacity="0.2"/>
  <line stroke="blue" x1="688" y1="812" x2="656" y2="844"/>
  <line stroke="blue" x1="688" y1="812" x2="750" y2="812"/>
  <line stroke="red" x1="812" y1="688" x2="844" y2="656"/>
  <line stroke="blue" x1="812" y1="688" x2="844" y2="688"/>
  <line stroke="red" x1="656" y1="844" x2="625" y2="875"/>
  <line stroke="red" x1="656" y1="844" x2="656" y2="875" opacity="0.2"/>
  <line stroke="red" x1="656" y1="844" x2="750" y2="844"/>
  <line stroke="blue" x1="844" y1="656" x2="875" y2="625"/>
  <line stroke="red" x1="844" y1="656" x2="844" y2="688" opacity="0.2"/>
  <line stroke="red" x1="844" y1="656" x2="969" y2="656"/>
  <line stroke="blue" x1="625" y1="875" x2="594" y2="906"/>
  <line stroke="blue" x1="625" y1="875" x2="656" y2="875"/>
  <line stroke="blue" x1="875" y1="625" x2="1000" y2="625"/>
  <line stroke="red" x1="469" y1="969" x2="500" y2="1000"/>
  <line stroke="red" x1="531" y1="938" x2="562" y2="938" opacity="0.2"/>
  <line stroke="red" x1="531" y1="938" x2="531" y2="969"/>
  <line stroke="red" x1="750" y1="781" x2="750" y2="812" opacity="0.2"/>
  <line stroke="red" x1="750" y1="781" x2="844" y2="781"/>
  <line stroke="red" x1="594" y1="906" x2="562" y2="938"/>
  <line stroke="red" x1="594" y1="906" x2="656" y2="906"/>
  <line stroke="blue" x1="844" y1="688" x2="844" y2="719" opacity="0.2"/>
  <line stroke="blue" x1="844" y1="688" x2="938" y2="688"/>
  <line stroke="blue" x1="969" y1="500" x2="1000" y2="500"/>
  <line stroke="red" x1="969" y1="500" x2="969" y2="594"/>
  <line stroke="blue" x1="562" y1="938" x2="531" y2="969"/>
  <line stroke="blue" x1="562" y1="938" x2="562" y2="969" opacity="0.2"/>
  <line stroke="blue" x1="562" y1="938" x2="656" y2="938"/>
  <line stroke="blue" x1="938" y1="562" x2="969" y2="594"/>
  <line stroke="blue" x1="656" y1="875" x2="656" y2="906" opacity="0.2"/>
  <line stroke="blue" x1="656" y1="875" x2="750" y2="875"/>
  <line stroke="red" x1="531" y1="969" x2="500" y2="1000"/>
  <line stroke="red" x1="531" y1="969" x2="562" y2="969"/>
  <line stroke="blue" x1="750" y1="812" x2="750" y2="844" opacity="0.2"/>
  <line stroke="blue" x1="750" y1="812" x2="844" y2="812"/>
  <line stroke="red" x1="844" y1="719" x2="844" y2="750" opacity="0.2"/>
  <line stroke="red" x1="844" y1="719" x2="906" y2="719"/>
  <line stroke="red" x1="656" y1="906" x2="656" y2="938" opacity="0.2"/>
  <line stroke="red" x1="656" y1="906" x2="750" y2="906"/>
  <line stroke="red" x1="562" y1="969" x2="562" y2="1000" opacity="0.2"/>
  <line stroke="red" x1="562" y1="969" x2="656" y2="969"/>
  <line stroke="red" x1="750" y1="844" x2="750" y2="875" opacity="0.2"/>
  <line stroke="red" x1="750" y1="844" x2="844" y2="844"/>
  <line stroke="blue" x1="844" y1="750" x2="844" y2="781" opacity="0.2"/>
  <line stroke="blue" x1="844" y1="750" x2="875" y2="750"/>
  <line stroke="red" x1="969" y1="594" x2="1000" y2="625"/>
  <line stroke="blue" x1="656" y1="938" x2="656" y2="969" opacity="0.2"/>
  <line stroke="blue" x1="656" y1="938" x2="750" y2="938"/>
  <line stroke="red" x1="844" y1="781" x2="844" y2="812" opacity="0.2"/>
  <line stroke="red" x1="844" y1="781" x2="906" y2="781"/>
  <line stroke="blue" x1="750" y1="875" x2="750" y2="906" opacity="0.2"/>
  <line stroke="blue" x1="750" y1="875" x2="844" y2="875"/>
  <line stroke="blue" x1="875" y1="750" x2="906" y2="719"/>
  <line stroke="red" x1="875" y1="750" x2="906" y2="750"/>
  <line stroke="blue" x1="875" y1="750" x2="906" y2="781"/>
  <line stroke="red" x1="906" y1="719" x2="938" y2="688"/>
  <line stroke="red" x1="906" y1="719" x2="906" y2="750"/>
  <line stroke="blue" x1="938" y1="688" x2="969" y2="656"/>
  <line stroke="blue" x1="938" y1="688" x2="938" y2="750"/>
  <line stroke="red" x1="656" y1="969" x2="656" y2="1000" opacity="0.2"/>
  <line stroke="red" x1="656" y1="969" x2="750" y2="969"/>
  <line stroke="red" x1="969" y1="656" x2="1000" y2="625"/>
  <line stroke="red" x1="969" y1="656" x2="969" y2="750"/>
  <line stroke="blue" x1="844" y1="812" x2="844" y2="844" opacity="0.2"/>
  <line stroke="blue" x1="844" y1="812" x2="938" y2="812"/>
  <line stroke="red" x1="750" y1="906" x2="750" y2="938" opacity="0.2"/>
  <line stroke="red" x1="750" y1="906" x2="844" y2="906"/>
  <line stroke="red" x1="906" y1="750" x2="906" y2="781"/>
  <line stroke="blue" x1="906" y1="750" x2="938" y2="750"/>
  <line stroke="red" x1="844" y1="844" x2="844" y2="875" opacity="0.2"/>
  <line stroke="red" x1="844" y1="844" x2="969" y2="844"/>
  <line stroke="red" x1="906" y1="781" x2="938" y2="812"/>
  <line stroke="blue" x1="750" y1="938" x2="750" y2="969" opacity="0.2"/>
  <line stroke="blue" x1="750" y1="938" x2="844" y2="938"/>
  <line stroke="red" x1="938" y1="750" x2="969" y2="750"/>
  <line stroke="blue" x1="938" y1="750" x2="938" y2="812"/>
  <line stroke="blue" x1="844" y1="875" x2="844" y2="906" opacity="0.2"/>
  <line stroke="blue" x1="844" y1="875" x2="1000" y2="875"/>
  <line stroke="red" x1="750" y1="969" x2="750" y2="1000" opacity="0.2"/>
  <line stroke="red" x1="750" y1="969" x2="844" y2="969"/>
  <line stroke="blue" x1="969" y1="750" x2="1000" y2="750"/>
  <line stroke="red" x1="969" y1="750" x2="969" y2="844"/>
  <line stroke="red" x1="844" y1="906" x2="844" y2="938" opacity="0.2"/>
  <line stroke="red" x1="844" y1="906" x2="969" y2="906"/>
  <line stroke="blue" x1="938" y1="812" x2="969" y2="844"/>
  <line stroke="blue" x1="844" y1="938" x2="844" y2="969" opacity="0.2"/>
  <line stroke="blue" x1="844" y1="938" x2="938" y2="938"/>
  <line stroke="red" x1="844" y1="969" x2="844" y2="1000" opacity="0.2"/>
  <line stroke="red" x1="844" y1="969" x2="906" y2="969"/>
  <line stroke="red" x1="969" y1="844" x2="1000" y2="875"/>
  <line stroke="red" x1="938" y1="938" x2="906" y2="969"/>
  <line stroke="blue" x1="938" y1="938" x2="969" y2="906"/>
  <line stroke="blue" x1="938" y1="938" x2="938" y2="1000"/>
  <line stroke="blue" x1="906" y1="969" x2="875" y2="1000"/>
  <line stroke="red" x1="906" y1="969" x2="906" y2="1000"/>
  <line stroke="red" x1="969" y1="906" x2="1000" y2="875"/>
  <line stroke="red" x1="969" y1="906" x2="969" y2="1000"/>
</svg>""",
)

SNAKE_UNMERGED_SHAPING_HEURISTICS_TEST = BoxPleatingShapingExample(
    base_example=SNAKE,
    stick_figure=stick_figure_examples.SNAKE,
    central_crease_removed=False,
    rectangles=[
        shaping_elements.ShapingFlap.from_centers(
            name="body",
            circle_centers=[
                point_utils.Point(x=1, y=1, require_on_paper=False),
            ],
            radius=math.sqrt(2),
            grid_size=SNAKE.grid_size,
        ).extend_creases([
            crease
            for crease in SNAKE.base_creases
            if crease.is_horizontal or crease.is_vertical
        ])
    ],
    rivers=[],
    zigzag_shaping_params=[
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="segment_2",
            zigzag_params=shaping_types.ZigzagParams(
                target_flap_angle=0.9272952180016122,
                fold_order=_FOLD_ORDERS["MV"],
                even_phase=True,
                depth=0,
                fold_percent=1.0,
                target_zigzag_depth=None,
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="segment_3",
            zigzag_params=shaping_types.ZigzagParams(
                target_flap_angle=1.5707963267948966,
                fold_order=_FOLD_ORDERS["MV"],
                even_phase=True,
                depth=0,
                fold_percent=1.0,
                target_zigzag_depth=None,
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="segment_4",
            zigzag_params=shaping_types.ZigzagParams(
                target_flap_angle=2.356194490192345,
                fold_order=_FOLD_ORDERS["MV"],
                even_phase=True,
                depth=0,
                fold_percent=1.0,
                target_zigzag_depth=None,
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="segment_5",
            zigzag_params=shaping_types.ZigzagParams(
                target_flap_angle=1.5707963267948966,
                fold_order=_FOLD_ORDERS["MV"],
                even_phase=True,
                depth=0,
                fold_percent=1.0,
                target_zigzag_depth=None,
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="segment_6",
            zigzag_params=shaping_types.ZigzagParams(
                target_flap_angle=0.9272952180016122,
                fold_order=_FOLD_ORDERS["MV"],
                even_phase=False,
                depth=0,
                fold_percent=1.0,
                target_zigzag_depth=None,
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="segment_7",
            zigzag_params=shaping_types.ZigzagParams(
                target_flap_angle=1.5707963267948966,
                fold_order=_FOLD_ORDERS["MV"],
                even_phase=False,
                depth=0,
                fold_percent=1.0,
                target_zigzag_depth=None,
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="segment_8",
            zigzag_params=shaping_types.ZigzagParams(
                target_flap_angle=0.7853981633974483,
                fold_order=_FOLD_ORDERS["MV"],
                even_phase=False,
                depth=0,
                fold_percent=1.0,
                target_zigzag_depth=None,
            ),
        ),
    ],
    inscribed_shaping_params=[],
)

SNAKE_SQUASH_SINK_SHAPING_TEST = BoxPleatingShapingExample(
    base_example=SNAKE,
    stick_figure=stick_figure_examples.SNAKE,
    central_crease_removed=False,
    rectangles=SNAKE_UNMERGED_SHAPING_HEURISTICS_TEST.rectangles,
    rivers=[],
    squash_sink_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
      <line stroke="black" x1="0" y1="0" x2="0" y2="62.5"/>
      <line stroke="black" x1="0" y1="0" x2="62.5" y2="0"/>
      <line stroke="blue" x1="0" y1="0" x2="62.5" y2="62.5"/>
      <line stroke="red" x1="0" y1="62.5" x2="62.5" y2="62.5" elevation="1"/>
      <line stroke="black" x1="0" y1="62.5" x2="0" y2="125"/>
      <line stroke="red" x1="62.5" y1="0" x2="62.5" y2="62.5" elevation="1"/>
      <line stroke="black" x1="62.5" y1="0" x2="125" y2="0"/>
      <line stroke="red" x1="62.5" y1="62.5" x2="125" y2="125"/>
      <line stroke="blue" x1="0" y1="125" x2="125" y2="125" elevation="0"/>
      <line stroke="black" x1="0" y1="125" x2="0" y2="187.5"/>
      <line stroke="blue" x1="125" y1="0" x2="125" y2="125" elevation="0"/>
      <line stroke="black" x1="125" y1="0" x2="187.5" y2="0"/>
      <line stroke="red" x1="125" y1="125" x2="143.3" y2="143.3"/>
      <line stroke="blue" x1="125" y1="125" x2="99.1" y2="187.5"/>
      <line stroke="blue" x1="125" y1="125" x2="187.5" y2="99.1"/>
      <line stroke="red" x1="0" y1="187.5" x2="99.1" y2="187.5" elevation="1"/>
      <line stroke="black" x1="0" y1="187.5" x2="0" y2="250"/>
      <line stroke="red" x1="187.5" y1="0" x2="187.5" y2="99.1" elevation="1"/>
      <line stroke="black" x1="187.5" y1="0" x2="250" y2="0"/>
      <line stroke="red" x1="143.3" y1="143.3" x2="99.1" y2="187.5"/>
      <line stroke="red" x1="143.3" y1="143.3" x2="187.5" y2="99.1"/>
      <line stroke="blue" x1="143.3" y1="143.3" x2="250" y2="250"/>
      <line stroke="red" x1="99.1" y1="187.5" x2="250" y2="250"/>
      <line stroke="red" x1="187.5" y1="99.1" x2="250" y2="250"/>
      <line stroke="black" x1="0" y1="250" x2="0" y2="312.5"/>
      <line stroke="blue" x1="0" y1="250" x2="250" y2="250" elevation="0"/>
      <line stroke="black" x1="250" y1="0" x2="312.5" y2="0"/>
      <line stroke="blue" x1="250" y1="0" x2="250" y2="250" elevation="0"/>
      <line stroke="black" x1="0" y1="312.5" x2="0" y2="375"/>
      <line stroke="red" x1="0" y1="312.5" x2="224.1" y2="312.5" elevation="1"/>
      <line stroke="black" x1="312.5" y1="0" x2="375" y2="0"/>
      <line stroke="red" x1="312.5" y1="0" x2="312.5" y2="224.1" elevation="1"/>
      <line stroke="red" x1="250" y1="250" x2="268.3" y2="268.3"/>
      <line stroke="blue" x1="250" y1="250" x2="224.1" y2="312.5"/>
      <line stroke="blue" x1="250" y1="250" x2="312.5" y2="224.1"/>
      <line stroke="black" x1="0" y1="375" x2="0" y2="437.5"/>
      <line stroke="blue" x1="0" y1="375" x2="375" y2="375" elevation="0"/>
      <line stroke="black" x1="375" y1="0" x2="437.5" y2="0"/>
      <line stroke="blue" x1="375" y1="0" x2="375" y2="375" elevation="0"/>
      <line stroke="red" x1="268.3" y1="268.3" x2="224.1" y2="312.5"/>
      <line stroke="red" x1="268.3" y1="268.3" x2="312.5" y2="224.1"/>
      <line stroke="blue" x1="268.3" y1="268.3" x2="375" y2="375"/>
      <line stroke="red" x1="224.1" y1="312.5" x2="375" y2="375"/>
      <line stroke="red" x1="312.5" y1="224.1" x2="375" y2="375"/>
      <line stroke="black" x1="0" y1="437.5" x2="0" y2="500"/>
      <line stroke="red" x1="0" y1="437.5" x2="349.1" y2="437.5" elevation="1"/>
      <line stroke="black" x1="437.5" y1="0" x2="500" y2="0"/>
      <line stroke="red" x1="437.5" y1="0" x2="437.5" y2="349.1" elevation="1"/>
      <line stroke="black" x1="0" y1="500" x2="0" y2="562.5"/>
      <line stroke="blue" x1="0" y1="500" x2="500" y2="500" elevation="0"/>
      <line stroke="black" x1="500" y1="0" x2="562.5" y2="0"/>
      <line stroke="blue" x1="500" y1="0" x2="500" y2="500" elevation="0"/>
      <line stroke="red" x1="375" y1="375" x2="393.3" y2="393.3"/>
      <line stroke="blue" x1="375" y1="375" x2="349.1" y2="437.5"/>
      <line stroke="blue" x1="375" y1="375" x2="437.5" y2="349.1"/>
      <line stroke="red" x1="393.3" y1="393.3" x2="349.1" y2="437.5"/>
      <line stroke="red" x1="393.3" y1="393.3" x2="437.5" y2="349.1"/>
      <line stroke="blue" x1="393.3" y1="393.3" x2="500" y2="500"/>
      <line stroke="red" x1="349.1" y1="437.5" x2="500" y2="500"/>
      <line stroke="red" x1="437.5" y1="349.1" x2="500" y2="500"/>
      <line stroke="black" x1="0" y1="562.5" x2="0" y2="625"/>
      <line stroke="red" x1="0" y1="562.5" x2="562.5" y2="562.5" elevation="1"/>
      <line stroke="black" x1="562.5" y1="0" x2="625" y2="0"/>
      <line stroke="red" x1="562.5" y1="0" x2="562.5" y2="562.5" elevation="1"/>
      <line stroke="black" x1="0" y1="625" x2="0" y2="687.5"/>
      <line stroke="blue" x1="0" y1="625" x2="625" y2="625" elevation="0"/>
      <line stroke="black" x1="625" y1="0" x2="687.5" y2="0"/>
      <line stroke="blue" x1="625" y1="0" x2="625" y2="625" elevation="0"/>
      <line stroke="black" x1="0" y1="687.5" x2="0" y2="750"/>
      <line stroke="red" x1="0" y1="687.5" x2="687.5" y2="687.5" elevation="1"/>
      <line stroke="black" x1="687.5" y1="0" x2="750" y2="0"/>
      <line stroke="red" x1="687.5" y1="0" x2="687.5" y2="687.5" elevation="1"/>
      <line stroke="blue" x1="500" y1="500" x2="562.5" y2="562.5"/>
      <line stroke="black" x1="0" y1="750" x2="0" y2="812.5"/>
      <line stroke="blue" x1="0" y1="750" x2="750" y2="750" elevation="0"/>
      <line stroke="black" x1="750" y1="0" x2="812.5" y2="0"/>
      <line stroke="blue" x1="750" y1="0" x2="750" y2="750" elevation="0"/>
      <line stroke="red" x1="562.5" y1="562.5" x2="625" y2="625"/>
      <line stroke="black" x1="0" y1="812.5" x2="0" y2="875"/>
      <line stroke="red" x1="0" y1="812.5" x2="812.5" y2="812.5" elevation="1"/>
      <line stroke="black" x1="812.5" y1="0" x2="875" y2="0"/>
      <line stroke="red" x1="812.5" y1="0" x2="812.5" y2="812.5" elevation="1"/>
      <line stroke="black" x1="0" y1="875" x2="0" y2="937.5"/>
      <line stroke="blue" x1="0" y1="875" x2="875" y2="875" elevation="0"/>
      <line stroke="black" x1="875" y1="0" x2="937.5" y2="0"/>
      <line stroke="blue" x1="875" y1="0" x2="875" y2="875" elevation="0"/>
      <line stroke="blue" x1="625" y1="625" x2="687.5" y2="687.5"/>
      <line stroke="black" x1="0" y1="937.5" x2="0" y2="1000"/>
      <line stroke="red" x1="0" y1="937.5" x2="937.5" y2="937.5" elevation="1"/>
      <line stroke="black" x1="937.5" y1="0" x2="1000" y2="0"/>
      <line stroke="red" x1="937.5" y1="0" x2="937.5" y2="937.5" elevation="1"/>
      <line stroke="red" x1="687.5" y1="687.5" x2="750" y2="750"/>
      <line stroke="black" x1="0" y1="1000" x2="1000" y2="1000"/>
      <line stroke="black" x1="1000" y1="0" x2="1000" y2="1000"/>
      <line stroke="blue" x1="750" y1="750" x2="812.5" y2="812.5"/>
      <line stroke="red" x1="812.5" y1="812.5" x2="875" y2="875"/>
      <line stroke="blue" x1="875" y1="875" x2="937.5" y2="937.5"/>
      <line stroke="red" x1="937.5" y1="937.5" x2="1000" y2="1000"/>
    </svg>""",
)

JAGUAR_HEURISTICS_TEST = BoxPleatingShapingExample(
    base_example=JAGUAR,
    stick_figure=stick_figure_examples.JAGUAR,
    central_crease_removed=JAGUAR.central_crease_removed,
    assigned_hinge_name_pairs=[
        ("head", "front_right_leg"),
        ("head", "front_left_leg"),
        ("rear_right_leg", "body"),
        ("rear_left_leg", "tail"),
    ],
    rectangles=[
        shaping_elements.ShapingFlap.from_centers(
            circle_centers=[
                point_utils.Point(
                    x=0.5, y=0.07142857142857142, require_on_paper=False
                ),
                point_utils.Point(
                    x=0.5, y=-0.07142857142857142, require_on_paper=False
                ),
            ],
            radius=0.20203050891044208,
            grid_size=14,
            name="head",
        ).extend_creases(JAGUAR.base_creases),
        shaping_elements.ShapingFlap.from_centers(
            circle_centers=[
                point_utils.Point(
                    x=-0.07142857142857142, y=0.0, require_on_paper=False
                ),
                point_utils.Point(
                    x=0.14285714285714285, y=0.0, require_on_paper=False
                ),
            ],
            radius=0.30304576336566325,
            grid_size=14,
            name="front_right_leg",
        ).extend_creases(JAGUAR.base_creases),
        shaping_elements.ShapingFlap.from_centers(
            circle_centers=[
                point_utils.Point(
                    x=0.0, y=0.7857142857142857, require_on_paper=False
                ),
                point_utils.Point(x=0.0, y=1.0, require_on_paper=False),
            ],
            radius=0.30304576336566325,
            grid_size=14,
            name="rear_right_leg",
        ).extend_creases(JAGUAR.base_creases),
        shaping_elements.ShapingFlap.from_centers(
            circle_centers=[
                point_utils.Point(
                    x=0.5, y=0.8571428571428571, require_on_paper=False
                ),
                point_utils.Point(
                    x=0.5, y=1.1428571428571428, require_on_paper=False
                ),
            ],
            radius=0.4040610178208843,
            grid_size=14,
            name="tail",
        ).extend_creases(JAGUAR.base_creases),
        shaping_elements.ShapingFlap.from_centers(
            circle_centers=[
                point_utils.Point(
                    x=0.8571428571428571, y=0.0, require_on_paper=False
                ),
                point_utils.Point(
                    x=1.0714285714285714, y=0.0, require_on_paper=False
                ),
            ],
            radius=0.30304576336566325,
            grid_size=14,
            name="front_left_leg",
        ).extend_creases(JAGUAR.base_creases),
        shaping_elements.ShapingFlap.from_centers(
            circle_centers=[
                point_utils.Point(
                    x=1.0, y=0.7857142857142857, require_on_paper=False
                ),
                point_utils.Point(x=1.0, y=1.0, require_on_paper=False),
            ],
            radius=0.30304576336566313,
            grid_size=14,
            name="rear_left_leg",
        ).extend_creases(JAGUAR.base_creases),
    ],
    rivers=[
        shaping_elements.ShapingRiver(
            name="body",
            bank_1=[
                crease_utils.Crease(
                    point_utils.Point(
                        x=0.0, y=0.21428571428571427, require_on_paper=False
                    ),
                    point_utils.Point(
                        x=1.0, y=0.21428571428571427, require_on_paper=False
                    ),
                    fold=crease_utils.Fold.HINGE,
                )
            ],
            bank_2=[
                crease_utils.Crease(
                    point_utils.Point(
                        x=0.0, y=0.5714285714285714, require_on_paper=False
                    ),
                    point_utils.Point(
                        x=1.0, y=0.5714285714285714, require_on_paper=False
                    ),
                    fold=crease_utils.Fold.HINGE,
                )
            ],
            width=5,
            grid_size=14,
        ).extend_creases(JAGUAR.base_creases)
    ],
    zigzag_shaping_params=[
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="rear_left_leg",
            zigzag_params=shaping_types.ZigzagParams(
                target_flap_angle=1.5707963267948966,
                fold_order=[
                    crease_utils.Fold.MOUNTAIN,
                    crease_utils.Fold.VALLEY,
                ],
                even_phase=False,
                depth=0,
                fold_percent=1.0,
                target_zigzag_depth=None,
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="rear_right_leg",
            zigzag_params=shaping_types.ZigzagParams(
                target_flap_angle=1.5707963267948966,
                fold_order=[
                    crease_utils.Fold.MOUNTAIN,
                    crease_utils.Fold.VALLEY,
                ],
                even_phase=True,
                depth=0,
                fold_percent=1.0,
                target_zigzag_depth=None,
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="front_left_leg",
            zigzag_params=shaping_types.ZigzagParams(
                target_flap_angle=1.5707963267948966,
                fold_order=[
                    crease_utils.Fold.MOUNTAIN,
                    crease_utils.Fold.VALLEY,
                ],
                even_phase=False,
                depth=0,
                fold_percent=1.0,
                target_zigzag_depth=None,
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="front_right_leg",
            zigzag_params=shaping_types.ZigzagParams(
                target_flap_angle=1.5707963267948966,
                fold_order=[
                    crease_utils.Fold.MOUNTAIN,
                    crease_utils.Fold.VALLEY,
                ],
                even_phase=False,
                depth=0,
                fold_percent=1.0,
                target_zigzag_depth=None,
            ),
        ),
    ],
    inscribed_shaping_params=[],
)

GOALIE = BoxPleatingShapingExample(
    base_example=HUMAN,
    assigned_hinge_name_pairs=[
        ("left_arm", "torso"),
        ("right_arm", "torso"),
    ],
    rectangles=[
        shaping_elements.ShapingFlap.from_centers(
            circle_centers=[
                point_utils.Point(x=0.5, y=1/4, require_on_paper=False),
                point_utils.Point(x=0.5, y=-1/4, require_on_paper=False)
            ],
            radius=1 / 8 * math.sqrt(2),
            name="head",
            grid_size=8,
        ),
        shaping_elements.ShapingFlap.from_centers(
            circle_centers=[
                point_utils.Point(x=0.0, y=0.0, require_on_paper=False)
            ],
            radius=3 / 8 * math.sqrt(2),
            name="left_arm",
            grid_size=8,
        ),
        shaping_elements.ShapingFlap.from_centers(
            circle_centers=[
                point_utils.Point(x=1.0, y=0.0, require_on_paper=False)
            ],
            radius=3 / 8 * math.sqrt(2),
            name="right_arm",
            grid_size=8,
        ),
        shaping_elements.ShapingFlap.from_centers(
            circle_centers=[
                point_utils.Point(x=0.0, y=1.0, require_on_paper=False)
            ],
            radius=3 / 8 * math.sqrt(2),
            name="left_leg",
            grid_size=8,
        ),
        shaping_elements.ShapingFlap.from_centers(
            circle_centers=[
                point_utils.Point(x=1.0, y=1.0, require_on_paper=False)
            ],
            radius=3 / 8 * math.sqrt(2),
            name="right_leg",
            grid_size=8,
        ),
    ],
    rivers=[
        shaping_elements.ShapingRiver(
            name="torso",
            bank_1=[_make_hinge_crease((0, 3), (8, 3), 8)],
            bank_2=[_make_hinge_crease((0, 4), (8, 4), 8)],
            width=1,
            grid_size=8,
        ).extend_creases(HUMAN.base_creases),
    ],
    zigzag_shaping_params=[
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="left_arm",
            zigzag_params=shaping_types.ZigzagParams(
                target_zigzag_depth=1,
                target_flap_angle=None,
                even_phase=True,
                fold_order=[
                    crease_utils.Fold.MOUNTAIN,
                    crease_utils.Fold.VALLEY,
                ],
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="right_arm",
            zigzag_params=shaping_types.ZigzagParams(
                target_zigzag_depth=1,
                target_flap_angle=None,
                even_phase=True,
                fold_order=[
                    crease_utils.Fold.MOUNTAIN,
                    crease_utils.Fold.VALLEY,
                ],
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="left_leg",
            zigzag_params=shaping_types.ZigzagParams(
                target_zigzag_depth=2,
                target_flap_angle=None,
                even_phase=True,
                fold_order=[
                    crease_utils.Fold.MOUNTAIN,
                    crease_utils.Fold.VALLEY,
                ],
            ),
        ),
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name="right_leg",
            zigzag_params=shaping_types.ZigzagParams(
                target_zigzag_depth=2,
                target_flap_angle=None,
                even_phase=True,
                fold_order=[
                    crease_utils.Fold.MOUNTAIN,
                    crease_utils.Fold.VALLEY,
                ],
            ),
        ),
    ],
    inscribed_rectangle_shaping_params=[
        inscribed_rectangle_shaping.InscribedRectangleFlapShapingParams(
            flap_name="left_arm",
            inscribed_rectangles_params=[
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=1,
                    fold_percent=0.5,
                    fold_order=[
                        crease_utils.Fold.VALLEY,
                        crease_utils.Fold.MOUNTAIN,
                    ],
                ),
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=2,
                    fold_percent=0.25,
                    fold_order=[
                        crease_utils.Fold.VALLEY,
                        crease_utils.Fold.MOUNTAIN,
                    ],
                ),
            ],
        ),
        inscribed_rectangle_shaping.InscribedRectangleFlapShapingParams(
            flap_name="right_arm",
            inscribed_rectangles_params=[
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=1,
                    fold_percent=0.5,
                    fold_order=[
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.VALLEY,
                    ],
                ),
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=2,
                    fold_percent=0.25,
                    fold_order=[
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.VALLEY,
                    ],
                ),
            ],
        ),
        inscribed_rectangle_shaping.InscribedRectangleFlapShapingParams(
            flap_name="left_leg",
            inscribed_rectangles_params=[
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=2,
                    fold_percent=0.5,
                    fold_order=[
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.VALLEY,
                    ],
                ),
            ],
        ),
        inscribed_rectangle_shaping.InscribedRectangleFlapShapingParams(
            flap_name="right_leg",
            inscribed_rectangles_params=[
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=2,
                    fold_percent=0.5,
                    fold_order=[
                        crease_utils.Fold.VALLEY,
                        crease_utils.Fold.MOUNTAIN,
                    ],
                ),
            ],
        ),
    ],
    central_crease_removed=True,
    zigzag_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="red" x1="0" y1="0" x2="125" y2="125"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="375"/>
  <line stroke="black" x1="0" y1="0" x2="375" y2="0"/>
  <line stroke="red" x1="125" y1="125" x2="125" y2="250"/>
  <line stroke="red" x1="125" y1="125" x2="250" y2="125"/>
  <line stroke="blue" x1="125" y1="125" x2="250" y2="250"/>
  <line stroke="blue" x1="125" y1="250" x2="0" y2="375" fold-step="1"/>
  <line stroke="red" x1="125" y1="250" x2="125" y2="375"/>
  <line stroke="red" x1="125" y1="250" x2="250" y2="375" fold-step="1"/>
  <line stroke="red" x1="250" y1="125" x2="375" y2="0" fold-step="1"/>
  <line stroke="red" x1="250" y1="125" x2="375" y2="125"/>
  <line stroke="blue" x1="250" y1="125" x2="375" y2="250" fold-step="1"/>
  <line stroke="red" x1="250" y1="250" x2="312" y2="312"/>
  <line stroke="blue" x1="250" y1="250" x2="250" y2="375"/>
  <line stroke="blue" x1="250" y1="250" x2="375" y2="250"/>
  <line stroke="blue" x1="0" y1="375" x2="125" y2="375"/>
  <line stroke="black" x1="0" y1="375" x2="0" y2="625"/>
  <line stroke="black" x1="375" y1="0" x2="500" y2="0"/>
  <line stroke="red" x1="125" y1="375" x2="250" y2="375"/>
  <line stroke="red" x1="125" y1="375" x2="125" y2="875"/>
  <line stroke="blue" x1="375" y1="125" x2="500" y2="0"/>
  <line stroke="red" x1="375" y1="125" x2="500" y2="250"/>
  <line stroke="red" x1="375" y1="125" x2="625" y2="125"/>
  <line stroke="blue" x1="312" y1="312" x2="250" y2="375" fold-step="1"/>
  <line stroke="red" x1="312" y1="312" x2="375" y2="250" fold-step="1"/>
  <line stroke="red" x1="312" y1="312" x2="375" y2="375"/>
  <line stroke="blue" x1="250" y1="375" x2="375" y2="375"/>
  <line stroke="blue" x1="250" y1="375" x2="250" y2="625"/>
  <line stroke="blue" x1="375" y1="250" x2="500" y2="250"/>
  <line stroke="black" x1="500" y1="0" x2="625" y2="0"/>
  <line stroke="blue" x1="500" y1="0" x2="625" y2="125"/>
  <line stroke="red" x1="375" y1="375" x2="500" y2="250"/>
  <line stroke="red" x1="375" y1="375" x2="375" y2="625"/>
  <line stroke="red" x1="500" y1="250" x2="625" y2="125"/>
  <line stroke="blue" x1="500" y1="250" x2="625" y2="250"/>
  <line stroke="red" x1="500" y1="250" x2="625" y2="375"/>
  <line stroke="red" x1="0" y1="625" x2="125" y2="875" fold-step="3"/>
  <line stroke="black" x1="0" y1="625" x2="0" y2="1000"/>
  <line stroke="blue" x1="625" y1="0" x2="750" y2="125" fold-step="2"/>
  <line stroke="black" x1="625" y1="0" x2="1000" y2="0"/>
  <line stroke="red" x1="625" y1="125" x2="750" y2="125"/>
  <line stroke="red" x1="250" y1="625" x2="292" y2="708" fold-step="3"/>
  <line stroke="blue" x1="250" y1="625" x2="250" y2="750"/>
  <line stroke="blue" x1="250" y1="625" x2="125" y2="875" fold-step="3"/>
  <line stroke="blue" x1="625" y1="250" x2="688" y2="312" fold-step="2"/>
  <line stroke="red" x1="625" y1="250" x2="750" y2="125" fold-step="2"/>
  <line stroke="blue" x1="625" y1="250" x2="750" y2="250"/>
  <line stroke="red" x1="375" y1="625" x2="292" y2="708"/>
  <line stroke="red" x1="375" y1="625" x2="625" y2="625"/>
  <line stroke="blue" x1="375" y1="625" x2="500" y2="750"/>
  <line stroke="red" x1="625" y1="375" x2="688" y2="312"/>
  <line stroke="blue" x1="625" y1="375" x2="750" y2="375"/>
  <line stroke="red" x1="625" y1="375" x2="625" y2="625"/>
  <line stroke="red" x1="688" y1="312" x2="750" y2="250"/>
  <line stroke="red" x1="688" y1="312" x2="750" y2="375" fold-step="2"/>
  <line stroke="red" x1="750" y1="125" x2="875" y2="125"/>
  <line stroke="red" x1="292" y1="708" x2="250" y2="750"/>
  <line stroke="blue" x1="292" y1="708" x2="375" y2="750" fold-step="3"/>
  <line stroke="blue" x1="250" y1="750" x2="375" y2="750"/>
  <line stroke="blue" x1="250" y1="750" x2="125" y2="875"/>
  <line stroke="blue" x1="750" y1="250" x2="750" y2="375"/>
  <line stroke="blue" x1="750" y1="250" x2="875" y2="125"/>
  <line stroke="red" x1="375" y1="750" x2="125" y2="875" fold-step="3"/>
  <line stroke="blue" x1="375" y1="750" x2="500" y2="750"/>
  <line stroke="blue" x1="750" y1="375" x2="875" y2="250" fold-step="2"/>
  <line stroke="red" x1="750" y1="375" x2="875" y2="375"/>
  <line stroke="blue" x1="750" y1="375" x2="750" y2="625"/>
  <line stroke="red" x1="125" y1="875" x2="375" y2="875"/>
  <line stroke="red" x1="125" y1="875" x2="0" y2="1000"/>
  <line stroke="blue" x1="125" y1="875" x2="375" y2="1000" fold-step="3"/>
  <line stroke="blue" x1="625" y1="625" x2="500" y2="750"/>
  <line stroke="red" x1="625" y1="625" x2="708" y2="708"/>
  <line stroke="red" x1="875" y1="125" x2="875" y2="250"/>
  <line stroke="red" x1="875" y1="125" x2="1000" y2="0"/>
  <line stroke="red" x1="500" y1="750" x2="375" y2="875"/>
  <line stroke="blue" x1="500" y1="750" x2="625" y2="750"/>
  <line stroke="red" x1="500" y1="750" x2="625" y2="875"/>
  <line stroke="red" x1="875" y1="250" x2="875" y2="375"/>
  <line stroke="red" x1="875" y1="250" x2="1000" y2="375" fold-step="2"/>
  <line stroke="red" x1="375" y1="875" x2="625" y2="875"/>
  <line stroke="blue" x1="375" y1="875" x2="500" y2="1000"/>
  <line stroke="blue" x1="875" y1="375" x2="1000" y2="375"/>
  <line stroke="red" x1="875" y1="375" x2="875" y2="875"/>
  <line stroke="red" x1="625" y1="750" x2="708" y2="708" fold-step="4"/>
  <line stroke="blue" x1="625" y1="750" x2="750" y2="750"/>
  <line stroke="blue" x1="625" y1="750" x2="875" y2="875" fold-step="4"/>
  <line stroke="blue" x1="750" y1="625" x2="708" y2="708" fold-step="4"/>
  <line stroke="blue" x1="750" y1="625" x2="750" y2="750"/>
  <line stroke="red" x1="750" y1="625" x2="875" y2="875" fold-step="4"/>
  <line stroke="black" x1="0" y1="1000" x2="375" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="375"/>
  <line stroke="red" x1="708" y1="708" x2="750" y2="750"/>
  <line stroke="blue" x1="750" y1="750" x2="875" y2="875"/>
  <line stroke="black" x1="375" y1="1000" x2="500" y2="1000"/>
  <line stroke="black" x1="1000" y1="375" x2="1000" y2="625"/>
  <line stroke="blue" x1="625" y1="875" x2="500" y2="1000"/>
  <line stroke="red" x1="625" y1="875" x2="875" y2="875"/>
  <line stroke="black" x1="500" y1="1000" x2="625" y2="1000"/>
  <line stroke="red" x1="625" y1="1000" x2="875" y2="875" fold-step="4"/>
  <line stroke="black" x1="625" y1="1000" x2="1000" y2="1000"/>
  <line stroke="blue" x1="1000" y1="625" x2="875" y2="875" fold-step="4"/>
  <line stroke="black" x1="1000" y1="625" x2="1000" y2="1000"/>
  <line stroke="red" x1="875" y1="875" x2="1000" y2="1000"/>
</svg>""",
    inscribed_rectangle_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="0" y2="125"/>
  <line stroke="black" x1="0" y1="0" x2="125" y2="0"/>
  <line stroke="red" x1="0" y1="0" x2="125" y2="125"/>
  <line stroke="red" x1="0" y1="125" x2="125" y2="125" opacity="0.25" fold-step="5"/>
  <line stroke="black" x1="0" y1="125" x2="0" y2="250"/>
  <line stroke="blue" x1="125" y1="0" x2="125" y2="125" opacity="0.25" fold-step="5"/>
  <line stroke="black" x1="125" y1="0" x2="250" y2="0"/>
  <line stroke="red" x1="125" y1="125" x2="125" y2="250"/>
  <line stroke="red" x1="125" y1="125" x2="250" y2="125"/>
  <line stroke="blue" x1="125" y1="125" x2="250" y2="250"/>
  <line stroke="red" x1="0" y1="250" x2="125" y2="250" opacity="0.5" fold-step="5"/>
  <line stroke="black" x1="0" y1="250" x2="0" y2="375"/>
  <line stroke="blue" x1="250" y1="0" x2="250" y2="125" opacity="0.5" fold-step="5"/>
  <line stroke="black" x1="250" y1="0" x2="375" y2="0"/>
  <line stroke="blue" x1="125" y1="250" x2="250" y2="250" opacity="0.5" fold-step="5"/>
  <line stroke="blue" x1="125" y1="250" x2="0" y2="375" fold-step="1"/>
  <line stroke="red" x1="125" y1="250" x2="125" y2="375"/>
  <line stroke="red" x1="125" y1="250" x2="250" y2="375" fold-step="1"/>
  <line stroke="red" x1="250" y1="125" x2="250" y2="250" opacity="0.5" fold-step="5"/>
  <line stroke="red" x1="250" y1="125" x2="375" y2="0" fold-step="1"/>
  <line stroke="red" x1="250" y1="125" x2="375" y2="125"/>
  <line stroke="blue" x1="250" y1="125" x2="375" y2="250" fold-step="1"/>
  <line stroke="red" x1="250" y1="250" x2="312" y2="312"/>
  <line stroke="blue" x1="250" y1="250" x2="250" y2="375"/>
  <line stroke="blue" x1="250" y1="250" x2="375" y2="250"/>
  <line stroke="blue" x1="0" y1="375" x2="125" y2="375"/>
  <line stroke="black" x1="0" y1="375" x2="0" y2="625"/>
  <line stroke="black" x1="375" y1="0" x2="500" y2="0"/>
  <line stroke="red" x1="125" y1="375" x2="250" y2="375"/>
  <line stroke="red" x1="125" y1="375" x2="125" y2="875"/>
  <line stroke="blue" x1="375" y1="125" x2="500" y2="0"/>
  <line stroke="red" x1="375" y1="125" x2="500" y2="250"/>
  <line stroke="red" x1="375" y1="125" x2="625" y2="125"/>
  <line stroke="blue" x1="312" y1="312" x2="250" y2="375" fold-step="1"/>
  <line stroke="red" x1="312" y1="312" x2="375" y2="250" fold-step="1"/>
  <line stroke="red" x1="312" y1="312" x2="375" y2="375"/>
  <line stroke="blue" x1="250" y1="375" x2="375" y2="375"/>
  <line stroke="blue" x1="250" y1="375" x2="250" y2="625"/>
  <line stroke="blue" x1="375" y1="250" x2="500" y2="250"/>
  <line stroke="black" x1="500" y1="0" x2="625" y2="0"/>
  <line stroke="blue" x1="500" y1="0" x2="625" y2="125"/>
  <line stroke="red" x1="375" y1="375" x2="500" y2="250"/>
  <line stroke="red" x1="375" y1="375" x2="375" y2="625"/>
  <line stroke="red" x1="500" y1="250" x2="625" y2="125"/>
  <line stroke="blue" x1="500" y1="250" x2="625" y2="250"/>
  <line stroke="red" x1="500" y1="250" x2="625" y2="375"/>
  <line stroke="black" x1="0" y1="625" x2="0" y2="875"/>
  <line stroke="red" x1="0" y1="625" x2="125" y2="875" fold-step="3"/>
  <line stroke="black" x1="625" y1="0" x2="750" y2="0"/>
  <line stroke="blue" x1="625" y1="0" x2="750" y2="125" fold-step="2"/>
  <line stroke="red" x1="625" y1="125" x2="750" y2="125"/>
  <line stroke="red" x1="250" y1="625" x2="292" y2="708" fold-step="3"/>
  <line stroke="blue" x1="250" y1="625" x2="250" y2="750"/>
  <line stroke="blue" x1="250" y1="625" x2="125" y2="875" fold-step="3"/>
  <line stroke="blue" x1="625" y1="250" x2="688" y2="312" fold-step="2"/>
  <line stroke="red" x1="625" y1="250" x2="750" y2="125" fold-step="2"/>
  <line stroke="blue" x1="625" y1="250" x2="750" y2="250"/>
  <line stroke="red" x1="375" y1="625" x2="292" y2="708"/>
  <line stroke="red" x1="375" y1="625" x2="625" y2="625"/>
  <line stroke="blue" x1="375" y1="625" x2="500" y2="750"/>
  <line stroke="red" x1="625" y1="375" x2="688" y2="312"/>
  <line stroke="blue" x1="625" y1="375" x2="750" y2="375"/>
  <line stroke="red" x1="625" y1="375" x2="625" y2="625"/>
  <line stroke="blue" x1="750" y1="0" x2="750" y2="125" opacity="0.5" fold-step="6"/>
  <line stroke="black" x1="750" y1="0" x2="875" y2="0"/>
  <line stroke="red" x1="688" y1="312" x2="750" y2="250"/>
  <line stroke="red" x1="688" y1="312" x2="750" y2="375" fold-step="2"/>
  <line stroke="red" x1="750" y1="125" x2="750" y2="250" opacity="0.5" fold-step="6"/>
  <line stroke="red" x1="750" y1="125" x2="875" y2="125"/>
  <line stroke="red" x1="292" y1="708" x2="250" y2="750"/>
  <line stroke="blue" x1="292" y1="708" x2="375" y2="750" fold-step="3"/>
  <line stroke="blue" x1="250" y1="750" x2="375" y2="750"/>
  <line stroke="blue" x1="250" y1="750" x2="125" y2="875"/>
  <line stroke="blue" x1="750" y1="250" x2="750" y2="375"/>
  <line stroke="blue" x1="750" y1="250" x2="875" y2="125"/>
  <line stroke="blue" x1="750" y1="250" x2="875" y2="250" opacity="0.5" fold-step="6"/>
  <line stroke="red" x1="375" y1="750" x2="125" y2="875" fold-step="3"/>
  <line stroke="blue" x1="375" y1="750" x2="500" y2="750"/>
  <line stroke="blue" x1="750" y1="375" x2="875" y2="250" fold-step="2"/>
  <line stroke="red" x1="750" y1="375" x2="875" y2="375"/>
  <line stroke="blue" x1="750" y1="375" x2="750" y2="625"/>
  <line stroke="red" x1="0" y1="875" x2="125" y2="875" opacity="0.5" fold-step="7"/>
  <line stroke="black" x1="0" y1="875" x2="0" y2="1000"/>
  <line stroke="blue" x1="875" y1="0" x2="875" y2="125" opacity="0.25" fold-step="6"/>
  <line stroke="black" x1="875" y1="0" x2="1000" y2="0"/>
  <line stroke="red" x1="125" y1="875" x2="375" y2="875"/>
  <line stroke="red" x1="125" y1="875" x2="0" y2="1000"/>
  <line stroke="blue" x1="125" y1="875" x2="125" y2="1000" opacity="0.5" fold-step="7"/>
  <line stroke="blue" x1="125" y1="875" x2="375" y2="1000" fold-step="3"/>
  <line stroke="blue" x1="625" y1="625" x2="500" y2="750"/>
  <line stroke="red" x1="625" y1="625" x2="708" y2="708"/>
  <line stroke="red" x1="875" y1="125" x2="875" y2="250"/>
  <line stroke="red" x1="875" y1="125" x2="1000" y2="0"/>
  <line stroke="red" x1="875" y1="125" x2="1000" y2="125" opacity="0.25" fold-step="6"/>
  <line stroke="red" x1="500" y1="750" x2="375" y2="875"/>
  <line stroke="blue" x1="500" y1="750" x2="625" y2="750"/>
  <line stroke="red" x1="500" y1="750" x2="625" y2="875"/>
  <line stroke="red" x1="875" y1="250" x2="875" y2="375"/>
  <line stroke="red" x1="875" y1="250" x2="1000" y2="250" opacity="0.5" fold-step="6"/>
  <line stroke="red" x1="875" y1="250" x2="1000" y2="375" fold-step="2"/>
  <line stroke="red" x1="375" y1="875" x2="625" y2="875"/>
  <line stroke="blue" x1="375" y1="875" x2="500" y2="1000"/>
  <line stroke="blue" x1="875" y1="375" x2="1000" y2="375"/>
  <line stroke="red" x1="875" y1="375" x2="875" y2="875"/>
  <line stroke="red" x1="625" y1="750" x2="708" y2="708" fold-step="4"/>
  <line stroke="blue" x1="625" y1="750" x2="750" y2="750"/>
  <line stroke="blue" x1="625" y1="750" x2="875" y2="875" fold-step="4"/>
  <line stroke="blue" x1="750" y1="625" x2="708" y2="708" fold-step="4"/>
  <line stroke="blue" x1="750" y1="625" x2="750" y2="750"/>
  <line stroke="red" x1="750" y1="625" x2="875" y2="875" fold-step="4"/>
  <line stroke="black" x1="0" y1="1000" x2="125" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="125"/>
  <line stroke="red" x1="708" y1="708" x2="750" y2="750"/>
  <line stroke="black" x1="125" y1="1000" x2="375" y2="1000"/>
  <line stroke="black" x1="1000" y1="125" x2="1000" y2="250"/>
  <line stroke="black" x1="1000" y1="250" x2="1000" y2="375"/>
  <line stroke="blue" x1="750" y1="750" x2="875" y2="875"/>
  <line stroke="black" x1="375" y1="1000" x2="500" y2="1000"/>
  <line stroke="black" x1="1000" y1="375" x2="1000" y2="625"/>
  <line stroke="blue" x1="625" y1="875" x2="500" y2="1000"/>
  <line stroke="red" x1="625" y1="875" x2="875" y2="875"/>
  <line stroke="black" x1="500" y1="1000" x2="625" y2="1000"/>
  <line stroke="red" x1="625" y1="1000" x2="875" y2="875" fold-step="4"/>
  <line stroke="black" x1="625" y1="1000" x2="875" y2="1000"/>
  <line stroke="blue" x1="1000" y1="625" x2="875" y2="875" fold-step="4"/>
  <line stroke="black" x1="1000" y1="625" x2="1000" y2="875"/>
  <line stroke="blue" x1="875" y1="875" x2="875" y2="1000" opacity="0.5" fold-step="8"/>
  <line stroke="red" x1="875" y1="875" x2="1000" y2="875" opacity="0.5" fold-step="8"/>
  <line stroke="red" x1="875" y1="875" x2="1000" y2="1000"/>
  <line stroke="black" x1="875" y1="1000" x2="1000" y2="1000"/>
  <line stroke="black" x1="1000" y1="875" x2="1000" y2="1000"/>
</svg>""",
    narrowed_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="red" x1="0" y1="0" x2="62.5" y2="62.5"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="375"/>
  <line stroke="black" x1="0" y1="0" x2="375" y2="0"/>
  <line stroke="red" x1="62.5" y1="62.5" x2="62.5" y2="312.5"/>
  <line stroke="blue" x1="62.5" y1="62.5" x2="312.5" y2="62.5"/>
  <line stroke="red" x1="62.5" y1="62.5" x2="125" y2="125"/>
  <line stroke="blue" x1="125" y1="125" x2="187.5" y2="187.5"/>
  <line stroke="red" x1="125" y1="125" x2="125" y2="250"/>
  <line stroke="red" x1="125" y1="125" x2="250" y2="125"/>
  <line stroke="blue" x1="187.5" y1="187.5" x2="250" y2="250"/>
  <line stroke="blue" x1="187.5" y1="187.5" x2="187.5" y2="312.5"/>
  <line stroke="red" x1="187.5" y1="187.5" x2="312.5" y2="187.5"/>
  <line stroke="blue" x1="125" y1="250" x2="62.5" y2="312.5" fold-step="1"/>
  <line stroke="red" x1="125" y1="250" x2="125" y2="312.5"/>
  <line stroke="red" x1="125" y1="250" x2="187.5" y2="312.5" fold-step="1"/>
  <line stroke="red" x1="250" y1="125" x2="312.5" y2="62.5" fold-step="1"/>
  <line stroke="red" x1="250" y1="125" x2="312.5" y2="125"/>
  <line stroke="blue" x1="250" y1="125" x2="312.5" y2="187.5" fold-step="1"/>
  <line stroke="blue" x1="62.5" y1="312.5" x2="0" y2="375" fold-step="1"/>
  <line stroke="blue" x1="62.5" y1="312.5" x2="125" y2="312.5"/>
  <line stroke="red" x1="312.5" y1="62.5" x2="375" y2="0" fold-step="1"/>
  <line stroke="red" x1="312.5" y1="62.5" x2="312.5" y2="125"/>
  <line stroke="red" x1="125" y1="312.5" x2="187.5" y2="312.5"/>
  <line stroke="red" x1="125" y1="312.5" x2="125" y2="375"/>
  <line stroke="blue" x1="312.5" y1="125" x2="312.5" y2="187.5"/>
  <line stroke="red" x1="312.5" y1="125" x2="375" y2="125"/>
  <line stroke="red" x1="250" y1="250" x2="312.5" y2="312.5"/>
  <line stroke="blue" x1="250" y1="250" x2="250" y2="375"/>
  <line stroke="blue" x1="250" y1="250" x2="375" y2="250"/>
  <line stroke="red" x1="187.5" y1="312.5" x2="250" y2="375" fold-step="1"/>
  <line stroke="blue" x1="312.5" y1="187.5" x2="375" y2="250" fold-step="1"/>
  <line stroke="blue" x1="0" y1="375" x2="125" y2="375"/>
  <line stroke="black" x1="0" y1="375" x2="0" y2="625"/>
  <line stroke="black" x1="375" y1="0" x2="500" y2="0"/>
  <line stroke="red" x1="125" y1="375" x2="250" y2="375"/>
  <line stroke="red" x1="125" y1="375" x2="125" y2="796.9"/>
  <line stroke="blue" x1="375" y1="125" x2="500" y2="0"/>
  <line stroke="red" x1="375" y1="125" x2="500" y2="250"/>
  <line stroke="red" x1="375" y1="125" x2="625" y2="125"/>
  <line stroke="blue" x1="312.5" y1="312.5" x2="250" y2="375" fold-step="1"/>
  <line stroke="red" x1="312.5" y1="312.5" x2="375" y2="250" fold-step="1"/>
  <line stroke="red" x1="312.5" y1="312.5" x2="375" y2="375"/>
  <line stroke="blue" x1="250" y1="375" x2="375" y2="375"/>
  <line stroke="blue" x1="250" y1="375" x2="250" y2="625"/>
  <line stroke="blue" x1="375" y1="250" x2="500" y2="250"/>
  <line stroke="black" x1="500" y1="0" x2="625" y2="0"/>
  <line stroke="blue" x1="500" y1="0" x2="625" y2="125"/>
  <line stroke="red" x1="375" y1="375" x2="500" y2="250"/>
  <line stroke="red" x1="375" y1="375" x2="375" y2="625"/>
  <line stroke="red" x1="500" y1="250" x2="625" y2="125"/>
  <line stroke="blue" x1="500" y1="250" x2="625" y2="250"/>
  <line stroke="red" x1="500" y1="250" x2="625" y2="375"/>
  <line stroke="red" x1="0" y1="625" x2="62.5" y2="750" fold-step="3"/>
  <line stroke="black" x1="0" y1="625" x2="0" y2="1000"/>
  <line stroke="blue" x1="625" y1="0" x2="687.5" y2="62.5" fold-step="2"/>
  <line stroke="black" x1="625" y1="0" x2="1000" y2="0"/>
  <line stroke="red" x1="625" y1="125" x2="687.5" y2="125"/>
  <line stroke="red" x1="250" y1="625" x2="291.7" y2="708.3" fold-step="3"/>
  <line stroke="blue" x1="250" y1="625" x2="187.5" y2="750" fold-step="3"/>
  <line stroke="blue" x1="250" y1="625" x2="250" y2="750"/>
  <line stroke="red" x1="625" y1="250" x2="687.5" y2="187.5" fold-step="2"/>
  <line stroke="blue" x1="625" y1="250" x2="687.5" y2="312.5" fold-step="2"/>
  <line stroke="blue" x1="625" y1="250" x2="750" y2="250"/>
  <line stroke="blue" x1="687.5" y1="62.5" x2="687.5" y2="125"/>
  <line stroke="blue" x1="687.5" y1="62.5" x2="750" y2="125" fold-step="2"/>
  <line stroke="red" x1="687.5" y1="62.5" x2="937.5" y2="62.5"/>
  <line stroke="red" x1="687.5" y1="125" x2="687.5" y2="187.5"/>
  <line stroke="red" x1="687.5" y1="125" x2="750" y2="125"/>
  <line stroke="blue" x1="687.5" y1="187.5" x2="812.5" y2="187.5"/>
  <line stroke="red" x1="687.5" y1="187.5" x2="750" y2="125" fold-step="2"/>
  <line stroke="red" x1="375" y1="625" x2="291.7" y2="708.3"/>
  <line stroke="red" x1="375" y1="625" x2="625" y2="625"/>
  <line stroke="blue" x1="375" y1="625" x2="500" y2="750"/>
  <line stroke="red" x1="625" y1="375" x2="687.5" y2="312.5"/>
  <line stroke="blue" x1="625" y1="375" x2="750" y2="375"/>
  <line stroke="red" x1="625" y1="375" x2="625" y2="625"/>
  <line stroke="red" x1="62.5" y1="750" x2="125" y2="875" fold-step="3"/>
  <line stroke="red" x1="62.5" y1="750" x2="125" y2="796.9"/>
  <line stroke="blue" x1="62.5" y1="750" x2="62.5" y2="937.5"/>
  <line stroke="red" x1="687.5" y1="312.5" x2="750" y2="250"/>
  <line stroke="red" x1="687.5" y1="312.5" x2="750" y2="375" fold-step="2"/>
  <line stroke="red" x1="750" y1="125" x2="875" y2="125"/>
  <line stroke="red" x1="291.7" y1="708.3" x2="250" y2="750"/>
  <line stroke="blue" x1="291.7" y1="708.3" x2="375" y2="750" fold-step="3"/>
  <line stroke="red" x1="187.5" y1="750" x2="187.5" y2="812.5"/>
  <line stroke="blue" x1="187.5" y1="750" x2="125" y2="875" fold-step="3"/>
  <line stroke="blue" x1="187.5" y1="750" x2="125" y2="796.9"/>
  <line stroke="blue" x1="250" y1="750" x2="187.5" y2="812.5"/>
  <line stroke="blue" x1="250" y1="750" x2="375" y2="750"/>
  <line stroke="blue" x1="750" y1="250" x2="812.5" y2="187.5"/>
  <line stroke="blue" x1="750" y1="250" x2="750" y2="375"/>
  <line stroke="red" x1="125" y1="796.9" x2="125" y2="875"/>
  <line stroke="blue" x1="187.5" y1="812.5" x2="250" y2="812.5"/>
  <line stroke="blue" x1="187.5" y1="812.5" x2="125" y2="875"/>
  <line stroke="red" x1="812.5" y1="187.5" x2="812.5" y2="312.5"/>
  <line stroke="blue" x1="812.5" y1="187.5" x2="875" y2="125"/>
  <line stroke="red" x1="375" y1="750" x2="250" y2="812.5" fold-step="3"/>
  <line stroke="blue" x1="375" y1="750" x2="500" y2="750"/>
  <line stroke="blue" x1="750" y1="375" x2="812.5" y2="312.5" fold-step="2"/>
  <line stroke="red" x1="750" y1="375" x2="875" y2="375"/>
  <line stroke="blue" x1="750" y1="375" x2="750" y2="625"/>
  <line stroke="red" x1="250" y1="812.5" x2="203.1" y2="875"/>
  <line stroke="red" x1="250" y1="812.5" x2="125" y2="875" fold-step="3"/>
  <line stroke="blue" x1="812.5" y1="312.5" x2="875" y2="312.5"/>
  <line stroke="blue" x1="812.5" y1="312.5" x2="875" y2="250" fold-step="2"/>
  <line stroke="red" x1="125" y1="875" x2="203.1" y2="875"/>
  <line stroke="red" x1="125" y1="875" x2="62.5" y2="937.5"/>
  <line stroke="blue" x1="125" y1="875" x2="250" y2="937.5" fold-step="3"/>
  <line stroke="blue" x1="625" y1="625" x2="500" y2="750"/>
  <line stroke="red" x1="625" y1="625" x2="708.3" y2="708.3"/>
  <line stroke="red" x1="875" y1="125" x2="875" y2="250"/>
  <line stroke="red" x1="875" y1="125" x2="937.5" y2="62.5"/>
  <line stroke="red" x1="203.1" y1="875" x2="375" y2="875"/>
  <line stroke="blue" x1="203.1" y1="875" x2="250" y2="937.5"/>
  <line stroke="red" x1="500" y1="750" x2="375" y2="875"/>
  <line stroke="blue" x1="500" y1="750" x2="625" y2="750"/>
  <line stroke="red" x1="500" y1="750" x2="625" y2="875"/>
  <line stroke="red" x1="875" y1="250" x2="875" y2="312.5"/>
  <line stroke="red" x1="875" y1="250" x2="937.5" y2="312.5" fold-step="2"/>
  <line stroke="red" x1="875" y1="312.5" x2="875" y2="375"/>
  <line stroke="red" x1="875" y1="312.5" x2="937.5" y2="312.5"/>
  <line stroke="red" x1="62.5" y1="937.5" x2="250" y2="937.5"/>
  <line stroke="red" x1="62.5" y1="937.5" x2="0" y2="1000"/>
  <line stroke="red" x1="937.5" y1="62.5" x2="1000" y2="0"/>
  <line stroke="blue" x1="937.5" y1="62.5" x2="937.5" y2="312.5"/>
  <line stroke="red" x1="375" y1="875" x2="625" y2="875"/>
  <line stroke="blue" x1="375" y1="875" x2="500" y2="1000"/>
  <line stroke="blue" x1="875" y1="375" x2="1000" y2="375"/>
  <line stroke="red" x1="875" y1="375" x2="875" y2="796.9"/>
  <line stroke="blue" x1="250" y1="937.5" x2="375" y2="1000" fold-step="3"/>
  <line stroke="red" x1="625" y1="750" x2="708.3" y2="708.3" fold-step="4"/>
  <line stroke="blue" x1="625" y1="750" x2="750" y2="750"/>
  <line stroke="blue" x1="625" y1="750" x2="750" y2="812.5" fold-step="4"/>
  <line stroke="blue" x1="750" y1="625" x2="708.3" y2="708.3" fold-step="4"/>
  <line stroke="blue" x1="750" y1="625" x2="750" y2="750"/>
  <line stroke="red" x1="750" y1="625" x2="812.5" y2="750" fold-step="4"/>
  <line stroke="red" x1="937.5" y1="312.5" x2="1000" y2="375" fold-step="2"/>
  <line stroke="black" x1="0" y1="1000" x2="375" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="375"/>
  <line stroke="red" x1="708.3" y1="708.3" x2="750" y2="750"/>
  <line stroke="blue" x1="750" y1="750" x2="812.5" y2="812.5"/>
  <line stroke="black" x1="375" y1="1000" x2="500" y2="1000"/>
  <line stroke="black" x1="1000" y1="375" x2="1000" y2="625"/>
  <line stroke="blue" x1="625" y1="875" x2="500" y2="1000"/>
  <line stroke="red" x1="625" y1="875" x2="796.9" y2="875"/>
  <line stroke="blue" x1="750" y1="812.5" x2="796.9" y2="875"/>
  <line stroke="red" x1="812.5" y1="750" x2="875" y2="875" fold-step="4"/>
  <line stroke="red" x1="750" y1="812.5" x2="812.5" y2="812.5"/>
  <line stroke="blue" x1="750" y1="812.5" x2="875" y2="875" fold-step="4"/>
  <line stroke="blue" x1="812.5" y1="750" x2="812.5" y2="812.5"/>
  <line stroke="red" x1="812.5" y1="750" x2="875" y2="796.9"/>
  <line stroke="black" x1="500" y1="1000" x2="625" y2="1000"/>
  <line stroke="blue" x1="812.5" y1="812.5" x2="875" y2="875"/>
  <line stroke="red" x1="625" y1="1000" x2="750" y2="937.5" fold-step="4"/>
  <line stroke="black" x1="625" y1="1000" x2="1000" y2="1000"/>
  <line stroke="blue" x1="1000" y1="625" x2="937.5" y2="750" fold-step="4"/>
  <line stroke="black" x1="1000" y1="625" x2="1000" y2="1000"/>
  <line stroke="red" x1="796.9" y1="875" x2="750" y2="937.5"/>
  <line stroke="red" x1="796.9" y1="875" x2="875" y2="875"/>
  <line stroke="blue" x1="875" y1="796.9" x2="937.5" y2="750"/>
  <line stroke="red" x1="875" y1="796.9" x2="875" y2="875"/>
  <line stroke="red" x1="750" y1="937.5" x2="875" y2="875" fold-step="4"/>
  <line stroke="blue" x1="750" y1="937.5" x2="937.5" y2="937.5"/>
  <line stroke="red" x1="937.5" y1="750" x2="937.5" y2="937.5"/>
  <line stroke="blue" x1="937.5" y1="750" x2="875" y2="875" fold-step="4"/>
  <line stroke="red" x1="875" y1="875" x2="937.5" y2="937.5"/>
  <line stroke="red" x1="937.5" y1="937.5" x2="1000" y2="1000"/>
</svg>""",
)

PHOENIX_ZEBRA_BORDER_FLIPPED = BoxPleatingCreasePatternExample(
    name="phoenix_zebra_border_flipped",
    grid_size=16,
    packing_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">
  <line stroke="black" x1="0" y1="0" x2="0" y2="4"/>
  <line stroke="black" x1="0" y1="0" x2="5" y2="0"/>
  <line stroke="black" x1="0" y1="4" x2="0" y2="9"/>
  <line stroke="red" x1="0" y1="4" x2="5" y2="9"/>
  <line stroke="black" x1="5" y1="0" x2="7" y2="0"/>
  <line stroke="green" x1="5" y1="0" x2="5" y2="9"/>
  <line stroke="green" x1="7" y1="0" x2="7" y2="1"/>
  <line stroke="black" x1="7" y1="0" x2="8" y2="0"/>
  <line stroke="green" x1="7" y1="1" x2="7" y2="3"/>
  <line stroke="red" x1="7" y1="1" x2="8" y2="0"/>
  <line stroke="red" x1="7" y1="1" x2="8" y2="2"/>
  <line stroke="green" x1="7" y1="1" x2="9" y2="1"/>
  <line stroke="red" x1="7" y1="3" x2="8" y2="2"/>
  <line stroke="green" x1="7" y1="3" x2="7" y2="5"/>
  <line stroke="red" x1="7" y1="3" x2="8" y2="4"/>
  <line stroke="green" x1="7" y1="3" x2="9" y2="3"/>
  <line stroke="black" x1="8" y1="0" x2="9" y2="0"/>
  <line stroke="red" x1="8" y1="0" x2="9" y2="1"/>
  <line stroke="red" x1="8" y1="2" x2="9" y2="1"/>
  <line stroke="red" x1="8" y1="2" x2="9" y2="3"/>
  <line stroke="red" x1="7" y1="5" x2="8" y2="4"/>
  <line stroke="green" x1="7" y1="5" x2="7" y2="7"/>
  <line stroke="red" x1="7" y1="5" x2="8" y2="6"/>
  <line stroke="green" x1="7" y1="5" x2="9" y2="5"/>
  <line stroke="red" x1="8" y1="4" x2="9" y2="3"/>
  <line stroke="red" x1="8" y1="4" x2="9" y2="5"/>
  <line stroke="green" x1="0" y1="9" x2="5" y2="9"/>
  <line stroke="black" x1="0" y1="9" x2="0" y2="11"/>
  <line stroke="green" x1="9" y1="0" x2="9" y2="1"/>
  <line stroke="black" x1="9" y1="0"  x2="11" y2="0"/>
  <line stroke="green" x1="9" y1="1" x2="9" y2="3"/>
  <line stroke="green" x1="9" y1="3" x2="9" y2="5"/>
  <line stroke="red" x1="7" y1="7" x2="8" y2="6"/>
  <line stroke="red" x1="7" y1="7" x2="5" y2="9"/>
  <line stroke="green" x1="7" y1="7" x2="9" y2="7"/>
  <line stroke="red" x1="8" y1="6" x2="9" y2="5"/>
  <line stroke="red" x1="8" y1="6" x2="9" y2="7"/>
  <line stroke="green" x1="5" y1="9" x2="11" y2="9"/>
  <line stroke="green" x1="9" y1="5" x2="9" y2="7"/>
  <line stroke="green" x1="0" y1="11" x2="3" y2="11"/>
  <line stroke="black" x1="0" y1="11" x2="0" y2="14"/>
  <line stroke="green" x1="11" y1="0" x2="11" y2="9"/>
  <line stroke="black" x1="11" y1="0" x2="16" y2="0"/>
  <line stroke="red" x1="3" y1="11" x2="0" y2="14"/>
  <line stroke="green" x1="3" y1="11" x2="3" y2="16"/>
  <line stroke="green" x1="3" y1="11" x2="13" y2="11"/>
  <line stroke="red" x1="3" y1="11" x2="8" y2="16"/>
  <line stroke="red" x1="9" y1="7" x2="11" y2="9"/>
  <line stroke="black" x1="0" y1="14" x2="0" y2="16"/>
  <line stroke="red" x1="11" y1="9" x2="16" y2="4"/>
  <line stroke="green" x1="11" y1="9" x2="16" y2="9"/>
  <line stroke="black" x1="0" y1="16" x2="3" y2="16"/>
  <line stroke="black" x1="16" y1="0" x2="16" y2="4"/>
  <line stroke="black" x1="3" y1="16" x2="8" y2="16"/>
  <line stroke="black" x1="16" y1="4" x2="16" y2="9"/>
  <line stroke="red" x1="13" y1="11" x2="8" y2="16"/>
  <line stroke="green" x1="13" y1="11" x2="16" y2="11"/>
  <line stroke="green" x1="13" y1="11" x2="13" y2="16"/>
  <line stroke="red" x1="13" y1="11" x2="16" y2="14"/>
  <line stroke="black" x1="8" y1="16" x2="13" y2="16"/>
  <line stroke="black" x1="16" y1="9" x2="16" y2="11"/>
  <line stroke="black" x1="16" y1="11" x2="16" y2="14"/>
  <line stroke="black" x1="13" y1="16" x2="16" y2="16"/>
  <line stroke="black" x1="16" y1="14" x2="16" y2="16"/>
</svg>""",
    base_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">
  <line stroke="black" x1="0" y1="0" x2="0" y2="1"/>
  <line stroke="black" x1="0" y1="0" x2="5" y2="0"/>
  <line stroke="black" x1="0" y1="1" x2="0" y2="2"/>
  <line stroke="red" x1="0" y1="1" x2="5" y2="1"/>
  <line stroke="black" x1="0" y1="2" x2="0" y2="3"/>
  <line stroke="blue" x1="0" y1="2" x2="5" y2="2"/>
  <line stroke="black" x1="0" y1="3" x2="0" y2="4"/>
  <line stroke="red" x1="0" y1="3" x2="5" y2="3"/>
  <line stroke="red" x1="0" y1="4" x2="1" y2="5"/>
  <line stroke="blue" x1="0" y1="4" x2="5" y2="4"/>
  <line stroke="black" x1="0" y1="4" x2="0" y2="11"/>
  <line stroke="blue" x1="5" y1="0" x2="5" y2="1"/>
  <line stroke="black" x1="5" y1="0" x2="8" y2="0"/>
  <line stroke="blue" x1="1" y1="5" x2="2" y2="6"/>
  <line stroke="red" x1="1" y1="5" x2="5" y2="5"/>
  <line stroke="red" x1="1" y1="5" x2="1" y2="11"/>
  <line stroke="red" x1="5" y1="1" x2="5" y2="2"/>
  <line stroke="red" x1="5" y1="1" x2="7" y2="1"/>
  <line stroke="blue" x1="5" y1="2" x2="5" y2="3"/>
  <line stroke="blue" x1="5" y1="2" x2="8" y2="2"/>
  <line stroke="red" x1="5" y1="3" x2="5" y2="4"/>
  <line stroke="red" x1="5" y1="3" x2="7" y2="3"/>
  <line stroke="red" x1="2" y1="6" x2="3" y2="7"/>
  <line stroke="blue" x1="2" y1="6" x2="5" y2="6"/>
  <line stroke="blue" x1="2" y1="6" x2="2" y2="11"/>
  <line stroke="blue" x1="5" y1="4" x2="5" y2="5"/>
  <line stroke="blue" x1="5" y1="4" x2="8" y2="4"/>
  <line stroke="red" x1="5" y1="5" x2="5" y2="6"/>
  <line stroke="red" x1="5" y1="5" x2="7" y2="5"/>
  <line stroke="red" x1="7" y1="1" x2="8" y2="0"/>
  <line stroke="blue" x1="7" y1="1" x2="8" y2="1"/>
  <line stroke="red" x1="7" y1="1" x2="8" y2="2"/>
  <line stroke="red" x1="3" y1="7" x2="5" y2="7"/>
  <line stroke="blue" x1="3" y1="7" x2="4" y2="8"/>
  <line stroke="red" x1="3" y1="7" x2="3" y2="11"/>
  <line stroke="red" x1="7" y1="3" x2="8" y2="2"/>
  <line stroke="blue" x1="7" y1="3" x2="8" y2="3"/>
  <line stroke="red" x1="7" y1="3" x2="8" y2="4"/>
  <line stroke="blue" x1="5" y1="6" x2="5" y2="7"/>
  <line stroke="blue" x1="5" y1="6" x2="8" y2="6"/>
  <line stroke="blue" x1="8" y1="0" x2="8" y2="1"/>
  <line stroke="red" x1="8" y1="0" x2="9" y2="1"/>
  <line stroke="black" x1="8" y1="0" x2="11" y2="0"/>
  <line stroke="red" x1="8" y1="1" x2="8" y2="2"/>
  <line stroke="blue" x1="8" y1="1" x2="9" y2="1"/>
  <line stroke="blue" x1="8" y1="2" x2="8" y2="3"/>
  <line stroke="red" x1="8" y1="2" x2="9" y2="1"/>
  <line stroke="red" x1="8" y1="2" x2="9" y2="3"/>
  <line stroke="blue" x1="8" y1="2" x2="11" y2="2"/>
  <line stroke="red" x1="8" y1="3" x2="8" y2="4"/>
  <line stroke="blue" x1="8" y1="3" x2="9" y2="3"/>
  <line stroke="red" x1="5" y1="7" x2="5" y2="8"/>
  <line stroke="red" x1="5" y1="7" x2="7" y2="7"/>
  <line stroke="red" x1="7" y1="5" x2="8" y2="4"/>
  <line stroke="blue" x1="7" y1="5" x2="8" y2="5"/>
  <line stroke="red" x1="7" y1="5" x2="8" y2="6"/>
  <line stroke="blue" x1="4" y1="8" x2="5" y2="8"/>
  <line stroke="red" x1="4" y1="8" x2="5" y2="9"/>
  <line stroke="blue" x1="4" y1="8" x2="4" y2="12"/>
  <line stroke="blue" x1="8" y1="4" x2="8" y2="5"/>
  <line stroke="red" x1="8" y1="4" x2="9" y2="3"/>
  <line stroke="red" x1="8" y1="4" x2="9" y2="5"/>
  <line stroke="blue" x1="8" y1="4" x2="11" y2="4"/>
  <line stroke="red" x1="9" y1="1" x2="11" y2="1"/>
  <line stroke="blue" x1="5" y1="8" x2="6" y2="8"/>
  <line stroke="blue" x1="5" y1="8" x2="5" y2="9"/>
  <line stroke="red" x1="8" y1="5" x2="8" y2="6"/>
  <line stroke="blue" x1="8" y1="5" x2="9" y2="5"/>
  <line stroke="red" x1="9" y1="3" x2="11" y2="3"/>
  <line stroke="blue" x1="7" y1="7" x2="6" y2="8"/>
  <line stroke="red" x1="7" y1="7" x2="8" y2="6"/>
  <line stroke="red" x1="7" y1="7" x2="7" y2="15"/>
  <line stroke="red" x1="6" y1="8" x2="5" y2="9"/>
  <line stroke="blue" x1="6" y1="8" x2="6" y2="14"/>
  <line stroke="red" x1="8" y1="6" x2="9" y2="5"/>
  <line stroke="red" x1="8" y1="6" x2="9" y2="7"/>
  <line stroke="blue" x1="8" y1="6" x2="11" y2="6"/>
  <line stroke="blue" x1="8" y1="6" x2="8" y2="16"/>
  <line stroke="red" x1="5" y1="9" x2="5" y2="13"/>
  <line stroke="red" x1="9" y1="5" x2="11" y2="5"/>
  <line stroke="blue" x1="0" y1="11" x2="1" y2="11"/>
  <line stroke="black" x1="0" y1="11" x2="0" y2="14"/>
  <line stroke="blue" x1="11" y1="0" x2="11" y2="1"/>
  <line stroke="black" x1="11" y1="0" x2="16" y2="0"/>
  <line stroke="red" x1="1" y1="11" x2="2" y2="11"/>
  <line stroke="red" x1="1" y1="11" x2="1" y2="13"/>
  <line stroke="red" x1="11" y1="1" x2="11" y2="2"/>
  <line stroke="red" x1="11" y1="1" x2="16" y2="1"/>
  <line stroke="blue" x1="2" y1="11" x2="3" y2="11"/>
  <line stroke="blue" x1="2" y1="11" x2="2" y2="12"/>
  <line stroke="blue" x1="11" y1="2" x2="11" y2="3"/>
  <line stroke="blue" x1="11" y1="2" x2="16" y2="2"/>
  <line stroke="red" x1="3" y1="11" x2="2" y2="12"/>
  <line stroke="red" x1="3" y1="11" x2="4" y2="12"/>
  <line stroke="blue" x1="9" y1="7" x2="10" y2="8"/>
  <line stroke="red" x1="9" y1="7" x2="11" y2="7"/>
  <line stroke="red" x1="9" y1="7" x2="9" y2="15"/>
  <line stroke="red" x1="11" y1="3" x2="11" y2="4"/>
  <line stroke="red" x1="11" y1="3" x2="16" y2="3"/>
  <line stroke="blue" x1="11" y1="4" x2="11" y2="5"/>
  <line stroke="blue" x1="11" y1="4" x2="16" y2="4"/>
  <line stroke="red" x1="11" y1="5" x2="11" y2="6"/>
  <line stroke="red" x1="11" y1="5" x2="15" y2="5"/>
  <line stroke="blue" x1="2" y1="12" x2="4" y2="12"/>
  <line stroke="blue" x1="2" y1="12" x2="1" y2="13"/>
  <line stroke="blue" x1="11" y1="6" x2="11" y2="7"/>
  <line stroke="blue" x1="11" y1="6" x2="14" y2="6"/>
  <line stroke="blue" x1="4" y1="12" x2="5" y2="13"/>
  <line stroke="blue" x1="10" y1="8" x2="11" y2="8"/>
  <line stroke="red" x1="10" y1="8" x2="11" y2="9"/>
  <line stroke="blue" x1="10" y1="8" x2="10" y2="14"/>
  <line stroke="red" x1="1" y1="13" x2="5" y2="13"/>
  <line stroke="red" x1="1" y1="13" x2="0" y2="14"/>
  <line stroke="red" x1="11" y1="7" x2="11" y2="8"/>
  <line stroke="red" x1="11" y1="7" x2="13" y2="7"/>
  <line stroke="blue" x1="11" y1="8" x2="11" y2="9"/>
  <line stroke="blue" x1="11" y1="8" x2="12" y2="8"/>
  <line stroke="red" x1="5" y1="13" x2="6" y2="14"/>
  <line stroke="black" x1="0" y1="14" x2="0" y2="15"/>
  <line stroke="blue" x1="0" y1="14" x2="6" y2="14"/>
  <line stroke="red" x1="11" y1="9" x2="12" y2="8"/>
  <line stroke="red" x1="11" y1="9" x2="11" y2="13"/>
  <line stroke="blue" x1="12" y1="8" x2="13" y2="7"/>
  <line stroke="blue" x1="12" y1="8" x2="12" y2="12"/>
  <line stroke="red" x1="13" y1="7" x2="14" y2="6"/>
  <line stroke="red" x1="13" y1="7" x2="13" y2="11"/>
  <line stroke="black" x1="0" y1="15" x2="0" y2="16"/>
  <line stroke="red" x1="0" y1="15" x2="7" y2="15"/>
  <line stroke="blue" x1="6" y1="14" x2="7" y2="15"/>
  <line stroke="blue" x1="14" y1="6" x2="15" y2="5"/>
  <line stroke="blue" x1="14" y1="6" x2="14" y2="11"/>
  <line stroke="red" x1="15" y1="5" x2="16" y2="4"/>
  <line stroke="red" x1="15" y1="5" x2="15" y2="11"/>
  <line stroke="black" x1="0" y1="16" x2="8" y2="16"/>
  <line stroke="black" x1="16" y1="0" x2="16" y2="1"/>
  <line stroke="black" x1="16" y1="1" x2="16" y2="2"/>
  <line stroke="black" x1="16" y1="2" x2="16" y2="3"/>
  <line stroke="black" x1="16" y1="3" x2="16" y2="4"/>
  <line stroke="black" x1="16" y1="4" x2="16" y2="11"/>
  <line stroke="red" x1="7" y1="15" x2="8" y2="16"/>
  <line stroke="blue" x1="12" y1="12" x2="11" y2="13"/>
  <line stroke="red" x1="12" y1="12" x2="13" y2="11"/>
  <line stroke="blue" x1="12" y1="12" x2="14" y2="12"/>
  <line stroke="red" x1="11" y1="13" x2="10" y2="14"/>
  <line stroke="red" x1="11" y1="13" x2="15" y2="13"/>
  <line stroke="blue" x1="13" y1="11" x2="14" y2="11"/>
  <line stroke="red" x1="13" y1="11" x2="14" y2="12"/>
  <line stroke="blue" x1="10" y1="14" x2="9" y2="15"/>
  <line stroke="blue" x1="10" y1="14" x2="16" y2="14"/>
  <line stroke="red" x1="9" y1="15" x2="8" y2="16"/>
  <line stroke="red" x1="9" y1="15" x2="16" y2="15"/>
  <line stroke="blue" x1="14" y1="11" x2="14" y2="12"/>
  <line stroke="red" x1="14" y1="11" x2="15" y2="11"/>
  <line stroke="black" x1="8" y1="16" x2="16" y2="16"/>
  <line stroke="blue" x1="14" y1="12" x2="15" y2="13"/>
  <line stroke="blue" x1="15" y1="11" x2="16" y2="11"/>
  <line stroke="red" x1="15" y1="11" x2="15" y2="13"/>
  <line stroke="black" x1="16" y1="11" x2="16" y2="14"/>
  <line stroke="red" x1="15" y1="13" x2="16" y2="14"/>
  <line stroke="black" x1="16" y1="14" x2="16" y2="15"/>
  <line stroke="black" x1="16" y1="15" x2="16" y2="16"/>
</svg>""",
    rectangles=[
        shaping_utils.Rectangle(
            name="right_tail",
            x_start=0,
            y_start=0,
            x_len=0.3125,
            y_len=0.5625,
        ),
        shaping_utils.Rectangle(
            name="rear1",
            x_start=0.4375,
            y_start=0,
            x_len=0.125,
            y_len=0.0625,
        ),
        shaping_utils.Rectangle(
            name="rear2",
            x_start=0.4375,
            y_start=0.0625,
            x_len=0.125,
            y_len=0.125,
        ),
        shaping_utils.Rectangle(
            name="rear3",
            x_start=0.4375,
            y_start=0.1875,
            x_len=0.125,
            y_len=0.125,
        ),
        shaping_utils.Rectangle(
            name="rear4",
            x_start=0.4375,
            y_start=0.3125,
            x_len=0.125,
            y_len=0.125,
        ),
        shaping_utils.Rectangle(
            name="left_tail",
            x_start=0.6875,
            y_start=0,
            x_len=0.3125,
            y_len=0.5625,
        ),
        shaping_utils.Rectangle(
            name="right_leg",
            x_start=0,
            y_start=0.6875,
            x_len=0.1875,
            y_len=0.3125,
        ),
        shaping_utils.Rectangle(
            name="head",
            x_start=0.1875,
            y_start=0.6875,
            x_len=0.625,
            y_len=0.3125,
        ),
        shaping_utils.Rectangle(
            name="left_leg",
            x_start=0.8125,
            y_start=0.6875,
            x_len=0.1875,
            y_len=0.3125,
        ),
    ],
)

PHOENIX_FAN_OUT_RIGHT_TAIL = BoxPleatingShapingExample(
    base_example=PHOENIX_ZEBRA_BORDER_FLIPPED,
    fan_out_shaping_params=[
        fan_out_shaping.FanOutShapingParams(
            valid_fold_types=[crease_utils.Fold.VALLEY],
            valid_orientations=[fan_out_shaping.Orientation.HORIZONTAL],
            rectangles=[
                shaping_utils.Rectangle(
                    name="right_tail",
                    x_start=0,
                    y_start=0,
                    x_len=0.3125,
                    y_len=0.5625,
                )
            ],
            move_delta=-1.0,
        ),
    ],
    fan_out_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="0" y2="62"/>
  <line stroke="black" x1="0" y1="0" x2="312" y2="0"/>
  <line stroke="black" x1="0" y1="62" x2="0" y2="188"/>
  <line stroke="blue" x1="0" y1="62" x2="312" y2="125"/>
  <line stroke="red" x1="0" y1="62" x2="339" y2="62"/>
  <line stroke="black" x1="0" y1="188" x2="0" y2="250"/>
  <line stroke="red" x1="0" y1="188" x2="325" y2="188"/>
  <line stroke="blue" x1="0" y1="188" x2="312" y2="250"/>
  <line stroke="red" x1="0" y1="250" x2="62" y2="312"/>
  <line stroke="black" x1="0" y1="250" x2="0" y2="688"/>
  <line stroke="blue" x1="312" y1="0" x2="339" y2="62"/>
  <line stroke="black" x1="312" y1="0" x2="500" y2="0"/>
  <line stroke="blue" x1="62" y1="312" x2="125" y2="375"/>
  <line stroke="red" x1="62" y1="312" x2="312" y2="312"/>
  <line stroke="red" x1="62" y1="312" x2="62" y2="688"/>
  <line stroke="red" x1="312" y1="125" x2="339" y2="62"/>
  <line stroke="blue" x1="312" y1="125" x2="325" y2="188"/>
  <line stroke="blue" x1="312" y1="125" x2="500" y2="125"/>
  <line stroke="red" x1="339" y1="62" x2="438" y2="62"/>
  <line stroke="red" x1="325" y1="188" x2="312" y2="250"/>
  <line stroke="red" x1="325" y1="188" x2="438" y2="188"/>
  <line stroke="red" x1="125" y1="375" x2="188" y2="438"/>
  <line stroke="blue" x1="125" y1="375" x2="312" y2="375"/>
  <line stroke="blue" x1="125" y1="375" x2="125" y2="688"/>
  <line stroke="blue" x1="312" y1="250" x2="312" y2="312"/>
  <line stroke="blue" x1="312" y1="250" x2="500" y2="250"/>
  <line stroke="red" x1="312" y1="312" x2="312" y2="375"/>
  <line stroke="red" x1="312" y1="312" x2="438" y2="312"/>
  <line stroke="red" x1="438" y1="62" x2="500" y2="0"/>
  <line stroke="blue" x1="438" y1="62" x2="500" y2="62"/>
  <line stroke="red" x1="438" y1="62" x2="500" y2="125"/>
  <line stroke="red" x1="188" y1="438" x2="312" y2="438"/>
  <line stroke="blue" x1="188" y1="438" x2="250" y2="500"/>
  <line stroke="red" x1="188" y1="438" x2="188" y2="688"/>
  <line stroke="red" x1="438" y1="188" x2="500" y2="125"/>
  <line stroke="blue" x1="438" y1="188" x2="500" y2="188"/>
  <line stroke="red" x1="438" y1="188" x2="500" y2="250"/>
  <line stroke="blue" x1="312" y1="375" x2="312" y2="438"/>
  <line stroke="blue" x1="312" y1="375" x2="500" y2="375"/>
  <line stroke="blue" x1="500" y1="0" x2="500" y2="62"/>
  <line stroke="red" x1="500" y1="0" x2="562" y2="62"/>
  <line stroke="black" x1="500" y1="0" x2="688" y2="0"/>
  <line stroke="red" x1="500" y1="62" x2="500" y2="125"/>
  <line stroke="blue" x1="500" y1="62" x2="562" y2="62"/>
  <line stroke="blue" x1="500" y1="125" x2="500" y2="188"/>
  <line stroke="red" x1="500" y1="125" x2="562" y2="62"/>
  <line stroke="red" x1="500" y1="125" x2="562" y2="188"/>
  <line stroke="blue" x1="500" y1="125" x2="688" y2="125"/>
  <line stroke="red" x1="500" y1="188" x2="500" y2="250"/>
  <line stroke="blue" x1="500" y1="188" x2="562" y2="188"/>
  <line stroke="red" x1="312" y1="438" x2="312" y2="500"/>
  <line stroke="red" x1="312" y1="438" x2="438" y2="438"/>
  <line stroke="red" x1="438" y1="312" x2="500" y2="250"/>
  <line stroke="blue" x1="438" y1="312" x2="500" y2="312"/>
  <line stroke="red" x1="438" y1="312" x2="500" y2="375"/>
  <line stroke="blue" x1="250" y1="500" x2="312" y2="500"/>
  <line stroke="red" x1="250" y1="500" x2="312" y2="562"/>
  <line stroke="blue" x1="250" y1="500" x2="250" y2="750"/>
  <line stroke="blue" x1="500" y1="250" x2="500" y2="312"/>
  <line stroke="red" x1="500" y1="250" x2="562" y2="188"/>
  <line stroke="red" x1="500" y1="250" x2="562" y2="312"/>
  <line stroke="blue" x1="500" y1="250" x2="688" y2="250"/>
  <line stroke="red" x1="562" y1="62" x2="688" y2="62"/>
  <line stroke="blue" x1="312" y1="500" x2="375" y2="500"/>
  <line stroke="blue" x1="312" y1="500" x2="312" y2="562"/>
  <line stroke="red" x1="500" y1="312" x2="500" y2="375"/>
  <line stroke="blue" x1="500" y1="312" x2="562" y2="312"/>
  <line stroke="red" x1="562" y1="188" x2="688" y2="188"/>
  <line stroke="blue" x1="438" y1="438" x2="375" y2="500"/>
  <line stroke="red" x1="438" y1="438" x2="500" y2="375"/>
  <line stroke="red" x1="438" y1="438" x2="438" y2="938"/>
  <line stroke="red" x1="375" y1="500" x2="312" y2="562"/>
  <line stroke="blue" x1="375" y1="500" x2="375" y2="875"/>
  <line stroke="red" x1="500" y1="375" x2="562" y2="312"/>
  <line stroke="red" x1="500" y1="375" x2="562" y2="438"/>
  <line stroke="blue" x1="500" y1="375" x2="688" y2="375"/>
  <line stroke="blue" x1="500" y1="375" x2="500" y2="1000"/>
  <line stroke="red" x1="312" y1="562" x2="312" y2="812"/>
  <line stroke="red" x1="562" y1="312" x2="688" y2="312"/>
  <line stroke="blue" x1="0" y1="688" x2="62" y2="688"/>
  <line stroke="black" x1="0" y1="688" x2="0" y2="875"/>
  <line stroke="blue" x1="688" y1="0" x2="688" y2="62"/>
  <line stroke="black" x1="688" y1="0" x2="1000" y2="0"/>
  <line stroke="red" x1="62" y1="688" x2="125" y2="688"/>
  <line stroke="red" x1="62" y1="688" x2="62" y2="812"/>
  <line stroke="red" x1="688" y1="62" x2="688" y2="125"/>
  <line stroke="red" x1="688" y1="62" x2="1000" y2="62"/>
  <line stroke="blue" x1="125" y1="688" x2="188" y2="688"/>
  <line stroke="blue" x1="125" y1="688" x2="125" y2="750"/>
  <line stroke="blue" x1="688" y1="125" x2="688" y2="188"/>
  <line stroke="blue" x1="688" y1="125" x2="1000" y2="125"/>
  <line stroke="blue" x1="562" y1="438" x2="625" y2="500"/>
  <line stroke="red" x1="562" y1="438" x2="688" y2="438"/>
  <line stroke="red" x1="562" y1="438" x2="562" y2="938"/>
  <line stroke="red" x1="188" y1="688" x2="125" y2="750"/>
  <line stroke="red" x1="188" y1="688" x2="250" y2="750"/>
  <line stroke="red" x1="688" y1="188" x2="688" y2="250"/>
  <line stroke="red" x1="688" y1="188" x2="1000" y2="188"/>
  <line stroke="blue" x1="688" y1="250" x2="688" y2="312"/>
  <line stroke="blue" x1="688" y1="250" x2="1000" y2="250"/>
  <line stroke="red" x1="688" y1="312" x2="688" y2="375"/>
  <line stroke="red" x1="688" y1="312" x2="938" y2="312"/>
  <line stroke="blue" x1="125" y1="750" x2="250" y2="750"/>
  <line stroke="blue" x1="125" y1="750" x2="62" y2="812"/>
  <line stroke="blue" x1="688" y1="375" x2="688" y2="438"/>
  <line stroke="blue" x1="688" y1="375" x2="875" y2="375"/>
  <line stroke="blue" x1="250" y1="750" x2="312" y2="812"/>
  <line stroke="blue" x1="625" y1="500" x2="688" y2="500"/>
  <line stroke="red" x1="625" y1="500" x2="688" y2="562"/>
  <line stroke="blue" x1="625" y1="500" x2="625" y2="875"/>
  <line stroke="red" x1="62" y1="812" x2="312" y2="812"/>
  <line stroke="red" x1="62" y1="812" x2="0" y2="875"/>
  <line stroke="red" x1="688" y1="438" x2="688" y2="500"/>
  <line stroke="red" x1="688" y1="438" x2="812" y2="438"/>
  <line stroke="blue" x1="688" y1="500" x2="688" y2="562"/>
  <line stroke="blue" x1="688" y1="500" x2="750" y2="500"/>
  <line stroke="red" x1="312" y1="812" x2="375" y2="875"/>
  <line stroke="black" x1="0" y1="875" x2="0" y2="938"/>
  <line stroke="blue" x1="0" y1="875" x2="375" y2="875"/>
  <line stroke="red" x1="688" y1="562" x2="750" y2="500"/>
  <line stroke="red" x1="688" y1="562" x2="688" y2="812"/>
  <line stroke="blue" x1="750" y1="500" x2="812" y2="438"/>
  <line stroke="blue" x1="750" y1="500" x2="750" y2="750"/>
  <line stroke="red" x1="812" y1="438" x2="875" y2="375"/>
  <line stroke="red" x1="812" y1="438" x2="812" y2="688"/>
  <line stroke="black" x1="0" y1="938" x2="0" y2="1000"/>
  <line stroke="red" x1="0" y1="938" x2="438" y2="938"/>
  <line stroke="blue" x1="375" y1="875" x2="438" y2="938"/>
  <line stroke="blue" x1="875" y1="375" x2="938" y2="312"/>
  <line stroke="blue" x1="875" y1="375" x2="875" y2="688"/>
  <line stroke="red" x1="938" y1="312" x2="1000" y2="250"/>
  <line stroke="red" x1="938" y1="312" x2="938" y2="688"/>
  <line stroke="black" x1="0" y1="1000" x2="500" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="62"/>
  <line stroke="black" x1="1000" y1="62" x2="1000" y2="125"/>
  <line stroke="black" x1="1000" y1="125" x2="1000" y2="188"/>
  <line stroke="black" x1="1000" y1="188" x2="1000" y2="250"/>
  <line stroke="black" x1="1000" y1="250" x2="1000" y2="688"/>
  <line stroke="red" x1="438" y1="938" x2="500" y2="1000"/>
  <line stroke="blue" x1="750" y1="750" x2="688" y2="812"/>
  <line stroke="red" x1="750" y1="750" x2="812" y2="688"/>
  <line stroke="blue" x1="750" y1="750" x2="875" y2="750"/>
  <line stroke="red" x1="688" y1="812" x2="625" y2="875"/>
  <line stroke="red" x1="688" y1="812" x2="938" y2="812"/>
  <line stroke="blue" x1="812" y1="688" x2="875" y2="688"/>
  <line stroke="red" x1="812" y1="688" x2="875" y2="750"/>
  <line stroke="blue" x1="625" y1="875" x2="562" y2="938"/>
  <line stroke="blue" x1="625" y1="875" x2="1000" y2="875"/>
  <line stroke="red" x1="562" y1="938" x2="500" y2="1000"/>
  <line stroke="red" x1="562" y1="938" x2="1000" y2="938"/>
  <line stroke="blue" x1="875" y1="688" x2="875" y2="750"/>
  <line stroke="red" x1="875" y1="688" x2="938" y2="688"/>
  <line stroke="black" x1="500" y1="1000" x2="1000" y2="1000"/>
  <line stroke="blue" x1="875" y1="750" x2="938" y2="812"/>
  <line stroke="blue" x1="938" y1="688" x2="1000" y2="688"/>
  <line stroke="red" x1="938" y1="688" x2="938" y2="812"/>
  <line stroke="black" x1="1000" y1="688" x2="1000" y2="875"/>
  <line stroke="red" x1="938" y1="812" x2="1000" y2="875"/>
  <line stroke="black" x1="1000" y1="875" x2="1000" y2="938"/>
  <line stroke="black" x1="1000" y1="938" x2="1000" y2="1000"/>
</svg>""",
)

PHOENIX_FAN_OUT_LEFT_TAIL = BoxPleatingShapingExample(
    base_example=PHOENIX_ZEBRA_BORDER_FLIPPED,
    fan_out_shaping_params=[
        fan_out_shaping.FanOutShapingParams(
            valid_fold_types=[crease_utils.Fold.VALLEY],
            valid_orientations=[fan_out_shaping.Orientation.HORIZONTAL],
            rectangles=[
                shaping_utils.Rectangle(
                    name="left_tail",
                    x_start=0.6875,
                    y_start=0,
                    x_len=0.3125,
                    y_len=0.5625,
                )
            ],
            move_delta=-1.0,
        ),
    ],
    fan_out_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="0" y2="62"/>
  <line stroke="black" x1="0" y1="0" x2="312" y2="0"/>
  <line stroke="black" x1="0" y1="62" x2="0" y2="125"/>
  <line stroke="red" x1="0" y1="62" x2="312" y2="62"/>
  <line stroke="black" x1="0" y1="125" x2="0" y2="188"/>
  <line stroke="blue" x1="0" y1="125" x2="312" y2="125"/>
  <line stroke="black" x1="0" y1="188" x2="0" y2="250"/>
  <line stroke="red" x1="0" y1="188" x2="312" y2="188"/>
  <line stroke="red" x1="0" y1="250" x2="62" y2="312"/>
  <line stroke="blue" x1="0" y1="250" x2="312" y2="250"/>
  <line stroke="black" x1="0" y1="250" x2="0" y2="688"/>
  <line stroke="blue" x1="312" y1="0" x2="312" y2="62"/>
  <line stroke="black" x1="312" y1="0" x2="500" y2="0"/>
  <line stroke="blue" x1="62" y1="312" x2="125" y2="375"/>
  <line stroke="red" x1="62" y1="312" x2="312" y2="312"/>
  <line stroke="red" x1="62" y1="312" x2="62" y2="688"/>
  <line stroke="red" x1="312" y1="62" x2="312" y2="125"/>
  <line stroke="red" x1="312" y1="62" x2="438" y2="62"/>
  <line stroke="blue" x1="312" y1="125" x2="312" y2="188"/>
  <line stroke="blue" x1="312" y1="125" x2="500" y2="125"/>
  <line stroke="red" x1="312" y1="188" x2="312" y2="250"/>
  <line stroke="red" x1="312" y1="188" x2="438" y2="188"/>
  <line stroke="red" x1="125" y1="375" x2="188" y2="438"/>
  <line stroke="blue" x1="125" y1="375" x2="312" y2="375"/>
  <line stroke="blue" x1="125" y1="375" x2="125" y2="688"/>
  <line stroke="blue" x1="312" y1="250" x2="312" y2="312"/>
  <line stroke="blue" x1="312" y1="250" x2="500" y2="250"/>
  <line stroke="red" x1="312" y1="312" x2="312" y2="375"/>
  <line stroke="red" x1="312" y1="312" x2="438" y2="312"/>
  <line stroke="red" x1="438" y1="62" x2="500" y2="0"/>
  <line stroke="blue" x1="438" y1="62" x2="500" y2="62"/>
  <line stroke="red" x1="438" y1="62" x2="500" y2="125"/>
  <line stroke="red" x1="188" y1="438" x2="312" y2="438"/>
  <line stroke="blue" x1="188" y1="438" x2="250" y2="500"/>
  <line stroke="red" x1="188" y1="438" x2="188" y2="688"/>
  <line stroke="red" x1="438" y1="188" x2="500" y2="125"/>
  <line stroke="blue" x1="438" y1="188" x2="500" y2="188"/>
  <line stroke="red" x1="438" y1="188" x2="500" y2="250"/>
  <line stroke="blue" x1="312" y1="375" x2="312" y2="438"/>
  <line stroke="blue" x1="312" y1="375" x2="500" y2="375"/>
  <line stroke="blue" x1="500" y1="0" x2="500" y2="62"/>
  <line stroke="red" x1="500" y1="0" x2="562" y2="62"/>
  <line stroke="black" x1="500" y1="0" x2="688" y2="0"/>
  <line stroke="red" x1="500" y1="62" x2="500" y2="125"/>
  <line stroke="blue" x1="500" y1="62" x2="562" y2="62"/>
  <line stroke="blue" x1="500" y1="125" x2="500" y2="188"/>
  <line stroke="red" x1="500" y1="125" x2="562" y2="62"/>
  <line stroke="red" x1="500" y1="125" x2="562" y2="188"/>
  <line stroke="blue" x1="500" y1="125" x2="688" y2="125"/>
  <line stroke="red" x1="500" y1="188" x2="500" y2="250"/>
  <line stroke="blue" x1="500" y1="188" x2="562" y2="188"/>
  <line stroke="red" x1="312" y1="438" x2="312" y2="500"/>
  <line stroke="red" x1="312" y1="438" x2="438" y2="438"/>
  <line stroke="red" x1="438" y1="312" x2="500" y2="250"/>
  <line stroke="blue" x1="438" y1="312" x2="500" y2="312"/>
  <line stroke="red" x1="438" y1="312" x2="500" y2="375"/>
  <line stroke="blue" x1="250" y1="500" x2="312" y2="500"/>
  <line stroke="red" x1="250" y1="500" x2="312" y2="562"/>
  <line stroke="blue" x1="250" y1="500" x2="250" y2="750"/>
  <line stroke="blue" x1="500" y1="250" x2="500" y2="312"/>
  <line stroke="red" x1="500" y1="250" x2="562" y2="188"/>
  <line stroke="red" x1="500" y1="250" x2="562" y2="312"/>
  <line stroke="blue" x1="500" y1="250" x2="688" y2="250"/>
  <line stroke="red" x1="562" y1="62" x2="661" y2="62"/>
  <line stroke="blue" x1="312" y1="500" x2="375" y2="500"/>
  <line stroke="blue" x1="312" y1="500" x2="312" y2="562"/>
  <line stroke="red" x1="500" y1="312" x2="500" y2="375"/>
  <line stroke="blue" x1="500" y1="312" x2="562" y2="312"/>
  <line stroke="red" x1="562" y1="188" x2="675" y2="188"/>
  <line stroke="blue" x1="438" y1="438" x2="375" y2="500"/>
  <line stroke="red" x1="438" y1="438" x2="500" y2="375"/>
  <line stroke="red" x1="438" y1="438" x2="438" y2="938"/>
  <line stroke="red" x1="375" y1="500" x2="312" y2="562"/>
  <line stroke="blue" x1="375" y1="500" x2="375" y2="875"/>
  <line stroke="red" x1="500" y1="375" x2="562" y2="312"/>
  <line stroke="red" x1="500" y1="375" x2="562" y2="438"/>
  <line stroke="blue" x1="500" y1="375" x2="688" y2="375"/>
  <line stroke="blue" x1="500" y1="375" x2="500" y2="1000"/>
  <line stroke="red" x1="312" y1="562" x2="312" y2="812"/>
  <line stroke="red" x1="562" y1="312" x2="688" y2="312"/>
  <line stroke="blue" x1="661" y1="62" x2="688" y2="0"/>
  <line stroke="red" x1="661" y1="62" x2="688" y2="125"/>
  <line stroke="red" x1="661" y1="62" x2="1000" y2="62"/>
  <line stroke="blue" x1="0" y1="688" x2="62" y2="688"/>
  <line stroke="black" x1="0" y1="688" x2="0" y2="875"/>
  <line stroke="black" x1="688" y1="0" x2="1000" y2="0"/>
  <line stroke="red" x1="62" y1="688" x2="125" y2="688"/>
  <line stroke="red" x1="62" y1="688" x2="62" y2="812"/>
  <line stroke="blue" x1="125" y1="688" x2="188" y2="688"/>
  <line stroke="blue" x1="125" y1="688" x2="125" y2="750"/>
  <line stroke="blue" x1="688" y1="125" x2="675" y2="188"/>
  <line stroke="blue" x1="688" y1="125" x2="1000" y2="62"/>
  <line stroke="red" x1="675" y1="188" x2="688" y2="250"/>
  <line stroke="red" x1="675" y1="188" x2="1000" y2="188"/>
  <line stroke="blue" x1="562" y1="438" x2="625" y2="500"/>
  <line stroke="red" x1="562" y1="438" x2="688" y2="438"/>
  <line stroke="red" x1="562" y1="438" x2="562" y2="938"/>
  <line stroke="red" x1="188" y1="688" x2="125" y2="750"/>
  <line stroke="red" x1="188" y1="688" x2="250" y2="750"/>
  <line stroke="blue" x1="688" y1="250" x2="688" y2="312"/>
  <line stroke="blue" x1="688" y1="250" x2="1000" y2="188"/>
  <line stroke="red" x1="688" y1="312" x2="688" y2="375"/>
  <line stroke="red" x1="688" y1="312" x2="938" y2="312"/>
  <line stroke="blue" x1="125" y1="750" x2="250" y2="750"/>
  <line stroke="blue" x1="125" y1="750" x2="62" y2="812"/>
  <line stroke="blue" x1="688" y1="375" x2="688" y2="438"/>
  <line stroke="blue" x1="688" y1="375" x2="875" y2="375"/>
  <line stroke="blue" x1="250" y1="750" x2="312" y2="812"/>
  <line stroke="blue" x1="625" y1="500" x2="688" y2="500"/>
  <line stroke="red" x1="625" y1="500" x2="688" y2="562"/>
  <line stroke="blue" x1="625" y1="500" x2="625" y2="875"/>
  <line stroke="red" x1="62" y1="812" x2="312" y2="812"/>
  <line stroke="red" x1="62" y1="812" x2="0" y2="875"/>
  <line stroke="red" x1="688" y1="438" x2="688" y2="500"/>
  <line stroke="red" x1="688" y1="438" x2="812" y2="438"/>
  <line stroke="blue" x1="688" y1="500" x2="688" y2="562"/>
  <line stroke="blue" x1="688" y1="500" x2="750" y2="500"/>
  <line stroke="red" x1="312" y1="812" x2="375" y2="875"/>
  <line stroke="black" x1="0" y1="875" x2="0" y2="938"/>
  <line stroke="blue" x1="0" y1="875" x2="375" y2="875"/>
  <line stroke="red" x1="688" y1="562" x2="750" y2="500"/>
  <line stroke="red" x1="688" y1="562" x2="688" y2="812"/>
  <line stroke="blue" x1="750" y1="500" x2="812" y2="438"/>
  <line stroke="blue" x1="750" y1="500" x2="750" y2="750"/>
  <line stroke="red" x1="812" y1="438" x2="875" y2="375"/>
  <line stroke="red" x1="812" y1="438" x2="812" y2="688"/>
  <line stroke="black" x1="0" y1="938" x2="0" y2="1000"/>
  <line stroke="red" x1="0" y1="938" x2="438" y2="938"/>
  <line stroke="blue" x1="375" y1="875" x2="438" y2="938"/>
  <line stroke="blue" x1="875" y1="375" x2="938" y2="312"/>
  <line stroke="blue" x1="875" y1="375" x2="875" y2="688"/>
  <line stroke="red" x1="938" y1="312" x2="1000" y2="250"/>
  <line stroke="red" x1="938" y1="312" x2="938" y2="688"/>
  <line stroke="black" x1="0" y1="1000" x2="500" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="62"/>
  <line stroke="black" x1="1000" y1="62" x2="1000" y2="188"/>
  <line stroke="black" x1="1000" y1="188" x2="1000" y2="250"/>
  <line stroke="black" x1="1000" y1="250" x2="1000" y2="688"/>
  <line stroke="red" x1="438" y1="938" x2="500" y2="1000"/>
  <line stroke="blue" x1="750" y1="750" x2="688" y2="812"/>
  <line stroke="red" x1="750" y1="750" x2="812" y2="688"/>
  <line stroke="blue" x1="750" y1="750" x2="875" y2="750"/>
  <line stroke="red" x1="688" y1="812" x2="625" y2="875"/>
  <line stroke="red" x1="688" y1="812" x2="938" y2="812"/>
  <line stroke="blue" x1="812" y1="688" x2="875" y2="688"/>
  <line stroke="red" x1="812" y1="688" x2="875" y2="750"/>
  <line stroke="blue" x1="625" y1="875" x2="562" y2="938"/>
  <line stroke="blue" x1="625" y1="875" x2="1000" y2="875"/>
  <line stroke="red" x1="562" y1="938" x2="500" y2="1000"/>
  <line stroke="red" x1="562" y1="938" x2="1000" y2="938"/>
  <line stroke="blue" x1="875" y1="688" x2="875" y2="750"/>
  <line stroke="red" x1="875" y1="688" x2="938" y2="688"/>
  <line stroke="black" x1="500" y1="1000" x2="1000" y2="1000"/>
  <line stroke="blue" x1="875" y1="750" x2="938" y2="812"/>
  <line stroke="blue" x1="938" y1="688" x2="1000" y2="688"/>
  <line stroke="red" x1="938" y1="688" x2="938" y2="812"/>
  <line stroke="black" x1="1000" y1="688" x2="1000" y2="875"/>
  <line stroke="red" x1="938" y1="812" x2="1000" y2="875"/>
  <line stroke="black" x1="1000" y1="875" x2="1000" y2="938"/>
  <line stroke="black" x1="1000" y1="938" x2="1000" y2="1000"/>
</svg>""",
)

PHOENIX_FAN_OUT_BOTH_TAILS = BoxPleatingShapingExample(
    base_example=PHOENIX_ZEBRA_BORDER_FLIPPED,
    fan_out_shaping_params=[
        fan_out_shaping.FanOutShapingParams(
            valid_fold_types=[crease_utils.Fold.VALLEY],
            valid_orientations=[fan_out_shaping.Orientation.HORIZONTAL],
            rectangles=[
                shaping_utils.Rectangle(
                    name="right_tail",
                    x_start=0,
                    y_start=0,
                    x_len=0.3125,
                    y_len=0.5625,
                )
            ],
            move_delta=-1.0,
        ),
        fan_out_shaping.FanOutShapingParams(
            valid_fold_types=[crease_utils.Fold.VALLEY],
            valid_orientations=[fan_out_shaping.Orientation.HORIZONTAL],
            rectangles=[
                shaping_utils.Rectangle(
                    name="left_tail",
                    x_start=0.6875,
                    y_start=0,
                    x_len=0.3125,
                    y_len=0.5625,
                )
            ],
            move_delta=-1.0,
        ),
    ],
    fan_out_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="0" y2="62"/>
  <line stroke="black" x1="0" y1="0" x2="312" y2="0"/>
  <line stroke="black" x1="0" y1="62" x2="0" y2="188"/>
  <line stroke="blue" x1="0" y1="62" x2="312" y2="125"/>
  <line stroke="red" x1="0" y1="62" x2="339" y2="62"/>
  <line stroke="black" x1="0" y1="188" x2="0" y2="250"/>
  <line stroke="red" x1="0" y1="188" x2="325" y2="188"/>
  <line stroke="blue" x1="0" y1="188" x2="312" y2="250"/>
  <line stroke="red" x1="0" y1="250" x2="62" y2="312"/>
  <line stroke="black" x1="0" y1="250" x2="0" y2="688"/>
  <line stroke="blue" x1="312" y1="0" x2="339" y2="62"/>
  <line stroke="black" x1="312" y1="0" x2="500" y2="0"/>
  <line stroke="blue" x1="62" y1="312" x2="125" y2="375"/>
  <line stroke="red" x1="62" y1="312" x2="312" y2="312"/>
  <line stroke="red" x1="62" y1="312" x2="62" y2="688"/>
  <line stroke="red" x1="312" y1="125" x2="339" y2="62"/>
  <line stroke="blue" x1="312" y1="125" x2="325" y2="188"/>
  <line stroke="blue" x1="312" y1="125" x2="500" y2="125"/>
  <line stroke="red" x1="339" y1="62" x2="438" y2="62"/>
  <line stroke="red" x1="325" y1="188" x2="312" y2="250"/>
  <line stroke="red" x1="325" y1="188" x2="438" y2="188"/>
  <line stroke="red" x1="125" y1="375" x2="188" y2="438"/>
  <line stroke="blue" x1="125" y1="375" x2="312" y2="375"/>
  <line stroke="blue" x1="125" y1="375" x2="125" y2="688"/>
  <line stroke="blue" x1="312" y1="250" x2="312" y2="312"/>
  <line stroke="blue" x1="312" y1="250" x2="500" y2="250"/>
  <line stroke="red" x1="312" y1="312" x2="312" y2="375"/>
  <line stroke="red" x1="312" y1="312" x2="438" y2="312"/>
  <line stroke="red" x1="438" y1="62" x2="500" y2="0"/>
  <line stroke="blue" x1="438" y1="62" x2="500" y2="62"/>
  <line stroke="red" x1="438" y1="62" x2="500" y2="125"/>
  <line stroke="red" x1="188" y1="438" x2="312" y2="438"/>
  <line stroke="blue" x1="188" y1="438" x2="250" y2="500"/>
  <line stroke="red" x1="188" y1="438" x2="188" y2="688"/>
  <line stroke="red" x1="438" y1="188" x2="500" y2="125"/>
  <line stroke="blue" x1="438" y1="188" x2="500" y2="188"/>
  <line stroke="red" x1="438" y1="188" x2="500" y2="250"/>
  <line stroke="blue" x1="312" y1="375" x2="312" y2="438"/>
  <line stroke="blue" x1="312" y1="375" x2="500" y2="375"/>
  <line stroke="blue" x1="500" y1="0" x2="500" y2="62"/>
  <line stroke="red" x1="500" y1="0" x2="562" y2="62"/>
  <line stroke="black" x1="500" y1="0" x2="688" y2="0"/>
  <line stroke="red" x1="500" y1="62" x2="500" y2="125"/>
  <line stroke="blue" x1="500" y1="62" x2="562" y2="62"/>
  <line stroke="blue" x1="500" y1="125" x2="500" y2="188"/>
  <line stroke="red" x1="500" y1="125" x2="562" y2="62"/>
  <line stroke="red" x1="500" y1="125" x2="562" y2="188"/>
  <line stroke="blue" x1="500" y1="125" x2="688" y2="125"/>
  <line stroke="red" x1="500" y1="188" x2="500" y2="250"/>
  <line stroke="blue" x1="500" y1="188" x2="562" y2="188"/>
  <line stroke="red" x1="312" y1="438" x2="312" y2="500"/>
  <line stroke="red" x1="312" y1="438" x2="438" y2="438"/>
  <line stroke="red" x1="438" y1="312" x2="500" y2="250"/>
  <line stroke="blue" x1="438" y1="312" x2="500" y2="312"/>
  <line stroke="red" x1="438" y1="312" x2="500" y2="375"/>
  <line stroke="blue" x1="250" y1="500" x2="312" y2="500"/>
  <line stroke="red" x1="250" y1="500" x2="312" y2="562"/>
  <line stroke="blue" x1="250" y1="500" x2="250" y2="750"/>
  <line stroke="blue" x1="500" y1="250" x2="500" y2="312"/>
  <line stroke="red" x1="500" y1="250" x2="562" y2="188"/>
  <line stroke="red" x1="500" y1="250" x2="562" y2="312"/>
  <line stroke="blue" x1="500" y1="250" x2="688" y2="250"/>
  <line stroke="red" x1="562" y1="62" x2="661" y2="62"/>
  <line stroke="blue" x1="312" y1="500" x2="375" y2="500"/>
  <line stroke="blue" x1="312" y1="500" x2="312" y2="562"/>
  <line stroke="red" x1="500" y1="312" x2="500" y2="375"/>
  <line stroke="blue" x1="500" y1="312" x2="562" y2="312"/>
  <line stroke="red" x1="562" y1="188" x2="675" y2="188"/>
  <line stroke="blue" x1="438" y1="438" x2="375" y2="500"/>
  <line stroke="red" x1="438" y1="438" x2="500" y2="375"/>
  <line stroke="red" x1="438" y1="438" x2="438" y2="938"/>
  <line stroke="red" x1="375" y1="500" x2="312" y2="562"/>
  <line stroke="blue" x1="375" y1="500" x2="375" y2="875"/>
  <line stroke="red" x1="500" y1="375" x2="562" y2="312"/>
  <line stroke="red" x1="500" y1="375" x2="562" y2="438"/>
  <line stroke="blue" x1="500" y1="375" x2="688" y2="375"/>
  <line stroke="blue" x1="500" y1="375" x2="500" y2="1000"/>
  <line stroke="red" x1="312" y1="562" x2="312" y2="812"/>
  <line stroke="red" x1="562" y1="312" x2="688" y2="312"/>
  <line stroke="blue" x1="661" y1="62" x2="688" y2="0"/>
  <line stroke="red" x1="661" y1="62" x2="688" y2="125"/>
  <line stroke="red" x1="661" y1="62" x2="1000" y2="62"/>
  <line stroke="blue" x1="0" y1="688" x2="62" y2="688"/>
  <line stroke="black" x1="0" y1="688" x2="0" y2="875"/>
  <line stroke="black" x1="688" y1="0" x2="1000" y2="0"/>
  <line stroke="red" x1="62" y1="688" x2="125" y2="688"/>
  <line stroke="red" x1="62" y1="688" x2="62" y2="812"/>
  <line stroke="blue" x1="125" y1="688" x2="188" y2="688"/>
  <line stroke="blue" x1="125" y1="688" x2="125" y2="750"/>
  <line stroke="blue" x1="688" y1="125" x2="675" y2="188"/>
  <line stroke="blue" x1="688" y1="125" x2="1000" y2="62"/>
  <line stroke="red" x1="675" y1="188" x2="688" y2="250"/>
  <line stroke="red" x1="675" y1="188" x2="1000" y2="188"/>
  <line stroke="blue" x1="562" y1="438" x2="625" y2="500"/>
  <line stroke="red" x1="562" y1="438" x2="688" y2="438"/>
  <line stroke="red" x1="562" y1="438" x2="562" y2="938"/>
  <line stroke="red" x1="188" y1="688" x2="125" y2="750"/>
  <line stroke="red" x1="188" y1="688" x2="250" y2="750"/>
  <line stroke="blue" x1="688" y1="250" x2="688" y2="312"/>
  <line stroke="blue" x1="688" y1="250" x2="1000" y2="188"/>
  <line stroke="red" x1="688" y1="312" x2="688" y2="375"/>
  <line stroke="red" x1="688" y1="312" x2="938" y2="312"/>
  <line stroke="blue" x1="125" y1="750" x2="250" y2="750"/>
  <line stroke="blue" x1="125" y1="750" x2="62" y2="812"/>
  <line stroke="blue" x1="688" y1="375" x2="688" y2="438"/>
  <line stroke="blue" x1="688" y1="375" x2="875" y2="375"/>
  <line stroke="blue" x1="250" y1="750" x2="312" y2="812"/>
  <line stroke="blue" x1="625" y1="500" x2="688" y2="500"/>
  <line stroke="red" x1="625" y1="500" x2="688" y2="562"/>
  <line stroke="blue" x1="625" y1="500" x2="625" y2="875"/>
  <line stroke="red" x1="62" y1="812" x2="312" y2="812"/>
  <line stroke="red" x1="62" y1="812" x2="0" y2="875"/>
  <line stroke="red" x1="688" y1="438" x2="688" y2="500"/>
  <line stroke="red" x1="688" y1="438" x2="812" y2="438"/>
  <line stroke="blue" x1="688" y1="500" x2="688" y2="562"/>
  <line stroke="blue" x1="688" y1="500" x2="750" y2="500"/>
  <line stroke="red" x1="312" y1="812" x2="375" y2="875"/>
  <line stroke="black" x1="0" y1="875" x2="0" y2="938"/>
  <line stroke="blue" x1="0" y1="875" x2="375" y2="875"/>
  <line stroke="red" x1="688" y1="562" x2="750" y2="500"/>
  <line stroke="red" x1="688" y1="562" x2="688" y2="812"/>
  <line stroke="blue" x1="750" y1="500" x2="812" y2="438"/>
  <line stroke="blue" x1="750" y1="500" x2="750" y2="750"/>
  <line stroke="red" x1="812" y1="438" x2="875" y2="375"/>
  <line stroke="red" x1="812" y1="438" x2="812" y2="688"/>
  <line stroke="black" x1="0" y1="938" x2="0" y2="1000"/>
  <line stroke="red" x1="0" y1="938" x2="438" y2="938"/>
  <line stroke="blue" x1="375" y1="875" x2="438" y2="938"/>
  <line stroke="blue" x1="875" y1="375" x2="938" y2="312"/>
  <line stroke="blue" x1="875" y1="375" x2="875" y2="688"/>
  <line stroke="red" x1="938" y1="312" x2="1000" y2="250"/>
  <line stroke="red" x1="938" y1="312" x2="938" y2="688"/>
  <line stroke="black" x1="0" y1="1000" x2="500" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="62"/>
  <line stroke="black" x1="1000" y1="62" x2="1000" y2="188"/>
  <line stroke="black" x1="1000" y1="188" x2="1000" y2="250"/>
  <line stroke="black" x1="1000" y1="250" x2="1000" y2="688"/>
  <line stroke="red" x1="438" y1="938" x2="500" y2="1000"/>
  <line stroke="blue" x1="750" y1="750" x2="688" y2="812"/>
  <line stroke="red" x1="750" y1="750" x2="812" y2="688"/>
  <line stroke="blue" x1="750" y1="750" x2="875" y2="750"/>
  <line stroke="red" x1="688" y1="812" x2="625" y2="875"/>
  <line stroke="red" x1="688" y1="812" x2="938" y2="812"/>
  <line stroke="blue" x1="812" y1="688" x2="875" y2="688"/>
  <line stroke="red" x1="812" y1="688" x2="875" y2="750"/>
  <line stroke="blue" x1="625" y1="875" x2="562" y2="938"/>
  <line stroke="blue" x1="625" y1="875" x2="1000" y2="875"/>
  <line stroke="red" x1="562" y1="938" x2="500" y2="1000"/>
  <line stroke="red" x1="562" y1="938" x2="1000" y2="938"/>
  <line stroke="blue" x1="875" y1="688" x2="875" y2="750"/>
  <line stroke="red" x1="875" y1="688" x2="938" y2="688"/>
  <line stroke="black" x1="500" y1="1000" x2="1000" y2="1000"/>
  <line stroke="blue" x1="875" y1="750" x2="938" y2="812"/>
  <line stroke="blue" x1="938" y1="688" x2="1000" y2="688"/>
  <line stroke="red" x1="938" y1="688" x2="938" y2="812"/>
  <line stroke="black" x1="1000" y1="688" x2="1000" y2="875"/>
  <line stroke="red" x1="938" y1="812" x2="1000" y2="875"/>
  <line stroke="black" x1="1000" y1="875" x2="1000" y2="938"/>
  <line stroke="black" x1="1000" y1="938" x2="1000" y2="1000"/>
</svg>""",
)

VERTICAL_FAN_OUT_SHAPING_TEST_TOP_HALF = BoxPleatingShapingExample(
    base_example=VERTICAL_FAN_OUT_SHAPING_TEST1,
    fan_out_shaping_params=[
        fan_out_shaping.FanOutShapingParams(
            valid_fold_types=[
                crease_utils.Fold.MOUNTAIN,
                crease_utils.Fold.VALLEY,
            ],
            valid_orientations=[fan_out_shaping.Orientation.VERTICAL],
            rectangles=[
                shaping_utils.Rectangle(
                    name="top_half",
                    x_start=0,
                    y_start=0,
                    x_len=1,
                    y_len=0.5,
                )
            ],
            move_delta=0.5,
        ),
    ],
    fan_out_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="312" y2="0"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="469"/>
  <line stroke="blue" x1="312" y1="0" x2="254" y2="469"/>
  <line stroke="black" x1="312" y1="0" x2="562" y2="0"/>
  <line stroke="blue" x1="0" y1="469" x2="254" y2="469"/>
  <line stroke="black" x1="0" y1="469" x2="0" y2="1000"/>
  <line stroke="red" x1="254" y1="469" x2="500" y2="500"/>
  <line stroke="blue" x1="254" y1="469" x2="250" y2="1000"/>
  <line stroke="red" x1="562" y1="0" x2="500" y2="500"/>
  <line stroke="black" x1="562" y1="0" x2="812" y2="0"/>
  <line stroke="blue" x1="500" y1="500" x2="750" y2="500"/>
  <line stroke="red" x1="500" y1="500" x2="500" y2="1000"/>
  <line stroke="blue" x1="812" y1="0" x2="750" y2="500"/>
  <line stroke="black" x1="812" y1="0" x2="1000" y2="0"/>
  <line stroke="red" x1="750" y1="500" x2="1000" y2="531"/>
  <line stroke="blue" x1="750" y1="500" x2="750" y2="1000"/>
  <line stroke="black" x1="0" y1="1000" x2="250" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="531"/>
  <line stroke="black" x1="250" y1="1000" x2="500" y2="1000"/>
  <line stroke="black" x1="500" y1="1000" x2="750" y2="1000"/>
  <line stroke="black" x1="1000" y1="531" x2="1000" y2="1000"/>
  <line stroke="black" x1="750" y1="1000" x2="1000" y2="1000"/>
</svg>""",
)

VERTICAL_FAN_OUT_SHAPING_TEST_BOTH_HALVES = BoxPleatingShapingExample(
    base_example=VERTICAL_FAN_OUT_SHAPING_TEST2,
    fan_out_shaping_params=[
        fan_out_shaping.FanOutShapingParams(
            valid_fold_types=[crease_utils.Fold.VALLEY],
            valid_orientations=[fan_out_shaping.Orientation.VERTICAL],
            rectangles=[
                shaping_utils.Rectangle(
                    name="bottom_half",
                    x_start=0,
                    y_start=0.5,
                    x_len=1,
                    y_len=0.5,
                )
            ],
            move_delta=1.0,
        ),
        fan_out_shaping.FanOutShapingParams(
            valid_fold_types=[crease_utils.Fold.VALLEY],
            valid_orientations=[
                fan_out_shaping.Orientation.VERTICAL,
                fan_out_shaping.Orientation.HORIZONTAL,
            ],
            rectangles=[
                shaping_utils.Rectangle(
                    name="top_half",
                    x_start=0,
                    y_start=0,
                    x_len=1,
                    y_len=0.5,
                )
            ],
            move_delta=-1.0,
        ),
        fan_out_shaping.FanOutShapingParams(
            valid_fold_types=[crease_utils.Fold.MOUNTAIN],
            valid_orientations=[fan_out_shaping.Orientation.HORIZONTAL],
            rectangles=[
                shaping_utils.Rectangle(
                    name="bottom_half",
                    x_start=0,
                    y_start=0.5,
                    x_len=1,
                    y_len=0.5,
                )
            ],
            move_delta=1.0,
        ),
    ],
    fan_out_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="375" y2="0"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="625"/>
  <line stroke="red" x1="0" y1="0" x2="500" y2="500"/>
  <line stroke="blue" x1="375" y1="0" x2="500" y2="500"/>
  <line stroke="black" x1="375" y1="0" x2="1000" y2="0"/>
  <line stroke="red" x1="0" y1="625" x2="500" y2="500"/>
  <line stroke="black" x1="0" y1="625" x2="0" y2="1000"/>
  <line stroke="red" x1="500" y1="500" x2="1000" y2="0"/>
  <line stroke="blue" x1="500" y1="500" x2="1000" y2="375"/>
  <line stroke="red" x1="500" y1="500" x2="599" y2="1000"/>
  <line stroke="blue" x1="500" y1="500" x2="625" y2="1000"/>
  <line stroke="red" x1="500" y1="500" x2="1000" y2="1000"/>
  <line stroke="black" x1="0" y1="1000" x2="599" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="375"/>
  <line stroke="black" x1="1000" y1="375" x2="1000" y2="1000"/>
  <line stroke="black" x1="599" y1="1000" x2="625" y2="1000"/>
  <line stroke="black" x1="625" y1="1000" x2="1000" y2="1000"/>
</svg>""",
)

ZZ_PLANT_FAN_OUT = BoxPleatingShapingExample(
    base_example=BoxPleatingCreasePatternExample(
        name="zz_plant_base",
        grid_size=6,
        packing_svg="",
        base_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 6 6">
  <rect stroke="black" x="0" y="0" width="6" height="6"/>
  <line stroke="red" x1="1" y1="0" x2="1" y2="3" elevation="1"/>
  <line stroke="blue" x1="2" y1="0" x2="2" y2="4" elevation="0"/>
  <line stroke="blue" x1="0" y1="3" x2="1" y2="3"/>
  <line stroke="red" x1="1" y1="3" x2="0" y2="4"/>
  <line stroke="red" x1="1" y1="3" x2="2" y2="4"/>
  <line stroke="blue" x1="0" y1="4" x2="2" y2="4" elevation="0"/>
  <line stroke="blue" x1="4" y1="0" x2="4" y2="4" elevation="0"/>
  <line stroke="blue" x1="2" y1="4" x2="3" y2="5"/>
  <line stroke="red" x1="0" y1="5" x2="3" y2="5" elevation="1"/>
  <line stroke="red" x1="5" y1="0" x2="5" y2="3" elevation="1"/>
  <line stroke="blue" x1="4" y1="4" x2="3" y2="5"/>
  <line stroke="red" x1="4" y1="4" x2="5" y2="3"/>
  <line stroke="blue" x1="4" y1="4" x2="6" y2="4" elevation="0"/>
  <line stroke="blue" x1="3" y1="5" x2="2" y2="6"/>
  <line stroke="blue" x1="3" y1="5" x2="4" y2="6"/>
  <line stroke="red" x1="3" y1="5" x2="6" y2="5" elevation="1"/>
  <line stroke="blue" x1="5" y1="3" x2="6" y2="3"/>
  <line stroke="red" x1="5" y1="3" x2="6" y2="4"/>
</svg>""",
    ),
    fan_out_shaping_params=[
        fan_out_shaping.FanOutShapingParams(
            valid_fold_types=[crease_utils.Fold.MOUNTAIN],
            valid_orientations=[fan_out_shaping.Orientation.VERTICAL],
            rectangles=[
                shaping_utils.Rectangle(
                    name="top_left",
                    x_start=0,
                    y_start=0,
                    x_len=2 / 6,
                    y_len=0.5,
                ),
            ],
            move_delta=1.0,
        ),
        fan_out_shaping.FanOutShapingParams(
            valid_fold_types=[crease_utils.Fold.MOUNTAIN],
            valid_orientations=[fan_out_shaping.Orientation.VERTICAL],
            rectangles=[
                shaping_utils.Rectangle(
                    name="top_right",
                    x_start=4 / 6,
                    y_start=0,
                    x_len=2 / 6,
                    y_len=0.5,
                ),
            ],
            move_delta=-1.0,
        ),
    ],
    fan_out_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="333" y2="0"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="444"/>
  <line stroke="red" x1="333" y1="0" x2="167" y2="500"/>
  <line stroke="black" x1="333" y1="0" x2="667" y2="0"/>
  <line stroke="blue" x1="333" y1="0" x2="333" y2="667" elevation="0"/>
  <line stroke="blue" x1="0" y1="444" x2="167" y2="500"/>
  <line stroke="black" x1="0" y1="444" x2="0" y2="667"/>
  <line stroke="red" x1="167" y1="500" x2="0" y2="667"/>
  <line stroke="red" x1="167" y1="500" x2="333" y2="667"/>
  <line stroke="blue" x1="0" y1="667" x2="333" y2="667" elevation="0"/>
  <line stroke="black" x1="0" y1="667" x2="0" y2="833"/>
  <line stroke="blue" x1="667" y1="0" x2="667" y2="667" elevation="0"/>
  <line stroke="red" x1="667" y1="0" x2="833" y2="500"/>
  <line stroke="black" x1="667" y1="0" x2="1000" y2="0"/>
  <line stroke="blue" x1="333" y1="667" x2="500" y2="833"/>
  <line stroke="red" x1="0" y1="833" x2="500" y2="833" elevation="1"/>
  <line stroke="black" x1="0" y1="833" x2="0" y2="1000"/>
  <line stroke="blue" x1="667" y1="667" x2="500" y2="833"/>
  <line stroke="red" x1="667" y1="667" x2="833" y2="500"/>
  <line stroke="blue" x1="667" y1="667" x2="1000" y2="667" elevation="0"/>
  <line stroke="blue" x1="500" y1="833" x2="333" y2="1000"/>
  <line stroke="blue" x1="500" y1="833" x2="667" y2="1000"/>
  <line stroke="red" x1="500" y1="833" x2="1000" y2="833" elevation="1"/>
  <line stroke="blue" x1="833" y1="500" x2="1000" y2="444"/>
  <line stroke="red" x1="833" y1="500" x2="1000" y2="667"/>
  <line stroke="black" x1="0" y1="1000" x2="333" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="444"/>
  <line stroke="black" x1="333" y1="1000" x2="667" y2="1000"/>
  <line stroke="black" x1="1000" y1="444" x2="1000" y2="667"/>
  <line stroke="black" x1="667" y1="1000" x2="1000" y2="1000"/>
  <line stroke="black" x1="1000" y1="667" x2="1000" y2="833"/>
  <line stroke="black" x1="1000" y1="833" x2="1000" y2="1000"/>
</svg>""",
)

INSECT_LEG = BoxPleatingShapingExample(
    base_example=BoxPleatingCreasePatternExample(
        name="insect_leg",
        packing_svg="",
        grid_size=20,
    ),
    central_crease_removed=False,
    rectangles=[
        shaping_utils.Rectangle(
            name="wide_flap",
            x_start=0.6,
            y_start=0.2,
            x_len=0.8,
            y_len=0.8,
            circle_centers=[
                point_utils.Point(x=1.0, y=0.6, require_on_paper=False)
            ],
        )
    ],
    inscribed_rectangle_shaping_params=[
        inscribed_rectangle_shaping.InscribedRectangleFlapShapingParams(
            flap_name="wide_flap",
            inscribed_rectangles_params=[
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=2,
                    fold_order=[
                        crease_utils.Fold.VALLEY,
                        crease_utils.Fold.VALLEY,
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.MOUNTAIN,
                    ],
                    fold_percent=1.0,
                ),
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=4,
                    fold_order=[
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.VALLEY,
                        crease_utils.Fold.VALLEY,
                    ],
                    fold_percent=1.0,
                ),
            ],
        ),
    ],
    narrowed_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="0" y2="100"/>
  <line stroke="black" x1="0" y1="0" x2="800" y2="0"/>
  <line stroke="black" x1="0" y1="100" x2="0" y2="200"/>
  <line stroke="red" x1="0" y1="100" x2="700" y2="100"/>
  <line stroke="black" x1="0" y1="200" x2="0" y2="300"/>
  <line stroke="blue" x1="0" y1="200" x2="600" y2="200"/>
  <line stroke="black" x1="0" y1="300" x2="0" y2="400"/>
  <line stroke="red" x1="0" y1="300" x2="625" y2="300"/>
  <line stroke="black" x1="0" y1="400" x2="0" y2="500"/>
  <line stroke="blue" x1="0" y1="400" x2="600" y2="400"/>
  <line stroke="black" x1="0" y1="500" x2="0" y2="600"/>
  <line stroke="red" x1="0" y1="500" x2="625" y2="500"/>
  <line stroke="black" x1="0" y1="600" x2="0" y2="700"/>
  <line stroke="blue" x1="0" y1="600" x2="600" y2="600"/>
  <line stroke="red" x1="600" y1="200" x2="625" y2="250"/>
  <line stroke="blue" x1="600" y1="200" x2="650" y2="225"/>
  <line stroke="blue" x1="600" y1="200" x2="650" y2="250"/>
  <line stroke="blue" x1="600" y1="200" x2="700" y2="100"/>
  <line stroke="red" x1="600" y1="200" x2="700" y2="200"/>
  <line stroke="red" x1="625" y1="250" x2="625" y2="300"/>
  <line stroke="blue" x1="625" y1="250" x2="650" y2="250"/>
  <line stroke="red" x1="625" y1="250" x2="650" y2="300"/>
  <line stroke="red" x1="650" y1="225" x2="650" y2="250"/>
  <line stroke="blue" x1="650" y1="225" x2="700" y2="225"/>
  <line stroke="blue" x1="650" y1="225" x2="700" y2="250"/>
  <line stroke="red" x1="625" y1="300" x2="650" y2="300"/>
  <line stroke="blue" x1="625" y1="300" x2="625" y2="350"/>
  <line stroke="blue" x1="650" y1="250" x2="700" y2="300"/>
  <line stroke="black" x1="0" y1="700" x2="0" y2="800"/>
  <line stroke="red" x1="0" y1="700" x2="625" y2="700"/>
  <line stroke="red" x1="700" y1="100" x2="700" y2="200"/>
  <line stroke="red" x1="700" y1="100" x2="800" y2="0"/>
  <line stroke="blue" x1="650" y1="300" x2="625" y2="350"/>
  <line stroke="red" x1="650" y1="300" x2="700" y2="300"/>
  <line stroke="blue" x1="625" y1="350" x2="600" y2="400"/>
  <line stroke="red" x1="625" y1="350" x2="750" y2="350"/>
  <line stroke="red" x1="600" y1="400" x2="625" y2="450"/>
  <line stroke="blue" x1="600" y1="400" x2="800" y2="400"/>
  <line stroke="red" x1="700" y1="200" x2="700" y2="225"/>
  <line stroke="blue" x1="700" y1="200" x2="800" y2="200"/>
  <line stroke="red" x1="700" y1="225" x2="700" y2="250"/>
  <line stroke="red" x1="700" y1="225" x2="750" y2="225"/>
  <line stroke="red" x1="700" y1="250" x2="700" y2="300"/>
  <line stroke="red" x1="700" y1="250" x2="750" y2="225"/>
  <line stroke="red" x1="700" y1="300" x2="750" y2="350"/>
  <line stroke="red" x1="625" y1="450" x2="625" y2="500"/>
  <line stroke="red" x1="625" y1="450" x2="650" y2="500"/>
  <line stroke="blue" x1="625" y1="450" x2="850" y2="450"/>
  <line stroke="red" x1="750" y1="225" x2="800" y2="200"/>
  <line stroke="blue" x1="750" y1="225" x2="750" y2="350"/>
  <line stroke="black" x1="0" y1="800" x2="0" y2="900"/>
  <line stroke="blue" x1="0" y1="800" x2="600" y2="800"/>
  <line stroke="blue" x1="800" y1="0" x2="800" y2="200"/>
  <line stroke="black" x1="800" y1="0" x2="900" y2="0"/>
  <line stroke="red" x1="625" y1="500" x2="650" y2="500"/>
  <line stroke="blue" x1="625" y1="500" x2="625" y2="550"/>
  <line stroke="blue" x1="650" y1="500" x2="625" y2="550"/>
  <line stroke="red" x1="650" y1="500" x2="900" y2="500"/>
  <line stroke="blue" x1="800" y1="200" x2="850" y2="225"/>
  <line stroke="blue" x1="800" y1="200" x2="800" y2="400"/>
  <line stroke="red" x1="800" y1="200" x2="900" y2="200"/>
  <line stroke="red" x1="750" y1="350" x2="800" y2="400"/>
  <line stroke="blue" x1="625" y1="550" x2="600" y2="600"/>
  <line stroke="red" x1="625" y1="550" x2="950" y2="550"/>
  <line stroke="red" x1="600" y1="600" x2="625" y2="650"/>
  <line stroke="blue" x1="600" y1="600" x2="950" y2="600"/>
  <line stroke="blue" x1="850" y1="225" x2="900" y2="225"/>
  <line stroke="blue" x1="850" y1="225" x2="900" y2="250"/>
  <line stroke="red" x1="850" y1="225" x2="850" y2="450"/>
  <line stroke="blue" x1="800" y1="400" x2="850" y2="450"/>
  <line stroke="black" x1="0" y1="900" x2="0" y2="1000"/>
  <line stroke="red" x1="0" y1="900" x2="625" y2="900"/>
  <line stroke="red" x1="900" y1="0" x2="900" y2="200"/>
  <line stroke="black" x1="900" y1="0" x2="1000" y2="0"/>
  <line stroke="red" x1="625" y1="650" x2="625" y2="700"/>
  <line stroke="red" x1="625" y1="650" x2="650" y2="700"/>
  <line stroke="blue" x1="625" y1="650" x2="950" y2="650"/>
  <line stroke="red" x1="900" y1="200" x2="900" y2="225"/>
  <line stroke="blue" x1="900" y1="200" x2="1000" y2="200"/>
  <line stroke="red" x1="900" y1="225" x2="900" y2="250"/>
  <line stroke="red" x1="900" y1="225" x2="950" y2="225"/>
  <line stroke="red" x1="900" y1="250" x2="950" y2="225"/>
  <line stroke="red" x1="900" y1="250" x2="900" y2="500"/>
  <line stroke="red" x1="625" y1="700" x2="650" y2="700"/>
  <line stroke="blue" x1="625" y1="700" x2="625" y2="750"/>
  <line stroke="blue" x1="650" y1="700" x2="625" y2="750"/>
  <line stroke="red" x1="650" y1="700" x2="900" y2="700"/>
  <line stroke="blue" x1="850" y1="450" x2="900" y2="500"/>
  <line stroke="blue" x1="625" y1="750" x2="600" y2="800"/>
  <line stroke="red" x1="625" y1="750" x2="850" y2="750"/>
  <line stroke="red" x1="950" y1="225" x2="1000" y2="200"/>
  <line stroke="blue" x1="950" y1="225" x2="950" y2="550"/>
  <line stroke="black" x1="0" y1="1000" x2="600" y2="1000"/>
  <line stroke="red" x1="600" y1="800" x2="625" y2="850"/>
  <line stroke="blue" x1="600" y1="800" x2="800" y2="800"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="200"/>
  <line stroke="black" x1="1000" y1="200" x2="1000" y2="550"/>
  <line stroke="red" x1="900" y1="500" x2="950" y2="550"/>
  <line stroke="red" x1="625" y1="850" x2="625" y2="900"/>
  <line stroke="red" x1="625" y1="850" x2="650" y2="900"/>
  <line stroke="blue" x1="625" y1="850" x2="750" y2="850"/>
  <line stroke="red" x1="625" y1="900" x2="650" y2="900"/>
  <line stroke="blue" x1="625" y1="900" x2="625" y2="950"/>
  <line stroke="blue" x1="950" y1="550" x2="950" y2="600" opacity="0.306"/>
  <line stroke="red" x1="950" y1="550" x2="1000" y2="550" opacity="0.306"/>
  <line stroke="red" x1="950" y1="550" x2="1000" y2="600"/>
  <line stroke="blue" x1="650" y1="900" x2="625" y2="950"/>
  <line stroke="red" x1="650" y1="900" x2="700" y2="900"/>
  <line stroke="red" x1="950" y1="600" x2="950" y2="650" opacity="0.306"/>
  <line stroke="blue" x1="950" y1="600" x2="1000" y2="600"/>
  <line stroke="red" x1="800" y1="800" x2="750" y2="850"/>
  <line stroke="blue" x1="800" y1="800" x2="850" y2="750"/>
  <line stroke="blue" x1="800" y1="800" x2="800" y2="1000"/>
  <line stroke="red" x1="750" y1="850" x2="700" y2="900"/>
  <line stroke="red" x1="750" y1="850" x2="750" y2="975"/>
  <line stroke="blue" x1="850" y1="750" x2="900" y2="700"/>
  <line stroke="blue" x1="850" y1="750" x2="850" y2="975"/>
  <line stroke="red" x1="625" y1="950" x2="650" y2="950"/>
  <line stroke="blue" x1="625" y1="950" x2="600" y2="1000"/>
  <line stroke="blue" x1="700" y1="900" x2="650" y2="950"/>
  <line stroke="red" x1="700" y1="900" x2="700" y2="950"/>
  <line stroke="red" x1="900" y1="700" x2="950" y2="650"/>
  <line stroke="red" x1="900" y1="700" x2="900" y2="950"/>
  <line stroke="black" x1="1000" y1="550" x2="1000" y2="600"/>
  <line stroke="blue" x1="650" y1="950" x2="600" y2="1000"/>
  <line stroke="blue" x1="650" y1="950" x2="650" y2="975"/>
  <line stroke="red" x1="950" y1="650" x2="1000" y2="600"/>
  <line stroke="blue" x1="950" y1="650" x2="1000" y2="650" opacity="0.306"/>
  <line stroke="red" x1="950" y1="650" x2="950" y2="975"/>
  <line stroke="red" x1="600" y1="1000" x2="650" y2="975"/>
  <line stroke="black" x1="600" y1="1000" x2="700" y2="1000"/>
  <line stroke="black" x1="1000" y1="600" x2="1000" y2="650"/>
  <line stroke="red" x1="650" y1="975" x2="700" y2="950"/>
  <line stroke="red" x1="650" y1="975" x2="700" y2="975"/>
  <line stroke="red" x1="700" y1="950" x2="700" y2="975"/>
  <line stroke="blue" x1="700" y1="950" x2="750" y2="975"/>
  <line stroke="black" x1="1000" y1="650" x2="1000" y2="1000"/>
  <line stroke="red" x1="700" y1="975" x2="700" y2="1000"/>
  <line stroke="blue" x1="700" y1="975" x2="750" y2="975"/>
  <line stroke="black" x1="700" y1="1000" x2="800" y2="1000"/>
  <line stroke="blue" x1="750" y1="975" x2="800" y2="1000"/>
  <line stroke="red" x1="800" y1="1000" x2="850" y2="975"/>
  <line stroke="black" x1="800" y1="1000" x2="900" y2="1000"/>
  <line stroke="red" x1="850" y1="975" x2="900" y2="950"/>
  <line stroke="red" x1="850" y1="975" x2="900" y2="975"/>
  <line stroke="red" x1="900" y1="950" x2="900" y2="975"/>
  <line stroke="blue" x1="900" y1="950" x2="950" y2="975"/>
  <line stroke="red" x1="900" y1="975" x2="900" y2="1000"/>
  <line stroke="blue" x1="900" y1="975" x2="950" y2="975"/>
  <line stroke="black" x1="900" y1="1000" x2="1000" y2="1000"/>
  <line stroke="blue" x1="950" y1="975" x2="1000" y2="1000"/>
</svg>""",
    crimp_fold_shaping_params=[
        crimp_fold_shaping.CrimpFoldShapingParams(
            element_name="wide_flap",
            crimp_fold_params=[
                crimp_fold_shaping.CrimpFoldParams(
                    depth=2,
                    fold_order=[
                        crease_utils.Fold.VALLEY,
                        crease_utils.Fold.VALLEY,
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.MOUNTAIN,
                    ],
                    fold_percent=1.0,
                    zigzag_flap_angle=np.pi / 1.5,
                    zigzag_fold_order=[
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.VALLEY,
                        crease_utils.Fold.VALLEY,
                    ],
                    zigzag_even_phase=True,
                ),
                crimp_fold_shaping.CrimpFoldParams(
                    depth=4,
                    fold_order=[
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.VALLEY,
                        crease_utils.Fold.VALLEY,
                    ],
                    fold_percent=1.0,
                    zigzag_flap_angle=np.pi / 1.5,
                    zigzag_fold_order=[
                        crease_utils.Fold.VALLEY,
                        crease_utils.Fold.VALLEY,
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.MOUNTAIN,
                    ],
                    zigzag_even_phase=False,
                ),
            ],
        )
    ],
    crimp_fold_shaping_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="0" y2="100"/>
  <line stroke="black" x1="0" y1="0" x2="800" y2="0"/>
  <line stroke="black" x1="0" y1="100" x2="0" y2="200"/>
  <line stroke="red" x1="0" y1="100" x2="700" y2="100"/>
  <line stroke="black" x1="0" y1="200" x2="0" y2="300"/>
  <line stroke="blue" x1="0" y1="200" x2="600" y2="200"/>
  <line stroke="black" x1="0" y1="300" x2="0" y2="400"/>
  <line stroke="red" x1="0" y1="300" x2="625" y2="300"/>
  <line stroke="black" x1="0" y1="400" x2="0" y2="500"/>
  <line stroke="blue" x1="0" y1="400" x2="600" y2="400"/>
  <line stroke="black" x1="0" y1="500" x2="0" y2="600"/>
  <line stroke="red" x1="0" y1="500" x2="625" y2="500"/>
  <line stroke="black" x1="0" y1="600" x2="0" y2="700"/>
  <line stroke="blue" x1="0" y1="600" x2="600" y2="600"/>
  <line stroke="red" x1="600" y1="200" x2="625" y2="250"/>
  <line stroke="blue" x1="600" y1="200" x2="650" y2="225"/>
  <line stroke="blue" x1="600" y1="200" x2="650" y2="250"/>
  <line stroke="blue" x1="600" y1="200" x2="700" y2="100"/>
  <line stroke="red" x1="600" y1="200" x2="700" y2="200"/>
  <line stroke="red" x1="625" y1="250" x2="625" y2="300"/>
  <line stroke="blue" x1="625" y1="250" x2="650" y2="250"/>
  <line stroke="red" x1="625" y1="250" x2="650" y2="300"/>
  <line stroke="red" x1="650" y1="225" x2="650" y2="250"/>
  <line stroke="blue" x1="650" y1="225" x2="700" y2="225"/>
  <line stroke="blue" x1="650" y1="225" x2="700" y2="250"/>
  <line stroke="red" x1="625" y1="300" x2="650" y2="300"/>
  <line stroke="blue" x1="625" y1="300" x2="625" y2="350"/>
  <line stroke="blue" x1="650" y1="250" x2="700" y2="300"/>
  <line stroke="black" x1="0" y1="700" x2="0" y2="800"/>
  <line stroke="red" x1="0" y1="700" x2="625" y2="700"/>
  <line stroke="red" x1="700" y1="100" x2="700" y2="200"/>
  <line stroke="red" x1="700" y1="100" x2="800" y2="0"/>
  <line stroke="blue" x1="650" y1="300" x2="625" y2="350"/>
  <line stroke="red" x1="650" y1="300" x2="700" y2="300"/>
  <line stroke="blue" x1="625" y1="350" x2="600" y2="400"/>
  <line stroke="red" x1="625" y1="350" x2="700" y2="350"/>
  <line stroke="red" x1="600" y1="400" x2="625" y2="450"/>
  <line stroke="blue" x1="600" y1="400" x2="700" y2="400"/>
  <line stroke="red" x1="700" y1="200" x2="700" y2="225"/>
  <line stroke="blue" x1="700" y1="200" x2="800" y2="200"/>
  <line stroke="red" x1="700" y1="225" x2="700" y2="250"/>
  <line stroke="red" x1="700" y1="225" x2="750" y2="225"/>
  <line stroke="red" x1="700" y1="250" x2="700" y2="300"/>
  <line stroke="red" x1="700" y1="250" x2="750" y2="225"/>
  <line stroke="red" x1="700" y1="300" x2="729" y2="350" fold-step="1"/>
  <line stroke="blue" x1="700" y1="300" x2="750" y2="329" fold-step="1"/>
  <line stroke="blue" x1="700" y1="300" x2="700" y2="350" fold-step="1"/>
  <line stroke="red" x1="700" y1="300" x2="750" y2="300" fold-step="1"/>
  <line stroke="red" x1="700" y1="300" x2="750" y2="350"/>
  <line stroke="red" x1="625" y1="450" x2="625" y2="500"/>
  <line stroke="red" x1="625" y1="450" x2="650" y2="500"/>
  <line stroke="blue" x1="625" y1="450" x2="700" y2="450"/>
  <line stroke="blue" x1="700" y1="350" x2="700" y2="400" fold-step="1"/>
  <line stroke="blue" x1="700" y1="350" x2="729" y2="350"/>
  <line stroke="blue" x1="750" y1="225" x2="750" y2="300"/>
  <line stroke="red" x1="750" y1="225" x2="800" y2="200"/>
  <line stroke="black" x1="0" y1="800" x2="0" y2="900"/>
  <line stroke="blue" x1="0" y1="800" x2="600" y2="800"/>
  <line stroke="blue" x1="800" y1="0" x2="800" y2="200"/>
  <line stroke="black" x1="800" y1="0" x2="900" y2="0"/>
  <line stroke="red" x1="625" y1="500" x2="650" y2="500"/>
  <line stroke="blue" x1="625" y1="500" x2="625" y2="550"/>
  <line stroke="red" x1="700" y1="400" x2="729" y2="350" fold-step="1"/>
  <line stroke="blue" x1="700" y1="400" x2="729" y2="450" fold-step="1"/>
  <line stroke="blue" x1="700" y1="400" x2="800" y2="400"/>
  <line stroke="red" x1="700" y1="400" x2="700" y2="450" fold-step="1"/>
  <line stroke="red" x1="750" y1="300" x2="750" y2="329"/>
  <line stroke="red" x1="750" y1="300" x2="800" y2="300" fold-step="1"/>
  <line stroke="red" x1="729" y1="350" x2="750" y2="350"/>
  <line stroke="blue" x1="750" y1="329" x2="750" y2="350"/>
  <line stroke="blue" x1="750" y1="329" x2="800" y2="300" fold-step="1"/>
  <line stroke="blue" x1="650" y1="500" x2="625" y2="550"/>
  <line stroke="red" x1="650" y1="500" x2="700" y2="500"/>
  <line stroke="blue" x1="800" y1="200" x2="800" y2="300"/>
  <line stroke="blue" x1="800" y1="200" x2="850" y2="225"/>
  <line stroke="red" x1="800" y1="200" x2="900" y2="200"/>
  <line stroke="red" x1="750" y1="350" x2="800" y2="400"/>
  <line stroke="red" x1="700" y1="450" x2="729" y2="450"/>
  <line stroke="red" x1="700" y1="450" x2="700" y2="500" fold-step="1"/>
  <line stroke="blue" x1="625" y1="550" x2="600" y2="600"/>
  <line stroke="red" x1="625" y1="550" x2="700" y2="550"/>
  <line stroke="red" x1="600" y1="600" x2="625" y2="650"/>
  <line stroke="blue" x1="600" y1="600" x2="700" y2="600"/>
  <line stroke="blue" x1="800" y1="300" x2="800" y2="400"/>
  <line stroke="red" x1="800" y1="300" x2="850" y2="329" fold-step="1"/>
  <line stroke="blue" x1="800" y1="300" x2="850" y2="300" fold-step="1"/>
  <line stroke="blue" x1="729" y1="450" x2="700" y2="500" fold-step="1"/>
  <line stroke="blue" x1="729" y1="450" x2="800" y2="450"/>
  <line stroke="red" x1="700" y1="500" x2="729" y2="550" fold-step="1"/>
  <line stroke="red" x1="700" y1="500" x2="800" y2="500"/>
  <line stroke="blue" x1="700" y1="500" x2="700" y2="550" fold-step="1"/>
  <line stroke="red" x1="850" y1="225" x2="850" y2="300"/>
  <line stroke="blue" x1="850" y1="225" x2="900" y2="225"/>
  <line stroke="blue" x1="850" y1="225" x2="900" y2="250"/>
  <line stroke="blue" x1="700" y1="550" x2="729" y2="550"/>
  <line stroke="blue" x1="700" y1="550" x2="700" y2="600" fold-step="1"/>
  <line stroke="blue" x1="800" y1="400" x2="800" y2="450" fold-step="1"/>
  <line stroke="red" x1="800" y1="400" x2="850" y2="400" fold-step="1"/>
  <line stroke="blue" x1="800" y1="400" x2="818" y2="418"/>
  <line stroke="black" x1="0" y1="900" x2="0" y2="1000"/>
  <line stroke="red" x1="0" y1="900" x2="625" y2="900"/>
  <line stroke="red" x1="900" y1="0" x2="900" y2="200"/>
  <line stroke="black" x1="900" y1="0" x2="1000" y2="0"/>
  <line stroke="blue" x1="850" y1="300" x2="850" y2="329"/>
  <line stroke="blue" x1="850" y1="300" x2="900" y2="300" fold-step="1"/>
  <line stroke="red" x1="625" y1="650" x2="625" y2="700"/>
  <line stroke="blue" x1="625" y1="650" x2="700" y2="650"/>
  <line stroke="red" x1="625" y1="650" x2="650" y2="700"/>
  <line stroke="red" x1="850" y1="329" x2="850" y2="400"/>
  <line stroke="red" x1="850" y1="329" x2="900" y2="300" fold-step="1"/>
  <line stroke="red" x1="729" y1="550" x2="700" y2="600" fold-step="1"/>
  <line stroke="red" x1="729" y1="550" x2="800" y2="550"/>
  <line stroke="blue" x1="800" y1="450" x2="800" y2="500" fold-step="1"/>
  <line stroke="blue" x1="800" y1="450" x2="850" y2="450"/>
  <line stroke="red" x1="800" y1="450" x2="818" y2="418" fold-step="1"/>
  <line stroke="red" x1="800" y1="450" x2="829" y2="500" fold-step="1"/>
  <line stroke="blue" x1="818" y1="418" x2="850" y2="400" fold-step="1"/>
  <line stroke="blue" x1="818" y1="418" x2="850" y2="450"/>
  <line stroke="blue" x1="700" y1="600" x2="729" y2="650" fold-step="1"/>
  <line stroke="blue" x1="700" y1="600" x2="800" y2="600"/>
  <line stroke="red" x1="700" y1="600" x2="700" y2="650" fold-step="1"/>
  <line stroke="red" x1="900" y1="200" x2="900" y2="225"/>
  <line stroke="blue" x1="900" y1="200" x2="1000" y2="200"/>
  <line stroke="red" x1="900" y1="225" x2="900" y2="250"/>
  <line stroke="red" x1="900" y1="225" x2="950" y2="225"/>
  <line stroke="red" x1="900" y1="250" x2="900" y2="300"/>
  <line stroke="red" x1="900" y1="250" x2="950" y2="225"/>
  <line stroke="red" x1="625" y1="700" x2="650" y2="700"/>
  <line stroke="blue" x1="625" y1="700" x2="625" y2="750"/>
  <line stroke="red" x1="850" y1="400" x2="850" y2="450"/>
  <line stroke="red" x1="850" y1="400" x2="900" y2="400" fold-step="1"/>
  <line stroke="blue" x1="850" y1="400" x2="900" y2="429" fold-step="1"/>
  <line stroke="red" x1="800" y1="500" x2="829" y2="500"/>
  <line stroke="red" x1="800" y1="500" x2="800" y2="550" fold-step="1"/>
  <line stroke="red" x1="900" y1="300" x2="900" y2="400"/>
  <line stroke="blue" x1="900" y1="300" x2="950" y2="329" fold-step="1"/>
  <line stroke="red" x1="900" y1="300" x2="950" y2="300" fold-step="1"/>
  <line stroke="red" x1="700" y1="650" x2="729" y2="650"/>
  <line stroke="red" x1="700" y1="650" x2="700" y2="700" fold-step="1"/>
  <line stroke="blue" x1="650" y1="700" x2="625" y2="750"/>
  <line stroke="red" x1="650" y1="700" x2="700" y2="700"/>
  <line stroke="blue" x1="850" y1="450" x2="900" y2="500"/>
  <line stroke="blue" x1="829" y1="500" x2="800" y2="550" fold-step="1"/>
  <line stroke="red" x1="829" y1="500" x2="900" y2="500"/>
  <line stroke="red" x1="800" y1="550" x2="800" y2="600" fold-step="1"/>
  <line stroke="blue" x1="800" y1="550" x2="829" y2="600" fold-step="1"/>
  <line stroke="red" x1="800" y1="550" x2="950" y2="550"/>
  <line stroke="blue" x1="625" y1="750" x2="600" y2="800"/>
  <line stroke="red" x1="625" y1="750" x2="700" y2="750"/>
  <line stroke="blue" x1="950" y1="225" x2="950" y2="300"/>
  <line stroke="red" x1="950" y1="225" x2="1000" y2="200"/>
  <line stroke="blue" x1="729" y1="650" x2="700" y2="700" fold-step="1"/>
  <line stroke="blue" x1="729" y1="650" x2="800" y2="650"/>
  <line stroke="red" x1="900" y1="400" x2="900" y2="429"/>
  <line stroke="blue" x1="900" y1="400" x2="950" y2="400" fold-step="1"/>
  <line stroke="red" x1="700" y1="700" x2="729" y2="750" fold-step="1"/>
  <line stroke="blue" x1="700" y1="700" x2="700" y2="750" fold-step="1"/>
  <line stroke="red" x1="700" y1="700" x2="800" y2="700"/>
  <line stroke="red" x1="950" y1="300" x2="950" y2="329"/>
  <line stroke="red" x1="950" y1="300" x2="1000" y2="300" fold-step="1"/>
  <line stroke="red" x1="900" y1="429" x2="900" y2="500"/>
  <line stroke="red" x1="900" y1="429" x2="950" y2="400" fold-step="1"/>
  <line stroke="blue" x1="800" y1="600" x2="829" y2="600"/>
  <line stroke="blue" x1="800" y1="600" x2="800" y2="650" fold-step="1"/>
  <line stroke="black" x1="0" y1="1000" x2="600" y2="1000"/>
  <line stroke="red" x1="600" y1="800" x2="625" y2="850"/>
  <line stroke="blue" x1="600" y1="800" x2="700" y2="800"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="200"/>
  <line stroke="blue" x1="950" y1="329" x2="950" y2="400"/>
  <line stroke="blue" x1="950" y1="329" x2="1000" y2="300" fold-step="1"/>
  <line stroke="black" x1="1000" y1="200" x2="1000" y2="300"/>
  <line stroke="red" x1="829" y1="600" x2="800" y2="650" fold-step="1"/>
  <line stroke="blue" x1="829" y1="600" x2="950" y2="600"/>
  <line stroke="blue" x1="700" y1="750" x2="729" y2="750"/>
  <line stroke="blue" x1="700" y1="750" x2="700" y2="800" fold-step="1"/>
  <line stroke="red" x1="900" y1="500" x2="950" y2="550"/>
  <line stroke="blue" x1="800" y1="650" x2="800" y2="700" fold-step="1"/>
  <line stroke="blue" x1="800" y1="650" x2="950" y2="650"/>
  <line stroke="blue" x1="950" y1="400" x2="1000" y2="400" fold-step="1"/>
  <line stroke="blue" x1="950" y1="400" x2="950" y2="550"/>
  <line stroke="red" x1="800" y1="650" x2="829" y2="700" fold-step="1"/>
  <line stroke="red" x1="950" y1="400" x2="1000" y2="429" fold-step="1"/>
  <line stroke="black" x1="1000" y1="300" x2="1000" y2="400"/>
  <line stroke="red" x1="729" y1="750" x2="700" y2="800" fold-step="1"/>
  <line stroke="red" x1="729" y1="750" x2="800" y2="750"/>
  <line stroke="red" x1="625" y1="850" x2="625" y2="900"/>
  <line stroke="blue" x1="625" y1="850" x2="700" y2="850"/>
  <line stroke="red" x1="625" y1="850" x2="650" y2="900"/>
  <line stroke="blue" x1="700" y1="800" x2="729" y2="850" fold-step="1"/>
  <line stroke="red" x1="700" y1="800" x2="700" y2="850" fold-step="1"/>
  <line stroke="blue" x1="700" y1="800" x2="800" y2="800"/>
  <line stroke="red" x1="800" y1="700" x2="829" y2="700"/>
  <line stroke="red" x1="800" y1="700" x2="800" y2="750" fold-step="1"/>
  <line stroke="black" x1="1000" y1="400" x2="1000" y2="429"/>
  <line stroke="blue" x1="829" y1="700" x2="800" y2="750" fold-step="1"/>
  <line stroke="red" x1="829" y1="700" x2="900" y2="700"/>
  <line stroke="black" x1="1000" y1="429" x2="1000" y2="550"/>
  <line stroke="red" x1="625" y1="900" x2="650" y2="900"/>
  <line stroke="blue" x1="625" y1="900" x2="625" y2="950"/>
  <line stroke="red" x1="800" y1="750" x2="800" y2="800" fold-step="1"/>
  <line stroke="blue" x1="800" y1="750" x2="818" y2="782" fold-step="1"/>
  <line stroke="red" x1="800" y1="750" x2="850" y2="750"/>
  <line stroke="blue" x1="950" y1="550" x2="950" y2="600" opacity="0.306"/>
  <line stroke="red" x1="950" y1="550" x2="1000" y2="550" opacity="0.306"/>
  <line stroke="red" x1="950" y1="550" x2="1000" y2="600"/>
  <line stroke="red" x1="700" y1="850" x2="729" y2="850"/>
  <line stroke="red" x1="700" y1="850" x2="700" y2="900" fold-step="1"/>
  <line stroke="blue" x1="650" y1="900" x2="625" y2="950"/>
  <line stroke="red" x1="650" y1="900" x2="700" y2="900"/>
  <line stroke="blue" x1="729" y1="850" x2="750" y2="850"/>
  <line stroke="blue" x1="729" y1="850" x2="700" y2="900" fold-step="1"/>
  <line stroke="red" x1="950" y1="600" x2="950" y2="650" opacity="0.306"/>
  <line stroke="blue" x1="950" y1="600" x2="1000" y2="600"/>
  <line stroke="blue" x1="800" y1="800" x2="850" y2="800" fold-step="1"/>
  <line stroke="blue" x1="800" y1="800" x2="818" y2="782"/>
  <line stroke="red" x1="800" y1="800" x2="750" y2="850"/>
  <line stroke="blue" x1="800" y1="800" x2="800" y2="900"/>
  <line stroke="blue" x1="818" y1="782" x2="850" y2="750"/>
  <line stroke="red" x1="818" y1="782" x2="850" y2="800" fold-step="1"/>
  <line stroke="red" x1="750" y1="850" x2="700" y2="900"/>
  <line stroke="red" x1="750" y1="850" x2="750" y2="871"/>
  <line stroke="blue" x1="850" y1="750" x2="900" y2="700"/>
  <line stroke="blue" x1="850" y1="750" x2="850" y2="800"/>
  <line stroke="red" x1="625" y1="950" x2="650" y2="950"/>
  <line stroke="blue" x1="625" y1="950" x2="600" y2="1000"/>
  <line stroke="red" x1="700" y1="900" x2="750" y2="871" fold-step="1"/>
  <line stroke="blue" x1="700" y1="900" x2="750" y2="900" fold-step="1"/>
  <line stroke="blue" x1="700" y1="900" x2="650" y2="950"/>
  <line stroke="red" x1="700" y1="900" x2="700" y2="950"/>
  <line stroke="red" x1="900" y1="700" x2="950" y2="650"/>
  <line stroke="red" x1="900" y1="700" x2="900" y2="771"/>
  <line stroke="black" x1="1000" y1="550" x2="1000" y2="600"/>
  <line stroke="blue" x1="750" y1="871" x2="750" y2="900"/>
  <line stroke="red" x1="750" y1="871" x2="800" y2="900" fold-step="1"/>
  <line stroke="blue" x1="650" y1="950" x2="600" y2="1000"/>
  <line stroke="blue" x1="650" y1="950" x2="650" y2="975"/>
  <line stroke="red" x1="950" y1="650" x2="1000" y2="600"/>
  <line stroke="blue" x1="950" y1="650" x2="1000" y2="650" opacity="0.306"/>
  <line stroke="red" x1="950" y1="650" x2="950" y2="800"/>
  <line stroke="red" x1="600" y1="1000" x2="650" y2="975"/>
  <line stroke="black" x1="600" y1="1000" x2="700" y2="1000"/>
  <line stroke="black" x1="1000" y1="600" x2="1000" y2="650"/>
  <line stroke="red" x1="850" y1="800" x2="900" y2="771" fold-step="1"/>
  <line stroke="blue" x1="850" y1="800" x2="900" y2="800" fold-step="1"/>
  <line stroke="blue" x1="850" y1="800" x2="850" y2="871"/>
  <line stroke="blue" x1="750" y1="900" x2="800" y2="900" fold-step="1"/>
  <line stroke="red" x1="750" y1="900" x2="750" y2="975"/>
  <line stroke="red" x1="650" y1="975" x2="700" y2="950"/>
  <line stroke="red" x1="650" y1="975" x2="700" y2="975"/>
  <line stroke="red" x1="700" y1="950" x2="700" y2="975"/>
  <line stroke="blue" x1="700" y1="950" x2="750" y2="975"/>
  <line stroke="red" x1="900" y1="771" x2="900" y2="800"/>
  <line stroke="blue" x1="900" y1="771" x2="950" y2="800" fold-step="1"/>
  <line stroke="black" x1="1000" y1="650" x2="1000" y2="771"/>
  <line stroke="red" x1="700" y1="975" x2="700" y2="1000"/>
  <line stroke="blue" x1="700" y1="975" x2="750" y2="975"/>
  <line stroke="blue" x1="800" y1="900" x2="850" y2="871" fold-step="1"/>
  <line stroke="blue" x1="800" y1="900" x2="800" y2="1000"/>
  <line stroke="red" x1="800" y1="900" x2="850" y2="900" fold-step="1"/>
  <line stroke="red" x1="900" y1="800" x2="950" y2="800" fold-step="1"/>
  <line stroke="red" x1="900" y1="800" x2="900" y2="900"/>
  <line stroke="red" x1="850" y1="871" x2="850" y2="900"/>
  <line stroke="blue" x1="850" y1="871" x2="900" y2="900" fold-step="1"/>
  <line stroke="black" x1="700" y1="1000" x2="800" y2="1000"/>
  <line stroke="blue" x1="750" y1="975" x2="800" y2="1000"/>
  <line stroke="red" x1="850" y1="900" x2="900" y2="900" fold-step="1"/>
  <line stroke="blue" x1="850" y1="900" x2="850" y2="975"/>
  <line stroke="blue" x1="950" y1="800" x2="1000" y2="771" fold-step="1"/>
  <line stroke="red" x1="950" y1="800" x2="1000" y2="800" fold-step="1"/>
  <line stroke="red" x1="950" y1="800" x2="950" y2="871"/>
  <line stroke="black" x1="1000" y1="771" x2="1000" y2="800"/>
  <line stroke="red" x1="900" y1="900" x2="950" y2="871" fold-step="1"/>
  <line stroke="red" x1="900" y1="900" x2="900" y2="950"/>
  <line stroke="blue" x1="900" y1="900" x2="950" y2="900" fold-step="1"/>
  <line stroke="red" x1="800" y1="1000" x2="850" y2="975"/>
  <line stroke="black" x1="800" y1="1000" x2="900" y2="1000"/>
  <line stroke="black" x1="1000" y1="800" x2="1000" y2="900"/>
  <line stroke="blue" x1="950" y1="871" x2="950" y2="900"/>
  <line stroke="red" x1="950" y1="871" x2="1000" y2="900" fold-step="1"/>
  <line stroke="red" x1="850" y1="975" x2="900" y2="950"/>
  <line stroke="red" x1="850" y1="975" x2="900" y2="975"/>
  <line stroke="red" x1="900" y1="950" x2="900" y2="975"/>
  <line stroke="blue" x1="900" y1="950" x2="950" y2="975"/>
  <line stroke="blue" x1="950" y1="900" x2="1000" y2="900" fold-step="1"/>
  <line stroke="red" x1="950" y1="900" x2="950" y2="975"/>
  <line stroke="red" x1="900" y1="975" x2="900" y2="1000"/>
  <line stroke="blue" x1="900" y1="975" x2="950" y2="975"/>
  <line stroke="black" x1="900" y1="1000" x2="1000" y2="1000"/>
  <line stroke="black" x1="1000" y1="900" x2="1000" y2="1000"/>
  <line stroke="blue" x1="950" y1="975" x2="1000" y2="1000"/>
</svg>""",
)

CRIMP_NARROWING = BoxPleatingShapingExample(
    base_example=BoxPleatingCreasePatternExample(
        name="crimp_narrowing",
        packing_svg="",
        grid_size=10,
        base_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
<line stroke="black" x1="0" y1="0" x2="0" y2="1"/>
<line stroke="black" x1="0" y1="0" x2="8" y2="0"/>
<line stroke="black" x1="0" y1="1" x2="0" y2="2"/>
<line stroke="red" x1="0" y1="1" x2="7" y2="1"/>
<line stroke="black" x1="0" y1="2" x2="0" y2="3"/>
<line stroke="blue" x1="0" y1="2" x2="6" y2="2"/>
<line stroke="black" x1="0" y1="3" x2="0" y2="4"/>
<line stroke="red" x1="0" y1="3" x2="6.5" y2="3"/>
<line stroke="black" x1="0" y1="4" x2="0" y2="5"/>
<line stroke="blue" x1="0" y1="4" x2="6" y2="4"/>
<line stroke="black" x1="0" y1="5" x2="0" y2="6"/>
<line stroke="red" x1="0" y1="5" x2="6.5" y2="5"/>
<line stroke="black" x1="0" y1="6" x2="0" y2="7"/>
<line stroke="blue" x1="0" y1="6" x2="6" y2="6"/>
<line stroke="blue" x1="6" y1="2" x2="7" y2="1"/>
<line stroke="red" x1="6" y1="2" x2="6.5" y2="3"/>
<line stroke="red" x1="6" y1="2" x2="7" y2="2"/>
<line stroke="blue" x1="6" y1="2" x2="7" y2="2.5"/>
<line stroke="blue" x1="6" y1="2" x2="7" y2="3"/>
<line stroke="black" x1="0" y1="7" x2="0" y2="8"/>
<line stroke="red" x1="0" y1="7" x2="6.5" y2="7"/>
<line stroke="red" x1="7" y1="1" x2="7" y2="2"/>
<line stroke="red" x1="7" y1="1" x2="8" y2="0"/>
<line stroke="blue" x1="6.5" y1="3" x2="6" y2="4"/>
<line stroke="red" x1="6.5" y1="3" x2="7" y2="3"/>
<line stroke="red" x1="6" y1="4" x2="6.5" y2="5"/>
<line stroke="blue" x1="6" y1="4" x2="8" y2="4"/>
<line stroke="red" x1="7" y1="2" x2="7" y2="2.5"/>
<line stroke="blue" x1="7" y1="2" x2="8" y2="2"/>
<line stroke="red" x1="7" y1="2.5" x2="7" y2="3"/>
<line stroke="red" x1="7" y1="2.5" x2="8" y2="2"/>
<line stroke="red" x1="7" y1="3" x2="8" y2="4"/>
<line stroke="black" x1="0" y1="8" x2="0" y2="9"/>
<line stroke="blue" x1="0" y1="8" x2="6" y2="8"/>
<line stroke="blue" x1="8" y1="0" x2="8" y2="2"/>
<line stroke="black" x1="8" y1="0" x2="9" y2="0"/>
<line stroke="blue" x1="6.5" y1="5" x2="6" y2="6"/>
<line stroke="red" x1="6.5" y1="5" x2="9" y2="5"/>
<line stroke="blue" x1="8" y1="2" x2="8" y2="4"/>
<line stroke="red" x1="8" y1="2" x2="9" y2="2"/>
<line stroke="blue" x1="8" y1="2" x2="9" y2="2.5"/>
<line stroke="red" x1="6" y1="6" x2="6.5" y2="7"/>
<line stroke="blue" x1="6" y1="6" x2="9" y2="6"/>
<line stroke="blue" x1="8" y1="4" x2="9" y2="5"/>
<line stroke="black" x1="0" y1="9" x2="0" y2="10"/>
<line stroke="red" x1="0" y1="9" x2="6.5" y2="9"/>
<line stroke="red" x1="9" y1="0" x2="9" y2="2"/>
<line stroke="black" x1="9" y1="0" x2="10" y2="0"/>
<line stroke="red" x1="9" y1="2" x2="9" y2="2.5"/>
<line stroke="blue" x1="9" y1="2" x2="10" y2="2"/>
<line stroke="red" x1="9" y1="2.5" x2="10" y2="2"/>
<line stroke="red" x1="9" y1="2.5" x2="9" y2="5"/>
<line stroke="blue" x1="6.5" y1="7" x2="6" y2="8"/>
<line stroke="red" x1="6.5" y1="7" x2="9" y2="7"/>
<line stroke="black" x1="0" y1="10" x2="6" y2="10"/>
<line stroke="red" x1="6" y1="8" x2="6.5" y2="9"/>
<line stroke="blue" x1="6" y1="8" x2="8" y2="8"/>
<line stroke="black" x1="10" y1="0" x2="10" y2="2"/>
<line stroke="black" x1="10" y1="2" x2="10" y2="5"/>
<line stroke="red" x1="9" y1="5" x2="9" y2="6"/>
<line stroke="blue" x1="9" y1="5" x2="10" y2="5"/>
<line stroke="red" x1="9" y1="5" x2="10" y2="6"/>
<line stroke="blue" x1="9" y1="6" x2="9" y2="7"/>
<line stroke="blue" x1="9" y1="6" x2="10" y2="6"/>
<line stroke="red" x1="6.5" y1="9" x2="7" y2="9"/>
<line stroke="blue" x1="6.5" y1="9" x2="6" y2="10"/>
<line stroke="black" x1="10" y1="5" x2="10" y2="6"/>
<line stroke="red" x1="8" y1="8" x2="7" y2="9"/>
<line stroke="blue" x1="8" y1="8" x2="9" y2="7"/>
<line stroke="blue" x1="8" y1="8" x2="8" y2="10"/>
<line stroke="blue" x1="7" y1="9" x2="6" y2="10"/>
<line stroke="red" x1="7" y1="9" x2="7" y2="9.5"/>
<line stroke="red" x1="9" y1="7" x2="10" y2="6"/>
<line stroke="red" x1="9" y1="7" x2="10" y2="7"/>
<line stroke="red" x1="9" y1="7" x2="9" y2="9.5"/>
<line stroke="red" x1="6" y1="10" x2="7" y2="9.5"/>
<line stroke="black" x1="6" y1="10" x2="7" y2="10"/>
<line stroke="black" x1="10" y1="6" x2="10" y2="7"/>
<line stroke="red" x1="7" y1="9.5" x2="7" y2="10"/>
<line stroke="blue" x1="7" y1="9.5" x2="8" y2="10"/>
<line stroke="black" x1="7" y1="10" x2="8" y2="10"/>
<line stroke="black" x1="10" y1="7" x2="10" y2="10"/>
<line stroke="red" x1="8" y1="10" x2="9" y2="9.5"/>
<line stroke="black" x1="8" y1="10" x2="9" y2="10"/>
<line stroke="red" x1="9" y1="9.5" x2="9" y2="10"/>
<line stroke="blue" x1="9" y1="9.5" x2="10" y2="10"/>
<line stroke="black" x1="9" y1="10" x2="10" y2="10"/>
</svg>""",
        zigzag_creases=[
            crease_utils.Crease(
                point_utils.Point(x=0.6, y=0.2),
                point_utils.Point(x=0.7, y=0.25),
                fold=crease_utils.Fold.VALLEY,
            ),
            crease_utils.Crease(
                point_utils.Point(x=0.7, y=0.25),
                point_utils.Point(x=0.8, y=0.2),
                fold=crease_utils.Fold.MOUNTAIN,
            ),
            crease_utils.Crease(
                point_utils.Point(x=0.8, y=0.2),
                point_utils.Point(x=0.9, y=0.25),
                fold=crease_utils.Fold.VALLEY,
            ),
            crease_utils.Crease(
                point_utils.Point(x=0.9, y=0.25),
                point_utils.Point(x=1.0, y=0.2),
                fold=crease_utils.Fold.MOUNTAIN,
            ),
            crease_utils.Crease(
                point_utils.Point(x=0.6, y=1.0),
                point_utils.Point(x=0.7, y=0.95),
                fold=crease_utils.Fold.MOUNTAIN,
            ),
            crease_utils.Crease(
                point_utils.Point(x=0.7, y=0.95),
                point_utils.Point(x=0.8, y=1.0),
                fold=crease_utils.Fold.VALLEY,
            ),
            crease_utils.Crease(
                point_utils.Point(x=0.8, y=1.0),
                point_utils.Point(x=0.9, y=0.95),
                fold=crease_utils.Fold.MOUNTAIN,
            ),
            crease_utils.Crease(
                point_utils.Point(x=0.9, y=0.95),
                point_utils.Point(x=1.0, y=1.0),
                fold=crease_utils.Fold.VALLEY,
            ),
            crease_utils.Crease(
                point_utils.Point(x=0.6, y=0.2),
                point_utils.Point(x=0.65, y=0.3),
                fold=crease_utils.Fold.MOUNTAIN,
            ),
            crease_utils.Crease(
                point_utils.Point(x=0.65, y=0.3),
                point_utils.Point(x=0.6, y=0.4),
                fold=crease_utils.Fold.VALLEY,
            ),
            crease_utils.Crease(
                point_utils.Point(x=0.6, y=0.4),
                point_utils.Point(x=0.65, y=0.5),
                fold=crease_utils.Fold.MOUNTAIN,
            ),
            crease_utils.Crease(
                point_utils.Point(x=0.65, y=0.5),
                point_utils.Point(x=0.6, y=0.6),
                fold=crease_utils.Fold.VALLEY,
            ),
            crease_utils.Crease(
                point_utils.Point(x=0.6, y=0.6),
                point_utils.Point(x=0.65, y=0.7),
                fold=crease_utils.Fold.MOUNTAIN,
            ),
            crease_utils.Crease(
                point_utils.Point(x=0.65, y=0.7),
                point_utils.Point(x=0.6, y=0.8),
                fold=crease_utils.Fold.VALLEY,
            ),
            crease_utils.Crease(
                point_utils.Point(x=0.6, y=0.8),
                point_utils.Point(x=0.65, y=0.9),
                fold=crease_utils.Fold.MOUNTAIN,
            ),
            crease_utils.Crease(
                point_utils.Point(x=0.65, y=0.9),
                point_utils.Point(x=0.6, y=1.0),
                fold=crease_utils.Fold.VALLEY,
            ),
        ],
    ),
    rectangles=[
        shaping_utils.Rectangle(
            name="leg", x_start=0.6, y_start=0.2, x_len=0.4, y_len=0.8
        )
    ],
    narrowed_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="0" y2="100"/>
  <line stroke="black" x1="0" y1="0" x2="800" y2="0"/>
  <line stroke="black" x1="0" y1="100" x2="0" y2="200"/>
  <line stroke="red" x1="0" y1="100" x2="700" y2="100"/>
  <line stroke="black" x1="0" y1="200" x2="0" y2="300"/>
  <line stroke="blue" x1="0" y1="200" x2="600" y2="200"/>
  <line stroke="black" x1="0" y1="300" x2="0" y2="400"/>
  <line stroke="red" x1="0" y1="300" x2="600" y2="300"/>
  <line stroke="black" x1="0" y1="400" x2="0" y2="500"/>
  <line stroke="blue" x1="0" y1="400" x2="600" y2="400"/>
  <line stroke="black" x1="0" y1="500" x2="0" y2="600"/>
  <line stroke="red" x1="0" y1="500" x2="600" y2="500"/>
  <line stroke="black" x1="0" y1="600" x2="0" y2="700"/>
  <line stroke="blue" x1="0" y1="600" x2="600" y2="600"/>
  <line stroke="red" x1="600" y1="200" x2="625" y2="250"/>
  <line stroke="blue" x1="600" y1="200" x2="650" y2="225"/>
  <line stroke="blue" x1="600" y1="200" x2="650" y2="250"/>
  <line stroke="blue" x1="600" y1="200" x2="700" y2="100"/>
  <line stroke="red" x1="600" y1="200" x2="700" y2="200"/>
  <line stroke="red" x1="600" y1="300" x2="625" y2="250"/>
  <line stroke="red" x1="600" y1="300" x2="650" y2="300"/>
  <line stroke="blue" x1="600" y1="300" x2="625" y2="350"/>
  <line stroke="blue" x1="625" y1="250" x2="650" y2="250"/>
  <line stroke="red" x1="625" y1="250" x2="663" y2="300"/>
  <line stroke="red" x1="650" y1="225" x2="650" y2="250"/>
  <line stroke="blue" x1="650" y1="225" x2="700" y2="200"/>
  <line stroke="blue" x1="650" y1="225" x2="700" y2="263"/>
  <line stroke="blue" x1="650" y1="250" x2="700" y2="300"/>
  <line stroke="black" x1="0" y1="700" x2="0" y2="800"/>
  <line stroke="red" x1="0" y1="700" x2="600" y2="700"/>
  <line stroke="red" x1="700" y1="100" x2="700" y2="200"/>
  <line stroke="red" x1="700" y1="100" x2="800" y2="0"/>
  <line stroke="red" x1="650" y1="300" x2="663" y2="300"/>
  <line stroke="blue" x1="625" y1="350" x2="600" y2="400"/>
  <line stroke="blue" x1="625" y1="350" x2="663" y2="300"/>
  <line stroke="red" x1="625" y1="350" x2="750" y2="350"/>
  <line stroke="red" x1="600" y1="400" x2="625" y2="450"/>
  <line stroke="blue" x1="600" y1="400" x2="800" y2="400"/>
  <line stroke="red" x1="663" y1="300" x2="700" y2="300"/>
  <line stroke="red" x1="700" y1="200" x2="700" y2="250"/>
  <line stroke="red" x1="700" y1="200" x2="750" y2="225"/>
  <line stroke="blue" x1="700" y1="200" x2="800" y2="200"/>
  <line stroke="red" x1="700" y1="250" x2="700" y2="263"/>
  <line stroke="red" x1="700" y1="263" x2="700" y2="300"/>
  <line stroke="red" x1="700" y1="263" x2="750" y2="225"/>
  <line stroke="red" x1="700" y1="300" x2="750" y2="350"/>
  <line stroke="red" x1="625" y1="450" x2="600" y2="500"/>
  <line stroke="red" x1="625" y1="450" x2="662" y2="500"/>
  <line stroke="blue" x1="625" y1="450" x2="850" y2="450"/>
  <line stroke="red" x1="600" y1="500" x2="650" y2="500"/>
  <line stroke="blue" x1="600" y1="500" x2="625" y2="550"/>
  <line stroke="red" x1="750" y1="225" x2="800" y2="200"/>
  <line stroke="blue" x1="750" y1="225" x2="750" y2="350"/>
  <line stroke="black" x1="0" y1="800" x2="0" y2="900"/>
  <line stroke="blue" x1="0" y1="800" x2="600" y2="800"/>
  <line stroke="blue" x1="800" y1="0" x2="800" y2="200"/>
  <line stroke="black" x1="800" y1="0" x2="900" y2="0"/>
  <line stroke="red" x1="650" y1="500" x2="662" y2="500"/>
  <line stroke="blue" x1="800" y1="200" x2="850" y2="225"/>
  <line stroke="blue" x1="800" y1="200" x2="800" y2="400"/>
  <line stroke="red" x1="800" y1="200" x2="900" y2="200"/>
  <line stroke="red" x1="750" y1="350" x2="800" y2="400"/>
  <line stroke="blue" x1="662" y1="500" x2="625" y2="550"/>
  <line stroke="red" x1="662" y1="500" x2="900" y2="500"/>
  <line stroke="blue" x1="625" y1="550" x2="600" y2="600"/>
  <line stroke="red" x1="625" y1="550" x2="900" y2="550"/>
  <line stroke="red" x1="600" y1="600" x2="625" y2="650"/>
  <line stroke="blue" x1="600" y1="600" x2="900" y2="600"/>
  <line stroke="blue" x1="850" y1="225" x2="900" y2="200"/>
  <line stroke="blue" x1="850" y1="225" x2="900" y2="263"/>
  <line stroke="red" x1="850" y1="225" x2="850" y2="450"/>
  <line stroke="blue" x1="800" y1="400" x2="850" y2="450"/>
  <line stroke="black" x1="0" y1="900" x2="0" y2="1000"/>
  <line stroke="red" x1="0" y1="900" x2="600" y2="900"/>
  <line stroke="red" x1="900" y1="0" x2="900" y2="200"/>
  <line stroke="black" x1="900" y1="0" x2="1000" y2="0"/>
  <line stroke="red" x1="625" y1="650" x2="600" y2="700"/>
  <line stroke="red" x1="625" y1="650" x2="663" y2="700"/>
  <line stroke="blue" x1="625" y1="650" x2="900" y2="650"/>
  <line stroke="red" x1="600" y1="700" x2="650" y2="700"/>
  <line stroke="blue" x1="600" y1="700" x2="625" y2="750"/>
  <line stroke="red" x1="900" y1="200" x2="900" y2="250"/>
  <line stroke="red" x1="900" y1="200" x2="950" y2="225"/>
  <line stroke="blue" x1="900" y1="200" x2="1000" y2="200"/>
  <line stroke="red" x1="900" y1="250" x2="900" y2="263"/>
  <line stroke="red" x1="900" y1="263" x2="950" y2="225"/>
  <line stroke="red" x1="900" y1="263" x2="900" y2="500"/>
  <line stroke="red" x1="650" y1="700" x2="663" y2="700"/>
  <line stroke="blue" x1="850" y1="450" x2="900" y2="500"/>
  <line stroke="blue" x1="663" y1="700" x2="625" y2="750"/>
  <line stroke="red" x1="663" y1="700" x2="900" y2="700"/>
  <line stroke="blue" x1="625" y1="750" x2="600" y2="800"/>
  <line stroke="red" x1="625" y1="750" x2="850" y2="750"/>
  <line stroke="red" x1="950" y1="225" x2="1000" y2="200"/>
  <line stroke="blue" x1="950" y1="225" x2="950" y2="500"/>
  <line stroke="black" x1="0" y1="1000" x2="600" y2="1000"/>
  <line stroke="red" x1="600" y1="800" x2="625" y2="850"/>
  <line stroke="blue" x1="600" y1="800" x2="800" y2="800"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="200"/>
  <line stroke="black" x1="1000" y1="200" x2="1000" y2="500"/>
  <line stroke="red" x1="900" y1="500" x2="900" y2="550"/>
  <line stroke="blue" x1="900" y1="500" x2="950" y2="500"/>
  <line stroke="red" x1="900" y1="500" x2="950" y2="550"/>
  <line stroke="red" x1="900" y1="550" x2="900" y2="600"/>
  <line stroke="blue" x1="900" y1="550" x2="950" y2="550"/>
  <line stroke="red" x1="625" y1="850" x2="600" y2="900"/>
  <line stroke="red" x1="625" y1="850" x2="662" y2="900"/>
  <line stroke="blue" x1="625" y1="850" x2="750" y2="850"/>
  <line stroke="red" x1="950" y1="500" x2="950" y2="550"/>
  <line stroke="blue" x1="950" y1="500" x2="1000" y2="500"/>
  <line stroke="red" x1="600" y1="900" x2="650" y2="900"/>
  <line stroke="blue" x1="600" y1="900" x2="625" y2="950"/>
  <line stroke="blue" x1="900" y1="600" x2="900" y2="650"/>
  <line stroke="blue" x1="900" y1="600" x2="1000" y2="600"/>
  <line stroke="red" x1="950" y1="550" x2="1000" y2="600"/>
  <line stroke="red" x1="650" y1="900" x2="662" y2="900"/>
  <line stroke="blue" x1="900" y1="650" x2="900" y2="700"/>
  <line stroke="red" x1="900" y1="650" x2="950" y2="650"/>
  <line stroke="blue" x1="662" y1="900" x2="625" y2="950"/>
  <line stroke="red" x1="662" y1="900" x2="700" y2="900"/>
  <line stroke="black" x1="1000" y1="500" x2="1000" y2="600"/>
  <line stroke="red" x1="800" y1="800" x2="750" y2="850"/>
  <line stroke="blue" x1="800" y1="800" x2="850" y2="750"/>
  <line stroke="blue" x1="800" y1="800" x2="800" y2="1000"/>
  <line stroke="red" x1="750" y1="850" x2="700" y2="900"/>
  <line stroke="red" x1="750" y1="850" x2="750" y2="975"/>
  <line stroke="blue" x1="850" y1="750" x2="900" y2="700"/>
  <line stroke="blue" x1="850" y1="750" x2="850" y2="975"/>
  <line stroke="red" x1="625" y1="950" x2="650" y2="950"/>
  <line stroke="blue" x1="625" y1="950" x2="600" y2="1000"/>
  <line stroke="blue" x1="700" y1="900" x2="650" y2="950"/>
  <line stroke="red" x1="700" y1="900" x2="700" y2="937"/>
  <line stroke="red" x1="900" y1="700" x2="950" y2="650"/>
  <line stroke="red" x1="900" y1="700" x2="950" y2="700"/>
  <line stroke="red" x1="900" y1="700" x2="900" y2="937"/>
  <line stroke="blue" x1="650" y1="950" x2="600" y2="1000"/>
  <line stroke="blue" x1="650" y1="950" x2="650" y2="975"/>
  <line stroke="red" x1="950" y1="650" x2="1000" y2="600"/>
  <line stroke="blue" x1="950" y1="650" x2="950" y2="700"/>
  <line stroke="red" x1="600" y1="1000" x2="650" y2="975"/>
  <line stroke="black" x1="600" y1="1000" x2="700" y2="1000"/>
  <line stroke="black" x1="1000" y1="600" x2="1000" y2="700"/>
  <line stroke="blue" x1="700" y1="937" x2="750" y2="975"/>
  <line stroke="red" x1="700" y1="938" x2="650" y2="975"/>
  <line stroke="red" x1="700" y1="938" x2="700" y2="950"/>
  <line stroke="red" x1="650" y1="975" x2="700" y2="1000"/>
  <line stroke="red" x1="700" y1="950" x2="700" y2="1000"/>
  <line stroke="red" x1="950" y1="700" x2="1000" y2="700"/>
  <line stroke="red" x1="950" y1="700" x2="950" y2="975"/>
  <line stroke="blue" x1="700" y1="1000" x2="750" y2="975"/>
  <line stroke="black" x1="700" y1="1000" x2="800" y2="1000"/>
  <line stroke="black" x1="1000" y1="700" x2="1000" y2="1000"/>
  <line stroke="blue" x1="750" y1="975" x2="800" y2="1000"/>
  <line stroke="red" x1="800" y1="1000" x2="850" y2="975"/>
  <line stroke="black" x1="800" y1="1000" x2="900" y2="1000"/>
  <line stroke="red" x1="850" y1="975" x2="900" y2="938"/>
  <line stroke="red" x1="850" y1="975" x2="900" y2="1000"/>
  <line stroke="blue" x1="900" y1="937" x2="950" y2="975"/>
  <line stroke="red" x1="900" y1="938" x2="900" y2="950"/>
  <line stroke="red" x1="900" y1="950" x2="900" y2="1000"/>
  <line stroke="blue" x1="900" y1="1000" x2="950" y2="975"/>
  <line stroke="black" x1="900" y1="1000" x2="1000" y2="1000"/>
  <line stroke="blue" x1="950" y1="975" x2="1000" y2="1000"/>
</svg>""",
)

BASIC_NARROW_45_ZIGZAG_TOP = BoxPleatingShapingExample(
    base_example=BASIC_NARROW_45_ZIGZAG_BASE,
    rectangles=[
        shaping_utils.Rectangle(
            name="top_flap", x_start=0, y_start=0, x_len=1, y_len=0.5
        ),
    ],
    narrowed_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="125" y2="0"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="500"/>
  <line stroke="black" x1="125" y1="0" x2="250" y2="0"/>
  <line stroke="blue" x1="125" y1="0" x2="125" y2="375"/>
  <line stroke="red" x1="250" y1="0" x2="250" y2="250"/>
  <line stroke="black" x1="250" y1="0" x2="375" y2="0"/>
  <line stroke="red" x1="250" y1="250" x2="125" y2="375"/>
  <line stroke="red" x1="250" y1="250" x2="250" y2="375"/>
  <line stroke="blue" x1="250" y1="250" x2="375" y2="375"/>
  <line stroke="black" x1="375" y1="0" x2="500" y2="0"/>
  <line stroke="red" x1="375" y1="0" x2="375" y2="375"/>
  <line stroke="red" x1="125" y1="375" x2="250" y2="375"/>
  <line stroke="red" x1="125" y1="375" x2="0" y2="500"/>
  <line stroke="blue" x1="250" y1="375" x2="375" y2="375"/>
  <line stroke="red" x1="250" y1="375" x2="250" y2="1000"/>
  <line stroke="black" x1="0" y1="500" x2="0" y2="1000"/>
  <line stroke="black" x1="500" y1="0" x2="625" y2="0"/>
  <line stroke="blue" x1="500" y1="0" x2="500" y2="500"/>
  <line stroke="blue" x1="375" y1="375" x2="500" y2="500"/>
  <line stroke="red" x1="625" y1="0" x2="625" y2="375"/>
  <line stroke="black" x1="625" y1="0" x2="750" y2="0"/>
  <line stroke="red" x1="500" y1="500" x2="625" y2="375"/>
  <line stroke="blue" x1="500" y1="500" x2="500" y2="1000"/>
  <line stroke="red" x1="625" y1="375" x2="750" y2="250"/>
  <line stroke="blue" x1="625" y1="375" x2="750" y2="375"/>
  <line stroke="red" x1="750" y1="0" x2="750" y2="250"/>
  <line stroke="black" x1="750" y1="0" x2="875" y2="0"/>
  <line stroke="red" x1="750" y1="250" x2="750" y2="375"/>
  <line stroke="blue" x1="750" y1="250" x2="875" y2="375"/>
  <line stroke="red" x1="750" y1="375" x2="875" y2="375"/>
  <line stroke="red" x1="750" y1="375" x2="750" y2="1000"/>
  <line stroke="blue" x1="875" y1="0" x2="875" y2="375"/>
  <line stroke="black" x1="875" y1="0" x2="1000" y2="0"/>
  <line stroke="blue" x1="875" y1="375" x2="1000" y2="500"/>
  <line stroke="black" x1="0" y1="1000" x2="250" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="500"/>
  <line stroke="black" x1="250" y1="1000" x2="500" y2="1000"/>
  <line stroke="black" x1="500" y1="1000" x2="750" y2="1000"/>
  <line stroke="black" x1="1000" y1="500" x2="1000" y2="1000"/>
  <line stroke="black" x1="750" y1="1000" x2="1000" y2="1000"/>
</svg>""",
)

BASIC_NARROW_45_ZIGZAG_BOTTOM = BoxPleatingShapingExample(
    base_example=BASIC_NARROW_45_ZIGZAG_BASE,
    rectangles=[
        shaping_utils.Rectangle(
            name="bottom_flap", x_start=0, y_start=0.25, x_len=1, y_len=0.75
        ),
    ],
    narrowed_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="250" y2="0"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="375"/>
  <line stroke="red" x1="250" y1="0" x2="250" y2="250"/>
  <line stroke="black" x1="250" y1="0" x2="500" y2="0"/>
  <line stroke="red" x1="250" y1="250" x2="125" y2="375"/>
  <line stroke="blue" x1="250" y1="250" x2="375" y2="375"/>
  <line stroke="red" x1="250" y1="250" x2="250" y2="1000"/>
  <line stroke="blue" x1="0" y1="375" x2="125" y2="375"/>
  <line stroke="black" x1="0" y1="375" x2="0" y2="500"/>
  <line stroke="red" x1="125" y1="375" x2="0" y2="500"/>
  <line stroke="red" x1="125" y1="375" x2="125" y2="1000"/>
  <line stroke="black" x1="0" y1="500" x2="0" y2="1000"/>
  <line stroke="blue" x1="500" y1="0" x2="500" y2="375"/>
  <line stroke="black" x1="500" y1="0" x2="750" y2="0"/>
  <line stroke="blue" x1="375" y1="375" x2="500" y2="375"/>
  <line stroke="blue" x1="375" y1="375" x2="500" y2="500"/>
  <line stroke="red" x1="375" y1="375" x2="375" y2="1000"/>
  <line stroke="blue" x1="500" y1="375" x2="500" y2="500"/>
  <line stroke="red" x1="500" y1="375" x2="625" y2="375"/>
  <line stroke="red" x1="500" y1="500" x2="625" y2="375"/>
  <line stroke="blue" x1="500" y1="500" x2="500" y2="1000"/>
  <line stroke="red" x1="625" y1="375" x2="750" y2="250"/>
  <line stroke="blue" x1="625" y1="375" x2="625" y2="1000"/>
  <line stroke="red" x1="750" y1="0" x2="750" y2="250"/>
  <line stroke="black" x1="750" y1="0" x2="1000" y2="0"/>
  <line stroke="blue" x1="750" y1="250" x2="875" y2="375"/>
  <line stroke="red" x1="750" y1="250" x2="750" y2="1000"/>
  <line stroke="blue" x1="875" y1="375" x2="1000" y2="375"/>
  <line stroke="blue" x1="875" y1="375" x2="1000" y2="500"/>
  <line stroke="red" x1="875" y1="375" x2="875" y2="1000"/>
  <line stroke="black" x1="0" y1="1000" x2="125" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="375"/>
  <line stroke="black" x1="125" y1="1000" x2="250" y2="1000"/>
  <line stroke="black" x1="250" y1="1000" x2="375" y2="1000"/>
  <line stroke="black" x1="375" y1="1000" x2="500" y2="1000"/>
  <line stroke="black" x1="1000" y1="375" x2="1000" y2="500"/>
  <line stroke="black" x1="500" y1="1000" x2="625" y2="1000"/>
  <line stroke="black" x1="1000" y1="500" x2="1000" y2="1000"/>
  <line stroke="black" x1="625" y1="1000" x2="750" y2="1000"/>
  <line stroke="black" x1="750" y1="1000" x2="875" y2="1000"/>
  <line stroke="black" x1="875" y1="1000" x2="1000" y2="1000"/>
</svg>""",
)

BASIC_NARROW_22P5_ZIGZAG_TOP = BoxPleatingShapingExample(
    base_example=BASIC_NARROW_22P5_ZIGZAG_BASE,
    rectangles=[
        shaping_utils.Rectangle(
            name="top_flap", x_start=0, y_start=0, x_len=1, y_len=0.5
        ),
    ],
    narrowed_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="125" y2="0"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="500"/>
  <line stroke="black" x1="125" y1="0" x2="250" y2="0"/>
  <line stroke="blue" x1="125" y1="0" x2="125" y2="438"/>
  <line stroke="black" x1="250" y1="0" x2="375" y2="0"/>
  <line stroke="red" x1="250" y1="0" x2="250" y2="344"/>
  <line stroke="black" x1="375" y1="0" x2="500" y2="0"/>
  <line stroke="red" x1="375" y1="0" x2="375" y2="438"/>
  <line stroke="red" x1="250" y1="344" x2="250" y2="375"/>
  <line stroke="red" x1="250" y1="344" x2="125" y2="438"/>
  <line stroke="blue" x1="250" y1="344" x2="375" y2="438"/>
  <line stroke="red" x1="250" y1="375" x2="250" y2="500"/>
  <line stroke="red" x1="125" y1="438" x2="0" y2="500"/>
  <line stroke="red" x1="125" y1="438" x2="250" y2="500"/>
  <line stroke="black" x1="0" y1="500" x2="0" y2="1000"/>
  <line stroke="black" x1="500" y1="0" x2="625" y2="0"/>
  <line stroke="blue" x1="500" y1="0" x2="500" y2="500"/>
  <line stroke="blue" x1="250" y1="500" x2="375" y2="438"/>
  <line stroke="red" x1="250" y1="500" x2="250" y2="1000"/>
  <line stroke="blue" x1="375" y1="438" x2="500" y2="500"/>
  <line stroke="black" x1="625" y1="0" x2="750" y2="0"/>
  <line stroke="blue" x1="625" y1="0" x2="625" y2="438"/>
  <line stroke="red" x1="500" y1="500" x2="625" y2="438"/>
  <line stroke="blue" x1="500" y1="500" x2="500" y2="1000"/>
  <line stroke="red" x1="750" y1="0" x2="750" y2="344"/>
  <line stroke="black" x1="750" y1="0" x2="875" y2="0"/>
  <line stroke="red" x1="625" y1="438" x2="750" y2="344"/>
  <line stroke="red" x1="625" y1="438" x2="750" y2="500"/>
  <line stroke="red" x1="750" y1="344" x2="750" y2="375"/>
  <line stroke="blue" x1="750" y1="344" x2="875" y2="438"/>
  <line stroke="red" x1="750" y1="375" x2="750" y2="500"/>
  <line stroke="red" x1="875" y1="0" x2="875" y2="438"/>
  <line stroke="black" x1="875" y1="0" x2="1000" y2="0"/>
  <line stroke="blue" x1="750" y1="500" x2="875" y2="438"/>
  <line stroke="red" x1="750" y1="500" x2="750" y2="1000"/>
  <line stroke="blue" x1="875" y1="438" x2="1000" y2="500"/>
  <line stroke="black" x1="0" y1="1000" x2="250" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="500"/>
  <line stroke="black" x1="250" y1="1000" x2="500" y2="1000"/>
  <line stroke="black" x1="500" y1="1000" x2="750" y2="1000"/>
  <line stroke="black" x1="1000" y1="500" x2="1000" y2="1000"/>
  <line stroke="black" x1="750" y1="1000" x2="1000" y2="1000"/>
</svg>""",
)

BASIC_NARROW_56P31_ZIGZAG_TOP = BoxPleatingShapingExample(
    base_example=BASIC_NARROW_56P31_ZIGZAG_BASE,
    rectangles=[
        shaping_utils.Rectangle(
            name="top_flap", x_start=0, y_start=0, x_len=1, y_len=0.5
        ),
    ],
    narrowed_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="125" y2="0"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="500"/>
  <line stroke="black" x1="125" y1="0" x2="250" y2="0"/>
  <line stroke="blue" x1="125" y1="0" x2="125" y2="312.5"/>
  <line stroke="red" x1="250" y1="0" x2="250" y2="125"/>
  <line stroke="black" x1="250" y1="0" x2="375" y2="0"/>
  <line stroke="red" x1="250" y1="125" x2="125" y2="312.5"/>
  <line stroke="red" x1="250" y1="125" x2="250" y2="260.4"/>
  <line stroke="blue" x1="250" y1="125" x2="375" y2="312.5"/>
  <line stroke="red" x1="125" y1="312.5" x2="250" y2="260.4"/>
  <line stroke="red" x1="125" y1="312.5" x2="0" y2="500"/>
  <line stroke="red" x1="250" y1="260.4" x2="250" y2="1000"/>
  <line stroke="blue" x1="250" y1="260.4" x2="375" y2="312.5"/>
  <line stroke="red" x1="375" y1="0" x2="375" y2="312.5"/>
  <line stroke="black" x1="375" y1="0" x2="500" y2="0"/>
  <line stroke="blue" x1="375" y1="312.5" x2="500" y2="500"/>
  <line stroke="black" x1="0" y1="500" x2="0" y2="1000"/>
  <line stroke="black" x1="500" y1="0" x2="625" y2="0"/>
  <line stroke="blue" x1="500" y1="0" x2="500" y2="500"/>
  <line stroke="red" x1="625" y1="0" x2="625" y2="312.5"/>
  <line stroke="black" x1="625" y1="0" x2="750" y2="0"/>
  <line stroke="red" x1="625" y1="312.5" x2="500" y2="500"/>
  <line stroke="red" x1="625" y1="312.5" x2="750" y2="125"/>
  <line stroke="blue" x1="625" y1="312.5" x2="750" y2="260.4"/>
  <line stroke="blue" x1="500" y1="500" x2="500" y2="1000"/>
  <line stroke="red" x1="750" y1="0" x2="750" y2="125"/>
  <line stroke="black" x1="750" y1="0" x2="875" y2="0"/>
  <line stroke="red" x1="750" y1="125" x2="750" y2="260.4"/>
  <line stroke="blue" x1="750" y1="125" x2="875" y2="312.5"/>
  <line stroke="red" x1="750" y1="260.4" x2="875" y2="312.5"/>
  <line stroke="red" x1="750" y1="260.4" x2="750" y2="1000"/>
  <line stroke="blue" x1="875" y1="0" x2="875" y2="312.5"/>
  <line stroke="black" x1="875" y1="0" x2="1000" y2="0"/>
  <line stroke="blue" x1="875" y1="312.5" x2="1000" y2="500"/>
  <line stroke="black" x1="0" y1="1000" x2="250" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="500"/>
  <line stroke="black" x1="250" y1="1000" x2="500" y2="1000"/>
  <line stroke="black" x1="500" y1="1000" x2="750" y2="1000"/>
  <line stroke="black" x1="1000" y1="500" x2="1000" y2="1000"/>
  <line stroke="black" x1="750" y1="1000" x2="1000" y2="1000"/>
</svg>""",
)

BASIC_NARROW_56P31_ZIGZAG_BOTTOM = BoxPleatingShapingExample(
    base_example=BoxPleatingCreasePatternExample(
        name="basic_narrow_22p5_zigzag",
        packing_svg="",
        grid_size=4,
        base_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="250" y2="0"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="625"/>
  <line stroke="red" x1="250" y1="0" x2="250" y2="250"/>
  <line stroke="black" x1="250" y1="0" x2="500" y2="0"/>
  <line stroke="red" x1="250" y1="250" x2="0" y2="625"/>
  <line stroke="blue" x1="250" y1="250" x2="500" y2="625"/>
  <line stroke="red" x1="250" y1="250" x2="250" y2="1000"/>
  <line stroke="black" x1="500" y1="0" x2="750" y2="0"/>
  <line stroke="blue" x1="500" y1="0" x2="500" y2="625"/>
  <line stroke="black" x1="0" y1="625" x2="0" y2="1000"/>
  <line stroke="red" x1="750" y1="0" x2="750" y2="250"/>
  <line stroke="black" x1="750" y1="0" x2="1000" y2="0"/>
  <line stroke="red" x1="750" y1="250" x2="500" y2="625"/>
  <line stroke="blue" x1="750" y1="250" x2="1000" y2="625"/>
  <line stroke="red" x1="750" y1="250" x2="750" y2="1000"/>
  <line stroke="blue" x1="500" y1="625" x2="500" y2="1000"/>
  <line stroke="black" x1="0" y1="1000" x2="250" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="625"/>
  <line stroke="black" x1="250" y1="1000" x2="500" y2="1000"/>
  <line stroke="black" x1="500" y1="1000" x2="750" y2="1000"/>
  <line stroke="black" x1="1000" y1="625" x2="1000" y2="1000"/>
  <line stroke="black" x1="750" y1="1000" x2="1000" y2="1000"/>
</svg>""",
        zigzag_creases=[
            crease_utils.Crease(
                point_utils.Point(x=0 / 8, y=5 / 8),
                point_utils.Point(x=2 / 8, y=2 / 8),
                fold=crease_utils.Fold.MOUNTAIN,
            ),
            crease_utils.Crease(
                point_utils.Point(x=2 / 8, y=2 / 8),
                point_utils.Point(x=4 / 8, y=5 / 8),
                fold=crease_utils.Fold.VALLEY,
            ),
            crease_utils.Crease(
                point_utils.Point(x=4 / 8, y=5 / 8),
                point_utils.Point(x=6 / 8, y=2 / 8),
                fold=crease_utils.Fold.MOUNTAIN,
            ),
            crease_utils.Crease(
                point_utils.Point(x=6 / 8, y=2 / 8),
                point_utils.Point(x=8 / 8, y=5 / 8),
                fold=crease_utils.Fold.VALLEY,
            ),
        ],
    ),
    rectangles=[
        shaping_utils.Rectangle(
            name="bottom_flap",
            x_start=0 / 8,
            y_start=2 / 8,
            x_len=8 / 8,
            y_len=6 / 8,
        ),
    ],
    narrowed_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="250" y2="0"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="490"/>
  <line stroke="red" x1="250" y1="0" x2="250" y2="250"/>
  <line stroke="black" x1="250" y1="0" x2="500" y2="0"/>
  <line stroke="red" x1="250" y1="250" x2="125" y2="438"/>
  <line stroke="blue" x1="250" y1="250" x2="375" y2="438"/>
  <line stroke="red" x1="250" y1="250" x2="250" y2="1000"/>
  <line stroke="blue" x1="125" y1="438" x2="0" y2="490"/>
  <line stroke="red" x1="125" y1="438" x2="0" y2="625"/>
  <line stroke="red" x1="125" y1="438" x2="125" y2="1000"/>
  <line stroke="black" x1="0" y1="490" x2="0" y2="625"/>
  <line stroke="blue" x1="500" y1="0" x2="500" y2="490"/>
  <line stroke="black" x1="500" y1="0" x2="750" y2="0"/>
  <line stroke="blue" x1="375" y1="438" x2="500" y2="490"/>
  <line stroke="blue" x1="375" y1="438" x2="500" y2="625"/>
  <line stroke="red" x1="375" y1="438" x2="375" y2="1000"/>
  <line stroke="black" x1="0" y1="625" x2="0" y2="1000"/>
  <line stroke="red" x1="500" y1="490" x2="625" y2="438"/>
  <line stroke="blue" x1="500" y1="490" x2="500" y2="625"/>
  <line stroke="red" x1="750" y1="0" x2="750" y2="250"/>
  <line stroke="black" x1="750" y1="0" x2="1000" y2="0"/>
  <line stroke="red" x1="625" y1="438" x2="750" y2="250"/>
  <line stroke="red" x1="625" y1="438" x2="500" y2="625"/>
  <line stroke="blue" x1="625" y1="438" x2="625" y2="1000"/>
  <line stroke="blue" x1="750" y1="250" x2="875" y2="438"/>
  <line stroke="red" x1="750" y1="250" x2="750" y2="1000"/>
  <line stroke="blue" x1="500" y1="625" x2="500" y2="1000"/>
  <line stroke="blue" x1="875" y1="438" x2="1000" y2="490"/>
  <line stroke="blue" x1="875" y1="438" x2="1000" y2="625"/>
  <line stroke="red" x1="875" y1="438" x2="875" y2="1000"/>
  <line stroke="black" x1="0" y1="1000" x2="125" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="490"/>
  <line stroke="black" x1="125" y1="1000" x2="250" y2="1000"/>
  <line stroke="black" x1="250" y1="1000" x2="375" y2="1000"/>
  <line stroke="black" x1="375" y1="1000" x2="500" y2="1000"/>
  <line stroke="black" x1="1000" y1="490" x2="1000" y2="625"/>
  <line stroke="black" x1="500" y1="1000" x2="625" y2="1000"/>
  <line stroke="black" x1="625" y1="1000" x2="750" y2="1000"/>
  <line stroke="black" x1="1000" y1="625" x2="1000" y2="1000"/>
  <line stroke="black" x1="750" y1="1000" x2="875" y2="1000"/>
  <line stroke="black" x1="875" y1="1000" x2="1000" y2="1000"/>
</svg>""",
)

BASIC_NARROW_26P56_ZIGZAG_TOP = BoxPleatingShapingExample(
    base_example=BoxPleatingCreasePatternExample(
        name="basic_narrow_22p5_zigzag",
        packing_svg="",
        grid_size=4,
        base_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="250" y2="0"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="500"/>
  <line stroke="blue" x1="250" y1="0" x2="250" y2="375"/>
  <line stroke="black" x1="250" y1="0" x2="500" y2="0"/>
  <line stroke="blue" x1="250" y1="375" x2="0" y2="500"/>
  <line stroke="red" x1="250" y1="375" x2="500" y2="500"/>
  <line stroke="blue" x1="250" y1="375" x2="250" y2="1000"/>
  <line stroke="black" x1="0" y1="500" x2="0" y2="1000"/>
  <line stroke="red" x1="500" y1="0" x2="500" y2="500"/>
  <line stroke="black" x1="500" y1="0" x2="750" y2="0"/>
  <line stroke="blue" x1="500" y1="500" x2="750" y2="375"/>
  <line stroke="red" x1="500" y1="500" x2="500" y2="1000"/>
  <line stroke="blue" x1="750" y1="0" x2="750" y2="375"/>
  <line stroke="black" x1="750" y1="0" x2="1000" y2="0"/>
  <line stroke="red" x1="750" y1="375" x2="1000" y2="500"/>
  <line stroke="blue" x1="750" y1="375" x2="750" y2="1000"/>
  <line stroke="black" x1="0" y1="1000" x2="250" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="500"/>
  <line stroke="black" x1="250" y1="1000" x2="500" y2="1000"/>
  <line stroke="black" x1="500" y1="1000" x2="750" y2="1000"/>
  <line stroke="black" x1="1000" y1="500" x2="1000" y2="1000"/>
  <line stroke="black" x1="750" y1="1000" x2="1000" y2="1000"/>
</svg>""",
        zigzag_creases=[
            crease_utils.Crease(
                p1_init=point_utils.Point(x=0 / 8, y=4 / 8),
                p2_init=point_utils.Point(x=2 / 8, y=3 / 8),
                fold=crease_utils.Fold.VALLEY,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(x=2 / 8, y=3 / 8),
                p2_init=point_utils.Point(x=4 / 8, y=4 / 8),
                fold=crease_utils.Fold.MOUNTAIN,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(x=4 / 8, y=4 / 8),
                p2_init=point_utils.Point(x=6 / 8, y=3 / 8),
                fold=crease_utils.Fold.VALLEY,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(x=6 / 8, y=3 / 8),
                p2_init=point_utils.Point(x=8 / 8, y=4 / 8),
                fold=crease_utils.Fold.MOUNTAIN,
            ),
        ],
    ),
    rectangles=[
        shaping_utils.Rectangle(
            name="top",
            x_start=0 / 8,
            y_start=0 / 8,
            x_len=8 / 8,
            y_len=4 / 8,
        ),
        shaping_utils.Rectangle(
            name="bottom",
            x_start=0 / 8,
            y_start=4 / 8,
            x_len=8 / 8,
            y_len=4 / 8,
        ),
    ],
    narrowed_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="125" y2="0"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="500"/>
  <line stroke="black" x1="125" y1="0" x2="250" y2="0"/>
  <line stroke="red" x1="125" y1="0" x2="125" y2="438"/>
  <line stroke="black" x1="250" y1="0" x2="375" y2="0"/>
  <line stroke="blue" x1="250" y1="0" x2="250" y2="344"/>
  <line stroke="black" x1="375" y1="0" x2="500" y2="0"/>
  <line stroke="blue" x1="375" y1="0" x2="375" y2="438"/>
  <line stroke="blue" x1="250" y1="344" x2="250" y2="375"/>
  <line stroke="blue" x1="250" y1="344" x2="125" y2="438"/>
  <line stroke="red" x1="250" y1="344" x2="375" y2="438"/>
  <line stroke="blue" x1="250" y1="375" x2="250" y2="500"/>
  <line stroke="blue" x1="125" y1="438" x2="0" y2="500"/>
  <line stroke="blue" x1="125" y1="438" x2="250" y2="500"/>
  <line stroke="black" x1="0" y1="500" x2="0" y2="1000"/>
  <line stroke="black" x1="500" y1="0" x2="625" y2="0"/>
  <line stroke="red" x1="500" y1="0" x2="500" y2="500"/>
  <line stroke="red" x1="250" y1="500" x2="375" y2="438"/>
  <line stroke="blue" x1="250" y1="500" x2="250" y2="1000"/>
  <line stroke="red" x1="375" y1="438" x2="500" y2="500"/>
  <line stroke="black" x1="625" y1="0" x2="750" y2="0"/>
  <line stroke="red" x1="625" y1="0" x2="625" y2="438"/>
  <line stroke="blue" x1="500" y1="500" x2="625" y2="438"/>
  <line stroke="red" x1="500" y1="500" x2="500" y2="1000"/>
  <line stroke="blue" x1="750" y1="0" x2="750" y2="344"/>
  <line stroke="black" x1="750" y1="0" x2="875" y2="0"/>
  <line stroke="blue" x1="625" y1="438" x2="750" y2="344"/>
  <line stroke="blue" x1="625" y1="438" x2="750" y2="500"/>
  <line stroke="blue" x1="750" y1="344" x2="750" y2="375"/>
  <line stroke="red" x1="750" y1="344" x2="875" y2="438"/>
  <line stroke="blue" x1="750" y1="375" x2="750" y2="500"/>
  <line stroke="blue" x1="875" y1="0" x2="875" y2="438"/>
  <line stroke="black" x1="875" y1="0" x2="1000" y2="0"/>
  <line stroke="red" x1="750" y1="500" x2="875" y2="438"/>
  <line stroke="blue" x1="750" y1="500" x2="750" y2="1000"/>
  <line stroke="red" x1="875" y1="438" x2="1000" y2="500"/>
  <line stroke="black" x1="0" y1="1000" x2="250" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="500"/>
  <line stroke="black" x1="250" y1="1000" x2="500" y2="1000"/>
  <line stroke="black" x1="500" y1="1000" x2="750" y2="1000"/>
  <line stroke="black" x1="1000" y1="500" x2="1000" y2="1000"/>
  <line stroke="black" x1="750" y1="1000" x2="1000" y2="1000"/>
</svg>""",
)

NARROW_EDGE_CASE1 = BoxPleatingShapingExample(
    base_example=BoxPleatingCreasePatternExample(
        name="narrow_edge_case1",
        packing_svg="",
        grid_size=26,
        base_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="0" y2="38"/>
  <line stroke="black" x1="0" y1="0" x2="192" y2="0"/>
  <line stroke="black" x1="0" y1="38" x2="0" y2="77"/>
  <line stroke="red" x1="0" y1="38" x2="192" y2="38" elevation="1"/>
  <line stroke="black" x1="0" y1="77" x2="0" y2="115"/>
  <line stroke="blue" x1="0" y1="77" x2="192" y2="77" elevation="0"/>
  <line stroke="black" x1="0" y1="115" x2="0" y2="154"/>
  <line stroke="red" x1="0" y1="115" x2="192" y2="115" elevation="1"/>
  <line stroke="black" x1="0" y1="154" x2="0" y2="192"/>
  <line stroke="blue" x1="0" y1="154" x2="192" y2="154" elevation="0"/>
  <line stroke="black" x1="0" y1="192" x2="0" y2="231"/>
  <line stroke="red" x1="0" y1="192" x2="192" y2="192" elevation="1"/>
  <line stroke="blue" x1="192" y1="0" x2="192" y2="38"/>
  <line stroke="black" x1="192" y1="0" x2="385" y2="0"/>
  <line stroke="red" x1="192" y1="38" x2="192" y2="77"/>
  <line stroke="red" x1="192" y1="38" x2="346" y2="38" elevation="1"/>
  <line stroke="blue" x1="192" y1="77" x2="192" y2="115"/>
  <line stroke="blue" x1="192" y1="77" x2="308" y2="77" elevation="0"/>
  <line stroke="red" x1="192" y1="115" x2="192" y2="154"/>
  <line stroke="red" x1="192" y1="115" x2="269" y2="115" elevation="1"/>
  <line stroke="red" x1="0" y1="231" x2="38" y2="269"/>
  <line stroke="blue" x1="0" y1="231" x2="192" y2="231" elevation="0"/>
  <line stroke="black" x1="0" y1="231" x2="0" y2="615"/>
  <line stroke="blue" x1="192" y1="154" x2="192" y2="192"/>
  <line stroke="blue" x1="192" y1="154" x2="231" y2="154" elevation="0"/>
  <line stroke="red" x1="192" y1="192" x2="231" y2="154"/>
  <line stroke="red" x1="192" y1="192" x2="192" y2="231"/>
  <line stroke="blue" x1="192" y1="192" x2="231" y2="192"/>
  <line stroke="red" x1="192" y1="192" x2="231" y2="231"/>
  <line stroke="blue" x1="38" y1="269" x2="77" y2="308"/>
  <line stroke="red" x1="38" y1="269" x2="192" y2="269" elevation="1"/>
  <line stroke="red" x1="38" y1="269" x2="38" y2="577" elevation="1"/>
  <line stroke="blue" x1="231" y1="154" x2="269" y2="115"/>
  <line stroke="blue" x1="231" y1="154" x2="231" y2="192" elevation="0"/>
  <line stroke="red" x1="269" y1="115" x2="308" y2="77"/>
  <line stroke="red" x1="269" y1="115" x2="269" y2="192" elevation="1"/>
  <line stroke="blue" x1="192" y1="231" x2="231" y2="231" elevation="0"/>
  <line stroke="blue" x1="192" y1="231" x2="192" y2="269"/>
  <line stroke="blue" x1="231" y1="192" x2="231" y2="231" elevation="0"/>
  <line stroke="red" x1="231" y1="192" x2="269" y2="192"/>
  <line stroke="blue" x1="77" y1="308" x2="192" y2="308" elevation="0"/>
  <line stroke="red" x1="77" y1="308" x2="115" y2="346"/>
  <line stroke="blue" x1="77" y1="308" x2="77" y2="538" elevation="0"/>
  <line stroke="blue" x1="308" y1="77" x2="346" y2="38"/>
  <line stroke="blue" x1="308" y1="77" x2="308" y2="192" elevation="0"/>
  <line stroke="blue" x1="231" y1="231" x2="269" y2="269"/>
  <line stroke="red" x1="192" y1="269" x2="192" y2="308"/>
  <line stroke="red" x1="192" y1="269" x2="269" y2="269" elevation="1"/>
  <line stroke="blue" x1="269" y1="192" x2="308" y2="192"/>
  <line stroke="red" x1="269" y1="192" x2="269" y2="269" elevation="1"/>
  <line stroke="red" x1="346" y1="38" x2="385" y2="0"/>
  <line stroke="red" x1="346" y1="38" x2="346" y2="192" elevation="1"/>
  <line stroke="blue" x1="192" y1="308" x2="192" y2="346"/>
  <line stroke="blue" x1="192" y1="308" x2="308" y2="308" elevation="0"/>
  <line stroke="red" x1="308" y1="192" x2="346" y2="192"/>
  <line stroke="blue" x1="308" y1="192" x2="308" y2="308" elevation="0"/>
  <line stroke="red" x1="115" y1="346" x2="192" y2="346" elevation="1"/>
  <line stroke="blue" x1="115" y1="346" x2="154" y2="385"/>
  <line stroke="red" x1="115" y1="346" x2="115" y2="500" elevation="1"/>
  <line stroke="red" x1="269" y1="269" x2="308" y2="308"/>
  <line stroke="black" x1="385" y1="0" x2="423" y2="0"/>
  <line stroke="blue" x1="385" y1="0" x2="385" y2="192" elevation="0"/>
  <line stroke="red" x1="192" y1="346" x2="192" y2="385"/>
  <line stroke="red" x1="192" y1="346" x2="346" y2="346" elevation="1"/>
  <line stroke="blue" x1="346" y1="192" x2="385" y2="192"/>
  <line stroke="red" x1="346" y1="192" x2="346" y2="346" elevation="1"/>
  <line stroke="blue" x1="154" y1="385" x2="192" y2="385" elevation="0"/>
  <line stroke="red" x1="154" y1="385" x2="192" y2="423"/>
  <line stroke="blue" x1="154" y1="385" x2="154" y2="462" elevation="0"/>
  <line stroke="black" x1="423" y1="0" x2="462" y2="0"/>
  <line stroke="red" x1="423" y1="0" x2="423" y2="192" elevation="1"/>
  <line stroke="blue" x1="192" y1="385" x2="192" y2="423"/>
  <line stroke="blue" x1="192" y1="385" x2="385" y2="385" elevation="0"/>
  <line stroke="red" x1="385" y1="192" x2="423" y2="192"/>
  <line stroke="blue" x1="385" y1="192" x2="385" y2="385" elevation="0"/>
  <line stroke="blue" x1="308" y1="308" x2="346" y2="346"/>
  <line stroke="black" x1="462" y1="0" x2="500" y2="0"/>
  <line stroke="blue" x1="462" y1="0" x2="462" y2="192" elevation="0"/>
  <line stroke="red" x1="192" y1="423" x2="154" y2="462"/>
  <line stroke="red" x1="192" y1="423" x2="423" y2="423" elevation="1"/>
  <line stroke="blue" x1="423" y1="192" x2="462" y2="192"/>
  <line stroke="red" x1="423" y1="192" x2="423" y2="423" elevation="1"/>
  <line stroke="blue" x1="154" y1="462" x2="115" y2="500"/>
  <line stroke="blue" x1="154" y1="462" x2="462" y2="462" elevation="0"/>
  <line stroke="red" x1="346" y1="346" x2="385" y2="385"/>
  <line stroke="blue" x1="500" y1="0" x2="500" y2="192" elevation="1"/>
  <line stroke="black" x1="500" y1="0" x2="538" y2="0"/>
  <line stroke="red" x1="462" y1="192" x2="500" y2="192"/>
  <line stroke="blue" x1="462" y1="192" x2="462" y2="462" elevation="0"/>
  <line stroke="red" x1="115" y1="500" x2="77" y2="538"/>
  <line stroke="red" x1="115" y1="500" x2="500" y2="500" elevation="1"/>
  <line stroke="red" x1="500" y1="192" x2="538" y2="192"/>
  <line stroke="red" x1="500" y1="192" x2="500" y2="500" elevation="1"/>
  <line stroke="blue" x1="538" y1="0" x2="538" y2="192" elevation="0"/>
  <line stroke="black" x1="538" y1="0" x2="577" y2="0"/>
  <line stroke="blue" x1="77" y1="538" x2="38" y2="577"/>
  <line stroke="blue" x1="77" y1="538" x2="462" y2="538" elevation="0"/>
  <line stroke="blue" x1="385" y1="385" x2="423" y2="423"/>
  <line stroke="blue" x1="538" y1="192" x2="577" y2="192"/>
  <line stroke="blue" x1="538" y1="192" x2="538" y2="462" elevation="0"/>
  <line stroke="red" x1="577" y1="0" x2="577" y2="192" elevation="1"/>
  <line stroke="black" x1="577" y1="0" x2="615" y2="0"/>
  <line stroke="red" x1="38" y1="577" x2="0" y2="615"/>
  <line stroke="red" x1="38" y1="577" x2="423" y2="577" elevation="1"/>
  <line stroke="red" x1="423" y1="423" x2="462" y2="462"/>
  <line stroke="red" x1="577" y1="192" x2="615" y2="192"/>
  <line stroke="red" x1="577" y1="192" x2="577" y2="423" elevation="1"/>
  <line stroke="red" x1="0" y1="615" x2="38" y2="654"/>
  <line stroke="blue" x1="0" y1="615" x2="385" y2="615" elevation="0"/>
  <line stroke="black" x1="0" y1="615" x2="0" y2="808"/>
  <line stroke="blue" x1="615" y1="0" x2="615" y2="192" elevation="0"/>
  <line stroke="red" x1="615" y1="0" x2="654" y2="38"/>
  <line stroke="black" x1="615" y1="0" x2="1000" y2="0"/>
  <line stroke="blue" x1="615" y1="192" x2="654" y2="192"/>
  <line stroke="blue" x1="615" y1="192" x2="615" y2="385" elevation="0"/>
  <line stroke="blue" x1="462" y1="462" x2="500" y2="500"/>
  <line stroke="blue" x1="38" y1="654" x2="77" y2="692"/>
  <line stroke="red" x1="38" y1="654" x2="346" y2="654" elevation="1"/>
  <line stroke="red" x1="38" y1="654" x2="38" y2="808" elevation="1"/>
  <line stroke="red" x1="654" y1="38" x2="654" y2="192" elevation="1"/>
  <line stroke="blue" x1="654" y1="38" x2="692" y2="77"/>
  <line stroke="red" x1="654" y1="38" x2="1000" y2="38" elevation="1"/>
  <line stroke="red" x1="654" y1="192" x2="692" y2="192"/>
  <line stroke="red" x1="654" y1="192" x2="654" y2="346" elevation="1"/>
  <line stroke="red" x1="77" y1="692" x2="115" y2="731"/>
  <line stroke="blue" x1="77" y1="692" x2="308" y2="692" elevation="0"/>
  <line stroke="blue" x1="77" y1="692" x2="77" y2="808" elevation="0"/>
  <line stroke="blue" x1="692" y1="77" x2="692" y2="192" elevation="0"/>
  <line stroke="red" x1="692" y1="77" x2="731" y2="115"/>
  <line stroke="blue" x1="692" y1="77" x2="1000" y2="77" elevation="0"/>
  <line stroke="blue" x1="500" y1="500" x2="462" y2="538"/>
  <line stroke="blue" x1="500" y1="500" x2="538" y2="462"/>
  <line stroke="blue" x1="500" y1="500" x2="538" y2="538"/>
  <line stroke="blue" x1="500" y1="500" x2="500" y2="808" elevation="1"/>
  <line stroke="red" x1="500" y1="500" x2="808" y2="500" elevation="1"/>
  <line stroke="red" x1="462" y1="538" x2="423" y2="577"/>
  <line stroke="blue" x1="462" y1="538" x2="462" y2="808" elevation="0"/>
  <line stroke="red" x1="538" y1="462" x2="577" y2="423"/>
  <line stroke="blue" x1="538" y1="462" x2="808" y2="462" elevation="0"/>
  <line stroke="blue" x1="423" y1="577" x2="385" y2="615"/>
  <line stroke="red" x1="423" y1="577" x2="423" y2="808" elevation="1"/>
  <line stroke="blue" x1="577" y1="423" x2="615" y2="385"/>
  <line stroke="red" x1="577" y1="423" x2="808" y2="423" elevation="1"/>
  <line stroke="blue" x1="692" y1="192" x2="731" y2="192"/>
  <line stroke="blue" x1="692" y1="192" x2="692" y2="308" elevation="0"/>
  <line stroke="red" x1="385" y1="615" x2="346" y2="654"/>
  <line stroke="blue" x1="385" y1="615" x2="385" y2="808" elevation="0"/>
  <line stroke="red" x1="615" y1="385" x2="654" y2="346"/>
  <line stroke="blue" x1="615" y1="385" x2="846" y2="385" elevation="0"/>
  <line stroke="blue" x1="346" y1="654" x2="308" y2="692"/>
  <line stroke="red" x1="346" y1="654" x2="346" y2="808" elevation="1"/>
  <line stroke="blue" x1="654" y1="346" x2="692" y2="308"/>
  <line stroke="red" x1="654" y1="346" x2="885" y2="346" elevation="1"/>
  <line stroke="red" x1="115" y1="731" x2="269" y2="731" elevation="1"/>
  <line stroke="blue" x1="115" y1="731" x2="154" y2="769"/>
  <line stroke="red" x1="115" y1="731" x2="115" y2="808" elevation="1"/>
  <line stroke="red" x1="731" y1="115" x2="731" y2="192" elevation="1"/>
  <line stroke="blue" x1="731" y1="115" x2="769" y2="154"/>
  <line stroke="red" x1="731" y1="115" x2="1000" y2="115" elevation="1"/>
  <line stroke="red" x1="731" y1="192" x2="731" y2="269" elevation="1"/>
  <line stroke="red" x1="731" y1="192" x2="769" y2="192"/>
  <line stroke="red" x1="308" y1="692" x2="269" y2="731"/>
  <line stroke="blue" x1="308" y1="692" x2="308" y2="808" elevation="0"/>
  <line stroke="red" x1="692" y1="308" x2="731" y2="269"/>
  <line stroke="blue" x1="692" y1="308" x2="923" y2="308" elevation="0"/>
  <line stroke="red" x1="538" y1="538" x2="577" y2="577"/>
  <line stroke="blue" x1="538" y1="538" x2="538" y2="808" elevation="0"/>
  <line stroke="blue" x1="538" y1="538" x2="808" y2="538" elevation="0"/>
  <line stroke="blue" x1="269" y1="731" x2="231" y2="769"/>
  <line stroke="red" x1="269" y1="731" x2="269" y2="846" elevation="1"/>
  <line stroke="blue" x1="731" y1="269" x2="769" y2="231"/>
  <line stroke="red" x1="731" y1="269" x2="962" y2="269" elevation="1"/>
  <line stroke="blue" x1="154" y1="769" x2="231" y2="769" elevation="0"/>
  <line stroke="blue" x1="154" y1="769" x2="154" y2="808" elevation="0"/>
  <line stroke="red" x1="154" y1="769" x2="192" y2="808"/>
  <line stroke="blue" x1="769" y1="154" x2="769" y2="192" elevation="0"/>
  <line stroke="red" x1="769" y1="154" x2="808" y2="192"/>
  <line stroke="blue" x1="769" y1="154" x2="1000" y2="154" elevation="0"/>
  <line stroke="blue" x1="769" y1="192" x2="769" y2="231" elevation="0"/>
  <line stroke="blue" x1="769" y1="192" x2="808" y2="192"/>
  <line stroke="red" x1="231" y1="769" x2="192" y2="808"/>
  <line stroke="blue" x1="231" y1="769" x2="231" y2="885" elevation="0"/>
  <line stroke="red" x1="769" y1="231" x2="808" y2="192"/>
  <line stroke="blue" x1="769" y1="231" x2="1000" y2="231" elevation="0"/>
  <line stroke="blue" x1="0" y1="808" x2="38" y2="808"/>
  <line stroke="black" x1="0" y1="808" x2="0" y2="1000"/>
  <line stroke="red" x1="38" y1="808" x2="77" y2="808"/>
  <line stroke="red" x1="38" y1="808" x2="38" y2="1000" elevation="1"/>
  <line stroke="blue" x1="77" y1="808" x2="115" y2="808"/>
  <line stroke="blue" x1="77" y1="808" x2="77" y2="1000" elevation="0"/>
  <line stroke="blue" x1="577" y1="577" x2="615" y2="615"/>
  <line stroke="red" x1="577" y1="577" x2="577" y2="808" elevation="1"/>
  <line stroke="red" x1="577" y1="577" x2="808" y2="577" elevation="1"/>
  <line stroke="red" x1="115" y1="808" x2="154" y2="808"/>
  <line stroke="red" x1="115" y1="808" x2="115" y2="1000" elevation="1"/>
  <line stroke="blue" x1="154" y1="808" x2="192" y2="808"/>
  <line stroke="blue" x1="154" y1="808" x2="154" y2="962" elevation="0"/>
  <line stroke="red" x1="192" y1="808" x2="192" y2="923" elevation="1"/>
  <line stroke="red" x1="808" y1="192" x2="1000" y2="192" elevation="1"/>
  <line stroke="red" x1="308" y1="808" x2="346" y2="808"/>
  <line stroke="blue" x1="308" y1="808" x2="269" y2="846"/>
  <line stroke="blue" x1="308" y1="808" x2="338" y2="838"/>
  <line stroke="red" x1="615" y1="615" x2="654" y2="654"/>
  <line stroke="blue" x1="615" y1="615" x2="615" y2="808" elevation="0"/>
  <line stroke="blue" x1="615" y1="615" x2="808" y2="615" elevation="0"/>
  <line stroke="blue" x1="346" y1="808" x2="385" y2="808"/>
  <line stroke="blue" x1="346" y1="808" x2="338" y2="838" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="346" y1="808" x2="346" y2="846" elevation="1"/>
  <line stroke="red" x1="346" y1="808" x2="360" y2="860" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="269" y1="846" x2="308" y2="846" elevation="1"/>
  <line stroke="red" x1="269" y1="846" x2="231" y2="885"/>
  <line stroke="red" x1="385" y1="808" x2="423" y2="808"/>
  <line stroke="blue" x1="385" y1="808" x2="385" y2="885" elevation="0"/>
  <line stroke="red" x1="308" y1="846" x2="338" y2="838" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="308" y1="846" x2="346" y2="846" elevation="1"/>
  <line stroke="blue" x1="308" y1="846" x2="360" y2="860" opacity="0.5" fold-step="1"/>
  <line stroke="blue" x1="338" y1="838" x2="346" y2="846"/>
  <line stroke="blue" x1="423" y1="808" x2="462" y2="808"/>
  <line stroke="blue" x1="423" y1="808" x2="399" y2="899" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="423" y1="808" x2="423" y2="923" elevation="1"/>
  <line stroke="red" x1="423" y1="808" x2="462" y2="951" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="808" y1="423" x2="846" y2="385"/>
  <line stroke="blue" x1="808" y1="423" x2="808" y2="462"/>
  <line stroke="red" x1="808" y1="423" x2="846" y2="462"/>
  <line stroke="red" x1="346" y1="846" x2="360" y2="860"/>
  <line stroke="blue" x1="231" y1="885" x2="192" y2="923"/>
  <line stroke="blue" x1="231" y1="885" x2="385" y2="885" elevation="0"/>
  <line stroke="blue" x1="654" y1="654" x2="692" y2="692"/>
  <line stroke="red" x1="654" y1="654" x2="654" y2="808" elevation="1"/>
  <line stroke="red" x1="654" y1="654" x2="808" y2="654" elevation="1"/>
  <line stroke="blue" x1="846" y1="385" x2="885" y2="346"/>
  <line stroke="blue" x1="846" y1="385" x2="846" y2="462" elevation="0"/>
  <line stroke="red" x1="462" y1="808" x2="500" y2="808"/>
  <line stroke="blue" x1="462" y1="808" x2="462" y2="951" elevation="0"/>
  <line stroke="red" x1="808" y1="462" x2="808" y2="500"/>
  <line stroke="blue" x1="808" y1="462" x2="846" y2="462" elevation="0"/>
  <line stroke="red" x1="360" y1="860" x2="385" y2="885"/>
  <line stroke="red" x1="192" y1="923" x2="308" y2="923" elevation="1"/>
  <line stroke="red" x1="192" y1="923" x2="154" y2="962"/>
  <line stroke="red" x1="500" y1="808" x2="538" y2="808"/>
  <line stroke="blue" x1="500" y1="808" x2="462" y2="951" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="500" y1="808" x2="538" y2="951" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="500" y1="808" x2="500" y2="1000" elevation="1"/>
  <line stroke="blue" x1="808" y1="500" x2="808" y2="538"/>
  <line stroke="red" x1="808" y1="500" x2="885" y2="500" elevation="1"/>
  <line stroke="red" x1="885" y1="346" x2="923" y2="308"/>
  <line stroke="red" x1="885" y1="346" x2="885" y2="500" elevation="1"/>
  <line stroke="blue" x1="846" y1="462" x2="885" y2="500"/>
  <line stroke="blue" x1="385" y1="885" x2="399" y2="899"/>
  <line stroke="blue" x1="538" y1="808" x2="577" y2="808"/>
  <line stroke="blue" x1="538" y1="808" x2="538" y2="951" elevation="0"/>
  <line stroke="red" x1="808" y1="538" x2="808" y2="577"/>
  <line stroke="blue" x1="808" y1="538" x2="923" y2="538" elevation="0"/>
  <line stroke="red" x1="308" y1="923" x2="399" y2="899" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="308" y1="923" x2="423" y2="923" elevation="1"/>
  <line stroke="blue" x1="308" y1="923" x2="451" y2="962" opacity="0.5" fold-step="1"/>
  <line stroke="blue" x1="923" y1="308" x2="962" y2="269"/>
  <line stroke="blue" x1="923" y1="308" x2="923" y2="538" elevation="0"/>
  <line stroke="blue" x1="154" y1="962" x2="115" y2="1000"/>
  <line stroke="blue" x1="154" y1="962" x2="451" y2="962" elevation="0"/>
  <line stroke="red" x1="692" y1="692" x2="731" y2="731"/>
  <line stroke="blue" x1="692" y1="692" x2="692" y2="808" elevation="0"/>
  <line stroke="blue" x1="692" y1="692" x2="808" y2="692" elevation="0"/>
  <line stroke="blue" x1="399" y1="899" x2="423" y2="923"/>
  <line stroke="red" x1="577" y1="808" x2="615" y2="808"/>
  <line stroke="red" x1="577" y1="808" x2="601" y2="899" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="577" y1="808" x2="577" y2="923" elevation="1"/>
  <line stroke="blue" x1="577" y1="808" x2="538" y2="951" opacity="0.5" fold-step="1"/>
  <line stroke="blue" x1="808" y1="577" x2="808" y2="615"/>
  <line stroke="red" x1="808" y1="577" x2="962" y2="577" elevation="1"/>
  <line stroke="red" x1="962" y1="269" x2="1000" y2="231"/>
  <line stroke="red" x1="962" y1="269" x2="962" y2="577" elevation="1"/>
  <line stroke="black" x1="0" y1="1000" x2="38" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="38"/>
  <line stroke="black" x1="38" y1="1000" x2="77" y2="1000"/>
  <line stroke="black" x1="1000" y1="38" x2="1000" y2="77"/>
  <line stroke="black" x1="77" y1="1000" x2="115" y2="1000"/>
  <line stroke="black" x1="1000" y1="77" x2="1000" y2="115"/>
  <line stroke="black" x1="115" y1="1000" x2="308" y2="1000"/>
  <line stroke="black" x1="1000" y1="115" x2="1000" y2="154"/>
  <line stroke="black" x1="1000" y1="154" x2="1000" y2="192"/>
  <line stroke="red" x1="423" y1="923" x2="462" y2="962"/>
  <line stroke="blue" x1="615" y1="808" x2="654" y2="808"/>
  <line stroke="blue" x1="615" y1="808" x2="615" y2="885" elevation="0"/>
  <line stroke="blue" x1="808" y1="615" x2="808" y2="654"/>
  <line stroke="red" x1="808" y1="615" x2="1000" y2="615" elevation="0"/>
  <line stroke="red" x1="885" y1="500" x2="923" y2="538"/>
  <line stroke="black" x1="1000" y1="192" x2="1000" y2="231"/>
  <line stroke="black" x1="1000" y1="231" x2="1000" y2="615"/>
  <line stroke="blue" x1="731" y1="731" x2="769" y2="769"/>
  <line stroke="red" x1="731" y1="731" x2="808" y2="731" elevation="1"/>
  <line stroke="red" x1="731" y1="731" x2="731" y2="846" elevation="1"/>
  <line stroke="red" x1="654" y1="808" x2="692" y2="808"/>
  <line stroke="red" x1="654" y1="808" x2="662" y2="838" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="654" y1="808" x2="654" y2="846" elevation="1"/>
  <line stroke="blue" x1="654" y1="808" x2="640" y2="860" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="808" y1="654" x2="808" y2="692"/>
  <line stroke="red" x1="808" y1="654" x2="962" y2="654" elevation="1"/>
  <line stroke="red" x1="308" y1="1000" x2="451" y2="962" opacity="0.5" fold-step="1"/>
  <line stroke="black" x1="308" y1="1000" x2="500" y2="1000"/>
  <line stroke="blue" x1="462" y1="951" x2="462" y2="962" elevation="0"/>
  <line stroke="blue" x1="451" y1="962" x2="462" y2="962" elevation="0"/>
  <line stroke="blue" x1="692" y1="808" x2="662" y2="838"/>
  <line stroke="blue" x1="692" y1="808" x2="731" y2="846"/>
  <line stroke="blue" x1="808" y1="692" x2="808" y2="731"/>
  <line stroke="blue" x1="808" y1="692" x2="923" y2="692" elevation="0"/>
  <line stroke="blue" x1="462" y1="962" x2="500" y2="1000"/>
  <line stroke="blue" x1="662" y1="838" x2="654" y2="846"/>
  <line stroke="blue" x1="662" y1="838" x2="692" y2="846" opacity="0.5" fold-step="1"/>
  <line stroke="blue" x1="923" y1="538" x2="962" y2="577"/>
  <line stroke="red" x1="654" y1="846" x2="640" y2="860"/>
  <line stroke="red" x1="654" y1="846" x2="692" y2="846" elevation="1"/>
  <line stroke="red" x1="640" y1="860" x2="615" y2="885"/>
  <line stroke="red" x1="640" y1="860" x2="692" y2="846" opacity="0.5" fold-step="1"/>
  <line stroke="blue" x1="615" y1="885" x2="601" y2="899"/>
  <line stroke="blue" x1="615" y1="885" x2="769" y2="885" elevation="0"/>
  <line stroke="blue" x1="601" y1="899" x2="577" y2="923"/>
  <line stroke="blue" x1="601" y1="899" x2="692" y2="923" opacity="0.5" fold-step="1"/>
  <line stroke="blue" x1="769" y1="769" x2="808" y2="769" elevation="0"/>
  <line stroke="red" x1="769" y1="769" x2="808" y2="808"/>
  <line stroke="blue" x1="769" y1="769" x2="769" y2="885" elevation="0"/>
  <line stroke="red" x1="577" y1="923" x2="538" y2="962"/>
  <line stroke="red" x1="577" y1="923" x2="692" y2="923" elevation="1"/>
  <line stroke="red" x1="808" y1="731" x2="808" y2="769"/>
  <line stroke="red" x1="808" y1="731" x2="885" y2="731" elevation="1"/>
  <line stroke="blue" x1="538" y1="951" x2="538" y2="962" elevation="0"/>
  <line stroke="red" x1="692" y1="846" x2="731" y2="846" elevation="1"/>
  <line stroke="blue" x1="538" y1="962" x2="549" y2="962" elevation="0"/>
  <line stroke="blue" x1="538" y1="962" x2="500" y2="1000"/>
  <line stroke="red" x1="549" y1="962" x2="692" y2="923" opacity="0.5" fold-step="1"/>
  <line stroke="blue" x1="549" y1="962" x2="692" y2="1000" opacity="0.5" fold-step="1"/>
  <line stroke="blue" x1="549" y1="962" x2="846" y2="962" elevation="0"/>
  <line stroke="blue" x1="808" y1="769" x2="808" y2="808"/>
  <line stroke="blue" x1="808" y1="769" x2="846" y2="769" elevation="0"/>
  <line stroke="black" x1="500" y1="1000" x2="692" y2="1000"/>
  <line stroke="red" x1="731" y1="846" x2="769" y2="885"/>
  <line stroke="red" x1="962" y1="577" x2="1000" y2="615"/>
  <line stroke="red" x1="808" y1="808" x2="846" y2="769"/>
  <line stroke="red" x1="808" y1="808" x2="808" y2="923" elevation="1"/>
  <line stroke="blue" x1="846" y1="769" x2="885" y2="731"/>
  <line stroke="blue" x1="846" y1="769" x2="846" y2="962" elevation="0"/>
  <line stroke="red" x1="885" y1="731" x2="923" y2="692"/>
  <line stroke="red" x1="885" y1="731" x2="885" y2="1000" elevation="1"/>
  <line stroke="red" x1="692" y1="923" x2="808" y2="923" elevation="1"/>
  <line stroke="blue" x1="923" y1="692" x2="962" y2="654"/>
  <line stroke="blue" x1="923" y1="692" x2="923" y2="1000" elevation="0"/>
  <line stroke="red" x1="962" y1="654" x2="1000" y2="615"/>
  <line stroke="red" x1="962" y1="654" x2="962" y2="1000" elevation="1"/>
  <line stroke="blue" x1="769" y1="885" x2="808" y2="923"/>
  <line stroke="black" x1="1000" y1="615" x2="1000" y2="1000"/>
  <line stroke="black" x1="692" y1="1000" x2="885" y2="1000"/>
  <line stroke="red" x1="808" y1="923" x2="846" y2="962"/>
  <line stroke="blue" x1="846" y1="962" x2="885" y2="1000"/>
  <line stroke="black" x1="885" y1="1000" x2="923" y2="1000"/>
  <line stroke="black" x1="923" y1="1000" x2="962" y2="1000"/>
  <line stroke="black" x1="962" y1="1000" x2="1000" y2="1000"/>
</svg>""",
        zigzag_creases=[
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.5, y=0.8076923076923077, require_on_paper=True
                ),
                p2_init=point_utils.Point(
                    x=0.5384615384615384,
                    y=0.9512327233680338,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.MOUNTAIN,
                fold_percent=0.5,
                fold_step=1,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.5384615384615384,
                    y=0.9512327233680338,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.5769230769230769,
                    y=0.8076923076923077,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.VALLEY,
                fold_percent=0.5,
                fold_step=1,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.5769230769230769,
                    y=0.8076923076923077,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.60130671523906,
                    y=0.89869328476094,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.MOUNTAIN,
                fold_percent=0.5,
                fold_step=1,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.60130671523906,
                    y=0.89869328476094,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.6923076923076923,
                    y=0.9230769230769231,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.VALLEY,
                fold_percent=0.5,
                fold_step=1,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.6923076923076923,
                    y=0.9230769230769231,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.5487672766319662,
                    y=0.9615384615384616,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.MOUNTAIN,
                fold_percent=0.5,
                fold_step=1,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.5487672766319662,
                    y=0.9615384615384616,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.6923076923076923, y=1.0, require_on_paper=True
                ),
                fold=crease_utils.Fold.VALLEY,
                fold_percent=0.5,
                fold_step=1,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.30769230769230793,
                    y=0.9999999999999994,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.451232723368034,
                    y=0.961538461538461,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.MOUNTAIN,
                fold_percent=0.5,
                fold_step=1,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.451232723368034,
                    y=0.961538461538461,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.30769230769230793,
                    y=0.9230769230769226,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.VALLEY,
                fold_percent=0.5,
                fold_step=1,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.30769230769230793,
                    y=0.9230769230769226,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.39869328476093985,
                    y=0.8986932847609397,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.MOUNTAIN,
                fold_percent=0.5,
                fold_step=1,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.39869328476093985,
                    y=0.8986932847609397,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.4230769230769228,
                    y=0.8076923076923078,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.VALLEY,
                fold_percent=0.5,
                fold_step=1,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.4230769230769228,
                    y=0.8076923076923078,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.46153846153846123,
                    y=0.9512327233680339,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.MOUNTAIN,
                fold_percent=0.5,
                fold_step=1,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.46153846153846123,
                    y=0.9512327233680339,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.49999999999999967,
                    y=0.8076923076923078,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.VALLEY,
                fold_percent=0.5,
                fold_step=1,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.6538461538461539,
                    y=0.8076923076923077,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.6619740332848149,
                    y=0.8380259667151851,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.MOUNTAIN,
                fold_percent=0.5,
                fold_step=1,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.6619740332848149,
                    y=0.8380259667151851,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.6923076923076923,
                    y=0.8461538461538461,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.VALLEY,
                fold_percent=0.5,
                fold_step=1,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.6923076923076923,
                    y=0.8461538461538461,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.6397682537005992,
                    y=0.8602317462994009,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.MOUNTAIN,
                fold_percent=0.5,
                fold_step=1,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.6397682537005992,
                    y=0.8602317462994009,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.653846153846154,
                    y=0.8076923076923078,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.VALLEY,
                fold_percent=0.5,
                fold_step=1,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.3076923076923077,
                    y=0.8461538461538461,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.3380259667151851,
                    y=0.8380259667151849,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.MOUNTAIN,
                fold_percent=0.5,
                fold_step=1,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.3380259667151851,
                    y=0.8380259667151849,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.3461538461538463,
                    y=0.8076923076923075,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.VALLEY,
                fold_percent=0.5,
                fold_step=1,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.3461538461538463,
                    y=0.8076923076923075,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.3602317462994017,
                    y=0.8602317462994015,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.MOUNTAIN,
                fold_percent=0.5,
                fold_step=1,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.3602317462994017,
                    y=0.8602317462994015,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.3076923076923077,
                    y=0.8461538461538461,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.VALLEY,
                fold_percent=0.5,
                fold_step=1,
                elevation=-1,
            ),
        ],
    ),
    rectangles=[
        shaping_utils.Rectangle(
            name="trunk",
            x_start=0.1923076923076923,
            y_start=0.1923076923076923,
            x_len=0.6153846153846154,
            y_len=0.6153846153846154,
            circle_centers=None,
            require_on_paper=False,
            radius=0.435142634576337,
        ),
        shaping_utils.Rectangle(
            name="frond_1a",
            x_start=0.3076923076923077,
            y_start=0.8076923076923077,
            x_len=0.3846153846153846,
            y_len=0.3846153846153846,
            circle_centers=None,
            require_on_paper=False,
            radius=0.27196414661021057,
        ),
        shaping_utils.Rectangle(
            name="frond_1b",
            x_start=0.38461538461538464,
            y_start=0.8846153846153846,
            x_len=0.23076923076923078,
            y_len=0.23076923076923084,
            circle_centers=None,
            require_on_paper=False,
            radius=0.16317848796612638,
        ),
        shaping_utils.Rectangle(
            name="frond_2a",
            x_start=-0.2692307692307693,
            y_start=0.8076923076923077,
            x_len=0.576923076923077,
            y_len=0.3846153846153846,
            circle_centers=None,
            require_on_paper=False,
            radius=0.27196414661021057,
        ),
        shaping_utils.Rectangle(
            name="frond_2b",
            x_start=-0.1923076923076923,
            y_start=0.8846153846153846,
            x_len=0.423076923076923,
            y_len=0.23076923076923084,
            circle_centers=None,
            require_on_paper=False,
            radius=0.1631784879661264,
        ),
        shaping_utils.Rectangle(
            name="frond_3a",
            x_start=-0.19230769230769232,
            y_start=0.42307692307692313,
            x_len=0.38461538461538464,
            y_len=0.3846153846153846,
            circle_centers=None,
            require_on_paper=False,
            radius=0.27196414661021057,
        ),
        shaping_utils.Rectangle(
            name="frond_3b",
            x_start=-0.11538461538461536,
            y_start=0.5,
            x_len=0.23076923076923073,
            y_len=0.23076923076923084,
            circle_centers=None,
            require_on_paper=False,
            radius=0.16317848796612633,
        ),
        shaping_utils.Rectangle(
            name="frond_4a",
            x_start=-0.19230769230769232,
            y_start=-0.23076923076923078,
            x_len=0.38461538461538464,
            y_len=0.653846153846154,
            circle_centers=None,
            require_on_paper=False,
            radius=0.2719641466102106,
        ),
        shaping_utils.Rectangle(
            name="frond_4b",
            x_start=-0.11538461538461539,
            y_start=-0.15384615384615385,
            x_len=0.23076923076923078,
            y_len=0.5,
            circle_centers=None,
            require_on_paper=False,
            radius=0.16317848796612638,
        ),
        shaping_utils.Rectangle(
            name="frond_5a",
            x_start=0.19230769230769232,
            y_start=-0.19230769230769232,
            x_len=0.6153846153846154,
            y_len=0.38461538461538464,
            circle_centers=None,
            require_on_paper=False,
            radius=0.2719641466102106,
        ),
        shaping_utils.Rectangle(
            name="frond_5b",
            x_start=0.2692307692307693,
            y_start=-0.11538461538461539,
            x_len=0.46153846153846156,
            y_len=0.23076923076923078,
            circle_centers=None,
            require_on_paper=False,
            radius=0.16317848796612638,
        ),
        shaping_utils.Rectangle(
            name="frond_8a",
            x_start=0.6923076923076923,
            y_start=0.8076923076923077,
            x_len=0.5769230769230769,
            y_len=0.3846153846153846,
            circle_centers=None,
            require_on_paper=False,
            radius=0.27196414661021057,
        ),
        shaping_utils.Rectangle(
            name="frond_8b",
            x_start=0.7692307692307692,
            y_start=0.8846153846153846,
            x_len=0.42307692307692313,
            y_len=0.23076923076923084,
            circle_centers=None,
            require_on_paper=False,
            radius=0.1631784879661264,
        ),
        shaping_utils.Rectangle(
            name="frond_7a",
            x_start=0.8076923076923077,
            y_start=0.42307692307692313,
            x_len=0.3846153846153846,
            y_len=0.3846153846153846,
            circle_centers=None,
            require_on_paper=False,
            radius=0.27196414661021057,
        ),
        shaping_utils.Rectangle(
            name="frond_7b",
            x_start=0.8846153846153846,
            y_start=0.5,
            x_len=0.23076923076923084,
            y_len=0.23076923076923084,
            circle_centers=None,
            require_on_paper=False,
            radius=0.1631784879661264,
        ),
        shaping_utils.Rectangle(
            name="frond_6a",
            x_start=0.8076923076923077,
            y_start=-0.23076923076923078,
            x_len=0.3846153846153846,
            y_len=0.653846153846154,
            circle_centers=None,
            require_on_paper=False,
            radius=0.27196414661021057,
        ),
        shaping_utils.Rectangle(
            name="frond_6b",
            x_start=0.8846153846153846,
            y_start=-0.15384615384615383,
            x_len=0.23076923076923084,
            y_len=0.5,
            circle_centers=None,
            require_on_paper=False,
            radius=0.1631784879661264,
        ),
    ],
    narrowed_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="0" y2="38"/>
  <line stroke="black" x1="0" y1="0" x2="192" y2="0"/>
  <line stroke="black" x1="0" y1="38" x2="0" y2="77"/>
  <line stroke="red" x1="0" y1="38" x2="192" y2="38" elevation="1"/>
  <line stroke="black" x1="0" y1="77" x2="0" y2="115"/>
  <line stroke="blue" x1="0" y1="77" x2="192" y2="77" elevation="0"/>
  <line stroke="black" x1="0" y1="115" x2="0" y2="154"/>
  <line stroke="red" x1="0" y1="115" x2="192" y2="115" elevation="1"/>
  <line stroke="black" x1="0" y1="154" x2="0" y2="192"/>
  <line stroke="blue" x1="0" y1="154" x2="192" y2="154" elevation="0"/>
  <line stroke="black" x1="0" y1="192" x2="0" y2="231"/>
  <line stroke="red" x1="0" y1="192" x2="192" y2="192" elevation="1"/>
  <line stroke="blue" x1="192" y1="0" x2="192" y2="38"/>
  <line stroke="black" x1="192" y1="0" x2="385" y2="0"/>
  <line stroke="red" x1="192" y1="38" x2="192" y2="77"/>
  <line stroke="red" x1="192" y1="38" x2="346" y2="38" elevation="1"/>
  <line stroke="blue" x1="192" y1="77" x2="192" y2="115"/>
  <line stroke="blue" x1="192" y1="77" x2="308" y2="77" elevation="0"/>
  <line stroke="red" x1="192" y1="115" x2="192" y2="154"/>
  <line stroke="red" x1="192" y1="115" x2="269" y2="115" elevation="1"/>
  <line stroke="red" x1="0" y1="231" x2="38" y2="269"/>
  <line stroke="blue" x1="0" y1="231" x2="192" y2="231" elevation="0"/>
  <line stroke="black" x1="0" y1="231" x2="0" y2="615"/>
  <line stroke="blue" x1="192" y1="154" x2="192" y2="192"/>
  <line stroke="blue" x1="192" y1="154" x2="231" y2="154" elevation="0"/>
  <line stroke="red" x1="192" y1="192" x2="231" y2="154"/>
  <line stroke="red" x1="192" y1="192" x2="192" y2="231"/>
  <line stroke="blue" x1="192" y1="192" x2="231" y2="192"/>
  <line stroke="red" x1="192" y1="192" x2="231" y2="231"/>
  <line stroke="blue" x1="38" y1="269" x2="77" y2="308"/>
  <line stroke="red" x1="38" y1="269" x2="192" y2="269" elevation="1"/>
  <line stroke="red" x1="38" y1="269" x2="38" y2="577" elevation="1"/>
  <line stroke="blue" x1="231" y1="154" x2="269" y2="115"/>
  <line stroke="blue" x1="231" y1="154" x2="231" y2="192" elevation="0"/>
  <line stroke="red" x1="269" y1="115" x2="308" y2="77"/>
  <line stroke="red" x1="269" y1="115" x2="269" y2="192" elevation="1"/>
  <line stroke="blue" x1="192" y1="231" x2="231" y2="231" elevation="0"/>
  <line stroke="blue" x1="192" y1="231" x2="192" y2="269"/>
  <line stroke="blue" x1="231" y1="192" x2="231" y2="231" elevation="0"/>
  <line stroke="red" x1="231" y1="192" x2="269" y2="192"/>
  <line stroke="blue" x1="77" y1="308" x2="192" y2="308" elevation="0"/>
  <line stroke="red" x1="77" y1="308" x2="115" y2="346"/>
  <line stroke="blue" x1="77" y1="308" x2="77" y2="538" elevation="0"/>
  <line stroke="blue" x1="308" y1="77" x2="346" y2="38"/>
  <line stroke="blue" x1="308" y1="77" x2="308" y2="192" elevation="0"/>
  <line stroke="blue" x1="231" y1="231" x2="269" y2="269"/>
  <line stroke="red" x1="192" y1="269" x2="192" y2="308"/>
  <line stroke="red" x1="192" y1="269" x2="269" y2="269" elevation="1"/>
  <line stroke="blue" x1="269" y1="192" x2="308" y2="192"/>
  <line stroke="red" x1="269" y1="192" x2="269" y2="269" elevation="1"/>
  <line stroke="red" x1="346" y1="38" x2="385" y2="0"/>
  <line stroke="red" x1="346" y1="38" x2="346" y2="192" elevation="1"/>
  <line stroke="blue" x1="192" y1="308" x2="192" y2="346"/>
  <line stroke="blue" x1="192" y1="308" x2="308" y2="308" elevation="0"/>
  <line stroke="red" x1="308" y1="192" x2="346" y2="192"/>
  <line stroke="blue" x1="308" y1="192" x2="308" y2="308" elevation="0"/>
  <line stroke="red" x1="115" y1="346" x2="192" y2="346" elevation="1"/>
  <line stroke="blue" x1="115" y1="346" x2="154" y2="385"/>
  <line stroke="red" x1="115" y1="346" x2="115" y2="500" elevation="1"/>
  <line stroke="red" x1="269" y1="269" x2="308" y2="308"/>
  <line stroke="black" x1="385" y1="0" x2="423" y2="0"/>
  <line stroke="blue" x1="385" y1="0" x2="385" y2="192" elevation="0"/>
  <line stroke="red" x1="192" y1="346" x2="192" y2="385"/>
  <line stroke="red" x1="192" y1="346" x2="346" y2="346" elevation="1"/>
  <line stroke="blue" x1="346" y1="192" x2="385" y2="192"/>
  <line stroke="red" x1="346" y1="192" x2="346" y2="346" elevation="1"/>
  <line stroke="blue" x1="154" y1="385" x2="192" y2="385" elevation="0"/>
  <line stroke="red" x1="154" y1="385" x2="192" y2="423"/>
  <line stroke="blue" x1="154" y1="385" x2="154" y2="462" elevation="0"/>
  <line stroke="black" x1="423" y1="0" x2="462" y2="0"/>
  <line stroke="red" x1="423" y1="0" x2="423" y2="192" elevation="1"/>
  <line stroke="blue" x1="192" y1="385" x2="192" y2="423"/>
  <line stroke="blue" x1="192" y1="385" x2="385" y2="385" elevation="0"/>
  <line stroke="red" x1="385" y1="192" x2="423" y2="192"/>
  <line stroke="blue" x1="385" y1="192" x2="385" y2="385" elevation="0"/>
  <line stroke="blue" x1="308" y1="308" x2="346" y2="346"/>
  <line stroke="black" x1="462" y1="0" x2="500" y2="0"/>
  <line stroke="blue" x1="462" y1="0" x2="462" y2="192" elevation="0"/>
  <line stroke="red" x1="192" y1="423" x2="154" y2="462"/>
  <line stroke="red" x1="192" y1="423" x2="423" y2="423" elevation="1"/>
  <line stroke="blue" x1="423" y1="192" x2="462" y2="192"/>
  <line stroke="red" x1="423" y1="192" x2="423" y2="423" elevation="1"/>
  <line stroke="blue" x1="154" y1="462" x2="115" y2="500"/>
  <line stroke="blue" x1="154" y1="462" x2="462" y2="462" elevation="0"/>
  <line stroke="red" x1="346" y1="346" x2="385" y2="385"/>
  <line stroke="blue" x1="500" y1="0" x2="500" y2="192" elevation="1"/>
  <line stroke="black" x1="500" y1="0" x2="538" y2="0"/>
  <line stroke="red" x1="462" y1="192" x2="500" y2="192"/>
  <line stroke="blue" x1="462" y1="192" x2="462" y2="462" elevation="0"/>
  <line stroke="red" x1="115" y1="500" x2="77" y2="538"/>
  <line stroke="red" x1="115" y1="500" x2="500" y2="500" elevation="1"/>
  <line stroke="red" x1="500" y1="192" x2="538" y2="192"/>
  <line stroke="red" x1="500" y1="192" x2="500" y2="500" elevation="1"/>
  <line stroke="blue" x1="538" y1="0" x2="538" y2="192" elevation="0"/>
  <line stroke="black" x1="538" y1="0" x2="577" y2="0"/>
  <line stroke="blue" x1="77" y1="538" x2="38" y2="577"/>
  <line stroke="blue" x1="77" y1="538" x2="462" y2="538" elevation="0"/>
  <line stroke="blue" x1="385" y1="385" x2="423" y2="423"/>
  <line stroke="blue" x1="538" y1="192" x2="577" y2="192"/>
  <line stroke="blue" x1="538" y1="192" x2="538" y2="462" elevation="0"/>
  <line stroke="red" x1="577" y1="0" x2="577" y2="192" elevation="1"/>
  <line stroke="black" x1="577" y1="0" x2="615" y2="0"/>
  <line stroke="red" x1="38" y1="577" x2="0" y2="615"/>
  <line stroke="red" x1="38" y1="577" x2="423" y2="577" elevation="1"/>
  <line stroke="red" x1="423" y1="423" x2="462" y2="462"/>
  <line stroke="red" x1="577" y1="192" x2="615" y2="192"/>
  <line stroke="red" x1="577" y1="192" x2="577" y2="423" elevation="1"/>
  <line stroke="red" x1="0" y1="615" x2="38" y2="654"/>
  <line stroke="blue" x1="0" y1="615" x2="385" y2="615" elevation="0"/>
  <line stroke="black" x1="0" y1="615" x2="0" y2="808"/>
  <line stroke="blue" x1="615" y1="0" x2="615" y2="192" elevation="0"/>
  <line stroke="red" x1="615" y1="0" x2="654" y2="38"/>
  <line stroke="black" x1="615" y1="0" x2="1000" y2="0"/>
  <line stroke="blue" x1="615" y1="192" x2="654" y2="192"/>
  <line stroke="blue" x1="615" y1="192" x2="615" y2="385" elevation="0"/>
  <line stroke="blue" x1="462" y1="462" x2="500" y2="500"/>
  <line stroke="blue" x1="38" y1="654" x2="77" y2="692"/>
  <line stroke="red" x1="38" y1="654" x2="346" y2="654" elevation="1"/>
  <line stroke="red" x1="38" y1="654" x2="38" y2="808" elevation="1"/>
  <line stroke="red" x1="654" y1="38" x2="654" y2="192" elevation="1"/>
  <line stroke="blue" x1="654" y1="38" x2="692" y2="77"/>
  <line stroke="red" x1="654" y1="38" x2="1000" y2="38" elevation="1"/>
  <line stroke="red" x1="654" y1="192" x2="692" y2="192"/>
  <line stroke="red" x1="654" y1="192" x2="654" y2="346" elevation="1"/>
  <line stroke="red" x1="77" y1="692" x2="115" y2="731"/>
  <line stroke="blue" x1="77" y1="692" x2="308" y2="692" elevation="0"/>
  <line stroke="blue" x1="77" y1="692" x2="77" y2="808" elevation="0"/>
  <line stroke="blue" x1="692" y1="77" x2="692" y2="192" elevation="0"/>
  <line stroke="red" x1="692" y1="77" x2="731" y2="115"/>
  <line stroke="blue" x1="692" y1="77" x2="1000" y2="77" elevation="0"/>
  <line stroke="blue" x1="500" y1="500" x2="462" y2="538"/>
  <line stroke="blue" x1="500" y1="500" x2="538" y2="462"/>
  <line stroke="blue" x1="500" y1="500" x2="538" y2="538"/>
  <line stroke="blue" x1="500" y1="500" x2="500" y2="808" elevation="1"/>
  <line stroke="red" x1="500" y1="500" x2="808" y2="500" elevation="1"/>
  <line stroke="red" x1="462" y1="538" x2="423" y2="577"/>
  <line stroke="blue" x1="462" y1="538" x2="462" y2="808" elevation="0"/>
  <line stroke="red" x1="538" y1="462" x2="577" y2="423"/>
  <line stroke="blue" x1="538" y1="462" x2="808" y2="462" elevation="0"/>
  <line stroke="blue" x1="423" y1="577" x2="385" y2="615"/>
  <line stroke="red" x1="423" y1="577" x2="423" y2="808" elevation="1"/>
  <line stroke="blue" x1="577" y1="423" x2="615" y2="385"/>
  <line stroke="red" x1="577" y1="423" x2="808" y2="423" elevation="1"/>
  <line stroke="blue" x1="692" y1="192" x2="731" y2="192"/>
  <line stroke="blue" x1="692" y1="192" x2="692" y2="308" elevation="0"/>
  <line stroke="red" x1="385" y1="615" x2="346" y2="654"/>
  <line stroke="blue" x1="385" y1="615" x2="385" y2="808" elevation="0"/>
  <line stroke="red" x1="615" y1="385" x2="654" y2="346"/>
  <line stroke="blue" x1="615" y1="385" x2="846" y2="385" elevation="0"/>
  <line stroke="blue" x1="346" y1="654" x2="308" y2="692"/>
  <line stroke="red" x1="346" y1="654" x2="346" y2="808" elevation="1"/>
  <line stroke="blue" x1="654" y1="346" x2="692" y2="308"/>
  <line stroke="red" x1="654" y1="346" x2="885" y2="346" elevation="1"/>
  <line stroke="red" x1="115" y1="731" x2="269" y2="731" elevation="1"/>
  <line stroke="blue" x1="115" y1="731" x2="154" y2="769"/>
  <line stroke="red" x1="115" y1="731" x2="115" y2="808" elevation="1"/>
  <line stroke="red" x1="731" y1="115" x2="731" y2="192" elevation="1"/>
  <line stroke="blue" x1="731" y1="115" x2="769" y2="154"/>
  <line stroke="red" x1="731" y1="115" x2="1000" y2="115" elevation="1"/>
  <line stroke="red" x1="731" y1="192" x2="731" y2="269" elevation="1"/>
  <line stroke="red" x1="731" y1="192" x2="769" y2="192"/>
  <line stroke="red" x1="308" y1="692" x2="269" y2="731"/>
  <line stroke="blue" x1="308" y1="692" x2="308" y2="808" elevation="0"/>
  <line stroke="red" x1="692" y1="308" x2="731" y2="269"/>
  <line stroke="blue" x1="692" y1="308" x2="923" y2="308" elevation="0"/>
  <line stroke="red" x1="538" y1="538" x2="577" y2="577"/>
  <line stroke="blue" x1="538" y1="538" x2="538" y2="808" elevation="0"/>
  <line stroke="blue" x1="538" y1="538" x2="808" y2="538" elevation="0"/>
  <line stroke="blue" x1="269" y1="731" x2="231" y2="769"/>
  <line stroke="red" x1="269" y1="731" x2="269" y2="846" elevation="1"/>
  <line stroke="blue" x1="731" y1="269" x2="769" y2="231"/>
  <line stroke="red" x1="731" y1="269" x2="962" y2="269" elevation="1"/>
  <line stroke="blue" x1="154" y1="769" x2="231" y2="769" elevation="0"/>
  <line stroke="blue" x1="154" y1="769" x2="154" y2="808" elevation="0"/>
  <line stroke="red" x1="154" y1="769" x2="192" y2="808"/>
  <line stroke="blue" x1="769" y1="154" x2="769" y2="192" elevation="0"/>
  <line stroke="red" x1="769" y1="154" x2="808" y2="192"/>
  <line stroke="blue" x1="769" y1="154" x2="1000" y2="154" elevation="0"/>
  <line stroke="blue" x1="769" y1="192" x2="769" y2="231" elevation="0"/>
  <line stroke="blue" x1="769" y1="192" x2="808" y2="192"/>
  <line stroke="red" x1="231" y1="769" x2="192" y2="808"/>
  <line stroke="blue" x1="231" y1="769" x2="231" y2="885" elevation="0"/>
  <line stroke="red" x1="769" y1="231" x2="808" y2="192"/>
  <line stroke="blue" x1="769" y1="231" x2="1000" y2="231" elevation="0"/>
  <line stroke="blue" x1="0" y1="808" x2="38" y2="808"/>
  <line stroke="black" x1="0" y1="808" x2="0" y2="1000"/>
  <line stroke="red" x1="38" y1="808" x2="77" y2="808"/>
  <line stroke="red" x1="38" y1="808" x2="38" y2="1000" elevation="1"/>
  <line stroke="blue" x1="77" y1="808" x2="115" y2="808"/>
  <line stroke="blue" x1="77" y1="808" x2="77" y2="1000" elevation="0"/>
  <line stroke="blue" x1="577" y1="577" x2="615" y2="615"/>
  <line stroke="red" x1="577" y1="577" x2="577" y2="808" elevation="1"/>
  <line stroke="red" x1="577" y1="577" x2="808" y2="577" elevation="1"/>
  <line stroke="red" x1="115" y1="808" x2="154" y2="808"/>
  <line stroke="red" x1="115" y1="808" x2="115" y2="1000" elevation="1"/>
  <line stroke="blue" x1="154" y1="808" x2="192" y2="808"/>
  <line stroke="blue" x1="154" y1="808" x2="154" y2="962" elevation="0"/>
  <line stroke="red" x1="192" y1="808" x2="192" y2="923" elevation="1"/>
  <line stroke="red" x1="808" y1="192" x2="1000" y2="192" elevation="1"/>
  <line stroke="red" x1="308" y1="808" x2="346" y2="808"/>
  <line stroke="blue" x1="308" y1="808" x2="269" y2="846"/>
  <line stroke="blue" x1="308" y1="808" x2="338" y2="838"/>
  <line stroke="red" x1="615" y1="615" x2="654" y2="654"/>
  <line stroke="blue" x1="615" y1="615" x2="615" y2="808" elevation="0"/>
  <line stroke="blue" x1="615" y1="615" x2="808" y2="615" elevation="0"/>
  <line stroke="blue" x1="346" y1="808" x2="385" y2="808"/>
  <line stroke="blue" x1="346" y1="808" x2="338" y2="838" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="346" y1="808" x2="346" y2="846" elevation="1"/>
  <line stroke="red" x1="346" y1="808" x2="360" y2="860" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="269" y1="846" x2="308" y2="846" elevation="1"/>
  <line stroke="red" x1="269" y1="846" x2="231" y2="885"/>
  <line stroke="red" x1="385" y1="808" x2="423" y2="808"/>
  <line stroke="blue" x1="385" y1="808" x2="385" y2="885" elevation="0"/>
  <line stroke="red" x1="308" y1="846" x2="338" y2="838" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="308" y1="846" x2="346" y2="846" elevation="1"/>
  <line stroke="blue" x1="308" y1="846" x2="360" y2="860" opacity="0.5" fold-step="1"/>
  <line stroke="blue" x1="338" y1="838" x2="346" y2="846"/>
  <line stroke="blue" x1="423" y1="808" x2="462" y2="808"/>
  <line stroke="blue" x1="423" y1="808" x2="399" y2="899" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="423" y1="808" x2="442" y2="879" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="423" y1="808" x2="423" y2="923" elevation="1"/>
  <line stroke="red" x1="808" y1="423" x2="846" y2="385"/>
  <line stroke="blue" x1="808" y1="423" x2="808" y2="462"/>
  <line stroke="red" x1="808" y1="423" x2="846" y2="462"/>
  <line stroke="red" x1="346" y1="846" x2="360" y2="860"/>
  <line stroke="blue" x1="231" y1="885" x2="192" y2="923"/>
  <line stroke="blue" x1="231" y1="885" x2="385" y2="885" elevation="0"/>
  <line stroke="blue" x1="654" y1="654" x2="692" y2="692"/>
  <line stroke="red" x1="654" y1="654" x2="654" y2="808" elevation="1"/>
  <line stroke="red" x1="654" y1="654" x2="808" y2="654" elevation="1"/>
  <line stroke="blue" x1="846" y1="385" x2="885" y2="346"/>
  <line stroke="blue" x1="846" y1="385" x2="846" y2="462" elevation="0"/>
  <line stroke="red" x1="462" y1="808" x2="500" y2="808"/>
  <line stroke="blue" x1="462" y1="808" x2="462" y2="912" elevation="0"/>
  <line stroke="red" x1="808" y1="462" x2="808" y2="500"/>
  <line stroke="blue" x1="808" y1="462" x2="846" y2="462" elevation="0"/>
  <line stroke="red" x1="360" y1="860" x2="385" y2="885"/>
  <line stroke="red" x1="192" y1="923" x2="308" y2="923" elevation="1"/>
  <line stroke="red" x1="192" y1="923" x2="154" y2="962"/>
  <line stroke="red" x1="500" y1="808" x2="538" y2="808"/>
  <line stroke="blue" x1="500" y1="808" x2="481" y2="880" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="500" y1="808" x2="519" y2="880" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="500" y1="808" x2="500" y2="1000" elevation="1"/>
  <line stroke="blue" x1="808" y1="500" x2="808" y2="538"/>
  <line stroke="red" x1="808" y1="500" x2="885" y2="500" elevation="1"/>
  <line stroke="red" x1="885" y1="346" x2="923" y2="308"/>
  <line stroke="red" x1="885" y1="346" x2="885" y2="500" elevation="1"/>
  <line stroke="blue" x1="846" y1="462" x2="885" y2="500"/>
  <line stroke="blue" x1="385" y1="885" x2="399" y2="899"/>
  <line stroke="blue" x1="538" y1="808" x2="577" y2="808"/>
  <line stroke="blue" x1="538" y1="808" x2="538" y2="913" elevation="0"/>
  <line stroke="red" x1="808" y1="538" x2="808" y2="577"/>
  <line stroke="blue" x1="808" y1="538" x2="923" y2="538" elevation="0"/>
  <line stroke="red" x1="308" y1="923" x2="399" y2="899" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="308" y1="923" x2="423" y2="923" elevation="1"/>
  <line stroke="blue" x1="308" y1="923" x2="379" y2="942" opacity="0.5" fold-step="1"/>
  <line stroke="blue" x1="923" y1="308" x2="962" y2="269"/>
  <line stroke="blue" x1="923" y1="308" x2="923" y2="538" elevation="0"/>
  <line stroke="blue" x1="154" y1="962" x2="115" y2="1000"/>
  <line stroke="blue" x1="154" y1="962" x2="412" y2="962" elevation="0"/>
  <line stroke="red" x1="692" y1="692" x2="731" y2="731"/>
  <line stroke="blue" x1="692" y1="692" x2="692" y2="808" elevation="0"/>
  <line stroke="blue" x1="692" y1="692" x2="808" y2="692" elevation="0"/>
  <line stroke="blue" x1="399" y1="899" x2="423" y2="923"/>
  <line stroke="red" x1="442" y1="879" x2="462" y2="912"/>
  <line stroke="blue" x1="442" y1="879" x2="442" y2="942"/>
  <line stroke="red" x1="442" y1="879" x2="462" y2="951" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="577" y1="808" x2="615" y2="808"/>
  <line stroke="blue" x1="577" y1="808" x2="558" y2="879" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="577" y1="808" x2="601" y2="899" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="577" y1="808" x2="577" y2="923" elevation="1"/>
  <line stroke="blue" x1="808" y1="577" x2="808" y2="615"/>
  <line stroke="red" x1="808" y1="577" x2="962" y2="577" elevation="1"/>
  <line stroke="red" x1="962" y1="269" x2="1000" y2="231"/>
  <line stroke="red" x1="962" y1="269" x2="962" y2="577" elevation="1"/>
  <line stroke="black" x1="0" y1="1000" x2="38" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="38"/>
  <line stroke="black" x1="38" y1="1000" x2="77" y2="1000"/>
  <line stroke="black" x1="1000" y1="38" x2="1000" y2="77"/>
  <line stroke="blue" x1="481" y1="880" x2="462" y2="912"/>
  <line stroke="blue" x1="481" y1="880" x2="462" y2="951" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="481" y1="880" x2="482" y2="982"/>
  <line stroke="black" x1="77" y1="1000" x2="115" y2="1000"/>
  <line stroke="black" x1="1000" y1="77" x2="1000" y2="115"/>
  <line stroke="black" x1="115" y1="1000" x2="308" y2="1000"/>
  <line stroke="black" x1="1000" y1="115" x2="1000" y2="154"/>
  <line stroke="black" x1="1000" y1="154" x2="1000" y2="192"/>
  <line stroke="red" x1="423" y1="923" x2="442" y2="942"/>
  <line stroke="red" x1="379" y1="942" x2="442" y2="942"/>
  <line stroke="blue" x1="379" y1="942" x2="412" y2="962"/>
  <line stroke="blue" x1="379" y1="942" x2="451" y2="962" opacity="0.5" fold-step="1"/>
  <line stroke="blue" x1="615" y1="808" x2="654" y2="808"/>
  <line stroke="blue" x1="615" y1="808" x2="615" y2="885" elevation="0"/>
  <line stroke="blue" x1="808" y1="615" x2="808" y2="654"/>
  <line stroke="red" x1="808" y1="615" x2="1000" y2="615" elevation="0"/>
  <line stroke="red" x1="885" y1="500" x2="923" y2="538"/>
  <line stroke="black" x1="1000" y1="192" x2="1000" y2="231"/>
  <line stroke="blue" x1="519" y1="880" x2="538" y2="913"/>
  <line stroke="red" x1="519" y1="880" x2="538" y2="951" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="519" y1="880" x2="519" y2="981"/>
  <line stroke="blue" x1="462" y1="912" x2="462" y2="951" elevation="0"/>
  <line stroke="black" x1="1000" y1="231" x2="1000" y2="615"/>
  <line stroke="blue" x1="731" y1="731" x2="769" y2="769"/>
  <line stroke="red" x1="731" y1="731" x2="808" y2="731" elevation="1"/>
  <line stroke="red" x1="731" y1="731" x2="731" y2="846" elevation="1"/>
  <line stroke="red" x1="654" y1="808" x2="692" y2="808"/>
  <line stroke="red" x1="654" y1="808" x2="662" y2="838" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="654" y1="808" x2="654" y2="846" elevation="1"/>
  <line stroke="blue" x1="654" y1="808" x2="640" y2="860" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="808" y1="654" x2="808" y2="692"/>
  <line stroke="red" x1="808" y1="654" x2="962" y2="654" elevation="1"/>
  <line stroke="red" x1="442" y1="942" x2="462" y2="962"/>
  <line stroke="red" x1="558" y1="879" x2="538" y2="913"/>
  <line stroke="blue" x1="558" y1="879" x2="538" y2="951" opacity="0.5" fold-step="1"/>
  <line stroke="blue" x1="558" y1="879" x2="557" y2="943"/>
  <line stroke="red" x1="308" y1="1000" x2="380" y2="981" opacity="0.5" fold-step="1"/>
  <line stroke="black" x1="308" y1="1000" x2="500" y2="1000"/>
  <line stroke="red" x1="412" y1="962" x2="380" y2="981"/>
  <line stroke="blue" x1="412" y1="962" x2="451" y2="962" elevation="0"/>
  <line stroke="red" x1="380" y1="981" x2="451" y2="962" opacity="0.5" fold-step="1"/>
  <line stroke="blue" x1="380" y1="981" x2="482" y2="982"/>
  <line stroke="blue" x1="462" y1="951" x2="462" y2="962" elevation="0"/>
  <line stroke="blue" x1="538" y1="913" x2="538" y2="951" elevation="0"/>
  <line stroke="blue" x1="451" y1="962" x2="462" y2="962" elevation="0"/>
  <line stroke="blue" x1="692" y1="808" x2="662" y2="838"/>
  <line stroke="blue" x1="692" y1="808" x2="731" y2="846"/>
  <line stroke="blue" x1="808" y1="692" x2="808" y2="731"/>
  <line stroke="blue" x1="808" y1="692" x2="923" y2="692" elevation="0"/>
  <line stroke="blue" x1="462" y1="962" x2="482" y2="982"/>
  <line stroke="blue" x1="662" y1="838" x2="654" y2="846"/>
  <line stroke="blue" x1="662" y1="838" x2="692" y2="846" opacity="0.5" fold-step="1"/>
  <line stroke="blue" x1="923" y1="538" x2="962" y2="577"/>
  <line stroke="red" x1="654" y1="846" x2="640" y2="860"/>
  <line stroke="red" x1="654" y1="846" x2="692" y2="846" elevation="1"/>
  <line stroke="red" x1="640" y1="860" x2="615" y2="885"/>
  <line stroke="red" x1="640" y1="860" x2="692" y2="846" opacity="0.5" fold-step="1"/>
  <line stroke="blue" x1="615" y1="885" x2="601" y2="899"/>
  <line stroke="blue" x1="615" y1="885" x2="769" y2="885" elevation="0"/>
  <line stroke="blue" x1="601" y1="899" x2="577" y2="923"/>
  <line stroke="blue" x1="601" y1="899" x2="692" y2="923" opacity="0.5" fold-step="1"/>
  <line stroke="blue" x1="769" y1="769" x2="808" y2="769" elevation="0"/>
  <line stroke="red" x1="769" y1="769" x2="808" y2="808"/>
  <line stroke="blue" x1="769" y1="769" x2="769" y2="885" elevation="0"/>
  <line stroke="red" x1="577" y1="923" x2="557" y2="943"/>
  <line stroke="red" x1="577" y1="923" x2="692" y2="923" elevation="1"/>
  <line stroke="red" x1="808" y1="731" x2="808" y2="769"/>
  <line stroke="red" x1="808" y1="731" x2="885" y2="731" elevation="1"/>
  <line stroke="blue" x1="538" y1="951" x2="538" y2="962" elevation="0"/>
  <line stroke="red" x1="692" y1="846" x2="731" y2="846" elevation="1"/>
  <line stroke="blue" x1="482" y1="982" x2="500" y2="1000"/>
  <line stroke="red" x1="557" y1="943" x2="538" y2="962"/>
  <line stroke="red" x1="557" y1="943" x2="621" y2="942"/>
  <line stroke="blue" x1="538" y1="962" x2="549" y2="962" elevation="0"/>
  <line stroke="blue" x1="538" y1="962" x2="519" y2="981"/>
  <line stroke="blue" x1="549" y1="962" x2="587" y2="962" elevation="0"/>
  <line stroke="red" x1="549" y1="962" x2="621" y2="942" opacity="0.5" fold-step="1"/>
  <line stroke="blue" x1="549" y1="962" x2="620" y2="981" opacity="0.5" fold-step="1"/>
  <line stroke="blue" x1="519" y1="981" x2="500" y2="1000"/>
  <line stroke="blue" x1="519" y1="981" x2="620" y2="981"/>
  <line stroke="blue" x1="808" y1="769" x2="808" y2="808"/>
  <line stroke="blue" x1="808" y1="769" x2="846" y2="769" elevation="0"/>
  <line stroke="black" x1="500" y1="1000" x2="692" y2="1000"/>
  <line stroke="red" x1="731" y1="846" x2="769" y2="885"/>
  <line stroke="red" x1="962" y1="577" x2="1000" y2="615"/>
  <line stroke="blue" x1="587" y1="962" x2="621" y2="942"/>
  <line stroke="red" x1="587" y1="962" x2="620" y2="981"/>
  <line stroke="blue" x1="587" y1="962" x2="846" y2="962" elevation="0"/>
  <line stroke="red" x1="621" y1="942" x2="692" y2="923" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="808" y1="808" x2="846" y2="769"/>
  <line stroke="red" x1="808" y1="808" x2="808" y2="923" elevation="1"/>
  <line stroke="blue" x1="846" y1="769" x2="885" y2="731"/>
  <line stroke="blue" x1="846" y1="769" x2="846" y2="962" elevation="0"/>
  <line stroke="red" x1="885" y1="731" x2="923" y2="692"/>
  <line stroke="red" x1="885" y1="731" x2="885" y2="1000" elevation="1"/>
  <line stroke="red" x1="692" y1="923" x2="808" y2="923" elevation="1"/>
  <line stroke="blue" x1="923" y1="692" x2="962" y2="654"/>
  <line stroke="blue" x1="923" y1="692" x2="923" y2="1000" elevation="0"/>
  <line stroke="blue" x1="620" y1="981" x2="692" y2="1000" opacity="0.5" fold-step="1"/>
  <line stroke="red" x1="962" y1="654" x2="1000" y2="615"/>
  <line stroke="red" x1="962" y1="654" x2="962" y2="1000" elevation="1"/>
  <line stroke="blue" x1="769" y1="885" x2="808" y2="923"/>
  <line stroke="black" x1="1000" y1="615" x2="1000" y2="1000"/>
  <line stroke="black" x1="692" y1="1000" x2="885" y2="1000"/>
  <line stroke="red" x1="808" y1="923" x2="846" y2="962"/>
  <line stroke="blue" x1="846" y1="962" x2="885" y2="1000"/>
  <line stroke="black" x1="885" y1="1000" x2="923" y2="1000"/>
  <line stroke="black" x1="923" y1="1000" x2="962" y2="1000"/>
  <line stroke="black" x1="962" y1="1000" x2="1000" y2="1000"/>
</svg>""",
)

NARROW_EDGE_CASE2 = BoxPleatingShapingExample(
    base_example=BoxPleatingCreasePatternExample(
        name="narrow_edge_case2",
        packing_svg="",
        grid_size=30,
        base_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="0" y2="33"/>
  <line stroke="black" x1="0" y1="0" x2="67" y2="0"/>
  <line stroke="black" x1="0" y1="33" x2="0" y2="67"/>
  <line stroke="red" x1="0" y1="33" x2="67" y2="33" elevation="1"/>
  <line stroke="blue" x1="0" y1="67" x2="67" y2="67" elevation="0"/>
  <line stroke="black" x1="0" y1="67" x2="0" y2="100"/>
  <line stroke="red" x1="67" y1="0" x2="67" y2="33"/>
  <line stroke="black" x1="67" y1="0" x2="367" y2="0"/>
  <line stroke="blue" x1="67" y1="33" x2="67" y2="67"/>
  <line stroke="red" x1="67" y1="33" x2="333" y2="33" elevation="1"/>
  <line stroke="red" x1="67" y1="67" x2="67" y2="100"/>
  <line stroke="blue" x1="67" y1="67" x2="300" y2="67" elevation="0"/>
  <line stroke="red" x1="0" y1="100" x2="67" y2="100" elevation="1"/>
  <line stroke="black" x1="0" y1="100" x2="0" y2="133"/>
  <line stroke="blue" x1="67" y1="100" x2="33" y2="133"/>
  <line stroke="blue" x1="67" y1="100" x2="100" y2="133"/>
  <line stroke="red" x1="67" y1="100" x2="67" y2="167" elevation="1"/>
  <line stroke="red" x1="67" y1="100" x2="267" y2="100" elevation="1"/>
  <line stroke="blue" x1="0" y1="133" x2="33" y2="133" elevation="0"/>
  <line stroke="black" x1="0" y1="133" x2="0" y2="167"/>
  <line stroke="red" x1="33" y1="133" x2="0" y2="167"/>
  <line stroke="blue" x1="33" y1="133" x2="33" y2="167" elevation="0"/>
  <line stroke="blue" x1="100" y1="133" x2="100" y2="167" elevation="0"/>
  <line stroke="red" x1="100" y1="133" x2="133" y2="167"/>
  <line stroke="blue" x1="100" y1="133" x2="233" y2="133" elevation="0"/>
  <line stroke="red" x1="0" y1="167" x2="33" y2="167"/>
  <line stroke="black" x1="0" y1="167" x2="0" y2="233"/>
  <line stroke="blue" x1="33" y1="167" x2="67" y2="167"/>
  <line stroke="blue" x1="33" y1="167" x2="33" y2="233" elevation="0"/>
  <line stroke="red" x1="67" y1="167" x2="100" y2="167"/>
  <line stroke="red" x1="67" y1="167" x2="67" y2="233" elevation="1"/>
  <line stroke="blue" x1="100" y1="167" x2="133" y2="167"/>
  <line stroke="blue" x1="100" y1="167" x2="100" y2="200" elevation="0"/>
  <line stroke="red" x1="133" y1="167" x2="100" y2="200"/>
  <line stroke="red" x1="133" y1="167" x2="267" y2="167" elevation="1"/>
  <line stroke="blue" x1="100" y1="200" x2="67" y2="233"/>
  <line stroke="blue" x1="100" y1="200" x2="300" y2="200" elevation="0"/>
  <line stroke="blue" x1="0" y1="233" x2="33" y2="233"/>
  <line stroke="black" x1="0" y1="233" x2="0" y2="533"/>
  <line stroke="red" x1="33" y1="233" x2="67" y2="233"/>
  <line stroke="blue" x1="33" y1="233" x2="33" y2="533" elevation="0"/>
  <line stroke="blue" x1="67" y1="233" x2="100" y2="267"/>
  <line stroke="red" x1="67" y1="233" x2="333" y2="233" elevation="1"/>
  <line stroke="red" x1="67" y1="233" x2="67" y2="533" elevation="1"/>
  <line stroke="blue" x1="233" y1="133" x2="267" y2="100"/>
  <line stroke="red" x1="233" y1="133" x2="267" y2="133"/>
  <line stroke="blue" x1="233" y1="133" x2="267" y2="167"/>
  <line stroke="red" x1="100" y1="267" x2="133" y2="300"/>
  <line stroke="blue" x1="100" y1="267" x2="367" y2="267" elevation="0"/>
  <line stroke="blue" x1="100" y1="267" x2="100" y2="533" elevation="0"/>
  <line stroke="red" x1="267" y1="100" x2="267" y2="133" elevation="1"/>
  <line stroke="red" x1="267" y1="100" x2="300" y2="67"/>
  <line stroke="red" x1="267" y1="133" x2="267" y2="167" elevation="1"/>
  <line stroke="blue" x1="267" y1="133" x2="300" y2="133"/>
  <line stroke="blue" x1="300" y1="67" x2="300" y2="133" elevation="0"/>
  <line stroke="blue" x1="300" y1="67" x2="333" y2="33"/>
  <line stroke="red" x1="267" y1="167" x2="300" y2="200"/>
  <line stroke="blue" x1="133" y1="300" x2="167" y2="333"/>
  <line stroke="red" x1="133" y1="300" x2="333" y2="300" elevation="1"/>
  <line stroke="red" x1="133" y1="300" x2="133" y2="533" elevation="1"/>
  <line stroke="red" x1="300" y1="133" x2="333" y2="133"/>
  <line stroke="blue" x1="300" y1="133" x2="300" y2="200" elevation="0"/>
  <line stroke="red" x1="333" y1="33" x2="333" y2="133" elevation="1"/>
  <line stroke="red" x1="333" y1="33" x2="367" y2="0"/>
  <line stroke="blue" x1="333" y1="133" x2="367" y2="133"/>
  <line stroke="red" x1="333" y1="133" x2="333" y2="233" elevation="1"/>
  <line stroke="blue" x1="300" y1="200" x2="333" y2="233"/>
  <line stroke="blue" x1="367" y1="0" x2="367" y2="133" elevation="0"/>
  <line stroke="red" x1="367" y1="0" x2="400" y2="33"/>
  <line stroke="black" x1="367" y1="0" x2="500" y2="0"/>
  <line stroke="red" x1="167" y1="333" x2="200" y2="367"/>
  <line stroke="blue" x1="167" y1="333" x2="300" y2="333" elevation="0"/>
  <line stroke="blue" x1="167" y1="333" x2="167" y2="533" elevation="0"/>
  <line stroke="red" x1="367" y1="133" x2="400" y2="133"/>
  <line stroke="blue" x1="367" y1="133" x2="367" y2="267" elevation="0"/>
  <line stroke="red" x1="400" y1="33" x2="400" y2="133" elevation="1"/>
  <line stroke="blue" x1="400" y1="33" x2="433" y2="67"/>
  <line stroke="red" x1="400" y1="33" x2="500" y2="33" elevation="1"/>
  <line stroke="red" x1="333" y1="233" x2="367" y2="267"/>
  <line stroke="red" x1="200" y1="367" x2="267" y2="367" elevation="1"/>
  <line stroke="blue" x1="200" y1="367" x2="233" y2="400"/>
  <line stroke="red" x1="200" y1="367" x2="200" y2="533" elevation="1"/>
  <line stroke="blue" x1="400" y1="133" x2="433" y2="133"/>
  <line stroke="red" x1="400" y1="133" x2="400" y2="233" elevation="1"/>
  <line stroke="blue" x1="433" y1="67" x2="433" y2="133" elevation="0"/>
  <line stroke="red" x1="433" y1="67" x2="467" y2="100"/>
  <line stroke="blue" x1="433" y1="67" x2="500" y2="67" elevation="0"/>
  <line stroke="blue" x1="300" y1="333" x2="333" y2="300"/>
  <line stroke="red" x1="300" y1="333" x2="267" y2="367"/>
  <line stroke="blue" x1="300" y1="333" x2="300" y2="400" elevation="0"/>
  <line stroke="red" x1="333" y1="300" x2="367" y2="267"/>
  <line stroke="red" x1="333" y1="300" x2="333" y2="400" elevation="1"/>
  <line stroke="blue" x1="433" y1="133" x2="433" y2="200" elevation="0"/>
  <line stroke="red" x1="433" y1="133" x2="467" y2="133"/>
  <line stroke="blue" x1="267" y1="367" x2="233" y2="400"/>
  <line stroke="red" x1="267" y1="367" x2="267" y2="400" elevation="1"/>
  <line stroke="red" x1="367" y1="267" x2="400" y2="233"/>
  <line stroke="red" x1="367" y1="267" x2="400" y2="300"/>
  <line stroke="red" x1="367" y1="267" x2="367" y2="400" elevation="0"/>
  <line stroke="blue" x1="367" y1="267" x2="500" y2="267" elevation="0"/>
  <line stroke="red" x1="233" y1="400" x2="267" y2="400"/>
  <line stroke="blue" x1="233" y1="400" x2="233" y2="533" elevation="0"/>
  <line stroke="blue" x1="400" y1="233" x2="433" y2="200"/>
  <line stroke="red" x1="400" y1="233" x2="500" y2="233" elevation="1"/>
  <line stroke="red" x1="433" y1="200" x2="467" y2="167"/>
  <line stroke="blue" x1="433" y1="200" x2="500" y2="200" elevation="0"/>
  <line stroke="red" x1="467" y1="100" x2="467" y2="133" elevation="1"/>
  <line stroke="red" x1="467" y1="100" x2="500" y2="100" elevation="1"/>
  <line stroke="blue" x1="467" y1="100" x2="500" y2="133"/>
  <line stroke="blue" x1="267" y1="400" x2="300" y2="400"/>
  <line stroke="red" x1="267" y1="400" x2="267" y2="533" elevation="1"/>
  <line stroke="red" x1="467" y1="133" x2="467" y2="167" elevation="1"/>
  <line stroke="blue" x1="467" y1="133" x2="500" y2="133"/>
  <line stroke="blue" x1="467" y1="167" x2="500" y2="133"/>
  <line stroke="red" x1="467" y1="167" x2="500" y2="167" elevation="1"/>
  <line stroke="red" x1="300" y1="400" x2="333" y2="400"/>
  <line stroke="blue" x1="300" y1="400" x2="300" y2="567" elevation="0"/>
  <line stroke="blue" x1="400" y1="300" x2="433" y2="333"/>
  <line stroke="red" x1="400" y1="300" x2="400" y2="400" elevation="1"/>
  <line stroke="red" x1="400" y1="300" x2="500" y2="300" elevation="1"/>
  <line stroke="blue" x1="500" y1="0" x2="500" y2="33"/>
  <line stroke="black" x1="500" y1="0" x2="633" y2="0"/>
  <line stroke="red" x1="500" y1="33" x2="500" y2="67"/>
  <line stroke="red" x1="500" y1="33" x2="600" y2="33" elevation="1"/>
  <line stroke="blue" x1="500" y1="67" x2="500" y2="100"/>
  <line stroke="blue" x1="500" y1="67" x2="567" y2="67" elevation="0"/>
  <line stroke="red" x1="500" y1="100" x2="500" y2="133"/>
  <line stroke="red" x1="500" y1="100" x2="533" y2="100" elevation="1"/>
  <line stroke="red" x1="500" y1="133" x2="500" y2="167"/>
  <line stroke="blue" x1="500" y1="133" x2="533" y2="100"/>
  <line stroke="red" x1="500" y1="133" x2="533" y2="133"/>
  <line stroke="blue" x1="500" y1="133" x2="533" y2="167"/>
  <line stroke="blue" x1="333" y1="400" x2="367" y2="400"/>
  <line stroke="red" x1="333" y1="400" x2="333" y2="600" elevation="1"/>
  <line stroke="blue" x1="500" y1="167" x2="500" y2="200"/>
  <line stroke="red" x1="500" y1="167" x2="533" y2="167" elevation="1"/>
  <line stroke="red" x1="0" y1="533" x2="33" y2="533"/>
  <line stroke="black" x1="0" y1="533" x2="0" y2="667"/>
  <line stroke="blue" x1="33" y1="533" x2="67" y2="533"/>
  <line stroke="blue" x1="33" y1="533" x2="33" y2="667" elevation="0"/>
  <line stroke="red" x1="67" y1="533" x2="100" y2="533"/>
  <line stroke="red" x1="67" y1="533" x2="67" y2="667" elevation="1"/>
  <line stroke="red" x1="500" y1="200" x2="500" y2="233"/>
  <line stroke="blue" x1="500" y1="200" x2="567" y2="200" elevation="0"/>
  <line stroke="blue" x1="100" y1="533" x2="133" y2="533"/>
  <line stroke="blue" x1="100" y1="533" x2="100" y2="667" elevation="0"/>
  <line stroke="red" x1="533" y1="100" x2="533" y2="133" elevation="1"/>
  <line stroke="red" x1="533" y1="100" x2="567" y2="67"/>
  <line stroke="blue" x1="367" y1="400" x2="400" y2="400"/>
  <line stroke="blue" x1="367" y1="400" x2="367" y2="633" elevation="0"/>
  <line stroke="blue" x1="433" y1="333" x2="433" y2="400" elevation="0"/>
  <line stroke="red" x1="433" y1="333" x2="467" y2="367"/>
  <line stroke="blue" x1="433" y1="333" x2="500" y2="333" elevation="0"/>
  <line stroke="red" x1="133" y1="533" x2="167" y2="533"/>
  <line stroke="red" x1="133" y1="533" x2="133" y2="667" elevation="1"/>
  <line stroke="red" x1="533" y1="133" x2="533" y2="167" elevation="1"/>
  <line stroke="blue" x1="533" y1="133" x2="567" y2="133"/>
  <line stroke="blue" x1="500" y1="233" x2="500" y2="267"/>
  <line stroke="red" x1="500" y1="233" x2="600" y2="233" elevation="1"/>
  <line stroke="blue" x1="167" y1="533" x2="200" y2="533"/>
  <line stroke="blue" x1="167" y1="533" x2="167" y2="633" elevation="0"/>
  <line stroke="red" x1="533" y1="167" x2="567" y2="200"/>
  <line stroke="red" x1="400" y1="400" x2="433" y2="400"/>
  <line stroke="red" x1="400" y1="400" x2="400" y2="667" elevation="1"/>
  <line stroke="red" x1="500" y1="267" x2="500" y2="300"/>
  <line stroke="blue" x1="500" y1="267" x2="633" y2="267" elevation="0"/>
  <line stroke="red" x1="200" y1="533" x2="233" y2="533"/>
  <line stroke="red" x1="200" y1="533" x2="200" y2="600" elevation="1"/>
  <line stroke="blue" x1="567" y1="67" x2="567" y2="133" elevation="0"/>
  <line stroke="blue" x1="567" y1="67" x2="600" y2="33"/>
  <line stroke="blue" x1="233" y1="533" x2="267" y2="533"/>
  <line stroke="blue" x1="233" y1="533" x2="233" y2="567" elevation="0"/>
  <line stroke="blue" x1="567" y1="133" x2="567" y2="200" elevation="0"/>
  <line stroke="red" x1="567" y1="133" x2="600" y2="133"/>
  <line stroke="blue" x1="500" y1="300" x2="500" y2="333"/>
  <line stroke="red" x1="500" y1="300" x2="600" y2="300" elevation="1"/>
  <line stroke="blue" x1="433" y1="400" x2="467" y2="400"/>
  <line stroke="blue" x1="433" y1="400" x2="433" y2="700" elevation="0"/>
  <line stroke="red" x1="467" y1="367" x2="467" y2="400" elevation="1"/>
  <line stroke="red" x1="467" y1="367" x2="500" y2="367" elevation="1"/>
  <line stroke="blue" x1="467" y1="367" x2="500" y2="400"/>
  <line stroke="red" x1="267" y1="533" x2="233" y2="567"/>
  <line stroke="red" x1="267" y1="533" x2="300" y2="567"/>
  <line stroke="red" x1="500" y1="333" x2="500" y2="367"/>
  <line stroke="blue" x1="500" y1="333" x2="567" y2="333" elevation="0"/>
  <line stroke="red" x1="600" y1="33" x2="600" y2="133" elevation="1"/>
  <line stroke="red" x1="600" y1="33" x2="633" y2="0"/>
  <line stroke="blue" x1="567" y1="200" x2="600" y2="233"/>
  <line stroke="blue" x1="233" y1="567" x2="200" y2="600"/>
  <line stroke="blue" x1="233" y1="567" x2="300" y2="567" elevation="0"/>
  <line stroke="red" x1="600" y1="133" x2="600" y2="233" elevation="1"/>
  <line stroke="blue" x1="600" y1="133" x2="633" y2="133"/>
  <line stroke="red" x1="467" y1="400" x2="500" y2="400"/>
  <line stroke="red" x1="467" y1="400" x2="467" y2="733" elevation="1"/>
  <line stroke="blue" x1="500" y1="367" x2="500" y2="400"/>
  <line stroke="red" x1="500" y1="367" x2="533" y2="367" elevation="1"/>
  <line stroke="red" x1="200" y1="600" x2="167" y2="633"/>
  <line stroke="red" x1="200" y1="600" x2="333" y2="600" elevation="1"/>
  <line stroke="blue" x1="633" y1="0" x2="633" y2="133" elevation="0"/>
  <line stroke="red" x1="633" y1="0" x2="667" y2="33"/>
  <line stroke="black" x1="633" y1="0" x2="767" y2="0"/>
  <line stroke="blue" x1="500" y1="400" x2="533" y2="367"/>
  <line stroke="red" x1="500" y1="400" x2="533" y2="400"/>
  <line stroke="blue" x1="500" y1="400" x2="500" y2="767" elevation="0"/>
  <line stroke="blue" x1="300" y1="567" x2="333" y2="600"/>
  <line stroke="red" x1="600" y1="233" x2="633" y2="267"/>
  <line stroke="blue" x1="633" y1="133" x2="667" y2="133"/>
  <line stroke="red" x1="633" y1="133" x2="633" y2="267" elevation="0"/>
  <line stroke="red" x1="533" y1="367" x2="567" y2="333"/>
  <line stroke="red" x1="533" y1="367" x2="533" y2="400" elevation="1"/>
  <line stroke="blue" x1="167" y1="633" x2="133" y2="667"/>
  <line stroke="blue" x1="167" y1="633" x2="367" y2="633" elevation="0"/>
  <line stroke="blue" x1="567" y1="333" x2="600" y2="300"/>
  <line stroke="blue" x1="567" y1="333" x2="567" y2="400" elevation="0"/>
  <line stroke="blue" x1="533" y1="400" x2="567" y2="400"/>
  <line stroke="red" x1="533" y1="400" x2="533" y2="733" elevation="1"/>
  <line stroke="blue" x1="0" y1="667" x2="33" y2="667"/>
  <line stroke="black" x1="0" y1="667" x2="0" y2="933"/>
  <line stroke="red" x1="33" y1="667" x2="67" y2="667"/>
  <line stroke="blue" x1="33" y1="667" x2="33" y2="933" elevation="0"/>
  <line stroke="red" x1="667" y1="33" x2="667" y2="133" elevation="1"/>
  <line stroke="blue" x1="667" y1="33" x2="700" y2="67"/>
  <line stroke="red" x1="667" y1="33" x2="767" y2="33" elevation="1"/>
  <line stroke="blue" x1="67" y1="667" x2="100" y2="667"/>
  <line stroke="red" x1="67" y1="667" x2="67" y2="933" elevation="1"/>
  <line stroke="red" x1="600" y1="300" x2="633" y2="267"/>
  <line stroke="red" x1="600" y1="300" x2="600" y2="400" elevation="1"/>
  <line stroke="red" x1="100" y1="667" x2="133" y2="667"/>
  <line stroke="blue" x1="100" y1="667" x2="100" y2="933" elevation="0"/>
  <line stroke="blue" x1="133" y1="667" x2="167" y2="700"/>
  <line stroke="red" x1="133" y1="667" x2="400" y2="667" elevation="1"/>
  <line stroke="red" x1="133" y1="667" x2="133" y2="933" elevation="1"/>
  <line stroke="red" x1="667" y1="133" x2="667" y2="233" elevation="1"/>
  <line stroke="red" x1="667" y1="133" x2="700" y2="133"/>
  <line stroke="red" x1="333" y1="600" x2="367" y2="633"/>
  <line stroke="red" x1="633" y1="267" x2="667" y2="233"/>
  <line stroke="red" x1="633" y1="267" x2="667" y2="300"/>
  <line stroke="blue" x1="633" y1="267" x2="633" y2="400" elevation="0"/>
  <line stroke="blue" x1="633" y1="267" x2="767" y2="267" elevation="0"/>
  <line stroke="red" x1="567" y1="400" x2="600" y2="400"/>
  <line stroke="blue" x1="567" y1="400" x2="567" y2="700" elevation="0"/>
  <line stroke="blue" x1="700" y1="67" x2="700" y2="133" elevation="0"/>
  <line stroke="red" x1="700" y1="67" x2="733" y2="100"/>
  <line stroke="blue" x1="700" y1="67" x2="767" y2="67" elevation="0"/>
  <line stroke="blue" x1="667" y1="233" x2="700" y2="200"/>
  <line stroke="red" x1="667" y1="233" x2="767" y2="233" elevation="1"/>
  <line stroke="blue" x1="700" y1="133" x2="700" y2="200" elevation="0"/>
  <line stroke="blue" x1="700" y1="133" x2="733" y2="133"/>
  <line stroke="red" x1="167" y1="700" x2="200" y2="733"/>
  <line stroke="blue" x1="167" y1="700" x2="433" y2="700" elevation="0"/>
  <line stroke="blue" x1="167" y1="700" x2="167" y2="900" elevation="0"/>
  <line stroke="blue" x1="600" y1="400" x2="633" y2="400"/>
  <line stroke="red" x1="600" y1="400" x2="600" y2="667" elevation="1"/>
  <line stroke="red" x1="700" y1="200" x2="733" y2="167"/>
  <line stroke="blue" x1="700" y1="200" x2="767" y2="200" elevation="0"/>
  <line stroke="blue" x1="667" y1="300" x2="700" y2="333"/>
  <line stroke="red" x1="667" y1="300" x2="667" y2="400" elevation="1"/>
  <line stroke="red" x1="667" y1="300" x2="767" y2="300" elevation="1"/>
  <line stroke="blue" x1="367" y1="633" x2="400" y2="667"/>
  <line stroke="red" x1="733" y1="100" x2="733" y2="133" elevation="1"/>
  <line stroke="red" x1="733" y1="100" x2="767" y2="100" elevation="1"/>
  <line stroke="blue" x1="733" y1="100" x2="767" y2="133"/>
  <line stroke="red" x1="733" y1="133" x2="733" y2="167" elevation="1"/>
  <line stroke="red" x1="733" y1="133" x2="767" y2="133"/>
  <line stroke="red" x1="633" y1="400" x2="667" y2="400"/>
  <line stroke="blue" x1="633" y1="400" x2="633" y2="633" elevation="0"/>
  <line stroke="blue" x1="733" y1="167" x2="767" y2="133"/>
  <line stroke="red" x1="733" y1="167" x2="767" y2="167" elevation="1"/>
  <line stroke="blue" x1="200" y1="733" x2="233" y2="767"/>
  <line stroke="red" x1="200" y1="733" x2="467" y2="733" elevation="1"/>
  <line stroke="red" x1="200" y1="733" x2="200" y2="867" elevation="1"/>
  <line stroke="blue" x1="767" y1="0" x2="767" y2="33"/>
  <line stroke="black" x1="767" y1="0" x2="867" y2="0"/>
  <line stroke="red" x1="767" y1="33" x2="767" y2="67"/>
  <line stroke="red" x1="767" y1="33" x2="924" y2="33" elevation="1"/>
  <line stroke="blue" x1="767" y1="67" x2="767" y2="100"/>
  <line stroke="blue" x1="767" y1="67" x2="867" y2="67" elevation="0"/>
  <line stroke="red" x1="767" y1="100" x2="767" y2="133"/>
  <line stroke="red" x1="767" y1="100" x2="924" y2="100" elevation="1"/>
  <line stroke="blue" x1="700" y1="333" x2="700" y2="400" elevation="0"/>
  <line stroke="red" x1="700" y1="333" x2="733" y2="367"/>
  <line stroke="blue" x1="700" y1="333" x2="767" y2="333" elevation="0"/>
  <line stroke="red" x1="400" y1="667" x2="433" y2="700"/>
  <line stroke="blue" x1="667" y1="400" x2="700" y2="400"/>
  <line stroke="red" x1="667" y1="400" x2="667" y2="600" elevation="1"/>
  <line stroke="blue" x1="767" y1="133" x2="767" y2="167"/>
  <line stroke="blue" x1="767" y1="133" x2="867" y2="133" elevation="0"/>
  <line stroke="red" x1="767" y1="167" x2="767" y2="200"/>
  <line stroke="red" x1="767" y1="167" x2="867" y2="167" elevation="1"/>
  <line stroke="blue" x1="767" y1="200" x2="767" y2="233"/>
  <line stroke="blue" x1="767" y1="200" x2="867" y2="200" elevation="0"/>
  <line stroke="red" x1="233" y1="767" x2="267" y2="800"/>
  <line stroke="blue" x1="233" y1="767" x2="233" y2="833" elevation="0"/>
  <line stroke="blue" x1="233" y1="767" x2="500" y2="767" elevation="0"/>
  <line stroke="red" x1="767" y1="233" x2="767" y2="267"/>
  <line stroke="red" x1="767" y1="233" x2="867" y2="233" elevation="1"/>
  <line stroke="red" x1="700" y1="400" x2="733" y2="400"/>
  <line stroke="blue" x1="700" y1="400" x2="700" y2="567" elevation="0"/>
  <line stroke="blue" x1="767" y1="267" x2="767" y2="300"/>
  <line stroke="blue" x1="767" y1="267" x2="867" y2="267" elevation="0"/>
  <line stroke="red" x1="733" y1="367" x2="733" y2="400" elevation="1"/>
  <line stroke="red" x1="733" y1="367" x2="767" y2="367" elevation="1"/>
  <line stroke="blue" x1="733" y1="367" x2="767" y2="400"/>
  <line stroke="blue" x1="433" y1="700" x2="467" y2="733"/>
  <line stroke="red" x1="767" y1="300" x2="767" y2="333"/>
  <line stroke="red" x1="767" y1="300" x2="867" y2="300" elevation="1"/>
  <line stroke="blue" x1="733" y1="400" x2="767" y2="400"/>
  <line stroke="red" x1="733" y1="400" x2="733" y2="533" elevation="1"/>
  <line stroke="blue" x1="767" y1="333" x2="767" y2="367"/>
  <line stroke="blue" x1="767" y1="333" x2="833" y2="333" elevation="0"/>
  <line stroke="red" x1="267" y1="800" x2="233" y2="833"/>
  <line stroke="blue" x1="267" y1="800" x2="267" y2="833"/>
  <line stroke="red" x1="267" y1="800" x2="500" y2="800" elevation="1"/>
  <line stroke="red" x1="767" y1="367" x2="767" y2="400"/>
  <line stroke="red" x1="767" y1="367" x2="800" y2="367" elevation="1"/>
  <line stroke="blue" x1="233" y1="833" x2="267" y2="833" elevation="0"/>
  <line stroke="blue" x1="233" y1="833" x2="200" y2="867"/>
  <line stroke="blue" x1="767" y1="400" x2="800" y2="367"/>
  <line stroke="red" x1="767" y1="400" x2="800" y2="400"/>
  <line stroke="blue" x1="767" y1="400" x2="767" y2="533" elevation="0"/>
  <line stroke="blue" x1="867" y1="0" x2="924" y2="33" opacity="0.8" fold-step="1"/>
  <line stroke="black" x1="867" y1="0" x2="933" y2="0"/>
  <line stroke="red" x1="467" y1="733" x2="500" y2="767"/>
  <line stroke="red" x1="867" y1="67" x2="924" y2="33" opacity="0.8" fold-step="1"/>
  <line stroke="blue" x1="867" y1="67" x2="924" y2="100" opacity="0.8" fold-step="1"/>
  <line stroke="blue" x1="867" y1="67" x2="933" y2="67" elevation="0"/>
  <line stroke="red" x1="267" y1="833" x2="267" y2="867"/>
  <line stroke="blue" x1="267" y1="833" x2="500" y2="833" elevation="0"/>
  <line stroke="blue" x1="867" y1="133" x2="888" y2="146" opacity="0.8" fold-step="1"/>
  <line stroke="blue" x1="867" y1="133" x2="900" y2="133" elevation="0"/>
  <line stroke="red" x1="867" y1="133" x2="924" y2="100" opacity="0.8" fold-step="1"/>
  <line stroke="red" x1="800" y1="367" x2="800" y2="400" elevation="1"/>
  <line stroke="red" x1="800" y1="367" x2="833" y2="333"/>
  <line stroke="blue" x1="867" y1="167" x2="867" y2="200"/>
  <line stroke="red" x1="867" y1="167" x2="888" y2="146"/>
  <line stroke="red" x1="867" y1="167" x2="900" y2="200"/>
  <line stroke="red" x1="200" y1="867" x2="267" y2="867" elevation="1"/>
  <line stroke="red" x1="200" y1="867" x2="167" y2="900"/>
  <line stroke="red" x1="867" y1="200" x2="867" y2="233"/>
  <line stroke="blue" x1="867" y1="200" x2="900" y2="200" elevation="0"/>
  <line stroke="blue" x1="800" y1="400" x2="833" y2="400"/>
  <line stroke="red" x1="800" y1="400" x2="800" y2="533" elevation="1"/>
  <line stroke="blue" x1="633" y1="633" x2="600" y2="667"/>
  <line stroke="red" x1="633" y1="633" x2="667" y2="600"/>
  <line stroke="blue" x1="633" y1="633" x2="833" y2="633" elevation="0"/>
  <line stroke="blue" x1="833" y1="333" x2="867" y2="300"/>
  <line stroke="blue" x1="833" y1="333" x2="833" y2="400" elevation="0"/>
  <line stroke="red" x1="600" y1="667" x2="567" y2="700"/>
  <line stroke="red" x1="600" y1="667" x2="867" y2="667" elevation="1"/>
  <line stroke="blue" x1="667" y1="600" x2="700" y2="567"/>
  <line stroke="red" x1="667" y1="600" x2="800" y2="600" elevation="1"/>
  <line stroke="blue" x1="867" y1="233" x2="867" y2="267"/>
  <line stroke="red" x1="867" y1="233" x2="933" y2="233" elevation="1"/>
  <line stroke="red" x1="888" y1="146" x2="900" y2="133"/>
  <line stroke="red" x1="888" y1="146" x2="900" y2="167" opacity="0.8" fold-step="1"/>
  <line stroke="blue" x1="567" y1="700" x2="533" y2="733"/>
  <line stroke="blue" x1="567" y1="700" x2="833" y2="700" elevation="0"/>
  <line stroke="red" x1="700" y1="567" x2="733" y2="533"/>
  <line stroke="blue" x1="700" y1="567" x2="767" y2="567" elevation="0"/>
  <line stroke="red" x1="533" y1="733" x2="500" y2="767"/>
  <line stroke="red" x1="533" y1="733" x2="800" y2="733" elevation="1"/>
  <line stroke="blue" x1="733" y1="533" x2="767" y2="533"/>
  <line stroke="red" x1="733" y1="533" x2="767" y2="567"/>
  <line stroke="blue" x1="267" y1="867" x2="267" y2="900"/>
  <line stroke="red" x1="267" y1="867" x2="500" y2="867" elevation="1"/>
  <line stroke="red" x1="867" y1="267" x2="867" y2="300"/>
  <line stroke="blue" x1="867" y1="267" x2="900" y2="267" elevation="0"/>
  <line stroke="blue" x1="900" y1="133" x2="900" y2="167" elevation="0"/>
  <line stroke="blue" x1="900" y1="133" x2="933" y2="100"/>
  <line stroke="blue" x1="167" y1="900" x2="267" y2="900" elevation="0"/>
  <line stroke="blue" x1="167" y1="900" x2="133" y2="933"/>
  <line stroke="blue" x1="900" y1="167" x2="900" y2="200" elevation="0"/>
  <line stroke="blue" x1="900" y1="167" x2="933" y2="109" opacity="0.8" fold-step="1"/>
  <line stroke="blue" x1="500" y1="767" x2="500" y2="800"/>
  <line stroke="blue" x1="500" y1="767" x2="767" y2="767" elevation="0"/>
  <line stroke="red" x1="867" y1="300" x2="900" y2="267"/>
  <line stroke="blue" x1="867" y1="300" x2="900" y2="300"/>
  <line stroke="red" x1="867" y1="300" x2="867" y2="400" elevation="1"/>
  <line stroke="blue" x1="900" y1="200" x2="933" y2="233"/>
  <line stroke="red" x1="833" y1="400" x2="867" y2="400"/>
  <line stroke="blue" x1="833" y1="400" x2="833" y2="533" elevation="0"/>
  <line stroke="red" x1="924" y1="33" x2="933" y2="33" elevation="1"/>
  <line stroke="red" x1="924" y1="100" x2="933" y2="100" elevation="1"/>
  <line stroke="blue" x1="0" y1="933" x2="33" y2="933"/>
  <line stroke="black" x1="0" y1="933" x2="0" y2="1000"/>
  <line stroke="red" x1="933" y1="0" x2="933" y2="33"/>
  <line stroke="black" x1="933" y1="0" x2="1000" y2="0"/>
  <line stroke="red" x1="33" y1="933" x2="67" y2="933"/>
  <line stroke="blue" x1="33" y1="933" x2="33" y2="1000" elevation="0"/>
  <line stroke="blue" x1="933" y1="33" x2="933" y2="67"/>
  <line stroke="red" x1="933" y1="33" x2="942" y2="33" elevation="1"/>
  <line stroke="blue" x1="767" y1="533" x2="767" y2="567" elevation="0"/>
  <line stroke="red" x1="767" y1="533" x2="800" y2="533"/>
  <line stroke="blue" x1="67" y1="933" x2="100" y2="933"/>
  <line stroke="red" x1="67" y1="933" x2="67" y2="1000" elevation="1"/>
  <line stroke="red" x1="933" y1="67" x2="933" y2="100"/>
  <line stroke="blue" x1="933" y1="67" x2="1000" y2="67" elevation="0"/>
  <line stroke="red" x1="100" y1="933" x2="133" y2="933"/>
  <line stroke="blue" x1="100" y1="933" x2="100" y2="1000" elevation="0"/>
  <line stroke="red" x1="933" y1="100" x2="933" y2="109" elevation="1"/>
  <line stroke="red" x1="933" y1="100" x2="942" y2="100" elevation="1"/>
  <line stroke="blue" x1="933" y1="100" x2="967" y2="133"/>
  <line stroke="red" x1="267" y1="900" x2="267" y2="933"/>
  <line stroke="blue" x1="267" y1="900" x2="500" y2="900" elevation="0"/>
  <line stroke="blue" x1="900" y1="267" x2="900" y2="300" elevation="0"/>
  <line stroke="blue" x1="900" y1="267" x2="933" y2="233"/>
  <line stroke="red" x1="933" y1="109" x2="933" y2="233" elevation="1"/>
  <line stroke="red" x1="933" y1="109" x2="967" y2="167" opacity="0.8" fold-step="1"/>
  <line stroke="red" x1="133" y1="933" x2="267" y2="933" elevation="1"/>
  <line stroke="blue" x1="133" y1="933" x2="167" y2="967"/>
  <line stroke="red" x1="133" y1="933" x2="133" y2="1000" elevation="1"/>
  <line stroke="red" x1="942" y1="33" x2="1000" y2="0" opacity="0.8" fold-step="1"/>
  <line stroke="red" x1="942" y1="33" x2="1000" y2="33" elevation="1"/>
  <line stroke="blue" x1="942" y1="33" x2="1000" y2="67" opacity="0.8" fold-step="1"/>
  <line stroke="red" x1="500" y1="800" x2="500" y2="833"/>
  <line stroke="red" x1="500" y1="800" x2="733" y2="800" elevation="1"/>
  <line stroke="red" x1="942" y1="100" x2="1000" y2="67" opacity="0.8" fold-step="1"/>
  <line stroke="red" x1="942" y1="100" x2="1000" y2="100" elevation="1"/>
  <line stroke="blue" x1="942" y1="100" x2="1000" y2="133" opacity="0.8" fold-step="1"/>
  <line stroke="red" x1="900" y1="300" x2="933" y2="300"/>
  <line stroke="blue" x1="900" y1="300" x2="900" y2="400" elevation="0"/>
  <line stroke="blue" x1="767" y1="567" x2="800" y2="600"/>
  <line stroke="blue" x1="867" y1="400" x2="900" y2="400"/>
  <line stroke="red" x1="867" y1="400" x2="867" y2="533" elevation="1"/>
  <line stroke="blue" x1="800" y1="533" x2="833" y2="533"/>
  <line stroke="red" x1="800" y1="533" x2="800" y2="600" elevation="1"/>
  <line stroke="red" x1="933" y1="233" x2="933" y2="300" elevation="1"/>
  <line stroke="red" x1="933" y1="233" x2="967" y2="233"/>
  <line stroke="blue" x1="267" y1="933" x2="267" y2="967"/>
  <line stroke="red" x1="267" y1="933" x2="500" y2="933" elevation="1"/>
  <line stroke="blue" x1="500" y1="833" x2="500" y2="867"/>
  <line stroke="blue" x1="500" y1="833" x2="733" y2="833" elevation="0"/>
  <line stroke="blue" x1="967" y1="133" x2="967" y2="167" elevation="0"/>
  <line stroke="red" x1="967" y1="133" x2="979" y2="146"/>
  <line stroke="blue" x1="967" y1="133" x2="1000" y2="133" elevation="0"/>
  <line stroke="blue" x1="933" y1="300" x2="967" y2="300"/>
  <line stroke="red" x1="933" y1="300" x2="933" y2="400" elevation="1"/>
  <line stroke="blue" x1="167" y1="967" x2="267" y2="967" elevation="0"/>
  <line stroke="blue" x1="167" y1="967" x2="167" y2="1000" elevation="0"/>
  <line stroke="red" x1="167" y1="967" x2="200" y2="1000"/>
  <line stroke="blue" x1="967" y1="167" x2="979" y2="146" opacity="0.8" fold-step="1"/>
  <line stroke="blue" x1="967" y1="167" x2="967" y2="233" elevation="0"/>
  <line stroke="red" x1="900" y1="400" x2="933" y2="400"/>
  <line stroke="blue" x1="900" y1="400" x2="900" y2="533" elevation="0"/>
  <line stroke="red" x1="833" y1="533" x2="867" y2="533"/>
  <line stroke="blue" x1="833" y1="533" x2="833" y2="633" elevation="0"/>
  <line stroke="red" x1="979" y1="146" x2="1000" y2="133" opacity="0.8" fold-step="1"/>
  <line stroke="red" x1="979" y1="146" x2="1000" y2="167"/>
  <line stroke="blue" x1="967" y1="233" x2="967" y2="300" elevation="0"/>
  <line stroke="blue" x1="967" y1="233" x2="1000" y2="233"/>
  <line stroke="black" x1="0" y1="1000" x2="33" y2="1000"/>
  <line stroke="red" x1="800" y1="600" x2="833" y2="633"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="33"/>
  <line stroke="black" x1="33" y1="1000" x2="67" y2="1000"/>
  <line stroke="black" x1="1000" y1="33" x2="1000" y2="67"/>
  <line stroke="red" x1="500" y1="867" x2="500" y2="900"/>
  <line stroke="red" x1="500" y1="867" x2="733" y2="867" elevation="1"/>
  <line stroke="black" x1="67" y1="1000" x2="100" y2="1000"/>
  <line stroke="black" x1="1000" y1="67" x2="1000" y2="100"/>
  <line stroke="red" x1="267" y1="967" x2="267" y2="1000"/>
  <line stroke="blue" x1="267" y1="967" x2="500" y2="967" elevation="0"/>
  <line stroke="black" x1="100" y1="1000" x2="133" y2="1000"/>
  <line stroke="black" x1="1000" y1="100" x2="1000" y2="133"/>
  <line stroke="black" x1="133" y1="1000" x2="167" y2="1000"/>
  <line stroke="black" x1="1000" y1="133" x2="1000" y2="167"/>
  <line stroke="red" x1="967" y1="300" x2="1000" y2="300"/>
  <line stroke="blue" x1="967" y1="300" x2="967" y2="400" elevation="0"/>
  <line stroke="black" x1="167" y1="1000" x2="200" y2="1000"/>
  <line stroke="black" x1="1000" y1="167" x2="1000" y2="233"/>
  <line stroke="blue" x1="933" y1="400" x2="967" y2="400"/>
  <line stroke="red" x1="933" y1="400" x2="933" y2="533" elevation="1"/>
  <line stroke="blue" x1="867" y1="533" x2="900" y2="533"/>
  <line stroke="red" x1="867" y1="533" x2="867" y2="667" elevation="1"/>
  <line stroke="black" x1="200" y1="1000" x2="267" y2="1000"/>
  <line stroke="black" x1="1000" y1="233" x2="1000" y2="300"/>
  <line stroke="blue" x1="500" y1="900" x2="500" y2="933"/>
  <line stroke="blue" x1="500" y1="900" x2="733" y2="900" elevation="0"/>
  <line stroke="black" x1="267" y1="1000" x2="500" y2="1000"/>
  <line stroke="black" x1="1000" y1="300" x2="1000" y2="400"/>
  <line stroke="red" x1="900" y1="533" x2="933" y2="533"/>
  <line stroke="blue" x1="900" y1="533" x2="900" y2="667" elevation="0"/>
  <line stroke="blue" x1="833" y1="633" x2="867" y2="667"/>
  <line stroke="red" x1="967" y1="400" x2="1000" y2="400"/>
  <line stroke="blue" x1="967" y1="400" x2="967" y2="533" elevation="0"/>
  <line stroke="red" x1="500" y1="933" x2="500" y2="967"/>
  <line stroke="red" x1="500" y1="933" x2="733" y2="933" elevation="1"/>
  <line stroke="blue" x1="933" y1="533" x2="967" y2="533"/>
  <line stroke="red" x1="933" y1="533" x2="933" y2="667" elevation="1"/>
  <line stroke="black" x1="1000" y1="400" x2="1000" y2="533"/>
  <line stroke="red" x1="767" y1="767" x2="733" y2="800"/>
  <line stroke="blue" x1="767" y1="767" x2="800" y2="733"/>
  <line stroke="blue" x1="767" y1="767" x2="767" y2="833" elevation="0"/>
  <line stroke="blue" x1="733" y1="800" x2="733" y2="833"/>
  <line stroke="red" x1="733" y1="800" x2="767" y2="833"/>
  <line stroke="red" x1="800" y1="733" x2="833" y2="700"/>
  <line stroke="red" x1="800" y1="733" x2="800" y2="867" elevation="1"/>
  <line stroke="blue" x1="833" y1="700" x2="867" y2="667"/>
  <line stroke="blue" x1="833" y1="700" x2="833" y2="900" elevation="0"/>
  <line stroke="blue" x1="500" y1="967" x2="500" y2="1000"/>
  <line stroke="blue" x1="500" y1="967" x2="733" y2="967" elevation="0"/>
  <line stroke="red" x1="867" y1="667" x2="900" y2="667"/>
  <line stroke="red" x1="867" y1="667" x2="867" y2="933" elevation="1"/>
  <line stroke="red" x1="967" y1="533" x2="1000" y2="533"/>
  <line stroke="blue" x1="967" y1="533" x2="967" y2="667" elevation="0"/>
  <line stroke="blue" x1="733" y1="833" x2="767" y2="833" elevation="0"/>
  <line stroke="red" x1="733" y1="833" x2="733" y2="867"/>
  <line stroke="black" x1="500" y1="1000" x2="733" y2="1000"/>
  <line stroke="blue" x1="900" y1="667" x2="933" y2="667"/>
  <line stroke="blue" x1="900" y1="667" x2="900" y2="933" elevation="0"/>
  <line stroke="blue" x1="767" y1="833" x2="800" y2="867"/>
  <line stroke="black" x1="1000" y1="533" x2="1000" y2="667"/>
  <line stroke="blue" x1="733" y1="867" x2="733" y2="900"/>
  <line stroke="red" x1="733" y1="867" x2="800" y2="867" elevation="1"/>
  <line stroke="red" x1="933" y1="667" x2="967" y2="667"/>
  <line stroke="red" x1="933" y1="667" x2="933" y2="933" elevation="1"/>
  <line stroke="red" x1="733" y1="900" x2="733" y2="933"/>
  <line stroke="blue" x1="733" y1="900" x2="833" y2="900" elevation="0"/>
  <line stroke="blue" x1="967" y1="667" x2="1000" y2="667"/>
  <line stroke="blue" x1="967" y1="667" x2="967" y2="933" elevation="0"/>
  <line stroke="red" x1="800" y1="867" x2="833" y2="900"/>
  <line stroke="blue" x1="733" y1="933" x2="733" y2="967"/>
  <line stroke="red" x1="733" y1="933" x2="867" y2="933" elevation="1"/>
  <line stroke="black" x1="1000" y1="667" x2="1000" y2="933"/>
  <line stroke="red" x1="733" y1="967" x2="733" y2="1000"/>
  <line stroke="blue" x1="733" y1="967" x2="833" y2="967" elevation="0"/>
  <line stroke="blue" x1="833" y1="900" x2="867" y2="933"/>
  <line stroke="black" x1="733" y1="1000" x2="800" y2="1000"/>
  <line stroke="blue" x1="867" y1="933" x2="833" y2="967"/>
  <line stroke="red" x1="867" y1="933" x2="900" y2="933"/>
  <line stroke="red" x1="867" y1="933" x2="867" y2="1000" elevation="1"/>
  <line stroke="red" x1="833" y1="967" x2="800" y2="1000"/>
  <line stroke="blue" x1="833" y1="967" x2="833" y2="1000" elevation="0"/>
  <line stroke="black" x1="800" y1="1000" x2="833" y2="1000"/>
  <line stroke="blue" x1="900" y1="933" x2="933" y2="933"/>
  <line stroke="blue" x1="900" y1="933" x2="900" y2="1000" elevation="0"/>
  <line stroke="black" x1="833" y1="1000" x2="867" y2="1000"/>
  <line stroke="red" x1="933" y1="933" x2="967" y2="933"/>
  <line stroke="red" x1="933" y1="933" x2="933" y2="1000" elevation="1"/>
  <line stroke="black" x1="867" y1="1000" x2="900" y2="1000"/>
  <line stroke="blue" x1="967" y1="933" x2="1000" y2="933"/>
  <line stroke="blue" x1="967" y1="933" x2="967" y2="1000" elevation="0"/>
  <line stroke="black" x1="900" y1="1000" x2="933" y2="1000"/>
  <line stroke="black" x1="933" y1="1000" x2="967" y2="1000"/>
  <line stroke="black" x1="1000" y1="933" x2="1000" y2="1000"/>
  <line stroke="black" x1="967" y1="1000" x2="1000" y2="1000"/>
</svg>""",
        zigzag_creases=[
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=1.0, y=-6.938893903907228e-17, require_on_paper=True
                ),
                p2_init=point_utils.Point(
                    x=0.9422649730810374,
                    y=0.033333333333333263,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.MOUNTAIN,
                fold_percent=1.0,
                fold_step=0,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.9422649730810374,
                    y=0.033333333333333263,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=1.0, y=0.0666666666666666, require_on_paper=True
                ),
                fold=crease_utils.Fold.VALLEY,
                fold_percent=1.0,
                fold_step=0,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=1.0, y=0.0666666666666666, require_on_paper=True
                ),
                p2_init=point_utils.Point(
                    x=0.9422649730810374,
                    y=0.09999999999999992,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.MOUNTAIN,
                fold_percent=1.0,
                fold_step=0,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.9422649730810374,
                    y=0.09999999999999992,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=1.0, y=0.13333333333333325, require_on_paper=True
                ),
                fold=crease_utils.Fold.VALLEY,
                fold_percent=1.0,
                fold_step=0,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=1.0, y=0.13333333333333325, require_on_paper=True
                ),
                p2_init=point_utils.Point(
                    x=0.9788675134594814,
                    y=0.14553418012614783,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.MOUNTAIN,
                fold_percent=1.0,
                fold_step=0,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.9788675134594814,
                    y=0.14553418012614783,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.9666666666666668,
                    y=0.1666666666666664,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.VALLEY,
                fold_percent=1.0,
                fold_step=0,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.9666666666666668,
                    y=0.1666666666666664,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.9333333333333335,
                    y=0.10893163974770381,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.MOUNTAIN,
                fold_percent=1.0,
                fold_step=0,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.9333333333333335,
                    y=0.10893163974770381,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.9000000000000001,
                    y=0.1666666666666664,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.VALLEY,
                fold_percent=1.0,
                fold_step=0,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.9000000000000001,
                    y=0.1666666666666664,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.8877991532071858,
                    y=0.145534180126148,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.MOUNTAIN,
                fold_percent=1.0,
                fold_step=0,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.8877991532071858,
                    y=0.145534180126148,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.8666666666666674,
                    y=0.13333333333333372,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.VALLEY,
                fold_percent=1.0,
                fold_step=0,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.8666666666666674,
                    y=0.13333333333333372,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.92440169358563,
                    y=0.1000000000000004,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.MOUNTAIN,
                fold_percent=1.0,
                fold_step=0,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.92440169358563,
                    y=0.1000000000000004,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.8666666666666674,
                    y=0.06666666666666707,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.VALLEY,
                fold_percent=1.0,
                fold_step=0,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.8666666666666674,
                    y=0.06666666666666707,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.92440169358563,
                    y=0.033333333333333735,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.MOUNTAIN,
                fold_percent=1.0,
                fold_step=0,
                elevation=-1,
            ),
            crease_utils.Crease(
                p1_init=point_utils.Point(
                    x=0.92440169358563,
                    y=0.033333333333333735,
                    require_on_paper=True,
                ),
                p2_init=point_utils.Point(
                    x=0.8666666666666674,
                    y=4.0245584642661925e-16,
                    require_on_paper=True,
                ),
                fold=crease_utils.Fold.VALLEY,
                fold_percent=1.0,
                fold_step=0,
                elevation=-1,
            ),
        ],
    ),
    rectangles=[
        shaping_utils.Rectangle(
            name="left_leg4",
            x_start=-0.13333333333333333,
            y_start=0.8,
            x_len=0.4,
            y_len=0.2666666666666666,
            circle_centers=None,
            require_on_paper=False,
            radius=0.18856180831641264,
        ),
        shaping_utils.Rectangle(
            name="left_leg3",
            x_start=-0.26666666666666666,
            y_start=0.5333333333333333,
            x_len=0.5333333333333333,
            y_len=0.2666666666666666,
            circle_centers=None,
            require_on_paper=False,
            radius=0.18856180831641264,
        ),
        shaping_utils.Rectangle(
            name="tail_segment1",
            x_start=0.26666666666666666,
            y_start=0.5333333333333334,
            x_len=0.46666666666666673,
            y_len=1.1666666666666665,
            circle_centers=None,
            require_on_paper=False,
            radius=0.32998316455372223,
        ),
        shaping_utils.Rectangle(
            name="tail_segment2",
            x_start=0.3333333333333333,
            y_start=0.6000000000000001,
            x_len=0.3333333333333334,
            y_len=1.0333333333333332,
            circle_centers=None,
            require_on_paper=False,
            radius=0.23570226039551592,
        ),
        shaping_utils.Rectangle(
            name="tail_segment3",
            x_start=0.39999999999999997,
            y_start=0.6666666666666667,
            x_len=0.20000000000000012,
            y_len=0.8999999999999999,
            circle_centers=None,
            require_on_paper=False,
            radius=0.14142135623730961,
        ),
        shaping_utils.Rectangle(
            name="stinger",
            x_start=0.4666666666666666,
            y_start=0.7333333333333333,
            x_len=0.06666666666666682,
            y_len=0.7666666666666667,
            circle_centers=None,
            require_on_paper=False,
            radius=0.04714045207910328,
        ),
        shaping_utils.Rectangle(
            name="left_pincer_upper",
            x_start=-0.06666666666666667,
            y_start=0.16666666666666669,
            x_len=0.2,
            y_len=0.1333333333333333,
            circle_centers=None,
            require_on_paper=False,
            radius=0.09428090415820632,
        ),
        shaping_utils.Rectangle(
            name="left_pincer_lower",
            x_start=0.0,
            y_start=-0.1,
            x_len=0.13333333333333333,
            y_len=0.2666666666666667,
            circle_centers=None,
            require_on_paper=False,
            radius=0.09428090415820634,
        ),
        shaping_utils.Rectangle(
            name="left_leg1",
            x_start=0.2333333333333333,
            y_start=-0.13333333333333333,
            x_len=0.2666666666666667,
            y_len=0.26666666666666666,
            circle_centers=None,
            require_on_paper=False,
            radius=0.18856180831641267,
        ),
        shaping_utils.Rectangle(
            name="left_leg2",
            x_start=0.2333333333333333,
            y_start=0.13333333333333333,
            x_len=0.2666666666666667,
            y_len=0.2666666666666667,
            circle_centers=None,
            require_on_paper=False,
            radius=0.18856180831641273,
        ),
        shaping_utils.Rectangle(
            name="right_leg4",
            x_start=0.7333333333333334,
            y_start=0.8,
            x_len=0.3999999999999999,
            y_len=0.2666666666666666,
            circle_centers=None,
            require_on_paper=False,
            radius=0.18856180831641264,
        ),
        shaping_utils.Rectangle(
            name="right_leg3",
            x_start=0.7333333333333334,
            y_start=0.5333333333333333,
            x_len=0.5333333333333332,
            y_len=0.2666666666666666,
            circle_centers=None,
            require_on_paper=False,
            radius=0.18856180831641264,
        ),
        shaping_utils.Rectangle(
            name="right_pincer_upper",
            x_start=0.8666666666666667,
            y_start=0.16666666666666669,
            x_len=0.19999999999999996,
            y_len=0.1333333333333333,
            circle_centers=None,
            require_on_paper=False,
            radius=0.09428090415820632,
        ),
        shaping_utils.Rectangle(
            name="right_pincer_lower",
            x_start=0.8666666666666667,
            y_start=-0.1,
            x_len=0.1333333333333333,
            y_len=0.2666666666666667,
            circle_centers=None,
            require_on_paper=False,
            radius=0.09428090415820632,
        ),
        shaping_utils.Rectangle(
            name="right_leg1",
            x_start=0.5,
            y_start=-0.13333333333333333,
            x_len=0.2666666666666666,
            y_len=0.26666666666666666,
            circle_centers=None,
            require_on_paper=False,
            radius=0.18856180831641264,
        ),
        shaping_utils.Rectangle(
            name="right_leg2",
            x_start=0.5,
            y_start=0.13333333333333333,
            x_len=0.2666666666666666,
            y_len=0.2666666666666667,
            circle_centers=None,
            require_on_paper=False,
            radius=0.18856180831641264,
        ),
    ],
    narrowed_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="black" x1="0" y1="0" x2="0" y2="33"/>
  <line stroke="black" x1="0" y1="0" x2="67" y2="0"/>
  <line stroke="black" x1="0" y1="33" x2="0" y2="67"/>
  <line stroke="red" x1="0" y1="33" x2="67" y2="33" elevation="1"/>
  <line stroke="blue" x1="0" y1="67" x2="67" y2="67" elevation="0"/>
  <line stroke="black" x1="0" y1="67" x2="0" y2="100"/>
  <line stroke="red" x1="67" y1="0" x2="67" y2="33"/>
  <line stroke="black" x1="67" y1="0" x2="367" y2="0"/>
  <line stroke="blue" x1="67" y1="33" x2="67" y2="67"/>
  <line stroke="red" x1="67" y1="33" x2="333" y2="33" elevation="1"/>
  <line stroke="red" x1="67" y1="67" x2="67" y2="100"/>
  <line stroke="blue" x1="67" y1="67" x2="300" y2="67" elevation="0"/>
  <line stroke="red" x1="0" y1="100" x2="67" y2="100" elevation="1"/>
  <line stroke="black" x1="0" y1="100" x2="0" y2="133"/>
  <line stroke="blue" x1="67" y1="100" x2="33" y2="133"/>
  <line stroke="blue" x1="67" y1="100" x2="100" y2="133"/>
  <line stroke="red" x1="67" y1="100" x2="67" y2="167" elevation="1"/>
  <line stroke="red" x1="67" y1="100" x2="267" y2="100" elevation="1"/>
  <line stroke="blue" x1="0" y1="133" x2="33" y2="133" elevation="0"/>
  <line stroke="black" x1="0" y1="133" x2="0" y2="167"/>
  <line stroke="red" x1="33" y1="133" x2="0" y2="167"/>
  <line stroke="blue" x1="33" y1="133" x2="33" y2="167" elevation="0"/>
  <line stroke="blue" x1="100" y1="133" x2="100" y2="167" elevation="0"/>
  <line stroke="red" x1="100" y1="133" x2="133" y2="167"/>
  <line stroke="blue" x1="100" y1="133" x2="233" y2="133" elevation="0"/>
  <line stroke="red" x1="0" y1="167" x2="33" y2="167"/>
  <line stroke="black" x1="0" y1="167" x2="0" y2="233"/>
  <line stroke="blue" x1="33" y1="167" x2="67" y2="167"/>
  <line stroke="blue" x1="33" y1="167" x2="33" y2="233" elevation="0"/>
  <line stroke="red" x1="67" y1="167" x2="100" y2="167"/>
  <line stroke="red" x1="67" y1="167" x2="67" y2="233" elevation="1"/>
  <line stroke="blue" x1="100" y1="167" x2="133" y2="167"/>
  <line stroke="blue" x1="100" y1="167" x2="100" y2="200" elevation="0"/>
  <line stroke="red" x1="133" y1="167" x2="100" y2="200"/>
  <line stroke="red" x1="133" y1="167" x2="267" y2="167" elevation="1"/>
  <line stroke="blue" x1="100" y1="200" x2="67" y2="233"/>
  <line stroke="blue" x1="100" y1="200" x2="300" y2="200" elevation="0"/>
  <line stroke="blue" x1="0" y1="233" x2="33" y2="233"/>
  <line stroke="black" x1="0" y1="233" x2="0" y2="533"/>
  <line stroke="red" x1="33" y1="233" x2="67" y2="233"/>
  <line stroke="blue" x1="33" y1="233" x2="33" y2="533" elevation="0"/>
  <line stroke="blue" x1="67" y1="233" x2="100" y2="267"/>
  <line stroke="red" x1="67" y1="233" x2="333" y2="233" elevation="1"/>
  <line stroke="red" x1="67" y1="233" x2="67" y2="533" elevation="1"/>
  <line stroke="blue" x1="233" y1="133" x2="267" y2="100"/>
  <line stroke="red" x1="233" y1="133" x2="267" y2="133"/>
  <line stroke="blue" x1="233" y1="133" x2="267" y2="167"/>
  <line stroke="red" x1="100" y1="267" x2="133" y2="300"/>
  <line stroke="blue" x1="100" y1="267" x2="367" y2="267" elevation="0"/>
  <line stroke="blue" x1="100" y1="267" x2="100" y2="533" elevation="0"/>
  <line stroke="red" x1="267" y1="100" x2="267" y2="133" elevation="1"/>
  <line stroke="red" x1="267" y1="100" x2="300" y2="67"/>
  <line stroke="red" x1="267" y1="133" x2="267" y2="167" elevation="1"/>
  <line stroke="blue" x1="267" y1="133" x2="300" y2="133"/>
  <line stroke="blue" x1="300" y1="67" x2="300" y2="133" elevation="0"/>
  <line stroke="blue" x1="300" y1="67" x2="333" y2="33"/>
  <line stroke="red" x1="267" y1="167" x2="300" y2="200"/>
  <line stroke="blue" x1="133" y1="300" x2="167" y2="333"/>
  <line stroke="red" x1="133" y1="300" x2="333" y2="300" elevation="1"/>
  <line stroke="red" x1="133" y1="300" x2="133" y2="533" elevation="1"/>
  <line stroke="red" x1="300" y1="133" x2="333" y2="133"/>
  <line stroke="blue" x1="300" y1="133" x2="300" y2="200" elevation="0"/>
  <line stroke="red" x1="333" y1="33" x2="333" y2="133" elevation="1"/>
  <line stroke="red" x1="333" y1="33" x2="367" y2="0"/>
  <line stroke="blue" x1="333" y1="133" x2="367" y2="133"/>
  <line stroke="red" x1="333" y1="133" x2="333" y2="233" elevation="1"/>
  <line stroke="blue" x1="300" y1="200" x2="333" y2="233"/>
  <line stroke="blue" x1="367" y1="0" x2="367" y2="133" elevation="0"/>
  <line stroke="red" x1="367" y1="0" x2="400" y2="33"/>
  <line stroke="black" x1="367" y1="0" x2="500" y2="0"/>
  <line stroke="red" x1="167" y1="333" x2="200" y2="367"/>
  <line stroke="blue" x1="167" y1="333" x2="300" y2="333" elevation="0"/>
  <line stroke="blue" x1="167" y1="333" x2="167" y2="533" elevation="0"/>
  <line stroke="red" x1="367" y1="133" x2="400" y2="133"/>
  <line stroke="blue" x1="367" y1="133" x2="367" y2="267" elevation="0"/>
  <line stroke="red" x1="400" y1="33" x2="400" y2="133" elevation="1"/>
  <line stroke="blue" x1="400" y1="33" x2="433" y2="67"/>
  <line stroke="red" x1="400" y1="33" x2="500" y2="33" elevation="1"/>
  <line stroke="red" x1="333" y1="233" x2="367" y2="267"/>
  <line stroke="red" x1="200" y1="367" x2="267" y2="367" elevation="1"/>
  <line stroke="blue" x1="200" y1="367" x2="233" y2="400"/>
  <line stroke="red" x1="200" y1="367" x2="200" y2="533" elevation="1"/>
  <line stroke="blue" x1="400" y1="133" x2="433" y2="133"/>
  <line stroke="red" x1="400" y1="133" x2="400" y2="233" elevation="1"/>
  <line stroke="blue" x1="433" y1="67" x2="433" y2="133" elevation="0"/>
  <line stroke="red" x1="433" y1="67" x2="467" y2="100"/>
  <line stroke="blue" x1="433" y1="67" x2="500" y2="67" elevation="0"/>
  <line stroke="blue" x1="300" y1="333" x2="333" y2="300"/>
  <line stroke="red" x1="300" y1="333" x2="267" y2="367"/>
  <line stroke="blue" x1="300" y1="333" x2="300" y2="400" elevation="0"/>
  <line stroke="red" x1="333" y1="300" x2="367" y2="267"/>
  <line stroke="red" x1="333" y1="300" x2="333" y2="400" elevation="1"/>
  <line stroke="blue" x1="433" y1="133" x2="433" y2="200" elevation="0"/>
  <line stroke="red" x1="433" y1="133" x2="467" y2="133"/>
  <line stroke="blue" x1="267" y1="367" x2="233" y2="400"/>
  <line stroke="red" x1="267" y1="367" x2="267" y2="400" elevation="1"/>
  <line stroke="red" x1="367" y1="267" x2="400" y2="233"/>
  <line stroke="red" x1="367" y1="267" x2="400" y2="300"/>
  <line stroke="red" x1="367" y1="267" x2="367" y2="400" elevation="0"/>
  <line stroke="blue" x1="367" y1="267" x2="500" y2="267" elevation="0"/>
  <line stroke="red" x1="233" y1="400" x2="267" y2="400"/>
  <line stroke="blue" x1="233" y1="400" x2="233" y2="533" elevation="0"/>
  <line stroke="blue" x1="400" y1="233" x2="433" y2="200"/>
  <line stroke="red" x1="400" y1="233" x2="500" y2="233" elevation="1"/>
  <line stroke="red" x1="433" y1="200" x2="467" y2="167"/>
  <line stroke="blue" x1="433" y1="200" x2="500" y2="200" elevation="0"/>
  <line stroke="red" x1="467" y1="100" x2="467" y2="133" elevation="1"/>
  <line stroke="red" x1="467" y1="100" x2="500" y2="100" elevation="1"/>
  <line stroke="blue" x1="467" y1="100" x2="500" y2="133"/>
  <line stroke="blue" x1="267" y1="400" x2="300" y2="400"/>
  <line stroke="red" x1="267" y1="400" x2="267" y2="533" elevation="1"/>
  <line stroke="red" x1="467" y1="133" x2="467" y2="167" elevation="1"/>
  <line stroke="blue" x1="467" y1="133" x2="500" y2="133"/>
  <line stroke="blue" x1="467" y1="167" x2="500" y2="133"/>
  <line stroke="red" x1="467" y1="167" x2="500" y2="167" elevation="1"/>
  <line stroke="red" x1="300" y1="400" x2="333" y2="400"/>
  <line stroke="blue" x1="300" y1="400" x2="300" y2="567" elevation="0"/>
  <line stroke="blue" x1="400" y1="300" x2="433" y2="333"/>
  <line stroke="red" x1="400" y1="300" x2="400" y2="400" elevation="1"/>
  <line stroke="red" x1="400" y1="300" x2="500" y2="300" elevation="1"/>
  <line stroke="blue" x1="500" y1="0" x2="500" y2="33"/>
  <line stroke="black" x1="500" y1="0" x2="633" y2="0"/>
  <line stroke="red" x1="500" y1="33" x2="500" y2="67"/>
  <line stroke="red" x1="500" y1="33" x2="600" y2="33" elevation="1"/>
  <line stroke="blue" x1="500" y1="67" x2="500" y2="100"/>
  <line stroke="blue" x1="500" y1="67" x2="567" y2="67" elevation="0"/>
  <line stroke="red" x1="500" y1="100" x2="500" y2="133"/>
  <line stroke="red" x1="500" y1="100" x2="533" y2="100" elevation="1"/>
  <line stroke="red" x1="500" y1="133" x2="500" y2="167"/>
  <line stroke="blue" x1="500" y1="133" x2="533" y2="100"/>
  <line stroke="red" x1="500" y1="133" x2="533" y2="133"/>
  <line stroke="blue" x1="500" y1="133" x2="533" y2="167"/>
  <line stroke="blue" x1="333" y1="400" x2="367" y2="400"/>
  <line stroke="red" x1="333" y1="400" x2="333" y2="600" elevation="1"/>
  <line stroke="blue" x1="500" y1="167" x2="500" y2="200"/>
  <line stroke="red" x1="500" y1="167" x2="533" y2="167" elevation="1"/>
  <line stroke="red" x1="0" y1="533" x2="33" y2="533"/>
  <line stroke="black" x1="0" y1="533" x2="0" y2="667"/>
  <line stroke="blue" x1="33" y1="533" x2="67" y2="533"/>
  <line stroke="blue" x1="33" y1="533" x2="33" y2="667" elevation="0"/>
  <line stroke="red" x1="67" y1="533" x2="100" y2="533"/>
  <line stroke="red" x1="67" y1="533" x2="67" y2="667" elevation="1"/>
  <line stroke="red" x1="500" y1="200" x2="500" y2="233"/>
  <line stroke="blue" x1="500" y1="200" x2="567" y2="200" elevation="0"/>
  <line stroke="blue" x1="100" y1="533" x2="133" y2="533"/>
  <line stroke="blue" x1="100" y1="533" x2="100" y2="667" elevation="0"/>
  <line stroke="red" x1="533" y1="100" x2="533" y2="133" elevation="1"/>
  <line stroke="red" x1="533" y1="100" x2="567" y2="67"/>
  <line stroke="blue" x1="367" y1="400" x2="400" y2="400"/>
  <line stroke="blue" x1="367" y1="400" x2="367" y2="633" elevation="0"/>
  <line stroke="blue" x1="433" y1="333" x2="433" y2="400" elevation="0"/>
  <line stroke="red" x1="433" y1="333" x2="467" y2="367"/>
  <line stroke="blue" x1="433" y1="333" x2="500" y2="333" elevation="0"/>
  <line stroke="red" x1="133" y1="533" x2="167" y2="533"/>
  <line stroke="red" x1="133" y1="533" x2="133" y2="667" elevation="1"/>
  <line stroke="red" x1="533" y1="133" x2="533" y2="167" elevation="1"/>
  <line stroke="blue" x1="533" y1="133" x2="567" y2="133"/>
  <line stroke="blue" x1="500" y1="233" x2="500" y2="267"/>
  <line stroke="red" x1="500" y1="233" x2="600" y2="233" elevation="1"/>
  <line stroke="blue" x1="167" y1="533" x2="200" y2="533"/>
  <line stroke="blue" x1="167" y1="533" x2="167" y2="633" elevation="0"/>
  <line stroke="red" x1="533" y1="167" x2="567" y2="200"/>
  <line stroke="red" x1="400" y1="400" x2="433" y2="400"/>
  <line stroke="red" x1="400" y1="400" x2="400" y2="667" elevation="1"/>
  <line stroke="red" x1="500" y1="267" x2="500" y2="300"/>
  <line stroke="blue" x1="500" y1="267" x2="633" y2="267" elevation="0"/>
  <line stroke="red" x1="200" y1="533" x2="233" y2="533"/>
  <line stroke="red" x1="200" y1="533" x2="200" y2="600" elevation="1"/>
  <line stroke="blue" x1="567" y1="67" x2="567" y2="133" elevation="0"/>
  <line stroke="blue" x1="567" y1="67" x2="600" y2="33"/>
  <line stroke="blue" x1="233" y1="533" x2="267" y2="533"/>
  <line stroke="blue" x1="233" y1="533" x2="233" y2="567" elevation="0"/>
  <line stroke="blue" x1="567" y1="133" x2="567" y2="200" elevation="0"/>
  <line stroke="red" x1="567" y1="133" x2="600" y2="133"/>
  <line stroke="blue" x1="500" y1="300" x2="500" y2="333"/>
  <line stroke="red" x1="500" y1="300" x2="600" y2="300" elevation="1"/>
  <line stroke="blue" x1="433" y1="400" x2="467" y2="400"/>
  <line stroke="blue" x1="433" y1="400" x2="433" y2="700" elevation="0"/>
  <line stroke="red" x1="467" y1="367" x2="467" y2="400" elevation="1"/>
  <line stroke="red" x1="467" y1="367" x2="500" y2="367" elevation="1"/>
  <line stroke="blue" x1="467" y1="367" x2="500" y2="400"/>
  <line stroke="red" x1="267" y1="533" x2="233" y2="567"/>
  <line stroke="red" x1="267" y1="533" x2="300" y2="567"/>
  <line stroke="red" x1="500" y1="333" x2="500" y2="367"/>
  <line stroke="blue" x1="500" y1="333" x2="567" y2="333" elevation="0"/>
  <line stroke="red" x1="600" y1="33" x2="600" y2="133" elevation="1"/>
  <line stroke="red" x1="600" y1="33" x2="633" y2="0"/>
  <line stroke="blue" x1="567" y1="200" x2="600" y2="233"/>
  <line stroke="blue" x1="233" y1="567" x2="200" y2="600"/>
  <line stroke="blue" x1="233" y1="567" x2="300" y2="567" elevation="0"/>
  <line stroke="red" x1="600" y1="133" x2="600" y2="233" elevation="1"/>
  <line stroke="blue" x1="600" y1="133" x2="633" y2="133"/>
  <line stroke="red" x1="467" y1="400" x2="500" y2="400"/>
  <line stroke="red" x1="467" y1="400" x2="467" y2="733" elevation="1"/>
  <line stroke="blue" x1="500" y1="367" x2="500" y2="400"/>
  <line stroke="red" x1="500" y1="367" x2="533" y2="367" elevation="1"/>
  <line stroke="red" x1="200" y1="600" x2="167" y2="633"/>
  <line stroke="red" x1="200" y1="600" x2="333" y2="600" elevation="1"/>
  <line stroke="blue" x1="633" y1="0" x2="633" y2="133" elevation="0"/>
  <line stroke="red" x1="633" y1="0" x2="667" y2="33"/>
  <line stroke="black" x1="633" y1="0" x2="767" y2="0"/>
  <line stroke="blue" x1="500" y1="400" x2="533" y2="367"/>
  <line stroke="red" x1="500" y1="400" x2="533" y2="400"/>
  <line stroke="blue" x1="500" y1="400" x2="500" y2="767" elevation="0"/>
  <line stroke="blue" x1="300" y1="567" x2="333" y2="600"/>
  <line stroke="red" x1="600" y1="233" x2="633" y2="267"/>
  <line stroke="blue" x1="633" y1="133" x2="667" y2="133"/>
  <line stroke="red" x1="633" y1="133" x2="633" y2="267" elevation="0"/>
  <line stroke="red" x1="533" y1="367" x2="567" y2="333"/>
  <line stroke="red" x1="533" y1="367" x2="533" y2="400" elevation="1"/>
  <line stroke="blue" x1="167" y1="633" x2="133" y2="667"/>
  <line stroke="blue" x1="167" y1="633" x2="367" y2="633" elevation="0"/>
  <line stroke="blue" x1="567" y1="333" x2="600" y2="300"/>
  <line stroke="blue" x1="567" y1="333" x2="567" y2="400" elevation="0"/>
  <line stroke="blue" x1="533" y1="400" x2="567" y2="400"/>
  <line stroke="red" x1="533" y1="400" x2="533" y2="733" elevation="1"/>
  <line stroke="blue" x1="0" y1="667" x2="33" y2="667"/>
  <line stroke="black" x1="0" y1="667" x2="0" y2="933"/>
  <line stroke="red" x1="33" y1="667" x2="67" y2="667"/>
  <line stroke="blue" x1="33" y1="667" x2="33" y2="933" elevation="0"/>
  <line stroke="red" x1="667" y1="33" x2="667" y2="133" elevation="1"/>
  <line stroke="blue" x1="667" y1="33" x2="700" y2="67"/>
  <line stroke="red" x1="667" y1="33" x2="767" y2="33" elevation="1"/>
  <line stroke="blue" x1="67" y1="667" x2="100" y2="667"/>
  <line stroke="red" x1="67" y1="667" x2="67" y2="933" elevation="1"/>
  <line stroke="red" x1="600" y1="300" x2="633" y2="267"/>
  <line stroke="red" x1="600" y1="300" x2="600" y2="400" elevation="1"/>
  <line stroke="red" x1="100" y1="667" x2="133" y2="667"/>
  <line stroke="blue" x1="100" y1="667" x2="100" y2="933" elevation="0"/>
  <line stroke="blue" x1="133" y1="667" x2="167" y2="700"/>
  <line stroke="red" x1="133" y1="667" x2="400" y2="667" elevation="1"/>
  <line stroke="red" x1="133" y1="667" x2="133" y2="933" elevation="1"/>
  <line stroke="red" x1="667" y1="133" x2="667" y2="233" elevation="1"/>
  <line stroke="red" x1="667" y1="133" x2="700" y2="133"/>
  <line stroke="red" x1="333" y1="600" x2="367" y2="633"/>
  <line stroke="red" x1="633" y1="267" x2="667" y2="233"/>
  <line stroke="red" x1="633" y1="267" x2="667" y2="300"/>
  <line stroke="blue" x1="633" y1="267" x2="633" y2="400" elevation="0"/>
  <line stroke="blue" x1="633" y1="267" x2="767" y2="267" elevation="0"/>
  <line stroke="red" x1="567" y1="400" x2="600" y2="400"/>
  <line stroke="blue" x1="567" y1="400" x2="567" y2="700" elevation="0"/>
  <line stroke="blue" x1="700" y1="67" x2="700" y2="133" elevation="0"/>
  <line stroke="red" x1="700" y1="67" x2="733" y2="100"/>
  <line stroke="blue" x1="700" y1="67" x2="767" y2="67" elevation="0"/>
  <line stroke="blue" x1="667" y1="233" x2="700" y2="200"/>
  <line stroke="red" x1="667" y1="233" x2="767" y2="233" elevation="1"/>
  <line stroke="blue" x1="700" y1="133" x2="700" y2="200" elevation="0"/>
  <line stroke="blue" x1="700" y1="133" x2="733" y2="133"/>
  <line stroke="red" x1="167" y1="700" x2="200" y2="733"/>
  <line stroke="blue" x1="167" y1="700" x2="433" y2="700" elevation="0"/>
  <line stroke="blue" x1="167" y1="700" x2="167" y2="900" elevation="0"/>
  <line stroke="blue" x1="600" y1="400" x2="633" y2="400"/>
  <line stroke="red" x1="600" y1="400" x2="600" y2="667" elevation="1"/>
  <line stroke="red" x1="700" y1="200" x2="733" y2="167"/>
  <line stroke="blue" x1="700" y1="200" x2="767" y2="200" elevation="0"/>
  <line stroke="blue" x1="667" y1="300" x2="700" y2="333"/>
  <line stroke="red" x1="667" y1="300" x2="667" y2="400" elevation="1"/>
  <line stroke="red" x1="667" y1="300" x2="767" y2="300" elevation="1"/>
  <line stroke="blue" x1="367" y1="633" x2="400" y2="667"/>
  <line stroke="red" x1="733" y1="100" x2="733" y2="133" elevation="1"/>
  <line stroke="red" x1="733" y1="100" x2="767" y2="100" elevation="1"/>
  <line stroke="blue" x1="733" y1="100" x2="767" y2="133"/>
  <line stroke="red" x1="733" y1="133" x2="733" y2="167" elevation="1"/>
  <line stroke="red" x1="733" y1="133" x2="767" y2="133"/>
  <line stroke="red" x1="633" y1="400" x2="667" y2="400"/>
  <line stroke="blue" x1="633" y1="400" x2="633" y2="633" elevation="0"/>
  <line stroke="blue" x1="733" y1="167" x2="767" y2="133"/>
  <line stroke="red" x1="733" y1="167" x2="767" y2="167" elevation="1"/>
  <line stroke="blue" x1="200" y1="733" x2="233" y2="767"/>
  <line stroke="red" x1="200" y1="733" x2="467" y2="733" elevation="1"/>
  <line stroke="red" x1="200" y1="733" x2="200" y2="867" elevation="1"/>
  <line stroke="blue" x1="767" y1="0" x2="767" y2="33"/>
  <line stroke="black" x1="767" y1="0" x2="867" y2="0"/>
  <line stroke="red" x1="767" y1="33" x2="767" y2="67"/>
  <line stroke="red" x1="767" y1="33" x2="905" y2="33" elevation="1"/>
  <line stroke="blue" x1="767" y1="67" x2="767" y2="100"/>
  <line stroke="blue" x1="767" y1="67" x2="867" y2="67" elevation="0"/>
  <line stroke="red" x1="767" y1="100" x2="767" y2="133"/>
  <line stroke="red" x1="767" y1="100" x2="905" y2="100" elevation="1"/>
  <line stroke="blue" x1="700" y1="333" x2="700" y2="400" elevation="0"/>
  <line stroke="red" x1="700" y1="333" x2="733" y2="367"/>
  <line stroke="blue" x1="700" y1="333" x2="767" y2="333" elevation="0"/>
  <line stroke="red" x1="400" y1="667" x2="433" y2="700"/>
  <line stroke="blue" x1="667" y1="400" x2="700" y2="400"/>
  <line stroke="red" x1="667" y1="400" x2="667" y2="600" elevation="1"/>
  <line stroke="blue" x1="767" y1="133" x2="767" y2="167"/>
  <line stroke="blue" x1="767" y1="133" x2="867" y2="133" elevation="0"/>
  <line stroke="red" x1="767" y1="167" x2="767" y2="200"/>
  <line stroke="red" x1="767" y1="167" x2="867" y2="167" elevation="1"/>
  <line stroke="blue" x1="767" y1="200" x2="767" y2="233"/>
  <line stroke="blue" x1="767" y1="200" x2="867" y2="200" elevation="0"/>
  <line stroke="red" x1="233" y1="767" x2="267" y2="800"/>
  <line stroke="blue" x1="233" y1="767" x2="233" y2="833" elevation="0"/>
  <line stroke="blue" x1="233" y1="767" x2="500" y2="767" elevation="0"/>
  <line stroke="red" x1="767" y1="233" x2="767" y2="267"/>
  <line stroke="red" x1="767" y1="233" x2="867" y2="233" elevation="1"/>
  <line stroke="red" x1="700" y1="400" x2="733" y2="400"/>
  <line stroke="blue" x1="700" y1="400" x2="700" y2="567" elevation="0"/>
  <line stroke="blue" x1="767" y1="267" x2="767" y2="300"/>
  <line stroke="blue" x1="767" y1="267" x2="867" y2="267" elevation="0"/>
  <line stroke="red" x1="733" y1="367" x2="733" y2="400" elevation="1"/>
  <line stroke="red" x1="733" y1="367" x2="767" y2="367" elevation="1"/>
  <line stroke="blue" x1="733" y1="367" x2="767" y2="400"/>
  <line stroke="blue" x1="433" y1="700" x2="467" y2="733"/>
  <line stroke="red" x1="767" y1="300" x2="767" y2="333"/>
  <line stroke="red" x1="767" y1="300" x2="867" y2="300" elevation="1"/>
  <line stroke="blue" x1="733" y1="400" x2="767" y2="400"/>
  <line stroke="red" x1="733" y1="400" x2="733" y2="533" elevation="1"/>
  <line stroke="blue" x1="767" y1="333" x2="767" y2="367"/>
  <line stroke="blue" x1="767" y1="333" x2="833" y2="333" elevation="0"/>
  <line stroke="red" x1="267" y1="800" x2="233" y2="833"/>
  <line stroke="blue" x1="267" y1="800" x2="267" y2="833"/>
  <line stroke="red" x1="267" y1="800" x2="500" y2="800" elevation="1"/>
  <line stroke="red" x1="767" y1="367" x2="767" y2="400"/>
  <line stroke="red" x1="767" y1="367" x2="800" y2="367" elevation="1"/>
  <line stroke="blue" x1="233" y1="833" x2="267" y2="833" elevation="0"/>
  <line stroke="blue" x1="233" y1="833" x2="200" y2="867"/>
  <line stroke="blue" x1="767" y1="400" x2="800" y2="367"/>
  <line stroke="red" x1="767" y1="400" x2="800" y2="400"/>
  <line stroke="blue" x1="767" y1="400" x2="767" y2="533" elevation="0"/>
  <line stroke="blue" x1="867" y1="0" x2="896" y2="17" opacity="0.8" fold-step="1"/>
  <line stroke="black" x1="867" y1="0" x2="933" y2="0"/>
  <line stroke="red" x1="467" y1="733" x2="500" y2="767"/>
  <line stroke="red" x1="867" y1="67" x2="896" y2="50" opacity="0.8" fold-step="1"/>
  <line stroke="blue" x1="867" y1="67" x2="895" y2="83" opacity="0.8" fold-step="1"/>
  <line stroke="blue" x1="867" y1="67" x2="933" y2="67" elevation="0"/>
  <line stroke="red" x1="267" y1="833" x2="267" y2="867"/>
  <line stroke="blue" x1="267" y1="833" x2="500" y2="833" elevation="0"/>
  <line stroke="blue" x1="867" y1="133" x2="888" y2="146" opacity="0.8" fold-step="1"/>
  <line stroke="red" x1="867" y1="133" x2="895" y2="117" opacity="0.8" fold-step="1"/>
  <line stroke="blue" x1="867" y1="133" x2="900" y2="133" elevation="0"/>
  <line stroke="red" x1="800" y1="367" x2="800" y2="400" elevation="1"/>
  <line stroke="red" x1="800" y1="367" x2="833" y2="333"/>
  <line stroke="blue" x1="867" y1="167" x2="867" y2="200"/>
  <line stroke="red" x1="867" y1="167" x2="888" y2="146"/>
  <line stroke="red" x1="867" y1="167" x2="900" y2="200"/>
  <line stroke="red" x1="200" y1="867" x2="267" y2="867" elevation="1"/>
  <line stroke="red" x1="200" y1="867" x2="167" y2="900"/>
  <line stroke="red" x1="867" y1="200" x2="867" y2="233"/>
  <line stroke="blue" x1="867" y1="200" x2="900" y2="200" elevation="0"/>
  <line stroke="blue" x1="800" y1="400" x2="833" y2="400"/>
  <line stroke="red" x1="800" y1="400" x2="800" y2="533" elevation="1"/>
  <line stroke="blue" x1="633" y1="633" x2="600" y2="667"/>
  <line stroke="red" x1="633" y1="633" x2="667" y2="600"/>
  <line stroke="blue" x1="633" y1="633" x2="833" y2="633" elevation="0"/>
  <line stroke="blue" x1="896" y1="17" x2="905" y2="33"/>
  <line stroke="blue" x1="896" y1="17" x2="924" y2="33" opacity="0.8" fold-step="1"/>
  <line stroke="red" x1="896" y1="17" x2="933" y2="16"/>
  <line stroke="blue" x1="833" y1="333" x2="867" y2="300"/>
  <line stroke="blue" x1="833" y1="333" x2="833" y2="400" elevation="0"/>
  <line stroke="red" x1="600" y1="667" x2="567" y2="700"/>
  <line stroke="red" x1="600" y1="667" x2="867" y2="667" elevation="1"/>
  <line stroke="blue" x1="667" y1="600" x2="700" y2="567"/>
  <line stroke="red" x1="667" y1="600" x2="800" y2="600" elevation="1"/>
  <line stroke="red" x1="896" y1="50" x2="905" y2="33"/>
  <line stroke="red" x1="896" y1="50" x2="924" y2="33" opacity="0.8" fold-step="1"/>
  <line stroke="blue" x1="896" y1="50" x2="933" y2="50"/>
  <line stroke="blue" x1="867" y1="233" x2="867" y2="267"/>
  <line stroke="red" x1="867" y1="233" x2="933" y2="233" elevation="1"/>
  <line stroke="red" x1="895" y1="83" x2="905" y2="100"/>
  <line stroke="blue" x1="895" y1="83" x2="924" y2="100" opacity="0.8" fold-step="1"/>
  <line stroke="blue" x1="895" y1="83" x2="933" y2="83"/>
  <line stroke="red" x1="888" y1="146" x2="900" y2="133"/>
  <line stroke="red" x1="888" y1="146" x2="900" y2="167" opacity="0.8" fold-step="1"/>
  <line stroke="blue" x1="567" y1="700" x2="533" y2="733"/>
  <line stroke="blue" x1="567" y1="700" x2="833" y2="700" elevation="0"/>
  <line stroke="red" x1="700" y1="567" x2="733" y2="533"/>
  <line stroke="blue" x1="700" y1="567" x2="767" y2="567" elevation="0"/>
  <line stroke="blue" x1="895" y1="117" x2="905" y2="100"/>
  <line stroke="red" x1="895" y1="117" x2="916" y2="117"/>
  <line stroke="red" x1="895" y1="117" x2="924" y2="100" opacity="0.8" fold-step="1"/>
  <line stroke="red" x1="905" y1="33" x2="924" y2="33" elevation="1"/>
  <line stroke="red" x1="533" y1="733" x2="500" y2="767"/>
  <line stroke="red" x1="533" y1="733" x2="800" y2="733" elevation="1"/>
  <line stroke="blue" x1="733" y1="533" x2="767" y2="533"/>
  <line stroke="red" x1="733" y1="533" x2="767" y2="567"/>
  <line stroke="blue" x1="267" y1="867" x2="267" y2="900"/>
  <line stroke="red" x1="267" y1="867" x2="500" y2="867" elevation="1"/>
  <line stroke="red" x1="867" y1="267" x2="867" y2="300"/>
  <line stroke="blue" x1="867" y1="267" x2="900" y2="267" elevation="0"/>
  <line stroke="blue" x1="900" y1="133" x2="900" y2="167" elevation="0"/>
  <line stroke="blue" x1="900" y1="133" x2="916" y2="117"/>
  <line stroke="red" x1="905" y1="100" x2="924" y2="100" elevation="1"/>
  <line stroke="blue" x1="167" y1="900" x2="267" y2="900" elevation="0"/>
  <line stroke="blue" x1="167" y1="900" x2="133" y2="933"/>
  <line stroke="blue" x1="900" y1="167" x2="900" y2="200" elevation="0"/>
  <line stroke="blue" x1="900" y1="167" x2="916" y2="138" opacity="0.8" fold-step="1"/>
  <line stroke="blue" x1="500" y1="767" x2="500" y2="800"/>
  <line stroke="blue" x1="500" y1="767" x2="767" y2="767" elevation="0"/>
  <line stroke="red" x1="867" y1="300" x2="900" y2="267"/>
  <line stroke="blue" x1="867" y1="300" x2="900" y2="300"/>
  <line stroke="red" x1="867" y1="300" x2="867" y2="400" elevation="1"/>
  <line stroke="blue" x1="900" y1="200" x2="933" y2="233"/>
  <line stroke="blue" x1="916" y1="117" x2="916" y2="138"/>
  <line stroke="blue" x1="916" y1="117" x2="933" y2="100"/>
  <line stroke="red" x1="833" y1="400" x2="867" y2="400"/>
  <line stroke="blue" x1="833" y1="400" x2="833" y2="533" elevation="0"/>
  <line stroke="red" x1="924" y1="33" x2="933" y2="33" elevation="1"/>
  <line stroke="blue" x1="916" y1="138" x2="933" y2="109" opacity="0.8" fold-step="1"/>
  <line stroke="red" x1="916" y1="138" x2="933" y2="128"/>
  <line stroke="red" x1="924" y1="100" x2="933" y2="100" elevation="1"/>
  <line stroke="blue" x1="0" y1="933" x2="33" y2="933"/>
  <line stroke="black" x1="0" y1="933" x2="0" y2="1000"/>
  <line stroke="red" x1="933" y1="0" x2="933" y2="16"/>
  <line stroke="black" x1="933" y1="0" x2="1000" y2="0"/>
  <line stroke="red" x1="933" y1="16" x2="933" y2="33"/>
  <line stroke="blue" x1="933" y1="16" x2="971" y2="17"/>
  <line stroke="red" x1="33" y1="933" x2="67" y2="933"/>
  <line stroke="blue" x1="33" y1="933" x2="33" y2="1000" elevation="0"/>
  <line stroke="blue" x1="933" y1="33" x2="933" y2="50"/>
  <line stroke="red" x1="933" y1="33" x2="942" y2="33" elevation="1"/>
  <line stroke="blue" x1="767" y1="533" x2="767" y2="567" elevation="0"/>
  <line stroke="red" x1="767" y1="533" x2="800" y2="533"/>
  <line stroke="blue" x1="933" y1="50" x2="933" y2="67"/>
  <line stroke="red" x1="933" y1="50" x2="971" y2="50"/>
  <line stroke="blue" x1="67" y1="933" x2="100" y2="933"/>
  <line stroke="red" x1="67" y1="933" x2="67" y2="1000" elevation="1"/>
  <line stroke="red" x1="933" y1="67" x2="933" y2="83"/>
  <line stroke="blue" x1="933" y1="67" x2="1000" y2="67" elevation="0"/>
  <line stroke="red" x1="933" y1="83" x2="933" y2="100"/>
  <line stroke="red" x1="933" y1="83" x2="971" y2="83"/>
  <line stroke="red" x1="100" y1="933" x2="133" y2="933"/>
  <line stroke="blue" x1="100" y1="933" x2="100" y2="1000" elevation="0"/>
  <line stroke="red" x1="933" y1="100" x2="933" y2="109" elevation="1"/>
  <line stroke="red" x1="933" y1="100" x2="942" y2="100" elevation="1"/>
  <line stroke="blue" x1="933" y1="100" x2="950" y2="116"/>
  <line stroke="red" x1="267" y1="900" x2="267" y2="933"/>
  <line stroke="blue" x1="267" y1="900" x2="500" y2="900" elevation="0"/>
  <line stroke="blue" x1="900" y1="267" x2="900" y2="300" elevation="0"/>
  <line stroke="blue" x1="900" y1="267" x2="933" y2="233"/>
  <line stroke="red" x1="933" y1="109" x2="933" y2="128" elevation="1"/>
  <line stroke="red" x1="933" y1="109" x2="950" y2="139" opacity="0.8" fold-step="1"/>
  <line stroke="blue" x1="933" y1="128" x2="950" y2="139"/>
  <line stroke="red" x1="933" y1="128" x2="933" y2="233" elevation="1"/>
  <line stroke="red" x1="133" y1="933" x2="267" y2="933" elevation="1"/>
  <line stroke="blue" x1="133" y1="933" x2="167" y2="967"/>
  <line stroke="red" x1="133" y1="933" x2="133" y2="1000" elevation="1"/>
  <line stroke="red" x1="942" y1="33" x2="961" y2="33" elevation="1"/>
  <line stroke="red" x1="942" y1="33" x2="971" y2="17" opacity="0.8" fold-step="1"/>
  <line stroke="blue" x1="942" y1="33" x2="971" y2="50" opacity="0.8" fold-step="1"/>
  <line stroke="red" x1="500" y1="800" x2="500" y2="833"/>
  <line stroke="red" x1="500" y1="800" x2="733" y2="800" elevation="1"/>
  <line stroke="red" x1="942" y1="100" x2="961" y2="100" elevation="1"/>
  <line stroke="red" x1="942" y1="100" x2="971" y2="84" opacity="0.8" fold-step="1"/>
  <line stroke="blue" x1="942" y1="100" x2="971" y2="116" opacity="0.8" fold-step="1"/>
  <line stroke="red" x1="900" y1="300" x2="933" y2="300"/>
  <line stroke="blue" x1="900" y1="300" x2="900" y2="400" elevation="0"/>
  <line stroke="blue" x1="767" y1="567" x2="800" y2="600"/>
  <line stroke="blue" x1="867" y1="400" x2="900" y2="400"/>
  <line stroke="red" x1="867" y1="400" x2="867" y2="533" elevation="1"/>
  <line stroke="red" x1="950" y1="116" x2="950" y2="139"/>
  <line stroke="blue" x1="950" y1="116" x2="967" y2="133"/>
  <line stroke="blue" x1="950" y1="116" x2="971" y2="116"/>
  <line stroke="red" x1="950" y1="139" x2="967" y2="167" opacity="0.8" fold-step="1"/>
  <line stroke="blue" x1="800" y1="533" x2="833" y2="533"/>
  <line stroke="red" x1="800" y1="533" x2="800" y2="600" elevation="1"/>
  <line stroke="red" x1="961" y1="33" x2="971" y2="17"/>
  <line stroke="blue" x1="961" y1="33" x2="971" y2="50"/>
  <line stroke="red" x1="961" y1="33" x2="1000" y2="33" elevation="1"/>
  <line stroke="red" x1="933" y1="233" x2="933" y2="300" elevation="1"/>
  <line stroke="red" x1="933" y1="233" x2="967" y2="233"/>
  <line stroke="blue" x1="961" y1="100" x2="971" y2="84"/>
  <line stroke="red" x1="961" y1="100" x2="971" y2="116"/>
  <line stroke="red" x1="961" y1="100" x2="1000" y2="100" elevation="1"/>
  <line stroke="blue" x1="267" y1="933" x2="267" y2="967"/>
  <line stroke="red" x1="267" y1="933" x2="500" y2="933" elevation="1"/>
  <line stroke="red" x1="971" y1="17" x2="1000" y2="0" opacity="0.8" fold-step="1"/>
  <line stroke="blue" x1="500" y1="833" x2="500" y2="867"/>
  <line stroke="blue" x1="500" y1="833" x2="733" y2="833" elevation="0"/>
  <line stroke="blue" x1="971" y1="50" x2="1000" y2="67" opacity="0.8" fold-step="1"/>
  <line stroke="red" x1="971" y1="84" x2="1000" y2="67" opacity="0.8" fold-step="1"/>
  <line stroke="blue" x1="967" y1="133" x2="967" y2="167" elevation="0"/>
  <line stroke="red" x1="967" y1="133" x2="979" y2="146"/>
  <line stroke="blue" x1="967" y1="133" x2="1000" y2="133" elevation="0"/>
  <line stroke="blue" x1="971" y1="116" x2="1000" y2="133" opacity="0.8" fold-step="1"/>
  <line stroke="blue" x1="933" y1="300" x2="967" y2="300"/>
  <line stroke="red" x1="933" y1="300" x2="933" y2="400" elevation="1"/>
  <line stroke="blue" x1="167" y1="967" x2="267" y2="967" elevation="0"/>
  <line stroke="blue" x1="167" y1="967" x2="167" y2="1000" elevation="0"/>
  <line stroke="red" x1="167" y1="967" x2="200" y2="1000"/>
  <line stroke="blue" x1="967" y1="167" x2="979" y2="146" opacity="0.8" fold-step="1"/>
  <line stroke="blue" x1="967" y1="167" x2="967" y2="233" elevation="0"/>
  <line stroke="red" x1="900" y1="400" x2="933" y2="400"/>
  <line stroke="blue" x1="900" y1="400" x2="900" y2="533" elevation="0"/>
  <line stroke="red" x1="833" y1="533" x2="867" y2="533"/>
  <line stroke="blue" x1="833" y1="533" x2="833" y2="633" elevation="0"/>
  <line stroke="red" x1="979" y1="146" x2="1000" y2="133" opacity="0.8" fold-step="1"/>
  <line stroke="red" x1="979" y1="146" x2="1000" y2="167"/>
  <line stroke="blue" x1="967" y1="233" x2="967" y2="300" elevation="0"/>
  <line stroke="blue" x1="967" y1="233" x2="1000" y2="233"/>
  <line stroke="black" x1="0" y1="1000" x2="33" y2="1000"/>
  <line stroke="red" x1="800" y1="600" x2="833" y2="633"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="33"/>
  <line stroke="black" x1="33" y1="1000" x2="67" y2="1000"/>
  <line stroke="black" x1="1000" y1="33" x2="1000" y2="67"/>
  <line stroke="red" x1="500" y1="867" x2="500" y2="900"/>
  <line stroke="red" x1="500" y1="867" x2="733" y2="867" elevation="1"/>
  <line stroke="black" x1="67" y1="1000" x2="100" y2="1000"/>
  <line stroke="black" x1="1000" y1="67" x2="1000" y2="100"/>
  <line stroke="red" x1="267" y1="967" x2="267" y2="1000"/>
  <line stroke="blue" x1="267" y1="967" x2="500" y2="967" elevation="0"/>
  <line stroke="black" x1="100" y1="1000" x2="133" y2="1000"/>
  <line stroke="black" x1="1000" y1="100" x2="1000" y2="133"/>
  <line stroke="black" x1="133" y1="1000" x2="167" y2="1000"/>
  <line stroke="black" x1="1000" y1="133" x2="1000" y2="167"/>
  <line stroke="red" x1="967" y1="300" x2="1000" y2="300"/>
  <line stroke="blue" x1="967" y1="300" x2="967" y2="400" elevation="0"/>
  <line stroke="black" x1="167" y1="1000" x2="200" y2="1000"/>
  <line stroke="black" x1="1000" y1="167" x2="1000" y2="233"/>
  <line stroke="blue" x1="933" y1="400" x2="967" y2="400"/>
  <line stroke="red" x1="933" y1="400" x2="933" y2="533" elevation="1"/>
  <line stroke="blue" x1="867" y1="533" x2="900" y2="533"/>
  <line stroke="red" x1="867" y1="533" x2="867" y2="667" elevation="1"/>
  <line stroke="black" x1="200" y1="1000" x2="267" y2="1000"/>
  <line stroke="black" x1="1000" y1="233" x2="1000" y2="300"/>
  <line stroke="blue" x1="500" y1="900" x2="500" y2="933"/>
  <line stroke="blue" x1="500" y1="900" x2="733" y2="900" elevation="0"/>
  <line stroke="black" x1="267" y1="1000" x2="500" y2="1000"/>
  <line stroke="black" x1="1000" y1="300" x2="1000" y2="400"/>
  <line stroke="red" x1="900" y1="533" x2="933" y2="533"/>
  <line stroke="blue" x1="900" y1="533" x2="900" y2="667" elevation="0"/>
  <line stroke="blue" x1="833" y1="633" x2="867" y2="667"/>
  <line stroke="red" x1="967" y1="400" x2="1000" y2="400"/>
  <line stroke="blue" x1="967" y1="400" x2="967" y2="533" elevation="0"/>
  <line stroke="red" x1="500" y1="933" x2="500" y2="967"/>
  <line stroke="red" x1="500" y1="933" x2="733" y2="933" elevation="1"/>
  <line stroke="blue" x1="933" y1="533" x2="967" y2="533"/>
  <line stroke="red" x1="933" y1="533" x2="933" y2="667" elevation="1"/>
  <line stroke="black" x1="1000" y1="400" x2="1000" y2="533"/>
  <line stroke="red" x1="767" y1="767" x2="733" y2="800"/>
  <line stroke="blue" x1="767" y1="767" x2="800" y2="733"/>
  <line stroke="blue" x1="767" y1="767" x2="767" y2="833" elevation="0"/>
  <line stroke="blue" x1="733" y1="800" x2="733" y2="833"/>
  <line stroke="red" x1="733" y1="800" x2="767" y2="833"/>
  <line stroke="red" x1="800" y1="733" x2="833" y2="700"/>
  <line stroke="red" x1="800" y1="733" x2="800" y2="867" elevation="1"/>
  <line stroke="blue" x1="833" y1="700" x2="867" y2="667"/>
  <line stroke="blue" x1="833" y1="700" x2="833" y2="900" elevation="0"/>
  <line stroke="blue" x1="500" y1="967" x2="500" y2="1000"/>
  <line stroke="blue" x1="500" y1="967" x2="733" y2="967" elevation="0"/>
  <line stroke="red" x1="867" y1="667" x2="900" y2="667"/>
  <line stroke="red" x1="867" y1="667" x2="867" y2="933" elevation="1"/>
  <line stroke="red" x1="967" y1="533" x2="1000" y2="533"/>
  <line stroke="blue" x1="967" y1="533" x2="967" y2="667" elevation="0"/>
  <line stroke="blue" x1="733" y1="833" x2="767" y2="833" elevation="0"/>
  <line stroke="red" x1="733" y1="833" x2="733" y2="867"/>
  <line stroke="black" x1="500" y1="1000" x2="733" y2="1000"/>
  <line stroke="blue" x1="900" y1="667" x2="933" y2="667"/>
  <line stroke="blue" x1="900" y1="667" x2="900" y2="933" elevation="0"/>
  <line stroke="blue" x1="767" y1="833" x2="800" y2="867"/>
  <line stroke="black" x1="1000" y1="533" x2="1000" y2="667"/>
  <line stroke="blue" x1="733" y1="867" x2="733" y2="900"/>
  <line stroke="red" x1="733" y1="867" x2="800" y2="867" elevation="1"/>
  <line stroke="red" x1="933" y1="667" x2="967" y2="667"/>
  <line stroke="red" x1="933" y1="667" x2="933" y2="933" elevation="1"/>
  <line stroke="red" x1="733" y1="900" x2="733" y2="933"/>
  <line stroke="blue" x1="733" y1="900" x2="833" y2="900" elevation="0"/>
  <line stroke="blue" x1="967" y1="667" x2="1000" y2="667"/>
  <line stroke="blue" x1="967" y1="667" x2="967" y2="933" elevation="0"/>
  <line stroke="red" x1="800" y1="867" x2="833" y2="900"/>
  <line stroke="blue" x1="733" y1="933" x2="733" y2="967"/>
  <line stroke="red" x1="733" y1="933" x2="867" y2="933" elevation="1"/>
  <line stroke="black" x1="1000" y1="667" x2="1000" y2="933"/>
  <line stroke="red" x1="733" y1="967" x2="733" y2="1000"/>
  <line stroke="blue" x1="733" y1="967" x2="833" y2="967" elevation="0"/>
  <line stroke="blue" x1="833" y1="900" x2="867" y2="933"/>
  <line stroke="black" x1="733" y1="1000" x2="800" y2="1000"/>
  <line stroke="blue" x1="867" y1="933" x2="833" y2="967"/>
  <line stroke="red" x1="867" y1="933" x2="900" y2="933"/>
  <line stroke="red" x1="867" y1="933" x2="867" y2="1000" elevation="1"/>
  <line stroke="red" x1="833" y1="967" x2="800" y2="1000"/>
  <line stroke="blue" x1="833" y1="967" x2="833" y2="1000" elevation="0"/>
  <line stroke="black" x1="800" y1="1000" x2="833" y2="1000"/>
  <line stroke="blue" x1="900" y1="933" x2="933" y2="933"/>
  <line stroke="blue" x1="900" y1="933" x2="900" y2="1000" elevation="0"/>
  <line stroke="black" x1="833" y1="1000" x2="867" y2="1000"/>
  <line stroke="red" x1="933" y1="933" x2="967" y2="933"/>
  <line stroke="red" x1="933" y1="933" x2="933" y2="1000" elevation="1"/>
  <line stroke="black" x1="867" y1="1000" x2="900" y2="1000"/>
  <line stroke="blue" x1="967" y1="933" x2="1000" y2="933"/>
  <line stroke="blue" x1="967" y1="933" x2="967" y2="1000" elevation="0"/>
  <line stroke="black" x1="900" y1="1000" x2="933" y2="1000"/>
  <line stroke="black" x1="933" y1="1000" x2="967" y2="1000"/>
  <line stroke="black" x1="1000" y1="933" x2="1000" y2="1000"/>
  <line stroke="black" x1="967" y1="1000" x2="1000" y2="1000"/>
</svg>""",
)

BASIC_REVERSE_FOLD_VERTICAL_TOP = BoxPleatingShapingExample(
    base_example=BoxPleatingCreasePatternExample(
        name="basic_reverse_fold_vertical_top",
        packing_svg="",
        grid_size=8,
        base_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8">
  <rect stroke="black" x="0" y="0" width="8" height="8"/>
  <line stroke="blue" x1="0" y1="1" x2="6" y2="1"/>
  <line stroke="red" x1="0" y1="2" x2="7" y2="2"/>
  <line stroke="blue" x1="0" y1="3" x2="8" y2="3"/>
  <line stroke="red" x1="0" y1="4" x2="7" y2="4"/>
  <line stroke="blue" x1="0" y1="5" x2="6" y2="5"/>
  <line stroke="red" x1="5" y1="0" x2="6" y2="1"/>
  <line stroke="red" x1="0" y1="6" x2="5" y2="6"/>
  <line stroke="blue" x1="6" y1="0" x2="6" y2="1"/>
  <line stroke="blue" x1="6" y1="1" x2="7" y2="2"/>
  <line stroke="blue" x1="0" y1="7" x2="4" y2="7"/>
  <line stroke="red" x1="7" y1="0" x2="7" y2="2"/>
  <line stroke="red" x1="7" y1="2" x2="8" y2="3"/>
  <line stroke="red" x1="5" y1="6" x2="6" y2="5"/>
  <line stroke="blue" x1="5" y1="6" x2="4" y2="7"/>
  <line stroke="red" x1="5" y1="6" x2="5" y2="7"/>
  <line stroke="blue" x1="6" y1="5" x2="7" y2="4"/>
  <line stroke="blue" x1="6" y1="5" x2="6" y2="7"/>
  <line stroke="red" x1="4" y1="7" x2="5" y2="7"/>
  <line stroke="blue" x1="4" y1="7" x2="5" y2="8"/>
  <line stroke="red" x1="7" y1="4" x2="8" y2="3"/>
  <line stroke="red" x1="7" y1="4" x2="7" y2="7"/>
  <line stroke="blue" x1="5" y1="7" x2="6" y2="7"/>
  <line stroke="red" x1="5" y1="7" x2="5" y2="8"/>
  <line stroke="red" x1="6" y1="7" x2="7" y2="7"/>
  <line stroke="blue" x1="6" y1="7" x2="6" y2="8"/>
  <line stroke="red" x1="7" y1="7" x2="7" y2="8"/>
  <line stroke="blue" x1="7" y1="7" x2="8" y2="7"/>
</svg>""",
    ),
    central_crease_removed=False,
    rectangles=[
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(x=1.0, y=0.375, require_on_paper=False)
            ],
            radius=np.sqrt(2) / 2,
            name="wide_flap",
        )
    ],
    reverse_fold_shaping_params=[
        reverse_fold_shaping.ReverseFoldShapingParams(
            flap_name="wide_flap",
            reverse_fold_params=[
                reverse_fold_shaping.ReverseFoldParams(
                    flap_angle=math.pi / 2,
                    fold_order=[
                        crease_utils.Fold.VALLEY,
                        crease_utils.Fold.MOUNTAIN,
                    ],
                    even_phase=True,
                ),
            ],
            orientation="vertical",
            fold_type=crease_utils.Fold.VALLEY,
        )
    ],
    reverse_fold_shaping_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8">
  <rect stroke="black" x="0" y="0" width="8" height="8"/>
  <line stroke="blue" x1="0" y1="1" x2="4" y2="1"/>
  <line stroke="red" x1="0" y1="2" x2="5" y2="2"/>
  <line stroke="blue" x1="0" y1="3" x2="4" y2="3"/>
  <line stroke="red" x1="0" y1="4" x2="5" y2="4"/>
  <line stroke="blue" x1="4" y1="1" x2="5" y2="-0" fold-step="1"/>
  <line stroke="red" x1="4" y1="1" x2="5" y2="2" fold-step="1"/>
  <line stroke="blue" x1="4" y1="1" x2="6" y2="1"/>
  <line stroke="blue" x1="0" y1="5" x2="4" y2="5"/>
  <line stroke="blue" x1="4" y1="3" x2="5" y2="2" fold-step="1"/>
  <line stroke="blue" x1="4" y1="3" x2="5" y2="4" fold-step="1"/>
  <line stroke="red" x1="4" y1="3" x2="8" y2="3"/>
  <line stroke="red" x1="5" y1="0" x2="6" y2="1"/>
  <line stroke="red" x1="5" y1="2" x2="7" y2="2"/>
  <line stroke="red" x1="0" y1="6" x2="5" y2="6"/>
  <line stroke="blue" x1="6" y1="0" x2="6" y2="1"/>
  <line stroke="blue" x1="6" y1="1" x2="7" y2="2"/>
  <line stroke="red" x1="4" y1="5" x2="5" y2="4" fold-step="1"/>
  <line stroke="blue" x1="4" y1="5" x2="5" y2="6" fold-step="1"/>
  <line stroke="blue" x1="4" y1="5" x2="6" y2="5"/>
  <line stroke="red" x1="5" y1="4" x2="7" y2="4"/>
  <line stroke="blue" x1="0" y1="7" x2="4" y2="7"/>
  <line stroke="red" x1="7" y1="0" x2="7" y2="2"/>
  <line stroke="red" x1="7" y1="2" x2="8" y2="3"/>
  <line stroke="red" x1="5" y1="6" x2="6" y2="5"/>
  <line stroke="blue" x1="5" y1="6" x2="4" y2="7"/>
  <line stroke="red" x1="5" y1="6" x2="5" y2="7"/>
  <line stroke="red" x1="5" y1="6" x2="6" y2="7" fold-step="1"/>
  <line stroke="blue" x1="6" y1="5" x2="7" y2="4"/>
  <line stroke="blue" x1="6" y1="5" x2="6" y2="7"/>
  <line stroke="red" x1="4" y1="7" x2="5" y2="7"/>
  <line stroke="blue" x1="4" y1="7" x2="5" y2="8"/>
  <line stroke="red" x1="7" y1="4" x2="8" y2="3"/>
  <line stroke="red" x1="7" y1="4" x2="7" y2="6"/>
  <line stroke="blue" x1="5" y1="7" x2="6" y2="7"/>
  <line stroke="red" x1="5" y1="7" x2="5" y2="8"/>
  <line stroke="blue" x1="6" y1="7" x2="7" y2="6" fold-step="1"/>
  <line stroke="red" x1="6" y1="7" x2="7" y2="7"/>
  <line stroke="blue" x1="6" y1="7" x2="6" y2="8"/>
  <line stroke="red" x1="7" y1="6" x2="7" y2="7"/>
  <line stroke="red" x1="7" y1="6" x2="8" y2="7" fold-step="1"/>
  <line stroke="red" x1="7" y1="7" x2="7" y2="8"/>
  <line stroke="blue" x1="7" y1="7" x2="8" y2="7"/>
</svg>""",
)

BASIC_REVERSE_FOLD_VERTICAL_BOTTOM = BoxPleatingShapingExample(
    base_example=BoxPleatingCreasePatternExample(
        name="basic_reverse_fold_vertical_bottom",
        packing_svg="",
        grid_size=8,
        base_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8">
  <rect stroke="black" x="0" y="0" width="8" height="8"/>
  <line stroke="blue" x1="0" y1="1" x2="4" y2="1"/>
  <line stroke="red" x1="0" y1="2" x2="5" y2="2"/>
  <line stroke="blue" x1="0" y1="3" x2="6" y2="3"/>
  <line stroke="red" x1="0" y1="4" x2="7" y2="4"/>
  <line stroke="blue" x1="4" y1="1" x2="5" y2="0"/>
  <line stroke="red" x1="4" y1="1" x2="5" y2="1"/>
  <line stroke="blue" x1="4" y1="1" x2="5" y2="2"/>
  <line stroke="blue" x1="0" y1="5" x2="8" y2="5"/>
  <line stroke="red" x1="5" y1="0" x2="5" y2="1"/>
  <line stroke="red" x1="5" y1="1" x2="5" y2="2"/>
  <line stroke="blue" x1="5" y1="1" x2="6" y2="1"/>
  <line stroke="red" x1="5" y1="2" x2="6" y2="3"/>
  <line stroke="red" x1="0" y1="6" x2="7" y2="6"/>
  <line stroke="blue" x1="6" y1="0" x2="6" y2="1"/>
  <line stroke="blue" x1="6" y1="1" x2="6" y2="3"/>
  <line stroke="red" x1="6" y1="1" x2="7" y2="1"/>
  <line stroke="blue" x1="6" y1="3" x2="7" y2="4"/>
  <line stroke="blue" x1="0" y1="7" x2="6" y2="7"/>
  <line stroke="red" x1="7" y1="0" x2="7" y2="1"/>
  <line stroke="red" x1="7" y1="1" x2="7" y2="4"/>
  <line stroke="blue" x1="7" y1="1" x2="8" y2="1"/>
  <line stroke="red" x1="7" y1="4" x2="8" y2="5"/>
  <line stroke="blue" x1="6" y1="7" x2="7" y2="6"/>
  <line stroke="red" x1="6" y1="7" x2="5" y2="8"/>
  <line stroke="blue" x1="6" y1="7" x2="6" y2="8"/>
  <line stroke="red" x1="7" y1="6" x2="8" y2="5"/>
  <line stroke="red" x1="7" y1="6" x2="7" y2="8"/>
</svg>""",
    ),
    central_crease_removed=False,
    rectangles=[
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(x=1.0, y=0.625, require_on_paper=False)
            ],
            radius=np.sqrt(2) / 2,
            name="wide_flap",
        )
    ],
    reverse_fold_shaping_params=[
        reverse_fold_shaping.ReverseFoldShapingParams(
            flap_name="wide_flap",
            reverse_fold_params=[
                reverse_fold_shaping.ReverseFoldParams(
                    flap_angle=math.pi / 2,
                    fold_order=[
                        crease_utils.Fold.VALLEY,
                        crease_utils.Fold.MOUNTAIN,
                    ],
                    even_phase=True,
                )
            ],
            orientation="vertical",
            fold_type=crease_utils.Fold.VALLEY,
        )
    ],
    reverse_fold_shaping_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8">
  <rect stroke="black" x="0" y="0" width="8" height="8"/>
  <line stroke="blue" x1="0" y1="1" x2="4" y2="1"/>
  <line stroke="red" x1="0" y1="2" x2="5" y2="2"/>
  <line stroke="blue" x1="0" y1="3" x2="4" y2="3"/>
  <line stroke="red" x1="0" y1="4" x2="5" y2="4"/>
  <line stroke="blue" x1="4" y1="1" x2="5" y2="0"/>
  <line stroke="red" x1="4" y1="1" x2="5" y2="1"/>
  <line stroke="blue" x1="4" y1="1" x2="5" y2="2"/>
  <line stroke="blue" x1="0" y1="5" x2="4" y2="5"/>
  <line stroke="blue" x1="4" y1="3" x2="5" y2="2" fold-step="1"/>
  <line stroke="red" x1="4" y1="3" x2="5" y2="4" fold-step="1"/>
  <line stroke="blue" x1="4" y1="3" x2="6" y2="3"/>
  <line stroke="red" x1="5" y1="0" x2="5" y2="1"/>
  <line stroke="red" x1="5" y1="1" x2="5" y2="2"/>
  <line stroke="blue" x1="5" y1="1" x2="6" y2="1"/>
  <line stroke="red" x1="5" y1="2" x2="6" y2="1" fold-step="1"/>
  <line stroke="red" x1="5" y1="2" x2="6" y2="3"/>
  <line stroke="red" x1="0" y1="6" x2="5" y2="6"/>
  <line stroke="blue" x1="6" y1="0" x2="6" y2="1"/>
  <line stroke="blue" x1="6" y1="1" x2="6" y2="3"/>
  <line stroke="red" x1="6" y1="1" x2="7" y2="1"/>
  <line stroke="blue" x1="6" y1="1" x2="7" y2="2" fold-step="1"/>
  <line stroke="blue" x1="4" y1="5" x2="5" y2="4" fold-step="1"/>
  <line stroke="blue" x1="4" y1="5" x2="5" y2="6" fold-step="1"/>
  <line stroke="red" x1="4" y1="5" x2="8" y2="5"/>
  <line stroke="red" x1="5" y1="4" x2="7" y2="4"/>
  <line stroke="blue" x1="6" y1="3" x2="7" y2="4"/>
  <line stroke="blue" x1="0" y1="7" x2="4" y2="7"/>
  <line stroke="red" x1="7" y1="0" x2="7" y2="1"/>
  <line stroke="red" x1="7" y1="1" x2="7" y2="2"/>
  <line stroke="blue" x1="7" y1="1" x2="8" y2="1"/>
  <line stroke="red" x1="7" y1="2" x2="7" y2="4"/>
  <line stroke="red" x1="7" y1="2" x2="8" y2="1" fold-step="1"/>
  <line stroke="red" x1="5" y1="6" x2="4" y2="7" fold-step="1"/>
  <line stroke="red" x1="5" y1="6" x2="7" y2="6"/>
  <line stroke="blue" x1="4" y1="7" x2="6" y2="7"/>
  <line stroke="blue" x1="4" y1="7" x2="5" y2="8" fold-step="1"/>
  <line stroke="red" x1="7" y1="4" x2="8" y2="5"/>
  <line stroke="blue" x1="6" y1="7" x2="7" y2="6"/>
  <line stroke="red" x1="6" y1="7" x2="5" y2="8"/>
  <line stroke="blue" x1="6" y1="7" x2="6" y2="8"/>
  <line stroke="red" x1="7" y1="6" x2="8" y2="5"/>
  <line stroke="red" x1="7" y1="6" x2="7" y2="8"/>
</svg>""",
)

BASIC_REVERSE_FOLD_HORIZONTAL_LEFT = BoxPleatingShapingExample(
    base_example=BoxPleatingCreasePatternExample(
        name="basic_reverse_fold_horizontal_left",
        packing_svg="",
        grid_size=8,
        base_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8">
  <rect stroke="black" x="0" y="0" width="8" height="8"/>
  <line stroke="blue" x1="1" y1="0" x2="1" y2="6"/>
  <line stroke="red" x1="2" y1="0" x2="2" y2="7"/>
  <line stroke="blue" x1="3" y1="0" x2="3" y2="8"/>
  <line stroke="red" x1="4" y1="0" x2="4" y2="7"/>
  <line stroke="red" x1="0" y1="5" x2="1" y2="6"/>
  <line stroke="blue" x1="5" y1="0" x2="5" y2="6"/>
  <line stroke="blue" x1="0" y1="6" x2="1" y2="6"/>
  <line stroke="red" x1="6" y1="0" x2="6" y2="5"/>
  <line stroke="blue" x1="1" y1="6" x2="2" y2="7"/>
  <line stroke="red" x1="0" y1="7" x2="2" y2="7"/>
  <line stroke="blue" x1="7" y1="0" x2="7" y2="4"/>
  <line stroke="red" x1="2" y1="7" x2="3" y2="8"/>
  <line stroke="red" x1="5" y1="6" x2="6" y2="5"/>
  <line stroke="blue" x1="5" y1="6" x2="4" y2="7"/>
  <line stroke="blue" x1="5" y1="6" x2="7" y2="6"/>
  <line stroke="blue" x1="6" y1="5" x2="7" y2="4"/>
  <line stroke="red" x1="6" y1="5" x2="7" y2="5"/>
  <line stroke="red" x1="4" y1="7" x2="3" y2="8"/>
  <line stroke="red" x1="4" y1="7" x2="7" y2="7"/>
  <line stroke="red" x1="7" y1="4" x2="7" y2="5"/>
  <line stroke="blue" x1="7" y1="4" x2="8" y2="5"/>
  <line stroke="blue" x1="7" y1="5" x2="7" y2="6"/>
  <line stroke="red" x1="7" y1="5" x2="8" y2="5"/>
  <line stroke="red" x1="7" y1="6" x2="7" y2="7"/>
  <line stroke="blue" x1="7" y1="6" x2="8" y2="6"/>
  <line stroke="blue" x1="7" y1="7" x2="7" y2="8"/>
  <line stroke="red" x1="7" y1="7" x2="8" y2="7"/>
</svg>""",
    ),
    central_crease_removed=False,
    rectangles=[
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(x=0.375, y=1.0, require_on_paper=False)
            ],
            radius=np.sqrt(2) / 2,
            name="wide_flap",
        )
    ],
    reverse_fold_shaping_params=[
        reverse_fold_shaping.ReverseFoldShapingParams(
            flap_name="wide_flap",
            reverse_fold_params=[
                reverse_fold_shaping.ReverseFoldParams(
                    flap_angle=math.pi / 2,
                    fold_order=[
                        crease_utils.Fold.VALLEY,
                        crease_utils.Fold.MOUNTAIN,
                    ],
                    even_phase=True,
                )
            ],
            orientation="horizontal",
            fold_type=crease_utils.Fold.VALLEY,
        )
    ],
    reverse_fold_shaping_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8">
  <rect stroke="black" x="0" y="0" width="8" height="8"/>
  <line stroke="blue" x1="1" y1="0" x2="1" y2="4"/>
  <line stroke="red" x1="2" y1="0" x2="2" y2="5"/>
  <line stroke="blue" x1="3" y1="0" x2="3" y2="4"/>
  <line stroke="red" x1="4" y1="0" x2="4" y2="5"/>
  <line stroke="blue" x1="1" y1="4" x2="0" y2="5" fold-step="1"/>
  <line stroke="red" x1="1" y1="4" x2="2" y2="5" fold-step="1"/>
  <line stroke="blue" x1="1" y1="4" x2="1" y2="6"/>
  <line stroke="red" x1="0" y1="5" x2="1" y2="6"/>
  <line stroke="blue" x1="3" y1="4" x2="2" y2="5" fold-step="1"/>
  <line stroke="blue" x1="3" y1="4" x2="4" y2="5" fold-step="1"/>
  <line stroke="red" x1="3" y1="4" x2="3" y2="8"/>
  <line stroke="blue" x1="5" y1="0" x2="5" y2="4"/>
  <line stroke="red" x1="2" y1="5" x2="2" y2="7"/>
  <line stroke="blue" x1="0" y1="6" x2="1" y2="6"/>
  <line stroke="red" x1="6" y1="0" x2="6" y2="5"/>
  <line stroke="blue" x1="1" y1="6" x2="2" y2="7"/>
  <line stroke="red" x1="4" y1="5" x2="5" y2="4" fold-step="1"/>
  <line stroke="red" x1="4" y1="5" x2="4" y2="7"/>
  <line stroke="blue" x1="5" y1="4" x2="5" y2="6"/>
  <line stroke="blue" x1="5" y1="4" x2="6" y2="5" fold-step="1"/>
  <line stroke="red" x1="0" y1="7" x2="2" y2="7"/>
  <line stroke="blue" x1="7" y1="0" x2="7" y2="4"/>
  <line stroke="red" x1="2" y1="7" x2="3" y2="8"/>
  <line stroke="red" x1="5" y1="6" x2="6" y2="5"/>
  <line stroke="blue" x1="5" y1="6" x2="4" y2="7"/>
  <line stroke="blue" x1="5" y1="6" x2="7" y2="6"/>
  <line stroke="blue" x1="6" y1="5" x2="7" y2="4"/>
  <line stroke="red" x1="6" y1="5" x2="7" y2="5"/>
  <line stroke="red" x1="6" y1="5" x2="7" y2="6" fold-step="1"/>
  <line stroke="red" x1="4" y1="7" x2="3" y2="8"/>
  <line stroke="red" x1="4" y1="7" x2="6" y2="7"/>
  <line stroke="red" x1="7" y1="4" x2="7" y2="5"/>
  <line stroke="blue" x1="7" y1="4" x2="8" y2="5"/>
  <line stroke="blue" x1="7" y1="5" x2="7" y2="6"/>
  <line stroke="red" x1="7" y1="5" x2="8" y2="5"/>
  <line stroke="blue" x1="6" y1="7" x2="7" y2="6" fold-step="1"/>
  <line stroke="red" x1="6" y1="7" x2="7" y2="7"/>
  <line stroke="red" x1="6" y1="7" x2="7" y2="8" fold-step="1"/>
  <line stroke="red" x1="7" y1="6" x2="7" y2="7"/>
  <line stroke="blue" x1="7" y1="6" x2="8" y2="6"/>
  <line stroke="blue" x1="7" y1="7" x2="7" y2="8"/>
  <line stroke="red" x1="7" y1="7" x2="8" y2="7"/>
</svg>""",
)

BASIC_REVERSE_FOLD_HORIZONTAL_RIGHT = BoxPleatingShapingExample(
    base_example=BoxPleatingCreasePatternExample(
        name="basic_reverse_fold_horizontal_right",
        packing_svg="",
        grid_size=8,
        base_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8">
  <rect stroke="black" x="0" y="0" width="8" height="8"/>
  <line stroke="blue" x1="1" y1="0" x2="1" y2="4"/>
  <line stroke="red" x1="2" y1="0" x2="2" y2="5"/>
  <line stroke="blue" x1="3" y1="0" x2="3" y2="6"/>
  <line stroke="red" x1="4" y1="0" x2="4" y2="7"/>
  <line stroke="blue" x1="1" y1="4" x2="0" y2="5"/>
  <line stroke="red" x1="1" y1="4" x2="1" y2="5"/>
  <line stroke="blue" x1="1" y1="4" x2="2" y2="5"/>
  <line stroke="red" x1="0" y1="5" x2="1" y2="5"/>
  <line stroke="blue" x1="5" y1="0" x2="5" y2="8"/>
  <line stroke="red" x1="1" y1="5" x2="2" y2="5"/>
  <line stroke="blue" x1="1" y1="5" x2="1" y2="6"/>
  <line stroke="red" x1="2" y1="5" x2="3" y2="6"/>
  <line stroke="blue" x1="0" y1="6" x2="1" y2="6"/>
  <line stroke="red" x1="6" y1="0" x2="6" y2="7"/>
  <line stroke="blue" x1="1" y1="6" x2="3" y2="6"/>
  <line stroke="red" x1="1" y1="6" x2="1" y2="7"/>
  <line stroke="blue" x1="3" y1="6" x2="4" y2="7"/>
  <line stroke="red" x1="0" y1="7" x2="1" y2="7"/>
  <line stroke="blue" x1="7" y1="0" x2="7" y2="6"/>
  <line stroke="blue" x1="1" y1="7" x2="1" y2="8"/>
  <line stroke="red" x1="1" y1="7" x2="4" y2="7"/>
  <line stroke="red" x1="4" y1="7" x2="5" y2="8"/>
  <line stroke="blue" x1="6" y1="7" x2="7" y2="6"/>
  <line stroke="red" x1="6" y1="7" x2="5" y2="8"/>
  <line stroke="red" x1="6" y1="7" x2="8" y2="7"/>
  <line stroke="red" x1="7" y1="6" x2="8" y2="5"/>
  <line stroke="blue" x1="7" y1="6" x2="8" y2="6"/>
</svg>""",
    ),
    central_crease_removed=False,
    rectangles=[
        shaping_utils.Rectangle.from_centers(
            circle_centers=[
                point_utils.Point(x=0.625, y=1.0, require_on_paper=False)
            ],
            radius=np.sqrt(2) / 2,
            name="wide_flap",
        )
    ],
    reverse_fold_shaping_params=[
        reverse_fold_shaping.ReverseFoldShapingParams(
            flap_name="wide_flap",
            reverse_fold_params=[
                reverse_fold_shaping.ReverseFoldParams(
                    flap_angle=math.pi / 2,
                    fold_order=[
                        crease_utils.Fold.VALLEY,
                        crease_utils.Fold.MOUNTAIN,
                    ],
                    even_phase=True,
                )
            ],
            orientation="horizontal",
            fold_type=crease_utils.Fold.VALLEY,
        )
    ],
    reverse_fold_shaping_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8">
  <rect stroke="black" x="0" y="0" width="8" height="8"/>
  <line stroke="blue" x1="1" y1="0" x2="1" y2="4"/>
  <line stroke="red" x1="2" y1="0" x2="2" y2="5"/>
  <line stroke="blue" x1="3" y1="0" x2="3" y2="4"/>
  <line stroke="red" x1="4" y1="0" x2="4" y2="5"/>
  <line stroke="blue" x1="1" y1="4" x2="0" y2="5"/>
  <line stroke="red" x1="1" y1="4" x2="1" y2="5"/>
  <line stroke="blue" x1="1" y1="4" x2="2" y2="5"/>
  <line stroke="red" x1="0" y1="5" x2="1" y2="5"/>
  <line stroke="blue" x1="3" y1="4" x2="2" y2="5" fold-step="1"/>
  <line stroke="red" x1="3" y1="4" x2="4" y2="5" fold-step="1"/>
  <line stroke="blue" x1="3" y1="4" x2="3" y2="6"/>
  <line stroke="blue" x1="5" y1="0" x2="5" y2="4"/>
  <line stroke="red" x1="1" y1="5" x2="2" y2="5"/>
  <line stroke="blue" x1="1" y1="5" x2="1" y2="6"/>
  <line stroke="red" x1="2" y1="5" x2="1" y2="6" fold-step="1"/>
  <line stroke="red" x1="2" y1="5" x2="3" y2="6"/>
  <line stroke="blue" x1="0" y1="6" x2="1" y2="6"/>
  <line stroke="red" x1="6" y1="0" x2="6" y2="5"/>
  <line stroke="blue" x1="1" y1="6" x2="3" y2="6"/>
  <line stroke="red" x1="1" y1="6" x2="1" y2="7"/>
  <line stroke="blue" x1="1" y1="6" x2="2" y2="7" fold-step="1"/>
  <line stroke="blue" x1="4" y1="5" x2="5" y2="4" fold-step="1"/>
  <line stroke="red" x1="4" y1="5" x2="4" y2="7"/>
  <line stroke="blue" x1="5" y1="4" x2="6" y2="5" fold-step="1"/>
  <line stroke="red" x1="5" y1="4" x2="5" y2="8"/>
  <line stroke="blue" x1="3" y1="6" x2="4" y2="7"/>
  <line stroke="red" x1="0" y1="7" x2="1" y2="7"/>
  <line stroke="blue" x1="7" y1="0" x2="7" y2="4"/>
  <line stroke="red" x1="1" y1="7" x2="2" y2="7"/>
  <line stroke="blue" x1="1" y1="7" x2="1" y2="8"/>
  <line stroke="red" x1="2" y1="7" x2="1" y2="8" fold-step="1"/>
  <line stroke="red" x1="2" y1="7" x2="4" y2="7"/>
  <line stroke="red" x1="6" y1="5" x2="7" y2="4" fold-step="1"/>
  <line stroke="red" x1="6" y1="5" x2="6" y2="7"/>
  <line stroke="red" x1="4" y1="7" x2="5" y2="8"/>
  <line stroke="blue" x1="7" y1="4" x2="7" y2="6"/>
  <line stroke="blue" x1="7" y1="4" x2="8" y2="5" fold-step="1"/>
  <line stroke="blue" x1="6" y1="7" x2="7" y2="6"/>
  <line stroke="red" x1="6" y1="7" x2="5" y2="8"/>
  <line stroke="red" x1="6" y1="7" x2="8" y2="7"/>
  <line stroke="red" x1="7" y1="6" x2="8" y2="5"/>
  <line stroke="blue" x1="7" y1="6" x2="8" y2="6"/>
</svg>""",
)


HUMAN_WITH_CENTER = BoxPleatingCreasePatternExample(
    name="human_with_center",
    grid_size=16,
    tree_data={
        "version": "0.7",
        "design": {
            "title": "human_with_center",
            "mode": "tree",
            "description": "",
            "layout": {
                "sheet": {"type": "rect", "height": 16, "width": 16},
                "flaps": [
                    {"id": 2, "x": 8, "y": 6, "width": 0, "height": 0},
                    {"id": 3, "x": 10, "y": 3, "width": 0, "height": 0},
                    {"id": 4, "x": 8, "y": 12, "width": 0, "height": 0},
                    {"id": 5, "x": 11, "y": 9, "width": 0, "height": 0},
                    {"id": 6, "x": 4, "y": 9, "width": 0, "height": 0},
                ],
                "stretches": [],
            },
            "tree": {
                "sheet": {"type": "rect", "height": 20, "width": 20},
                "nodes": [
                    {"id": 0, "name": "", "x": 10, "y": 10},
                    {"id": 1, "name": "", "x": 10, "y": 13},
                    {"id": 2, "name": "", "x": 6, "y": 4},
                    {"id": 3, "name": "", "x": 13, "y": 4},
                    {"id": 4, "name": "", "x": 10, "y": 15},
                    {"id": 5, "name": "", "x": 14, "y": 11},
                    {"id": 6, "name": "", "x": 5, "y": 11},
                ],
                "edges": [
                    {"n1": 0, "n2": 1, "length": 2},
                    {"n1": 0, "n2": 2, "length": 8},
                    {"n1": 0, "n2": 3, "length": 8},
                    {"n1": 1, "n2": 4, "length": 2},
                    {"n1": 1, "n2": 5, "length": 6},
                    {"n1": 1, "n2": 6, "length": 6},
                ],
            },
        },
    },
    packing_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">
  <line stroke="black" x1="0" y1="0" x2="0" y2="6"/>
  <line stroke="black" x1="0" y1="0" x2="6" y2="0"/>
  <line stroke="red" x1="0" y1="0" x2="6" y2="6"/>
  <line stroke="black" x1="0" y1="6" x2="0" y2="8"/>
  <line stroke="green" x1="0" y1="6" x2="6" y2="6"/>
  <line stroke="black" x1="6" y1="0" x2="8" y2="0"/>
  <line stroke="green" x1="6" y1="0" x2="6" y2="6"/>
  <line stroke="green" x1="0" y1="8" x2="8" y2="8"/>
  <line stroke="black" x1="0" y1="8" x2="0" y2="16"/>
  <line stroke="red" x1="8" y1="0" x2="8" y2="4"/>
  <line stroke="black" x1="8" y1="0" x2="10" y2="0"/>
  <line stroke="red" x1="6" y1="6" x2="8" y2="4"/>
  <line stroke="green" x1="6" y1="6" x2="10" y2="6"/>
  <line stroke="red" x1="8" y1="4" x2="10" y2="6"/>
  <line stroke="green" x1="10" y1="0" x2="10" y2="6"/>
  <line stroke="black" x1="10" y1="0" x2="16" y2="0"/>
  <line stroke="red" x1="8" y1="8" x2="0" y2="16"/>
  <line stroke="green" x1="8" y1="8" x2="8" y2="16"/>
  <line stroke="green" x1="8" y1="8" x2="16" y2="8"/>
  <line stroke="red" x1="8" y1="8" x2="16" y2="16"/>
  <line stroke="red" x1="10" y1="6" x2="16" y2="0"/>
  <line stroke="green" x1="10" y1="6" x2="16" y2="6"/>
  <line stroke="black" x1="0" y1="16" x2="8" y2="16"/>
  <line stroke="black" x1="16" y1="0" x2="16" y2="6"/>
  <line stroke="black" x1="16" y1="6" x2="16" y2="8"/>
  <line stroke="black" x1="8" y1="16" x2="16" y2="16"/>
  <line stroke="black" x1="16" y1="8" x2="16" y2="16"/>
</svg>""",
    base_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">
  <line stroke="red" x1="0" y1="0" x2="1" y2="1"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="6"/>
  <line stroke="black" x1="0" y1="0" x2="8" y2="0"/>
  <line stroke="blue" x1="1" y1="1" x2="2" y2="2"/>
  <line stroke="red" x1="1" y1="1" x2="1" y2="6"/>
  <line stroke="red" x1="1" y1="1" x2="8" y2="1"/>
  <line stroke="red" x1="2" y1="2" x2="3" y2="3"/>
  <line stroke="blue" x1="2" y1="2" x2="2" y2="6"/>
  <line stroke="blue" x1="2" y1="2" x2="8" y2="2"/>
  <line stroke="blue" x1="3" y1="3" x2="4" y2="4"/>
  <line stroke="red" x1="3" y1="3" x2="3" y2="6"/>
  <line stroke="red" x1="3" y1="3" x2="8" y2="3"/>
  <line stroke="red" x1="4" y1="4" x2="5" y2="5"/>
  <line stroke="blue" x1="4" y1="4" x2="4" y2="6"/>
  <line stroke="blue" x1="4" y1="4" x2="8" y2="4"/>
  <line stroke="blue" x1="0" y1="6" x2="1" y2="6"/>
  <line stroke="black" x1="0" y1="6" x2="0" y2="16"/>
  <line stroke="red" x1="1" y1="6" x2="2" y2="6"/>
  <line stroke="red" x1="1" y1="6" x2="1" y2="15"/>
  <line stroke="blue" x1="2" y1="6" x2="3" y2="6"/>
  <line stroke="blue" x1="2" y1="6" x2="2" y2="14"/>
  <line stroke="red" x1="3" y1="6" x2="4" y2="6"/>
  <line stroke="red" x1="3" y1="6" x2="3" y2="13"/>
  <line stroke="red" x1="5" y1="5" x2="5" y2="6"/>
  <line stroke="blue" x1="5" y1="5" x2="6" y2="6"/>
  <line stroke="red" x1="5" y1="5" x2="7" y2="5"/>
  <line stroke="blue" x1="4" y1="6" x2="5" y2="6"/>
  <line stroke="blue" x1="4" y1="6" x2="4" y2="12"/>
  <line stroke="red" x1="5" y1="6" x2="6" y2="6"/>
  <line stroke="red" x1="5" y1="6" x2="5" y2="11"/>
  <line stroke="red" x1="8" y1="0" x2="8" y2="1"/>
  <line stroke="black" x1="8" y1="0" x2="16" y2="0"/>
  <line stroke="blue" x1="8" y1="1" x2="8" y2="2"/>
  <line stroke="red" x1="8" y1="1" x2="15" y2="1"/>
  <line stroke="red" x1="8" y1="2" x2="8" y2="3"/>
  <line stroke="blue" x1="8" y1="2" x2="14" y2="2"/>
  <line stroke="blue" x1="6" y1="6" x2="7" y2="5"/>
  <line stroke="blue" x1="6" y1="6" x2="6" y2="10"/>
  <line stroke="blue" x1="8" y1="3" x2="8" y2="4"/>
  <line stroke="red" x1="8" y1="3" x2="13" y2="3"/>
  <line stroke="red" x1="7" y1="5" x2="8" y2="4"/>
  <line stroke="red" x1="7" y1="5" x2="7" y2="9"/>
  <line stroke="red" x1="8" y1="4" x2="9" y2="5"/>
  <line stroke="blue" x1="8" y1="4" x2="8" y2="8"/>
  <line stroke="blue" x1="8" y1="4" x2="12" y2="4"/>
  <line stroke="blue" x1="9" y1="5" x2="10" y2="6"/>
  <line stroke="red" x1="9" y1="5" x2="11" y2="5"/>
  <line stroke="red" x1="9" y1="5" x2="9" y2="9"/>
  <line stroke="blue" x1="8" y1="8" x2="7" y2="9"/>
  <line stroke="red" x1="8" y1="8" x2="8" y2="9"/>
  <line stroke="blue" x1="8" y1="8" x2="9" y2="9"/>
  <line stroke="red" x1="7" y1="9" x2="6" y2="10"/>
  <line stroke="red" x1="7" y1="9" x2="8" y2="9"/>
  <line stroke="blue" x1="6" y1="10" x2="5" y2="11"/>
  <line stroke="blue" x1="6" y1="10" x2="8" y2="10"/>
  <line stroke="blue" x1="10" y1="6" x2="11" y2="5"/>
  <line stroke="red" x1="10" y1="6" x2="11" y2="6"/>
  <line stroke="blue" x1="10" y1="6" x2="10" y2="10"/>
  <line stroke="red" x1="8" y1="9" x2="9" y2="9"/>
  <line stroke="blue" x1="8" y1="9" x2="8" y2="10"/>
  <line stroke="red" x1="5" y1="11" x2="4" y2="12"/>
  <line stroke="red" x1="5" y1="11" x2="8" y2="11"/>
  <line stroke="red" x1="11" y1="5" x2="11" y2="6"/>
  <line stroke="red" x1="11" y1="5" x2="12" y2="4"/>
  <line stroke="blue" x1="11" y1="6" x2="12" y2="6"/>
  <line stroke="red" x1="11" y1="6" x2="11" y2="11"/>
  <line stroke="blue" x1="4" y1="12" x2="3" y2="13"/>
  <line stroke="blue" x1="4" y1="12" x2="8" y2="12"/>
  <line stroke="blue" x1="12" y1="4" x2="13" y2="3"/>
  <line stroke="blue" x1="12" y1="4" x2="12" y2="6"/>
  <line stroke="red" x1="9" y1="9" x2="10" y2="10"/>
  <line stroke="red" x1="8" y1="10" x2="8" y2="11"/>
  <line stroke="blue" x1="8" y1="10" x2="10" y2="10"/>
  <line stroke="red" x1="3" y1="13" x2="2" y2="14"/>
  <line stroke="red" x1="3" y1="13" x2="8" y2="13"/>
  <line stroke="red" x1="13" y1="3" x2="14" y2="2"/>
  <line stroke="red" x1="13" y1="3" x2="13" y2="6"/>
  <line stroke="red" x1="12" y1="6" x2="13" y2="6"/>
  <line stroke="blue" x1="12" y1="6" x2="12" y2="12"/>
  <line stroke="blue" x1="8" y1="11" x2="8" y2="12"/>
  <line stroke="red" x1="8" y1="11" x2="11" y2="11"/>
  <line stroke="blue" x1="2" y1="14" x2="1" y2="15"/>
  <line stroke="blue" x1="2" y1="14" x2="8" y2="14"/>
  <line stroke="blue" x1="10" y1="10" x2="11" y2="11"/>
  <line stroke="blue" x1="14" y1="2" x2="15" y2="1"/>
  <line stroke="blue" x1="14" y1="2" x2="14" y2="6"/>
  <line stroke="blue" x1="13" y1="6" x2="14" y2="6"/>
  <line stroke="red" x1="13" y1="6" x2="13" y2="13"/>
  <line stroke="red" x1="8" y1="12" x2="8" y2="13"/>
  <line stroke="blue" x1="8" y1="12" x2="12" y2="12"/>
  <line stroke="red" x1="1" y1="15" x2="0" y2="16"/>
  <line stroke="red" x1="1" y1="15" x2="8" y2="15"/>
  <line stroke="red" x1="15" y1="1" x2="16" y2="0"/>
  <line stroke="red" x1="15" y1="1" x2="15" y2="6"/>
  <line stroke="red" x1="14" y1="6" x2="15" y2="6"/>
  <line stroke="blue" x1="14" y1="6" x2="14" y2="14"/>
  <line stroke="blue" x1="8" y1="13" x2="8" y2="14"/>
  <line stroke="red" x1="8" y1="13" x2="13" y2="13"/>
  <line stroke="red" x1="11" y1="11" x2="12" y2="12"/>
  <line stroke="black" x1="0" y1="16" x2="8" y2="16"/>
  <line stroke="black" x1="16" y1="0" x2="16" y2="6"/>
  <line stroke="red" x1="8" y1="14" x2="8" y2="15"/>
  <line stroke="blue" x1="8" y1="14" x2="14" y2="14"/>
  <line stroke="blue" x1="15" y1="6" x2="16" y2="6"/>
  <line stroke="red" x1="15" y1="6" x2="15" y2="15"/>
  <line stroke="blue" x1="12" y1="12" x2="13" y2="13"/>
  <line stroke="blue" x1="8" y1="15" x2="8" y2="16"/>
  <line stroke="red" x1="8" y1="15" x2="15" y2="15"/>
  <line stroke="black" x1="16" y1="6" x2="16" y2="16"/>
  <line stroke="black" x1="8" y1="16" x2="16" y2="16"/>
  <line stroke="red" x1="13" y1="13" x2="14" y2="14"/>
  <line stroke="blue" x1="14" y1="14" x2="15" y2="15"/>
  <line stroke="red" x1="15" y1="15" x2="16" y2="16"/>
</svg>""",
    rectangles=[
        shaping_utils.Rectangle(
            name="left_arm",
            x_start=0,
            y_start=0,
            x_len=0.375,
            y_len=0.375,
        ),
        shaping_utils.Rectangle(
            name="head",
            x_start=0.375,
            y_start=0,
            x_len=0.25,
            y_len=0.375,
        ),
        shaping_utils.Rectangle(
            name="right_arm",
            x_start=0.375 + 0.25,
            y_start=0,
            x_len=0.375,
            y_len=0.375,
        ),
        shaping_utils.Rectangle(
            name="left_leg",
            x_start=0,
            y_start=0.5,
            x_len=0.5,
            y_len=0.5,
        ),
        shaping_utils.Rectangle(
            name="right_leg",
            x_start=0.5,
            y_start=0.5,
            x_len=0.5,
            y_len=0.5,
        ),
    ],
    target_zigzag_depths=[
        1,
        None,
        1,
        None,
        None,
    ],
    target_flap_angles=[
        None,
        None,
        None,
        None,
        None,
    ],
    fold_orders=[
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
    ],
    even_phases=[
        False,
        True,
        True,
        True,
        True,
    ],
    zigzag_shaping_svg_sha=(
        "9c7fb4b90c201ddf121bf53e04ddd47b94458b2004ce6f8367de1019e907d3da"
    ),
)

SCORPION = BoxPleatingCreasePatternExample(
    name="scorpion",
    tree_data={
        "version": "0.7",
        "design": {
            "title": "Scorpion",
            "mode": "layout",
            "description": "",
            "layout": {
                "sheet": {"type": "rect", "height": 32, "width": 32},
                "flaps": [
                    {"id": 1, "x": 32, "y": 32, "width": 0, "height": 0},
                    {"id": 2, "x": 24, "y": 32, "width": 0, "height": 0},
                    {"id": 5, "x": 8, "y": 32, "width": 0, "height": 0},
                    {"id": 6, "x": 0, "y": 32, "width": 0, "height": 0},
                    {"id": 7, "x": 0, "y": 12, "width": 0, "height": 0},
                    {"id": 8, "x": 0, "y": 20, "width": 0, "height": 0},
                    {"id": 9, "x": 0, "y": 4, "width": 0, "height": 0},
                    {"id": 10, "x": 32, "y": 20, "width": 0, "height": 0},
                    {"id": 11, "x": 32, "y": 12, "width": 0, "height": 0},
                    {"id": 12, "x": 32, "y": 4, "width": 0, "height": 0},
                    {"id": 13, "x": 24, "y": 20, "width": 0, "height": 0},
                    {"id": 14, "x": 8, "y": 20, "width": 0, "height": 0},
                    {"id": 15, "x": 16, "y": 0, "width": 0, "height": 0},
                    {"id": 16, "x": 6, "y": 14, "width": 20, "height": 0},
                    {"id": 18, "x": 13, "y": 17, "width": 6, "height": 0},
                    {"id": 17, "x": 13, "y": 19, "width": 6, "height": 0},
                    {"id": 19, "x": 13, "y": 23, "width": 6, "height": 0},
                    {"id": 20, "x": 13, "y": 21, "width": 6, "height": 0},
                ],
                "stretches": [],
            },
            "tree": {
                "sheet": {"type": "rect", "height": 20, "width": 20},
                "nodes": [
                    {"id": 0, "name": "", "x": 6, "y": 13},
                    {"id": 1, "name": "", "x": 4, "y": 15},
                    {"id": 2, "name": "", "x": 8, "y": 15},
                    {"id": 3, "name": "", "x": 10, "y": 11},
                    {"id": 4, "name": "", "x": 14, "y": 13},
                    {"id": 5, "name": "", "x": 16, "y": 15},
                    {"id": 6, "name": "", "x": 12, "y": 15},
                    {"id": 7, "name": "", "x": 14, "y": 8},
                    {"id": 8, "name": "", "x": 14, "y": 11},
                    {"id": 9, "name": "", "x": 6, "y": 6},
                    {"id": 10, "name": "", "x": 6, "y": 10},
                    {"id": 11, "name": "", "x": 14, "y": 10},
                    {"id": 12, "name": "", "x": 14, "y": 6},
                    {"id": 13, "name": "", "x": 6, "y": 11},
                    {"id": 14, "name": "", "x": 6, "y": 8},
                    {"id": 15, "name": "", "x": 10, "y": 2},
                    {"id": 16, "name": "", "x": 9, "y": 9},
                    {"id": 17, "name": "", "x": 11, "y": 9},
                    {"id": 18, "name": "", "x": 10, "y": 12},
                    {"id": 19, "name": "", "x": 11, "y": 11},
                    {"id": 20, "name": "", "x": 10, "y": 10},
                ],
                "edges": [
                    {"n1": 3, "n2": 4, "length": 4},
                    {"n1": 3, "n2": 0, "length": 4},
                    {"n1": 3, "n2": 10, "length": 4},
                    {"n1": 3, "n2": 12, "length": 4},
                    {"n1": 3, "n2": 15, "length": 12},
                    {"n1": 3, "n2": 7, "length": 4},
                    {"n1": 3, "n2": 11, "length": 4},
                    {"n1": 3, "n2": 8, "length": 4},
                    {"n1": 3, "n2": 13, "length": 4},
                    {"n1": 3, "n2": 14, "length": 4},
                    {"n1": 3, "n2": 9, "length": 4},
                    {"n1": 3, "n2": 16, "length": 2},
                    {"n1": 3, "n2": 17, "length": 1},
                    {"n1": 3, "n2": 18, "length": 1},
                    {"n1": 3, "n2": 19, "length": 1},
                    {"n1": 3, "n2": 20, "length": 1},
                    {"n1": 4, "n2": 5, "length": 4},
                    {"n1": 4, "n2": 6, "length": 4},
                    {"n1": 0, "n2": 1, "length": 4},
                    {"n1": 0, "n2": 2, "length": 4},
                ],
            },
        },
    },
    grid_size=32,
    packing_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <line stroke="black" x1="0" y1="0" x2="0" y2="4"/>
  <line stroke="black" x1="0" y1="0" x2="4" y2="0"/>
  <line stroke="red" x1="0" y1="0" x2="4" y2="4"/>
  <line stroke="green" x1="0" y1="4" x2="4" y2="4"/>
  <line stroke="black" x1="0" y1="4" x2="0" y2="8"/>
  <line stroke="green" x1="4" y1="0" x2="4" y2="4"/>
  <line stroke="black" x1="4" y1="0" x2="8" y2="0"/>
  <line stroke="red" x1="4" y1="4" x2="8" y2="0"/>
  <line stroke="green" x1="4" y1="4" x2="12" y2="4"/>
  <line stroke="green" x1="0" y1="8" x2="4" y2="8"/>
  <line stroke="black" x1="0" y1="8" x2="0" y2="12"/>
  <line stroke="black" x1="8" y1="0" x2="12" y2="0"/>
  <line stroke="red" x1="8" y1="0" x2="12" y2="4"/>
  <line stroke="red" x1="4" y1="8" x2="0" y2="12"/>
  <line stroke="red" x1="4" y1="8" x2="8" y2="12"/>
  <line stroke="green" x1="4" y1="8" x2="12" y2="8"/>
  <line stroke="green" x1="4" y1="8" x2="4" y2="16"/>
  <line stroke="black" x1="0" y1="12" x2="0" y2="16"/>
  <line stroke="red" x1="0" y1="12" x2="4" y2="16"/>
  <line stroke="green" x1="12" y1="0" x2="12" y2="4"/>
  <line stroke="black" x1="12" y1="0" x2="16" y2="0"/>
  <line stroke="red" x1="12" y1="4" x2="16" y2="8"/>
  <line stroke="red" x1="8" y1="12" x2="12" y2="8"/>
  <line stroke="red" x1="8" y1="12" x2="4" y2="16"/>
  <line stroke="red" x1="8" y1="12" x2="12" y2="16"/>
  <line stroke="green" x1="12" y1="8" x2="12" y2="10"/>
  <line stroke="red" x1="12" y1="8" x2="13" y2="9"/>
  <line stroke="green" x1="12" y1="8" x2="16" y2="8"/>
  <line stroke="red" x1="12" y1="10" x2="13" y2="9"/>
  <line stroke="green" x1="12" y1="10" x2="12" y2="12"/>
  <line stroke="red" x1="12" y1="10" x2="13" y2="11"/>
  <line stroke="green" x1="12" y1="10" x2="20" y2="10"/>
  <line stroke="red" x1="13" y1="9" x2="19" y2="9"/>
  <line stroke="green" x1="0" y1="16" x2="4" y2="16"/>
  <line stroke="black" x1="0" y1="16" x2="0" y2="20"/>
  <line stroke="green" x1="16" y1="0" x2="16" y2="8"/>
  <line stroke="black" x1="16" y1="0" x2="20" y2="0"/>
  <line stroke="red" x1="4" y1="16" x2="6" y2="18"/>
  <line stroke="red" x1="4" y1="16" x2="0" y2="20"/>
  <line stroke="green" x1="4" y1="16" x2="12" y2="16"/>
  <line stroke="green" x1="4" y1="16" x2="4" y2="20"/>
  <line stroke="red" x1="12" y1="12" x2="13" y2="11"/>
  <line stroke="red" x1="12" y1="12" x2="13" y2="13"/>
  <line stroke="green" x1="12" y1="12" x2="12" y2="14"/>
  <line stroke="green" x1="12" y1="12" x2="20" y2="12"/>
  <line stroke="red" x1="13" y1="11" x2="19" y2="11"/>
  <line stroke="red" x1="16" y1="8" x2="20" y2="4"/>
  <line stroke="green" x1="16" y1="8" x2="20" y2="8"/>
  <line stroke="red" x1="13" y1="13" x2="12" y2="14"/>
  <line stroke="red" x1="13" y1="13" x2="19" y2="13"/>
  <line stroke="red" x1="12" y1="14" x2="13" y2="15"/>
  <line stroke="green" x1="12" y1="14" x2="12" y2="16"/>
  <line stroke="green" x1="12" y1="14" x2="20" y2="14"/>
  <line stroke="red" x1="6" y1="18" x2="4" y2="20"/>
  <line stroke="red" x1="6" y1="18" x2="26" y2="18"/>
  <line stroke="red" x1="13" y1="15" x2="12" y2="16"/>
  <line stroke="red" x1="13" y1="15" x2="19" y2="15"/>
  <line stroke="black" x1="0" y1="20" x2="0" y2="24"/>
  <line stroke="red" x1="0" y1="20" x2="4" y2="24"/>
  <line stroke="green" x1="12" y1="16" x2="20" y2="16"/>
  <line stroke="green" x1="20" y1="0" x2="20" y2="4"/>
  <line stroke="black" x1="20" y1="0" x2="24" y2="0"/>
  <line stroke="green" x1="4" y1="20" x2="4" y2="24"/>
  <line stroke="green" x1="4" y1="20" x2="28" y2="20"/>
  <line stroke="red" x1="4" y1="20" x2="16" y2="32"/>
  <line stroke="red" x1="20" y1="4" x2="24" y2="0"/>
  <line stroke="green" x1="20" y1="4" x2="28" y2="4"/>
  <line stroke="red" x1="19" y1="9" x2="20" y2="8"/>
  <line stroke="red" x1="19" y1="9" x2="20" y2="10"/>
  <line stroke="green" x1="20" y1="8" x2="20" y2="10"/>
  <line stroke="red" x1="20" y1="8" x2="24" y2="12"/>
  <line stroke="green" x1="20" y1="8" x2="28" y2="8"/>
  <line stroke="red" x1="19" y1="11" x2="20" y2="10"/>
  <line stroke="red" x1="19" y1="11" x2="20" y2="12"/>
  <line stroke="green" x1="20" y1="10" x2="20" y2="12"/>
  <line stroke="red" x1="19" y1="13" x2="20" y2="12"/>
  <line stroke="red" x1="19" y1="13" x2="20" y2="14"/>
  <line stroke="green" x1="20" y1="12" x2="20" y2="14"/>
  <line stroke="green" x1="0" y1="24" x2="4" y2="24"/>
  <line stroke="black" x1="0" y1="24" x2="0" y2="28"/>
  <line stroke="black" x1="24" y1="0" x2="28" y2="0"/>
  <line stroke="red" x1="24" y1="0" x2="28" y2="4"/>
  <line stroke="red" x1="19" y1="15" x2="20" y2="14"/>
  <line stroke="red" x1="19" y1="15" x2="20" y2="16"/>
  <line stroke="red" x1="4" y1="24" x2="0" y2="28"/>
  <line stroke="green" x1="4" y1="24" x2="4" y2="32"/>
  <line stroke="green" x1="20" y1="14" x2="20" y2="16"/>
  <line stroke="red" x1="20" y1="16" x2="24" y2="12"/>
  <line stroke="green" x1="20" y1="16" x2="28" y2="16"/>
  <line stroke="red" x1="24" y1="12" x2="28" y2="8"/>
  <line stroke="red" x1="24" y1="12" x2="28" y2="16"/>
  <line stroke="black" x1="0" y1="28" x2="0" y2="32"/>
  <line stroke="red" x1="0" y1="28" x2="4" y2="32"/>
  <line stroke="green" x1="28" y1="0" x2="28" y2="4"/>
  <line stroke="black" x1="28" y1="0" x2="32" y2="0"/>
  <line stroke="red" x1="28" y1="4" x2="32" y2="0"/>
  <line stroke="green" x1="28" y1="4" x2="32" y2="4"/>
  <line stroke="green" x1="28" y1="8" x2="28" y2="16"/>
  <line stroke="green" x1="28" y1="8" x2="32" y2="8"/>
  <line stroke="red" x1="28" y1="8" x2="32" y2="12"/>
  <line stroke="red" x1="26" y1="18" x2="28" y2="16"/>
  <line stroke="red" x1="26" y1="18" x2="28" y2="20"/>
  <line stroke="black" x1="0" y1="32" x2="4" y2="32"/>
  <line stroke="black" x1="32" y1="0" x2="32" y2="4"/>
  <line stroke="black" x1="4" y1="32" x2="16" y2="32"/>
  <line stroke="red" x1="28" y1="16" x2="32" y2="12"/>
  <line stroke="green" x1="28" y1="16" x2="28" y2="20"/>
  <line stroke="green" x1="28" y1="16" x2="32" y2="16"/>
  <line stroke="red" x1="28" y1="16" x2="32" y2="20"/>
  <line stroke="black" x1="32" y1="4" x2="32" y2="8"/>
  <line stroke="black" x1="32" y1="8" x2="32" y2="12"/>
  <line stroke="black" x1="32" y1="12" x2="32" y2="16"/>
  <line stroke="red" x1="28" y1="20" x2="16" y2="32"/>
  <line stroke="green" x1="28" y1="20" x2="28" y2="24"/>
  <line stroke="black" x1="16" y1="32" x2="28" y2="32"/>
  <line stroke="black" x1="32" y1="16" x2="32" y2="20"/>
  <line stroke="red" x1="28" y1="24" x2="32" y2="20"/>
  <line stroke="green" x1="28" y1="24" x2="32" y2="24"/>
  <line stroke="green" x1="28" y1="24" x2="28" y2="32"/>
  <line stroke="red" x1="28" y1="24" x2="32" y2="28"/>
  <line stroke="black" x1="32" y1="20" x2="32" y2="24"/>
  <line stroke="black" x1="32" y1="24" x2="32" y2="28"/>
  <line stroke="red" x1="28" y1="32" x2="32" y2="28"/>
  <line stroke="black" x1="28" y1="32" x2="32" y2="32"/>
  <line stroke="black" x1="32" y1="28" x2="32" y2="32"/>
</svg>
""",
    assigned_pleats_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <line stroke="black" x1="0" y1="0" x2="8" y2="0"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="12"/>
  <line stroke="red" x1="1" y1="1" x2="7" y2="1"/>
  <line stroke="red" x1="1" y1="1" x2="1" y2="11"/>
  <line stroke="blue" x1="2" y1="2" x2="6" y2="2"/>
  <line stroke="blue" x1="2" y1="2" x2="2" y2="10"/>
  <line stroke="red" x1="3" y1="3" x2="5" y2="3"/>
  <line stroke="red" x1="3" y1="3" x2="3" y2="9"/>
  <line stroke="blue" x1="4" y1="4" x2="4" y2="8"/>
  <line stroke="red" x1="5" y1="3" x2="5" y2="9"/>
  <line stroke="blue" x1="6" y1="2" x2="6" y2="10"/>
  <line stroke="red" x1="7" y1="1" x2="7" y2="11"/>
  <line stroke="blue" x1="8" y1="0" x2="8" y2="12"/>
  <line stroke="black" x1="8" y1="0" x2="24" y2="0"/>
  <line stroke="red" x1="9" y1="1" x2="9" y2="11"/>
  <line stroke="red" x1="9" y1="1" x2="23" y2="1"/>
  <line stroke="red" x1="3" y1="9" x2="5" y2="9"/>
  <line stroke="blue" x1="2" y1="10" x2="6" y2="10"/>
  <line stroke="blue" x1="10" y1="2" x2="10" y2="10"/>
  <line stroke="blue" x1="10" y1="2" x2="22" y2="2"/>
  <line stroke="red" x1="1" y1="11" x2="7" y2="11"/>
  <line stroke="red" x1="11" y1="3" x2="11" y2="9"/>
  <line stroke="red" x1="11" y1="3" x2="21" y2="3"/>
  <line stroke="blue" x1="0" y1="12" x2="8" y2="12"/>
  <line stroke="black" x1="0" y1="12" x2="0" y2="20"/>
  <line stroke="blue" x1="12" y1="4" x2="12" y2="8"/>
  <line stroke="blue" x1="12" y1="4" x2="20" y2="4"/>
  <line stroke="red" x1="1" y1="13" x2="7" y2="13"/>
  <line stroke="red" x1="1" y1="13" x2="1" y2="19"/>
  <line stroke="red" x1="13" y1="5" x2="13" y2="9"/>
  <line stroke="red" x1="13" y1="5" x2="19" y2="5"/>
  <line stroke="blue" x1="2" y1="14" x2="6" y2="14"/>
  <line stroke="blue" x1="2" y1="14" x2="2" y2="18"/>
  <line stroke="blue" x1="10" y1="10" x2="12" y2="10"/>
  <line stroke="red" x1="9" y1="11" x2="13" y2="11"/>
  <line stroke="red" x1="11" y1="9" x2="13" y2="9"/>
  <line stroke="blue" x1="8" y1="12" x2="12" y2="12"/>
  <line stroke="blue" x1="8" y1="12" x2="8" y2="24"/>
  <line stroke="red" x1="7" y1="13" x2="7" y2="23"/>
  <line stroke="blue" x1="6" y1="14" x2="6" y2="18"/>
  <line stroke="blue" x1="14" y1="6" x2="18" y2="6"/>
  <line stroke="blue" x1="14" y1="6" x2="14" y2="30"/>
  <line stroke="red" x1="3" y1="15" x2="5" y2="15"/>
  <line stroke="red" x1="3" y1="15" x2="3" y2="17"/>
  <line stroke="red" x1="5" y1="15" x2="5" y2="17"/>
  <line stroke="red" x1="9" y1="13" x2="13" y2="13"/>
  <line stroke="red" x1="9" y1="13" x2="9" y2="25"/>
  <line stroke="red" x1="13" y1="9" x2="13" y2="11"/>
  <line stroke="red" x1="15" y1="7" x2="17" y2="7"/>
  <line stroke="red" x1="15" y1="7" x2="15" y2="31"/>
  <line stroke="red" x1="13" y1="11" x2="13" y2="13"/>
  <line stroke="blue" x1="10" y1="14" x2="12" y2="14"/>
  <line stroke="blue" x1="10" y1="14" x2="10" y2="26"/>
  <line stroke="red" x1="3" y1="17" x2="5" y2="17"/>
  <line stroke="blue" x1="16" y1="8" x2="16" y2="32"/>
  <line stroke="blue" x1="2" y1="18" x2="6" y2="18"/>
  <line stroke="red" x1="13" y1="13" x2="13" y2="15"/>
  <line stroke="red" x1="17" y1="7" x2="17" y2="31"/>
  <line stroke="red" x1="11" y1="15" x2="13" y2="15"/>
  <line stroke="red" x1="11" y1="15" x2="11" y2="27"/>
  <line stroke="blue" x1="6" y1="18" x2="6" y2="22"/>
  <line stroke="blue" x1="18" y1="6" x2="18" y2="30"/>
  <line stroke="red" x1="1" y1="19" x2="5" y2="19"/>
  <line stroke="red" x1="5" y1="19" x2="5" y2="21"/>
  <line stroke="red" x1="19" y1="5" x2="19" y2="9"/>
  <line stroke="red" x1="13" y1="15" x2="13" y2="29"/>
  <line stroke="blue" x1="0" y1="20" x2="4" y2="20"/>
  <line stroke="black" x1="0" y1="20" x2="0" y2="28"/>
  <line stroke="blue" x1="12" y1="16" x2="12" y2="28"/>
  <line stroke="blue" x1="20" y1="4" x2="20" y2="8"/>
  <line stroke="red" x1="1" y1="21" x2="5" y2="21"/>
  <line stroke="red" x1="1" y1="21" x2="1" y2="27"/>
  <line stroke="red" x1="19" y1="9" x2="19" y2="11"/>
  <line stroke="red" x1="19" y1="9" x2="21" y2="9"/>
  <line stroke="red" x1="21" y1="3" x2="21" y2="9"/>
  <line stroke="red" x1="19" y1="11" x2="19" y2="13"/>
  <line stroke="red" x1="19" y1="11" x2="23" y2="11"/>
  <line stroke="blue" x1="2" y1="22" x2="6" y2="22"/>
  <line stroke="blue" x1="2" y1="22" x2="2" y2="26"/>
  <line stroke="blue" x1="22" y1="2" x2="22" y2="10"/>
  <line stroke="blue" x1="20" y1="10" x2="22" y2="10"/>
  <line stroke="red" x1="19" y1="13" x2="19" y2="15"/>
  <line stroke="red" x1="19" y1="13" x2="23" y2="13"/>
  <line stroke="red" x1="23" y1="1" x2="23" y2="11"/>
  <line stroke="red" x1="3" y1="23" x2="7" y2="23"/>
  <line stroke="red" x1="3" y1="23" x2="3" y2="25"/>
  <line stroke="blue" x1="20" y1="12" x2="24" y2="12"/>
  <line stroke="blue" x1="24" y1="0" x2="24" y2="12"/>
  <line stroke="black" x1="24" y1="0" x2="32" y2="0"/>
  <line stroke="red" x1="19" y1="15" x2="21" y2="15"/>
  <line stroke="red" x1="19" y1="15" x2="19" y2="29"/>
  <line stroke="blue" x1="4" y1="24" x2="8" y2="24"/>
  <line stroke="blue" x1="20" y1="14" x2="22" y2="14"/>
  <line stroke="red" x1="25" y1="1" x2="25" y2="11"/>
  <line stroke="red" x1="25" y1="1" x2="31" y2="1"/>
  <line stroke="red" x1="3" y1="25" x2="9" y2="25"/>
  <line stroke="blue" x1="20" y1="16" x2="20" y2="28"/>
  <line stroke="red" x1="21" y1="15" x2="21" y2="27"/>
  <line stroke="blue" x1="2" y1="26" x2="10" y2="26"/>
  <line stroke="blue" x1="22" y1="14" x2="22" y2="26"/>
  <line stroke="blue" x1="26" y1="2" x2="26" y2="10"/>
  <line stroke="blue" x1="26" y1="2" x2="30" y2="2"/>
  <line stroke="red" x1="23" y1="13" x2="23" y2="25"/>
  <line stroke="blue" x1="24" y1="12" x2="24" y2="24"/>
  <line stroke="blue" x1="24" y1="12" x2="32" y2="12"/>
  <line stroke="red" x1="1" y1="27" x2="11" y2="27"/>
  <line stroke="red" x1="27" y1="3" x2="27" y2="9"/>
  <line stroke="red" x1="27" y1="3" x2="29" y2="3"/>
  <line stroke="red" x1="25" y1="11" x2="31" y2="11"/>
  <line stroke="blue" x1="26" y1="10" x2="30" y2="10"/>
  <line stroke="blue" x1="0" y1="28" x2="12" y2="28"/>
  <line stroke="black" x1="0" y1="28" x2="0" y2="32"/>
  <line stroke="red" x1="25" y1="13" x2="31" y2="13"/>
  <line stroke="red" x1="25" y1="13" x2="25" y2="23"/>
  <line stroke="blue" x1="28" y1="4" x2="28" y2="8"/>
  <line stroke="red" x1="27" y1="9" x2="29" y2="9"/>
  <line stroke="red" x1="1" y1="29" x2="13" y2="29"/>
  <line stroke="red" x1="1" y1="29" x2="1" y2="32"/>
  <line stroke="red" x1="29" y1="3" x2="29" y2="9"/>
  <line stroke="blue" x1="26" y1="14" x2="26" y2="18"/>
  <line stroke="blue" x1="26" y1="14" x2="30" y2="14"/>
  <line stroke="blue" x1="2" y1="30" x2="2" y2="32"/>
  <line stroke="blue" x1="2" y1="30" x2="14" y2="30"/>
  <line stroke="blue" x1="30" y1="2" x2="30" y2="10"/>
  <line stroke="red" x1="27" y1="15" x2="27" y2="17"/>
  <line stroke="red" x1="27" y1="15" x2="29" y2="15"/>
  <line stroke="red" x1="31" y1="1" x2="31" y2="11"/>
  <line stroke="red" x1="3" y1="31" x2="3" y2="32"/>
  <line stroke="red" x1="3" y1="31" x2="15" y2="31"/>
  <line stroke="blue" x1="26" y1="18" x2="26" y2="22"/>
  <line stroke="blue" x1="26" y1="18" x2="30" y2="18"/>
  <line stroke="red" x1="27" y1="17" x2="29" y2="17"/>
  <line stroke="black" x1="0" y1="32" x2="1" y2="32"/>
  <line stroke="black" x1="32" y1="0" x2="32" y2="12"/>
  <line stroke="black" x1="1" y1="32" x2="2" y2="32"/>
  <line stroke="black" x1="2" y1="32" x2="3" y2="32"/>
  <line stroke="black" x1="3" y1="32" x2="16" y2="32"/>
  <line stroke="red" x1="29" y1="15" x2="29" y2="17"/>
  <line stroke="red" x1="27" y1="19" x2="27" y2="21"/>
  <line stroke="red" x1="27" y1="19" x2="31" y2="19"/>
  <line stroke="blue" x1="30" y1="14" x2="30" y2="18"/>
  <line stroke="red" x1="31" y1="13" x2="31" y2="19"/>
  <line stroke="blue" x1="24" y1="24" x2="28" y2="24"/>
  <line stroke="red" x1="23" y1="25" x2="29" y2="25"/>
  <line stroke="red" x1="25" y1="23" x2="29" y2="23"/>
  <line stroke="blue" x1="22" y1="26" x2="30" y2="26"/>
  <line stroke="blue" x1="26" y1="22" x2="30" y2="22"/>
  <line stroke="black" x1="32" y1="12" x2="32" y2="20"/>
  <line stroke="red" x1="21" y1="27" x2="31" y2="27"/>
  <line stroke="red" x1="27" y1="21" x2="31" y2="21"/>
  <line stroke="blue" x1="20" y1="28" x2="32" y2="28"/>
  <line stroke="blue" x1="28" y1="20" x2="32" y2="20"/>
  <line stroke="red" x1="19" y1="29" x2="31" y2="29"/>
  <line stroke="blue" x1="18" y1="30" x2="30" y2="30"/>
  <line stroke="red" x1="17" y1="31" x2="29" y2="31"/>
  <line stroke="black" x1="16" y1="32" x2="29" y2="32"/>
  <line stroke="red" x1="29" y1="23" x2="29" y2="25"/>
  <line stroke="blue" x1="30" y1="22" x2="30" y2="26"/>
  <line stroke="red" x1="31" y1="21" x2="31" y2="27"/>
  <line stroke="black" x1="32" y1="20" x2="32" y2="28"/>
  <line stroke="blue" x1="30" y1="30" x2="30" y2="32"/>
  <line stroke="red" x1="29" y1="31" x2="29" y2="32"/>
  <line stroke="red" x1="31" y1="29" x2="31" y2="32"/>
  <line stroke="black" x1="32" y1="28" x2="32" y2="32"/>
  <line stroke="black" x1="29" y1="32" x2="30" y2="32"/>
  <line stroke="black" x1="30" y1="32" x2="31" y2="32"/>
  <line stroke="black" x1="31" y1="32" x2="32" y2="32"/>
</svg>""",
    assigned_ridges_svg="""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="red" x1="0" y1="0" x2="31" y2="31"/>
  <line stroke="black" x1="0" y1="0" x2="250" y2="0"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="375"/>
  <line stroke="blue" x1="31" y1="31" x2="62" y2="62"/>
  <line stroke="red" x1="31" y1="31" x2="219" y2="31"/>
  <line stroke="red" x1="31" y1="31" x2="31" y2="344"/>
  <line stroke="red" x1="62" y1="62" x2="94" y2="94"/>
  <line stroke="blue" x1="62" y1="62" x2="188" y2="62"/>
  <line stroke="blue" x1="62" y1="62" x2="62" y2="312"/>
  <line stroke="blue" x1="94" y1="94" x2="125" y2="125"/>
  <line stroke="red" x1="94" y1="94" x2="156" y2="94"/>
  <line stroke="red" x1="94" y1="94" x2="94" y2="281"/>
  <line stroke="blue" x1="125" y1="125" x2="156" y2="94"/>
  <line stroke="blue" x1="125" y1="125" x2="125" y2="250"/>
  <line stroke="red" x1="156" y1="94" x2="188" y2="62"/>
  <line stroke="red" x1="156" y1="94" x2="156" y2="281"/>
  <line stroke="blue" x1="188" y1="62" x2="219" y2="31"/>
  <line stroke="blue" x1="188" y1="62" x2="188" y2="312"/>
  <line stroke="red" x1="219" y1="31" x2="250" y2="0"/>
  <line stroke="red" x1="219" y1="31" x2="219" y2="344"/>
  <line stroke="red" x1="250" y1="0" x2="281" y2="31"/>
  <line stroke="blue" x1="250" y1="0" x2="250" y2="375"/>
  <line stroke="black" x1="250" y1="0" x2="750" y2="0"/>
  <line stroke="blue" x1="125" y1="250" x2="94" y2="281"/>
  <line stroke="blue" x1="125" y1="250" x2="156" y2="281"/>
  <line stroke="blue" x1="281" y1="31" x2="312" y2="62"/>
  <line stroke="red" x1="281" y1="31" x2="281" y2="344"/>
  <line stroke="red" x1="281" y1="31" x2="719" y2="31"/>
  <line stroke="red" x1="94" y1="281" x2="62" y2="312"/>
  <line stroke="red" x1="94" y1="281" x2="156" y2="281"/>
  <line stroke="blue" x1="62" y1="312" x2="31" y2="344"/>
  <line stroke="blue" x1="62" y1="312" x2="188" y2="312"/>
  <line stroke="red" x1="312" y1="62" x2="344" y2="94"/>
  <line stroke="blue" x1="312" y1="62" x2="312" y2="312"/>
  <line stroke="blue" x1="312" y1="62" x2="688" y2="62"/>
  <line stroke="red" x1="156" y1="281" x2="188" y2="312"/>
  <line stroke="red" x1="31" y1="344" x2="0" y2="375"/>
  <line stroke="red" x1="31" y1="344" x2="219" y2="344"/>
  <line stroke="blue" x1="344" y1="94" x2="375" y2="125"/>
  <line stroke="red" x1="344" y1="94" x2="344" y2="281"/>
  <line stroke="red" x1="344" y1="94" x2="656" y2="94"/>
  <line stroke="blue" x1="188" y1="312" x2="219" y2="344"/>
  <line stroke="red" x1="0" y1="375" x2="31" y2="406"/>
  <line stroke="blue" x1="0" y1="375" x2="250" y2="375"/>
  <line stroke="black" x1="0" y1="375" x2="0" y2="625"/>
  <line stroke="red" x1="375" y1="125" x2="406" y2="156"/>
  <line stroke="blue" x1="375" y1="125" x2="375" y2="250"/>
  <line stroke="blue" x1="375" y1="125" x2="625" y2="125"/>
  <line stroke="blue" x1="31" y1="406" x2="62" y2="438"/>
  <line stroke="red" x1="31" y1="406" x2="219" y2="406"/>
  <line stroke="red" x1="31" y1="406" x2="31" y2="594"/>
  <line stroke="red" x1="219" y1="344" x2="250" y2="375"/>
  <line stroke="blue" x1="406" y1="156" x2="438" y2="188"/>
  <line stroke="red" x1="406" y1="156" x2="406" y2="281"/>
  <line stroke="red" x1="406" y1="156" x2="594" y2="156"/>
  <line stroke="blue" x1="62" y1="438" x2="188" y2="438"/>
  <line stroke="red" x1="62" y1="438" x2="94" y2="469"/>
  <line stroke="blue" x1="62" y1="438" x2="62" y2="562"/>
  <line stroke="blue" x1="312" y1="312" x2="281" y2="344"/>
  <line stroke="red" x1="312" y1="312" x2="344" y2="281"/>
  <line stroke="blue" x1="312" y1="312" x2="375" y2="312"/>
  <line stroke="red" x1="281" y1="344" x2="250" y2="375"/>
  <line stroke="red" x1="281" y1="344" x2="406" y2="344"/>
  <line stroke="blue" x1="344" y1="281" x2="375" y2="250"/>
  <line stroke="red" x1="344" y1="281" x2="406" y2="281"/>
  <line stroke="red" x1="250" y1="375" x2="219" y2="406"/>
  <line stroke="red" x1="250" y1="375" x2="281" y2="406"/>
  <line stroke="blue" x1="250" y1="375" x2="375" y2="375"/>
  <line stroke="blue" x1="250" y1="375" x2="250" y2="750"/>
  <line stroke="blue" x1="375" y1="250" x2="406" y2="281"/>
  <line stroke="blue" x1="219" y1="406" x2="188" y2="438"/>
  <line stroke="red" x1="219" y1="406" x2="219" y2="719"/>
  <line stroke="red" x1="188" y1="438" x2="156" y2="469"/>
  <line stroke="blue" x1="188" y1="438" x2="188" y2="562"/>
  <line stroke="red" x1="438" y1="188" x2="469" y2="219"/>
  <line stroke="blue" x1="438" y1="188" x2="562" y2="188"/>
  <line stroke="blue" x1="438" y1="188" x2="438" y2="938"/>
  <line stroke="red" x1="94" y1="469" x2="156" y2="469"/>
  <line stroke="blue" x1="94" y1="469" x2="125" y2="500"/>
  <line stroke="red" x1="94" y1="469" x2="94" y2="531"/>
  <line stroke="blue" x1="375" y1="312" x2="406" y2="281"/>
  <line stroke="blue" x1="375" y1="312" x2="406" y2="344"/>
  <line stroke="blue" x1="156" y1="469" x2="125" y2="500"/>
  <line stroke="red" x1="156" y1="469" x2="156" y2="531"/>
  <line stroke="blue" x1="281" y1="406" x2="312" y2="438"/>
  <line stroke="red" x1="281" y1="406" x2="406" y2="406"/>
  <line stroke="red" x1="281" y1="406" x2="281" y2="781"/>
  <line stroke="red" x1="406" y1="281" x2="406" y2="344"/>
  <line stroke="blue" x1="125" y1="500" x2="94" y2="531"/>
  <line stroke="blue" x1="125" y1="500" x2="156" y2="531"/>
  <line stroke="blue" x1="469" y1="219" x2="500" y2="250"/>
  <line stroke="red" x1="469" y1="219" x2="531" y2="219"/>
  <line stroke="red" x1="469" y1="219" x2="469" y2="969"/>
  <line stroke="blue" x1="375" y1="375" x2="406" y2="344"/>
  <line stroke="blue" x1="375" y1="375" x2="406" y2="406"/>
  <line stroke="red" x1="406" y1="344" x2="406" y2="406"/>
  <line stroke="blue" x1="312" y1="438" x2="375" y2="438"/>
  <line stroke="red" x1="312" y1="438" x2="344" y2="469"/>
  <line stroke="blue" x1="312" y1="438" x2="312" y2="812"/>
  <line stroke="red" x1="94" y1="531" x2="156" y2="531"/>
  <line stroke="red" x1="94" y1="531" x2="62" y2="562"/>
  <line stroke="red" x1="156" y1="531" x2="188" y2="562"/>
  <line stroke="blue" x1="500" y1="250" x2="531" y2="219"/>
  <line stroke="blue" x1="500" y1="250" x2="500" y2="1000"/>
  <line stroke="blue" x1="62" y1="562" x2="188" y2="562"/>
  <line stroke="blue" x1="62" y1="562" x2="31" y2="594"/>
  <line stroke="blue" x1="406" y1="406" x2="375" y2="438"/>
  <line stroke="red" x1="406" y1="406" x2="406" y2="469"/>
  <line stroke="red" x1="531" y1="219" x2="562" y2="188"/>
  <line stroke="red" x1="531" y1="219" x2="531" y2="969"/>
  <line stroke="blue" x1="375" y1="438" x2="406" y2="469"/>
  <line stroke="red" x1="344" y1="469" x2="406" y2="469"/>
  <line stroke="blue" x1="344" y1="469" x2="375" y2="500"/>
  <line stroke="red" x1="344" y1="469" x2="344" y2="844"/>
  <line stroke="red" x1="188" y1="562" x2="156" y2="594"/>
  <line stroke="blue" x1="188" y1="562" x2="188" y2="688"/>
  <line stroke="blue" x1="562" y1="188" x2="594" y2="156"/>
  <line stroke="blue" x1="562" y1="188" x2="562" y2="938"/>
  <line stroke="red" x1="31" y1="594" x2="156" y2="594"/>
  <line stroke="red" x1="31" y1="594" x2="0" y2="625"/>
  <line stroke="blue" x1="156" y1="594" x2="125" y2="625"/>
  <line stroke="red" x1="156" y1="594" x2="156" y2="656"/>
  <line stroke="red" x1="594" y1="156" x2="625" y2="125"/>
  <line stroke="red" x1="594" y1="156" x2="594" y2="281"/>
  <line stroke="blue" x1="406" y1="469" x2="375" y2="500"/>
  <line stroke="red" x1="406" y1="469" x2="406" y2="906"/>
  <line stroke="blue" x1="0" y1="625" x2="125" y2="625"/>
  <line stroke="red" x1="0" y1="625" x2="31" y2="656"/>
  <line stroke="black" x1="0" y1="625" x2="0" y2="875"/>
  <line stroke="blue" x1="375" y1="500" x2="375" y2="875"/>
  <line stroke="blue" x1="125" y1="625" x2="156" y2="656"/>
  <line stroke="blue" x1="625" y1="125" x2="656" y2="94"/>
  <line stroke="blue" x1="625" y1="125" x2="625" y2="250"/>
  <line stroke="red" x1="31" y1="656" x2="156" y2="656"/>
  <line stroke="blue" x1="31" y1="656" x2="62" y2="688"/>
  <line stroke="red" x1="31" y1="656" x2="31" y2="844"/>
  <line stroke="blue" x1="594" y1="281" x2="625" y2="250"/>
  <line stroke="red" x1="594" y1="281" x2="594" y2="344"/>
  <line stroke="blue" x1="594" y1="281" x2="625" y2="312"/>
  <line stroke="red" x1="594" y1="281" x2="656" y2="281"/>
  <line stroke="red" x1="656" y1="94" x2="688" y2="62"/>
  <line stroke="red" x1="656" y1="94" x2="656" y2="281"/>
  <line stroke="blue" x1="625" y1="250" x2="656" y2="281"/>
  <line stroke="red" x1="156" y1="656" x2="188" y2="688"/>
  <line stroke="blue" x1="594" y1="344" x2="625" y2="312"/>
  <line stroke="red" x1="594" y1="344" x2="594" y2="406"/>
  <line stroke="blue" x1="594" y1="344" x2="625" y2="375"/>
  <line stroke="red" x1="594" y1="344" x2="719" y2="344"/>
  <line stroke="blue" x1="62" y1="688" x2="188" y2="688"/>
  <line stroke="red" x1="62" y1="688" x2="94" y2="719"/>
  <line stroke="blue" x1="62" y1="688" x2="62" y2="812"/>
  <line stroke="blue" x1="688" y1="62" x2="719" y2="31"/>
  <line stroke="blue" x1="688" y1="62" x2="688" y2="312"/>
  <line stroke="blue" x1="625" y1="312" x2="688" y2="312"/>
  <line stroke="blue" x1="188" y1="688" x2="219" y2="719"/>
  <line stroke="red" x1="656" y1="281" x2="688" y2="312"/>
  <line stroke="blue" x1="594" y1="406" x2="625" y2="375"/>
  <line stroke="red" x1="594" y1="406" x2="594" y2="469"/>
  <line stroke="blue" x1="594" y1="406" x2="625" y2="438"/>
  <line stroke="red" x1="594" y1="406" x2="719" y2="406"/>
  <line stroke="red" x1="719" y1="31" x2="750" y2="0"/>
  <line stroke="red" x1="719" y1="31" x2="719" y2="344"/>
  <line stroke="red" x1="94" y1="719" x2="219" y2="719"/>
  <line stroke="blue" x1="94" y1="719" x2="125" y2="750"/>
  <line stroke="red" x1="94" y1="719" x2="94" y2="781"/>
  <line stroke="blue" x1="625" y1="375" x2="750" y2="375"/>
  <line stroke="red" x1="750" y1="0" x2="781" y2="31"/>
  <line stroke="blue" x1="750" y1="0" x2="750" y2="375"/>
  <line stroke="black" x1="750" y1="0" x2="1000" y2="0"/>
  <line stroke="red" x1="219" y1="719" x2="250" y2="750"/>
  <line stroke="blue" x1="688" y1="312" x2="719" y2="344"/>
  <line stroke="blue" x1="594" y1="469" x2="625" y2="438"/>
  <line stroke="blue" x1="594" y1="469" x2="625" y2="500"/>
  <line stroke="red" x1="594" y1="469" x2="656" y2="469"/>
  <line stroke="red" x1="594" y1="469" x2="594" y2="906"/>
  <line stroke="blue" x1="125" y1="750" x2="94" y2="781"/>
  <line stroke="blue" x1="125" y1="750" x2="250" y2="750"/>
  <line stroke="blue" x1="625" y1="438" x2="688" y2="438"/>
  <line stroke="blue" x1="781" y1="31" x2="812" y2="62"/>
  <line stroke="red" x1="781" y1="31" x2="781" y2="344"/>
  <line stroke="red" x1="781" y1="31" x2="969" y2="31"/>
  <line stroke="red" x1="94" y1="781" x2="62" y2="812"/>
  <line stroke="red" x1="94" y1="781" x2="281" y2="781"/>
  <line stroke="blue" x1="250" y1="750" x2="281" y2="781"/>
  <line stroke="red" x1="719" y1="344" x2="750" y2="375"/>
  <line stroke="blue" x1="625" y1="500" x2="656" y2="469"/>
  <line stroke="blue" x1="625" y1="500" x2="625" y2="875"/>
  <line stroke="red" x1="656" y1="469" x2="688" y2="438"/>
  <line stroke="red" x1="656" y1="469" x2="656" y2="844"/>
  <line stroke="blue" x1="62" y1="812" x2="31" y2="844"/>
  <line stroke="blue" x1="62" y1="812" x2="312" y2="812"/>
  <line stroke="blue" x1="688" y1="438" x2="719" y2="406"/>
  <line stroke="blue" x1="688" y1="438" x2="688" y2="812"/>
  <line stroke="red" x1="812" y1="62" x2="844" y2="94"/>
  <line stroke="blue" x1="812" y1="62" x2="812" y2="312"/>
  <line stroke="blue" x1="812" y1="62" x2="938" y2="62"/>
  <line stroke="red" x1="719" y1="406" x2="750" y2="375"/>
  <line stroke="red" x1="719" y1="406" x2="719" y2="781"/>
  <line stroke="red" x1="281" y1="781" x2="312" y2="812"/>
  <line stroke="red" x1="750" y1="375" x2="781" y2="344"/>
  <line stroke="red" x1="750" y1="375" x2="781" y2="406"/>
  <line stroke="blue" x1="750" y1="375" x2="750" y2="750"/>
  <line stroke="blue" x1="750" y1="375" x2="1000" y2="375"/>
  <line stroke="red" x1="31" y1="844" x2="0" y2="875"/>
  <line stroke="red" x1="31" y1="844" x2="344" y2="844"/>
  <line stroke="blue" x1="844" y1="94" x2="875" y2="125"/>
  <line stroke="red" x1="844" y1="94" x2="844" y2="281"/>
  <line stroke="red" x1="844" y1="94" x2="906" y2="94"/>
  <line stroke="blue" x1="781" y1="344" x2="812" y2="312"/>
  <line stroke="red" x1="781" y1="344" x2="969" y2="344"/>
  <line stroke="blue" x1="312" y1="812" x2="344" y2="844"/>
  <line stroke="red" x1="812" y1="312" x2="844" y2="281"/>
  <line stroke="blue" x1="812" y1="312" x2="938" y2="312"/>
  <line stroke="red" x1="0" y1="875" x2="31" y2="906"/>
  <line stroke="blue" x1="0" y1="875" x2="375" y2="875"/>
  <line stroke="black" x1="0" y1="875" x2="0" y2="1000"/>
  <line stroke="blue" x1="781" y1="406" x2="812" y2="438"/>
  <line stroke="red" x1="781" y1="406" x2="969" y2="406"/>
  <line stroke="red" x1="781" y1="406" x2="781" y2="719"/>
  <line stroke="blue" x1="875" y1="125" x2="875" y2="250"/>
  <line stroke="blue" x1="875" y1="125" x2="906" y2="94"/>
  <line stroke="blue" x1="844" y1="281" x2="875" y2="250"/>
  <line stroke="red" x1="844" y1="281" x2="906" y2="281"/>
  <line stroke="blue" x1="31" y1="906" x2="62" y2="938"/>
  <line stroke="red" x1="31" y1="906" x2="406" y2="906"/>
  <line stroke="red" x1="31" y1="906" x2="31" y2="1000"/>
  <line stroke="blue" x1="875" y1="250" x2="906" y2="281"/>
  <line stroke="red" x1="344" y1="844" x2="375" y2="875"/>
  <line stroke="red" x1="906" y1="94" x2="938" y2="62"/>
  <line stroke="red" x1="906" y1="94" x2="906" y2="281"/>
  <line stroke="red" x1="812" y1="438" x2="844" y2="469"/>
  <line stroke="blue" x1="812" y1="438" x2="812" y2="562"/>
  <line stroke="blue" x1="812" y1="438" x2="938" y2="438"/>
  <line stroke="red" x1="62" y1="938" x2="94" y2="969"/>
  <line stroke="blue" x1="62" y1="938" x2="62" y2="1000"/>
  <line stroke="blue" x1="62" y1="938" x2="438" y2="938"/>
  <line stroke="blue" x1="938" y1="62" x2="969" y2="31"/>
  <line stroke="blue" x1="938" y1="62" x2="938" y2="312"/>
  <line stroke="red" x1="906" y1="281" x2="938" y2="312"/>
  <line stroke="blue" x1="375" y1="875" x2="406" y2="906"/>
  <line stroke="red" x1="844" y1="469" x2="844" y2="531"/>
  <line stroke="blue" x1="844" y1="469" x2="875" y2="500"/>
  <line stroke="red" x1="844" y1="469" x2="906" y2="469"/>
  <line stroke="red" x1="969" y1="31" x2="1000" y2="0"/>
  <line stroke="red" x1="969" y1="31" x2="969" y2="344"/>
  <line stroke="red" x1="94" y1="969" x2="94" y2="1000"/>
  <line stroke="blue" x1="94" y1="969" x2="125" y2="1000"/>
  <line stroke="red" x1="94" y1="969" x2="469" y2="969"/>
  <line stroke="red" x1="812" y1="562" x2="844" y2="531"/>
  <line stroke="red" x1="812" y1="562" x2="844" y2="594"/>
  <line stroke="blue" x1="812" y1="562" x2="812" y2="688"/>
  <line stroke="blue" x1="812" y1="562" x2="938" y2="562"/>
  <line stroke="blue" x1="938" y1="312" x2="969" y2="344"/>
  <line stroke="red" x1="406" y1="906" x2="438" y2="938"/>
  <line stroke="blue" x1="844" y1="531" x2="875" y2="500"/>
  <line stroke="red" x1="844" y1="531" x2="906" y2="531"/>
  <line stroke="black" x1="0" y1="1000" x2="31" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="375"/>
  <line stroke="black" x1="31" y1="1000" x2="62" y2="1000"/>
  <line stroke="black" x1="62" y1="1000" x2="94" y2="1000"/>
  <line stroke="black" x1="94" y1="1000" x2="125" y2="1000"/>
  <line stroke="black" x1="125" y1="1000" x2="500" y2="1000"/>
  <line stroke="blue" x1="875" y1="500" x2="906" y2="469"/>
  <line stroke="blue" x1="875" y1="500" x2="906" y2="531"/>
  <line stroke="red" x1="906" y1="469" x2="938" y2="438"/>
  <line stroke="red" x1="906" y1="469" x2="906" y2="531"/>
  <line stroke="red" x1="969" y1="344" x2="1000" y2="375"/>
  <line stroke="red" x1="844" y1="594" x2="844" y2="656"/>
  <line stroke="blue" x1="844" y1="594" x2="875" y2="625"/>
  <line stroke="red" x1="844" y1="594" x2="969" y2="594"/>
  <line stroke="blue" x1="438" y1="938" x2="469" y2="969"/>
  <line stroke="blue" x1="938" y1="438" x2="969" y2="406"/>
  <line stroke="blue" x1="938" y1="438" x2="938" y2="562"/>
  <line stroke="red" x1="906" y1="531" x2="938" y2="562"/>
  <line stroke="red" x1="969" y1="406" x2="1000" y2="375"/>
  <line stroke="red" x1="969" y1="406" x2="969" y2="594"/>
  <line stroke="blue" x1="750" y1="750" x2="719" y2="781"/>
  <line stroke="red" x1="750" y1="750" x2="781" y2="719"/>
  <line stroke="blue" x1="750" y1="750" x2="875" y2="750"/>
  <line stroke="red" x1="719" y1="781" x2="688" y2="812"/>
  <line stroke="red" x1="719" y1="781" x2="906" y2="781"/>
  <line stroke="blue" x1="781" y1="719" x2="812" y2="688"/>
  <line stroke="red" x1="781" y1="719" x2="906" y2="719"/>
  <line stroke="blue" x1="688" y1="812" x2="656" y2="844"/>
  <line stroke="blue" x1="688" y1="812" x2="938" y2="812"/>
  <line stroke="red" x1="812" y1="688" x2="844" y2="656"/>
  <line stroke="blue" x1="812" y1="688" x2="938" y2="688"/>
  <line stroke="black" x1="1000" y1="375" x2="1000" y2="625"/>
  <line stroke="red" x1="656" y1="844" x2="625" y2="875"/>
  <line stroke="red" x1="656" y1="844" x2="969" y2="844"/>
  <line stroke="blue" x1="844" y1="656" x2="875" y2="625"/>
  <line stroke="red" x1="844" y1="656" x2="969" y2="656"/>
  <line stroke="blue" x1="625" y1="875" x2="594" y2="906"/>
  <line stroke="blue" x1="625" y1="875" x2="1000" y2="875"/>
  <line stroke="blue" x1="875" y1="625" x2="1000" y2="625"/>
  <line stroke="red" x1="469" y1="969" x2="500" y2="1000"/>
  <line stroke="red" x1="594" y1="906" x2="562" y2="938"/>
  <line stroke="red" x1="594" y1="906" x2="969" y2="906"/>
  <line stroke="blue" x1="562" y1="938" x2="531" y2="969"/>
  <line stroke="blue" x1="562" y1="938" x2="938" y2="938"/>
  <line stroke="blue" x1="938" y1="562" x2="969" y2="594"/>
  <line stroke="red" x1="531" y1="969" x2="500" y2="1000"/>
  <line stroke="red" x1="531" y1="969" x2="906" y2="969"/>
  <line stroke="black" x1="500" y1="1000" x2="875" y2="1000"/>
  <line stroke="red" x1="969" y1="594" x2="1000" y2="625"/>
  <line stroke="blue" x1="875" y1="750" x2="906" y2="719"/>
  <line stroke="blue" x1="875" y1="750" x2="906" y2="781"/>
  <line stroke="red" x1="906" y1="719" x2="938" y2="688"/>
  <line stroke="red" x1="906" y1="719" x2="906" y2="781"/>
  <line stroke="blue" x1="938" y1="688" x2="969" y2="656"/>
  <line stroke="blue" x1="938" y1="688" x2="938" y2="812"/>
  <line stroke="red" x1="969" y1="656" x2="1000" y2="625"/>
  <line stroke="red" x1="969" y1="656" x2="969" y2="844"/>
  <line stroke="black" x1="1000" y1="625" x2="1000" y2="875"/>
  <line stroke="red" x1="906" y1="781" x2="938" y2="812"/>
  <line stroke="blue" x1="938" y1="812" x2="969" y2="844"/>
  <line stroke="red" x1="969" y1="844" x2="1000" y2="875"/>
  <line stroke="red" x1="938" y1="938" x2="906" y2="969"/>
  <line stroke="blue" x1="938" y1="938" x2="969" y2="906"/>
  <line stroke="blue" x1="938" y1="938" x2="938" y2="1000"/>
  <line stroke="blue" x1="906" y1="969" x2="875" y2="1000"/>
  <line stroke="red" x1="906" y1="969" x2="906" y2="1000"/>
  <line stroke="red" x1="969" y1="906" x2="1000" y2="875"/>
  <line stroke="red" x1="969" y1="906" x2="969" y2="1000"/>
  <line stroke="black" x1="875" y1="1000" x2="906" y2="1000"/>
  <line stroke="black" x1="1000" y1="875" x2="1000" y2="1000"/>
  <line stroke="black" x1="906" y1="1000" x2="938" y2="1000"/>
  <line stroke="black" x1="938" y1="1000" x2="969" y2="1000"/>
  <line stroke="black" x1="969" y1="1000" x2="1000" y2="1000"/>
</svg>""",
    rectangles=[
        shaping_utils.Rectangle(
            name="left top claw",
            x_start=0,
            y_start=0,
            x_len=0.125,
            y_len=0.125,
            circle_centers=[point_utils.Point(x=0, y=0)],
        ),
        shaping_utils.Rectangle(
            name="left bottom claw",
            x_start=0.125,
            y_start=0,
            x_len=0.25,
            y_len=0.125,
            circle_centers=[point_utils.Point(x=0.25, y=0)],
        ),
        shaping_utils.Rectangle(
            name="left river",
            x_start=0,
            y_start=0,
            x_len=0.5 - 1 / 32,
            y_len=0.25 - 1 / 32,
        ),
        shaping_utils.Rectangle(
            name="second left leg",
            x_start=0,
            y_start=0.25,
            x_len=0.125,
            y_len=0.25,
        ),
        shaping_utils.Rectangle(
            name="first left leg",
            x_start=0.125,
            y_start=0.25,
            x_len=0.25,
            y_len=0.25,
        ),
        shaping_utils.Rectangle(
            name="third left leg",
            x_start=0,
            y_start=0.5,
            x_len=0.125,
            y_len=0.25,
            circle_centers=[point_utils.Point(x=0, y=0.5 + 0.125)],
        ),
        shaping_utils.Rectangle(
            name="fourth left leg",
            x_start=0,
            y_start=0.75,
            x_len=0.125,
            y_len=0.25,
            circle_centers=[point_utils.Point(x=0, y=0.75 + 0.125)],
        ),
        shaping_utils.Rectangle(
            name="right top claw",
            x_start=0.875,
            y_start=0,
            x_len=0.125,
            y_len=0.125,
            circle_centers=[point_utils.Point(x=1, y=0)],
        ),
        shaping_utils.Rectangle(
            name="right bottom claw",
            x_start=0.625,
            y_start=0,
            x_len=0.25,
            y_len=0.125,
            circle_centers=[point_utils.Point(x=0.625 + 0.125, y=0)],
        ),
        shaping_utils.Rectangle(
            name="right river",
            x_start=0.5 + 1 / 32,
            y_start=0,
            x_len=0.5 - 1 / 32,
            y_len=0.25 - 1 / 32,
        ),
        shaping_utils.Rectangle(
            name="second right leg",
            x_start=0.875,
            y_start=0.25,
            x_len=0.125,
            y_len=0.25,
        ),
        shaping_utils.Rectangle(
            name="first right leg",
            x_start=0.625,
            y_start=0.25,
            x_len=0.25,
            y_len=0.25,
        ),
        shaping_utils.Rectangle(
            name="third right leg",
            x_start=0.875,
            y_start=0.5,
            x_len=0.125,
            y_len=0.25,
            circle_centers=[point_utils.Point(x=1, y=0.5 + 0.125)],
        ),
        shaping_utils.Rectangle(
            name="fourth right leg",
            x_start=0.875,
            y_start=0.75,
            x_len=0.125,
            y_len=0.25,
            circle_centers=[point_utils.Point(x=1, y=0.75 + 0.125)],
        ),
        shaping_utils.Rectangle(
            name="head", x_start=0.375, y_start=0.25, x_len=0.25, y_len=0.0625
        ),
        shaping_utils.Rectangle(
            name="back", x_start=0.375, y_start=0.3125, x_len=0.25, y_len=0.0625
        ),
        shaping_utils.Rectangle(
            name="left hind leg",
            x_start=0.375,
            y_start=0.375,
            x_len=0.25,
            y_len=0.0625,
        ),
        shaping_utils.Rectangle(
            name="right hind leg",
            x_start=0.375,
            y_start=0.4375,
            x_len=0.25,
            y_len=0.0625,
        ),
        shaping_utils.Rectangle(
            name="?", x_start=0.125, y_start=0.5, x_len=0.75, y_len=0.125
        ),
        shaping_utils.Rectangle(
            name="tail",
            x_start=0.125,
            y_start=0.625,
            x_len=0.75,
            y_len=0.375,
            circle_centers=[point_utils.Point(x=0.5, y=1)],
        ),
    ],
    target_zigzag_depths=[
        2,
        None,
        0.5,
        None,
        None,
        None,
        None,
        2,
        None,
        0.5,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ],
    target_flap_angles=[
        None,
        None,
        None,
        np.pi / 2 + np.arcsin(1 / np.sqrt(17)) - 0.6,
        np.pi / 2,
        np.pi / 2 + np.arcsin(3 / 5),
        np.pi / 2 + np.arcsin(5 / np.sqrt(41)),
        None,
        None,
        None,
        np.pi / 2 + np.arcsin(1 / np.sqrt(17)) - 0.6,
        np.pi / 2,
        np.pi / 2 + np.arcsin(3 / 5),
        np.pi / 2 + np.arcsin(5 / np.sqrt(41)),
        None,
        None,
        None,
        None,
        None,
        None,
    ],
    fold_orders=[
        [crease_utils.Fold.VALLEY, crease_utils.Fold.MOUNTAIN],
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
        [crease_utils.Fold.VALLEY, crease_utils.Fold.MOUNTAIN],
        [crease_utils.Fold.VALLEY, crease_utils.Fold.MOUNTAIN],
        [crease_utils.Fold.VALLEY, crease_utils.Fold.MOUNTAIN],
        [crease_utils.Fold.VALLEY, crease_utils.Fold.MOUNTAIN],
        [crease_utils.Fold.VALLEY, crease_utils.Fold.MOUNTAIN],
        [crease_utils.Fold.VALLEY, crease_utils.Fold.MOUNTAIN],
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
        [crease_utils.Fold.VALLEY, crease_utils.Fold.MOUNTAIN],
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
        [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
    ],
    even_phases=[
        False,
        True,
        False,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ],
    base_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <line stroke="red" x1="0" y1="0" x2="1" y2="1"/>
  <line stroke="black" x1="0" y1="0" x2="4" y2="0"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="8"/>
  <line stroke="blue" x1="1" y1="1" x2="2" y2="2"/>
  <line stroke="red" x1="1" y1="1" x2="4" y2="1"/>
  <line stroke="red" x1="1" y1="1" x2="1" y2="8"/>
  <line stroke="red" x1="2" y1="2" x2="3" y2="3"/>
  <line stroke="blue" x1="2" y1="2" x2="4" y2="2"/>
  <line stroke="blue" x1="2" y1="2" x2="2" y2="8"/>
  <line stroke="blue" x1="4" y1="0" x2="4" y2="1"/>
  <line stroke="black" x1="4" y1="0" x2="8" y2="0"/>
  <line stroke="red" x1="4" y1="1" x2="4" y2="2"/>
  <line stroke="red" x1="4" y1="1" x2="7" y2="1"/>
  <line stroke="red" x1="3" y1="3" x2="4" y2="3"/>
  <line stroke="blue" x1="3" y1="3" x2="4" y2="4"/>
  <line stroke="red" x1="3" y1="3" x2="3" y2="8"/>
  <line stroke="blue" x1="4" y1="2" x2="4" y2="3"/>
  <line stroke="blue" x1="4" y1="2" x2="6" y2="2"/>
  <line stroke="red" x1="4" y1="3" x2="4" y2="4"/>
  <line stroke="red" x1="4" y1="3" x2="5" y2="3"/>
  <line stroke="blue" x1="4" y1="4" x2="5" y2="3"/>
  <line stroke="blue" x1="4" y1="4" x2="4" y2="8"/>
  <line stroke="red" x1="5" y1="3" x2="6" y2="2"/>
  <line stroke="red" x1="5" y1="3" x2="5" y2="8"/>
  <line stroke="blue" x1="6" y1="2" x2="7" y2="1"/>
  <line stroke="blue" x1="6" y1="2" x2="6" y2="8"/>
  <line stroke="red" x1="7" y1="1" x2="8" y2="0"/>
  <line stroke="red" x1="7" y1="1" x2="7" y2="8"/>
  <line stroke="blue" x1="0" y1="8" x2="1" y2="8"/>
  <line stroke="black" x1="0" y1="8" x2="0" y2="12"/>
  <line stroke="red" x1="8" y1="0" x2="9" y2="1"/>
  <line stroke="blue" x1="8" y1="0" x2="8" y2="8"/>
  <line stroke="black" x1="8" y1="0" x2="16" y2="0"/>
  <line stroke="red" x1="1" y1="8" x2="2" y2="8"/>
  <line stroke="red" x1="1" y1="8" x2="1" y2="11"/>
  <line stroke="blue" x1="2" y1="8" x2="3" y2="8"/>
  <line stroke="blue" x1="2" y1="8" x2="2" y2="10"/>
  <line stroke="red" x1="3" y1="8" x2="4" y2="8"/>
  <line stroke="red" x1="3" y1="8" x2="3" y2="9"/>
  <line stroke="blue" x1="4" y1="8" x2="5" y2="8"/>
  <line stroke="blue" x1="4" y1="8" x2="3" y2="9"/>
  <line stroke="red" x1="4" y1="8" x2="4" y2="9"/>
  <line stroke="blue" x1="4" y1="8" x2="5" y2="9"/>
  <line stroke="blue" x1="9" y1="1" x2="10" y2="2"/>
  <line stroke="red" x1="9" y1="1" x2="9" y2="8"/>
  <line stroke="red" x1="9" y1="1" x2="15" y2="1"/>
  <line stroke="red" x1="5" y1="8" x2="6" y2="8"/>
  <line stroke="red" x1="5" y1="8" x2="5" y2="9"/>
  <line stroke="red" x1="3" y1="9" x2="4" y2="9"/>
  <line stroke="red" x1="3" y1="9" x2="2" y2="10"/>
  <line stroke="red" x1="4" y1="9" x2="5" y2="9"/>
  <line stroke="blue" x1="4" y1="9" x2="4" y2="10"/>
  <line stroke="blue" x1="6" y1="8" x2="7" y2="8"/>
  <line stroke="blue" x1="6" y1="8" x2="6" y2="10"/>
  <line stroke="blue" x1="2" y1="10" x2="4" y2="10"/>
  <line stroke="blue" x1="2" y1="10" x2="1" y2="11"/>
  <line stroke="red" x1="10" y1="2" x2="11" y2="3"/>
  <line stroke="blue" x1="10" y1="2" x2="10" y2="8"/>
  <line stroke="blue" x1="10" y1="2" x2="16" y2="2"/>
  <line stroke="red" x1="5" y1="9" x2="6" y2="10"/>
  <line stroke="red" x1="7" y1="8" x2="8" y2="8"/>
  <line stroke="red" x1="7" y1="8" x2="7" y2="11"/>
  <line stroke="blue" x1="4" y1="10" x2="6" y2="10"/>
  <line stroke="red" x1="4" y1="10" x2="4" y2="11"/>
  <line stroke="red" x1="1" y1="11" x2="4" y2="11"/>
  <line stroke="red" x1="1" y1="11" x2="0" y2="12"/>
  <line stroke="blue" x1="8" y1="8" x2="9" y2="8"/>
  <line stroke="blue" x1="8" y1="8" x2="8" y2="12"/>
  <line stroke="blue" x1="11" y1="3" x2="12" y2="4"/>
  <line stroke="red" x1="11" y1="3" x2="11" y2="8"/>
  <line stroke="red" x1="11" y1="3" x2="15" y2="3"/>
  <line stroke="blue" x1="6" y1="10" x2="7" y2="11"/>
  <line stroke="blue" x1="4" y1="11" x2="4" y2="12"/>
  <line stroke="red" x1="4" y1="11" x2="7" y2="11"/>
  <line stroke="blue" x1="0" y1="12" x2="4" y2="12"/>
  <line stroke="red" x1="0" y1="12" x2="1" y2="13"/>
  <line stroke="black" x1="0" y1="12" x2="0" y2="16"/>
  <line stroke="red" x1="9" y1="8" x2="10" y2="8"/>
  <line stroke="red" x1="9" y1="8" x2="9" y2="11"/>
  <line stroke="blue" x1="4" y1="12" x2="4" y2="13"/>
  <line stroke="red" x1="4" y1="12" x2="8" y2="12"/>
  <line stroke="red" x1="12" y1="4" x2="13" y2="5"/>
  <line stroke="blue" x1="12" y1="4" x2="12" y2="8"/>
  <line stroke="blue" x1="12" y1="4" x2="16" y2="4"/>
  <line stroke="blue" x1="10" y1="8" x2="11" y2="8"/>
  <line stroke="blue" x1="10" y1="8" x2="10" y2="10"/>
  <line stroke="red" x1="1" y1="13" x2="4" y2="13"/>
  <line stroke="blue" x1="1" y1="13" x2="2" y2="14"/>
  <line stroke="red" x1="1" y1="13" x2="1" y2="16"/>
  <line stroke="red" x1="7" y1="11" x2="8" y2="12"/>
  <line stroke="red" x1="4" y1="13" x2="4" y2="14"/>
  <line stroke="red" x1="4" y1="13" x2="7" y2="13"/>
  <line stroke="red" x1="11" y1="8" x2="11" y2="9"/>
  <line stroke="red" x1="11" y1="8" x2="12" y2="8"/>
  <line stroke="blue" x1="13" y1="5" x2="14" y2="6"/>
  <line stroke="red" x1="13" y1="5" x2="13" y2="9"/>
  <line stroke="red" x1="13" y1="5" x2="15" y2="5"/>
  <line stroke="blue" x1="10" y1="10" x2="9" y2="11"/>
  <line stroke="red" x1="10" y1="10" x2="11" y2="9"/>
  <line stroke="blue" x1="10" y1="10" x2="12" y2="10"/>
  <line stroke="blue" x1="2" y1="14" x2="4" y2="14"/>
  <line stroke="red" x1="2" y1="14" x2="3" y2="15"/>
  <line stroke="blue" x1="2" y1="14" x2="2" y2="16"/>
  <line stroke="red" x1="9" y1="11" x2="8" y2="12"/>
  <line stroke="red" x1="9" y1="11" x2="13" y2="11"/>
  <line stroke="blue" x1="11" y1="9" x2="12" y2="8"/>
  <line stroke="red" x1="11" y1="9" x2="13" y2="9"/>
  <line stroke="red" x1="8" y1="12" x2="7" y2="13"/>
  <line stroke="red" x1="8" y1="12" x2="9" y2="13"/>
  <line stroke="blue" x1="8" y1="12" x2="12" y2="12"/>
  <line stroke="blue" x1="8" y1="12" x2="8" y2="18"/>
  <line stroke="blue" x1="12" y1="8" x2="13" y2="9"/>
  <line stroke="blue" x1="4" y1="14" x2="6" y2="14"/>
  <line stroke="blue" x1="4" y1="14" x2="4" y2="15"/>
  <line stroke="blue" x1="7" y1="13" x2="6" y2="14"/>
  <line stroke="red" x1="7" y1="13" x2="7" y2="18"/>
  <line stroke="blue" x1="15" y1="1" x2="16" y2="0"/>
  <line stroke="red" x1="15" y1="1" x2="16" y2="2"/>
  <line stroke="red" x1="15" y1="1" x2="17" y2="1"/>
  <line stroke="red" x1="6" y1="14" x2="5" y2="15"/>
  <line stroke="blue" x1="6" y1="14" x2="6" y2="18"/>
  <line stroke="red" x1="14" y1="6" x2="15" y2="7"/>
  <line stroke="blue" x1="14" y1="6" x2="14" y2="9"/>
  <line stroke="blue" x1="14" y1="6" x2="16" y2="6"/>
  <line stroke="red" x1="3" y1="15" x2="4" y2="15"/>
  <line stroke="red" x1="3" y1="15" x2="3" y2="16"/>
  <line stroke="blue" x1="3" y1="15" x2="4" y2="16"/>
  <line stroke="blue" x1="15" y1="3" x2="16" y2="2"/>
  <line stroke="red" x1="15" y1="3" x2="16" y2="4"/>
  <line stroke="red" x1="15" y1="3" x2="17" y2="3"/>
  <line stroke="red" x1="4" y1="15" x2="5" y2="15"/>
  <line stroke="red" x1="4" y1="15" x2="4" y2="16"/>
  <line stroke="blue" x1="12" y1="10" x2="13" y2="9"/>
  <line stroke="red" x1="12" y1="10" x2="13" y2="10"/>
  <line stroke="blue" x1="12" y1="10" x2="13" y2="11"/>
  <line stroke="blue" x1="9" y1="13" x2="10" y2="14"/>
  <line stroke="red" x1="9" y1="13" x2="13" y2="13"/>
  <line stroke="red" x1="9" y1="13" x2="9" y2="18"/>
  <line stroke="red" x1="13" y1="9" x2="13" y2="10"/>
  <line stroke="red" x1="13" y1="9" x2="14" y2="9"/>
  <line stroke="blue" x1="5" y1="15" x2="4" y2="16"/>
  <line stroke="red" x1="5" y1="15" x2="5" y2="17"/>
  <line stroke="blue" x1="15" y1="5" x2="16" y2="4"/>
  <line stroke="red" x1="15" y1="5" x2="16" y2="6"/>
  <line stroke="red" x1="15" y1="5" x2="17" y2="5"/>
  <line stroke="blue" x1="0" y1="16" x2="1" y2="16"/>
  <line stroke="black" x1="0" y1="16" x2="0" y2="20"/>
  <line stroke="blue" x1="16" y1="0" x2="17" y2="1"/>
  <line stroke="black" x1="16" y1="0" x2="24" y2="0"/>
  <line stroke="red" x1="1" y1="16" x2="2" y2="16"/>
  <line stroke="red" x1="1" y1="16" x2="1" y2="19"/>
  <line stroke="red" x1="16" y1="2" x2="17" y2="1"/>
  <line stroke="blue" x1="16" y1="2" x2="17" y2="3"/>
  <line stroke="blue" x1="16" y1="2" x2="22" y2="2"/>
  <line stroke="blue" x1="2" y1="16" x2="3" y2="16"/>
  <line stroke="blue" x1="2" y1="16" x2="2" y2="18"/>
  <line stroke="red" x1="3" y1="16" x2="4" y2="16"/>
  <line stroke="red" x1="3" y1="16" x2="3" y2="17"/>
  <line stroke="red" x1="13" y1="10" x2="13" y2="11"/>
  <line stroke="blue" x1="13" y1="10" x2="14" y2="10"/>
  <line stroke="blue" x1="4" y1="16" x2="3" y2="17"/>
  <line stroke="blue" x1="4" y1="16" x2="5" y2="17"/>
  <line stroke="red" x1="16" y1="4" x2="17" y2="3"/>
  <line stroke="blue" x1="16" y1="4" x2="17" y2="5"/>
  <line stroke="blue" x1="16" y1="4" x2="20" y2="4"/>
  <line stroke="blue" x1="15" y1="7" x2="16" y2="6"/>
  <line stroke="red" x1="15" y1="7" x2="15" y2="9"/>
  <line stroke="red" x1="15" y1="7" x2="17" y2="7"/>
  <line stroke="blue" x1="14" y1="9" x2="14" y2="10"/>
  <line stroke="blue" x1="14" y1="9" x2="15" y2="9"/>
  <line stroke="blue" x1="12" y1="12" x2="13" y2="11"/>
  <line stroke="red" x1="12" y1="12" x2="13" y2="12"/>
  <line stroke="blue" x1="12" y1="12" x2="13" y2="13"/>
  <line stroke="red" x1="17" y1="1" x2="23" y2="1"/>
  <line stroke="red" x1="13" y1="11" x2="13" y2="12"/>
  <line stroke="red" x1="13" y1="11" x2="14" y2="11"/>
  <line stroke="red" x1="16" y1="6" x2="17" y2="5"/>
  <line stroke="blue" x1="16" y1="6" x2="17" y2="7"/>
  <line stroke="blue" x1="16" y1="6" x2="18" y2="6"/>
  <line stroke="blue" x1="10" y1="14" x2="12" y2="14"/>
  <line stroke="red" x1="10" y1="14" x2="11" y2="15"/>
  <line stroke="blue" x1="10" y1="14" x2="10" y2="18"/>
  <line stroke="blue" x1="14" y1="10" x2="14" y2="11"/>
  <line stroke="red" x1="14" y1="10" x2="15" y2="10"/>
  <line stroke="red" x1="3" y1="17" x2="5" y2="17"/>
  <line stroke="red" x1="3" y1="17" x2="2" y2="18"/>
  <line stroke="red" x1="17" y1="3" x2="21" y2="3"/>
  <line stroke="red" x1="15" y1="9" x2="15" y2="10"/>
  <line stroke="red" x1="15" y1="9" x2="17" y2="9"/>
  <line stroke="red" x1="13" y1="12" x2="13" y2="13"/>
  <line stroke="blue" x1="13" y1="12" x2="14" y2="12"/>
  <line stroke="red" x1="5" y1="17" x2="6" y2="18"/>
  <line stroke="red" x1="17" y1="5" x2="19" y2="5"/>
  <line stroke="blue" x1="14" y1="11" x2="14" y2="12"/>
  <line stroke="blue" x1="14" y1="11" x2="15" y2="11"/>
  <line stroke="red" x1="15" y1="10" x2="15" y2="11"/>
  <line stroke="blue" x1="15" y1="10" x2="17" y2="10"/>
  <line stroke="blue" x1="2" y1="18" x2="6" y2="18"/>
  <line stroke="blue" x1="2" y1="18" x2="1" y2="19"/>
  <line stroke="blue" x1="13" y1="13" x2="12" y2="14"/>
  <line stroke="red" x1="13" y1="13" x2="13" y2="14"/>
  <line stroke="red" x1="13" y1="13" x2="14" y2="13"/>
  <line stroke="red" x1="17" y1="7" x2="18" y2="6"/>
  <line stroke="red" x1="17" y1="7" x2="17" y2="9"/>
  <line stroke="red" x1="12" y1="14" x2="13" y2="14"/>
  <line stroke="blue" x1="12" y1="14" x2="13" y2="15"/>
  <line stroke="blue" x1="14" y1="12" x2="14" y2="13"/>
  <line stroke="red" x1="14" y1="12" x2="15" y2="12"/>
  <line stroke="red" x1="11" y1="15" x2="13" y2="15"/>
  <line stroke="blue" x1="11" y1="15" x2="12" y2="16"/>
  <line stroke="red" x1="11" y1="15" x2="11" y2="18"/>
  <line stroke="red" x1="15" y1="11" x2="15" y2="12"/>
  <line stroke="red" x1="15" y1="11" x2="17" y2="11"/>
  <line stroke="blue" x1="6" y1="18" x2="7" y2="18"/>
  <line stroke="red" x1="6" y1="18" x2="5" y2="19"/>
  <line stroke="blue" x1="6" y1="18" x2="6" y2="20"/>
  <line stroke="blue" x1="18" y1="6" x2="19" y2="5"/>
  <line stroke="blue" x1="18" y1="6" x2="18" y2="9"/>
  <line stroke="red" x1="1" y1="19" x2="5" y2="19"/>
  <line stroke="red" x1="1" y1="19" x2="0" y2="20"/>
  <line stroke="blue" x1="13" y1="14" x2="14" y2="14"/>
  <line stroke="red" x1="13" y1="14" x2="13" y2="15"/>
  <line stroke="blue" x1="14" y1="13" x2="14" y2="14"/>
  <line stroke="blue" x1="14" y1="13" x2="15" y2="13"/>
  <line stroke="red" x1="15" y1="12" x2="15" y2="13"/>
  <line stroke="blue" x1="15" y1="12" x2="17" y2="12"/>
  <line stroke="red" x1="17" y1="9" x2="17" y2="10"/>
  <line stroke="blue" x1="17" y1="9" x2="18" y2="9"/>
  <line stroke="red" x1="7" y1="18" x2="8" y2="18"/>
  <line stroke="red" x1="7" y1="18" x2="7" y2="20"/>
  <line stroke="blue" x1="5" y1="19" x2="4" y2="20"/>
  <line stroke="red" x1="5" y1="19" x2="5" y2="20"/>
  <line stroke="red" x1="19" y1="5" x2="20" y2="4"/>
  <line stroke="red" x1="19" y1="5" x2="19" y2="9"/>
  <line stroke="blue" x1="8" y1="18" x2="9" y2="18"/>
  <line stroke="blue" x1="8" y1="18" x2="8" y2="20"/>
  <line stroke="red" x1="17" y1="10" x2="17" y2="11"/>
  <line stroke="red" x1="17" y1="10" x2="18" y2="10"/>
  <line stroke="blue" x1="14" y1="14" x2="14" y2="15"/>
  <line stroke="red" x1="14" y1="14" x2="15" y2="14"/>
  <line stroke="blue" x1="13" y1="15" x2="12" y2="16"/>
  <line stroke="red" x1="13" y1="15" x2="14" y2="15"/>
  <line stroke="red" x1="13" y1="15" x2="13" y2="16"/>
  <line stroke="red" x1="15" y1="13" x2="15" y2="14"/>
  <line stroke="red" x1="15" y1="13" x2="17" y2="13"/>
  <line stroke="blue" x1="0" y1="20" x2="4" y2="20"/>
  <line stroke="red" x1="0" y1="20" x2="1" y2="21"/>
  <line stroke="black" x1="0" y1="20" x2="0" y2="24"/>
  <line stroke="red" x1="12" y1="16" x2="13" y2="16"/>
  <line stroke="blue" x1="12" y1="16" x2="12" y2="18"/>
  <line stroke="red" x1="9" y1="18" x2="10" y2="18"/>
  <line stroke="red" x1="9" y1="18" x2="9" y2="20"/>
  <line stroke="blue" x1="18" y1="9" x2="18" y2="10"/>
  <line stroke="red" x1="18" y1="9" x2="19" y2="9"/>
  <line stroke="red" x1="17" y1="11" x2="17" y2="12"/>
  <line stroke="blue" x1="17" y1="11" x2="18" y2="11"/>
  <line stroke="red" x1="4" y1="20" x2="5" y2="20"/>
  <line stroke="blue" x1="4" y1="20" x2="5" y2="21"/>
  <line stroke="blue" x1="20" y1="4" x2="21" y2="3"/>
  <line stroke="blue" x1="20" y1="4" x2="20" y2="8"/>
  <line stroke="blue" x1="14" y1="15" x2="15" y2="15"/>
  <line stroke="blue" x1="14" y1="15" x2="14" y2="16"/>
  <line stroke="red" x1="15" y1="14" x2="15" y2="15"/>
  <line stroke="blue" x1="15" y1="14" x2="17" y2="14"/>
  <line stroke="blue" x1="10" y1="18" x2="11" y2="18"/>
  <line stroke="blue" x1="10" y1="18" x2="10" y2="20"/>
  <line stroke="blue" x1="18" y1="10" x2="18" y2="11"/>
  <line stroke="blue" x1="18" y1="10" x2="19" y2="10"/>
  <line stroke="blue" x1="13" y1="16" x2="14" y2="16"/>
  <line stroke="red" x1="13" y1="16" x2="13" y2="18"/>
  <line stroke="blue" x1="5" y1="20" x2="6" y2="20"/>
  <line stroke="red" x1="5" y1="20" x2="5" y2="21"/>
  <line stroke="red" x1="17" y1="12" x2="17" y2="13"/>
  <line stroke="red" x1="17" y1="12" x2="18" y2="12"/>
  <line stroke="red" x1="6" y1="20" x2="7" y2="20"/>
  <line stroke="blue" x1="6" y1="20" x2="6" y2="22"/>
  <line stroke="red" x1="1" y1="21" x2="5" y2="21"/>
  <line stroke="blue" x1="1" y1="21" x2="2" y2="22"/>
  <line stroke="red" x1="1" y1="21" x2="1" y2="24"/>
  <line stroke="red" x1="19" y1="9" x2="19" y2="10"/>
  <line stroke="blue" x1="19" y1="9" x2="20" y2="8"/>
  <line stroke="blue" x1="19" y1="9" x2="20" y2="10"/>
  <line stroke="red" x1="19" y1="9" x2="21" y2="9"/>
  <line stroke="red" x1="11" y1="18" x2="12" y2="18"/>
  <line stroke="red" x1="11" y1="18" x2="11" y2="20"/>
  <line stroke="blue" x1="18" y1="11" x2="18" y2="12"/>
  <line stroke="red" x1="18" y1="11" x2="19" y2="11"/>
  <line stroke="blue" x1="7" y1="20" x2="8" y2="20"/>
  <line stroke="red" x1="7" y1="20" x2="7" y2="23"/>
  <line stroke="red" x1="21" y1="3" x2="22" y2="2"/>
  <line stroke="red" x1="21" y1="3" x2="21" y2="8"/>
  <line stroke="red" x1="15" y1="15" x2="15" y2="16"/>
  <line stroke="red" x1="15" y1="15" x2="17" y2="15"/>
  <line stroke="red" x1="14" y1="16" x2="15" y2="16"/>
  <line stroke="blue" x1="14" y1="16" x2="14" y2="18"/>
  <line stroke="red" x1="17" y1="13" x2="17" y2="14"/>
  <line stroke="blue" x1="17" y1="13" x2="18" y2="13"/>
  <line stroke="red" x1="19" y1="10" x2="19" y2="11"/>
  <line stroke="red" x1="19" y1="10" x2="20" y2="10"/>
  <line stroke="red" x1="8" y1="20" x2="9" y2="20"/>
  <line stroke="blue" x1="8" y1="20" x2="8" y2="24"/>
  <line stroke="red" x1="20" y1="8" x2="21" y2="8"/>
  <line stroke="blue" x1="20" y1="8" x2="21" y2="9"/>
  <line stroke="red" x1="5" y1="21" x2="6" y2="22"/>
  <line stroke="blue" x1="12" y1="18" x2="13" y2="18"/>
  <line stroke="blue" x1="12" y1="18" x2="12" y2="20"/>
  <line stroke="blue" x1="18" y1="12" x2="18" y2="13"/>
  <line stroke="blue" x1="18" y1="12" x2="19" y2="12"/>
  <line stroke="blue" x1="9" y1="20" x2="10" y2="20"/>
  <line stroke="red" x1="9" y1="20" x2="9" y2="25"/>
  <line stroke="blue" x1="15" y1="16" x2="17" y2="16"/>
  <line stroke="red" x1="15" y1="16" x2="15" y2="18"/>
  <line stroke="blue" x1="19" y1="11" x2="20" y2="10"/>
  <line stroke="red" x1="19" y1="11" x2="19" y2="12"/>
  <line stroke="blue" x1="19" y1="11" x2="20" y2="12"/>
  <line stroke="red" x1="19" y1="11" x2="23" y2="11"/>
  <line stroke="red" x1="17" y1="14" x2="17" y2="15"/>
  <line stroke="red" x1="17" y1="14" x2="18" y2="14"/>
  <line stroke="blue" x1="2" y1="22" x2="6" y2="22"/>
  <line stroke="red" x1="2" y1="22" x2="3" y2="23"/>
  <line stroke="blue" x1="2" y1="22" x2="2" y2="24"/>
  <line stroke="blue" x1="22" y1="2" x2="23" y2="1"/>
  <line stroke="blue" x1="22" y1="2" x2="22" y2="8"/>
  <line stroke="red" x1="13" y1="18" x2="14" y2="18"/>
  <line stroke="red" x1="13" y1="18" x2="13" y2="20"/>
  <line stroke="blue" x1="18" y1="13" x2="18" y2="14"/>
  <line stroke="red" x1="18" y1="13" x2="19" y2="13"/>
  <line stroke="red" x1="10" y1="20" x2="11" y2="20"/>
  <line stroke="blue" x1="10" y1="20" x2="10" y2="26"/>
  <line stroke="blue" x1="20" y1="10" x2="22" y2="10"/>
  <line stroke="red" x1="21" y1="8" x2="21" y2="9"/>
  <line stroke="blue" x1="21" y1="8" x2="22" y2="8"/>
  <line stroke="red" x1="19" y1="12" x2="19" y2="13"/>
  <line stroke="red" x1="19" y1="12" x2="20" y2="12"/>
  <line stroke="red" x1="17" y1="15" x2="17" y2="16"/>
  <line stroke="blue" x1="17" y1="15" x2="18" y2="15"/>
  <line stroke="blue" x1="14" y1="18" x2="15" y2="18"/>
  <line stroke="blue" x1="14" y1="18" x2="14" y2="20"/>
  <line stroke="blue" x1="18" y1="14" x2="18" y2="15"/>
  <line stroke="blue" x1="18" y1="14" x2="19" y2="14"/>
  <line stroke="blue" x1="6" y1="22" x2="7" y2="23"/>
  <line stroke="blue" x1="11" y1="20" x2="12" y2="20"/>
  <line stroke="red" x1="11" y1="20" x2="11" y2="27"/>
  <line stroke="red" x1="21" y1="9" x2="22" y2="10"/>
  <line stroke="blue" x1="19" y1="13" x2="20" y2="12"/>
  <line stroke="red" x1="19" y1="13" x2="19" y2="14"/>
  <line stroke="blue" x1="19" y1="13" x2="20" y2="14"/>
  <line stroke="red" x1="19" y1="13" x2="23" y2="13"/>
  <line stroke="red" x1="23" y1="1" x2="24" y2="0"/>
  <line stroke="red" x1="23" y1="1" x2="23" y2="8"/>
  <line stroke="red" x1="3" y1="23" x2="7" y2="23"/>
  <line stroke="red" x1="3" y1="23" x2="3" y2="24"/>
  <line stroke="blue" x1="3" y1="23" x2="4" y2="24"/>
  <line stroke="red" x1="12" y1="20" x2="13" y2="20"/>
  <line stroke="blue" x1="12" y1="20" x2="12" y2="28"/>
  <line stroke="blue" x1="20" y1="12" x2="24" y2="12"/>
  <line stroke="red" x1="17" y1="16" x2="18" y2="16"/>
  <line stroke="red" x1="17" y1="16" x2="17" y2="18"/>
  <line stroke="red" x1="15" y1="18" x2="17" y2="18"/>
  <line stroke="red" x1="15" y1="18" x2="15" y2="20"/>
  <line stroke="blue" x1="18" y1="15" x2="18" y2="16"/>
  <line stroke="red" x1="18" y1="15" x2="19" y2="15"/>
  <line stroke="blue" x1="22" y1="8" x2="22" y2="10"/>
  <line stroke="red" x1="22" y1="8" x2="23" y2="8"/>
  <line stroke="red" x1="19" y1="14" x2="19" y2="15"/>
  <line stroke="red" x1="19" y1="14" x2="20" y2="14"/>
  <line stroke="blue" x1="13" y1="20" x2="14" y2="20"/>
  <line stroke="red" x1="13" y1="20" x2="13" y2="29"/>
  <line stroke="blue" x1="0" y1="24" x2="1" y2="24"/>
  <line stroke="black" x1="0" y1="24" x2="0" y2="28"/>
  <line stroke="red" x1="24" y1="0" x2="25" y2="1"/>
  <line stroke="blue" x1="24" y1="0" x2="24" y2="8"/>
  <line stroke="black" x1="24" y1="0" x2="28" y2="0"/>
  <line stroke="red" x1="1" y1="24" x2="2" y2="24"/>
  <line stroke="red" x1="1" y1="24" x2="1" y2="27"/>
  <line stroke="red" x1="7" y1="23" x2="8" y2="24"/>
  <line stroke="blue" x1="18" y1="16" x2="19" y2="16"/>
  <line stroke="blue" x1="18" y1="16" x2="18" y2="18"/>
  <line stroke="blue" x1="2" y1="24" x2="3" y2="24"/>
  <line stroke="blue" x1="2" y1="24" x2="2" y2="26"/>
  <line stroke="blue" x1="22" y1="10" x2="23" y2="11"/>
  <line stroke="red" x1="3" y1="24" x2="4" y2="24"/>
  <line stroke="red" x1="3" y1="24" x2="3" y2="25"/>
  <line stroke="blue" x1="19" y1="15" x2="20" y2="14"/>
  <line stroke="red" x1="19" y1="15" x2="19" y2="16"/>
  <line stroke="blue" x1="19" y1="15" x2="20" y2="16"/>
  <line stroke="red" x1="19" y1="15" x2="21" y2="15"/>
  <line stroke="blue" x1="4" y1="24" x2="3" y2="25"/>
  <line stroke="blue" x1="4" y1="24" x2="8" y2="24"/>
  <line stroke="blue" x1="23" y1="8" x2="24" y2="8"/>
  <line stroke="red" x1="23" y1="8" x2="23" y2="11"/>
  <line stroke="red" x1="14" y1="20" x2="15" y2="20"/>
  <line stroke="blue" x1="14" y1="20" x2="14" y2="30"/>
  <line stroke="blue" x1="20" y1="14" x2="22" y2="14"/>
  <line stroke="blue" x1="17" y1="18" x2="18" y2="18"/>
  <line stroke="red" x1="17" y1="18" x2="17" y2="20"/>
  <line stroke="red" x1="19" y1="16" x2="20" y2="16"/>
  <line stroke="red" x1="19" y1="16" x2="19" y2="18"/>
  <line stroke="blue" x1="15" y1="20" x2="17" y2="20"/>
  <line stroke="red" x1="15" y1="20" x2="15" y2="31"/>
  <line stroke="blue" x1="25" y1="1" x2="26" y2="2"/>
  <line stroke="red" x1="25" y1="1" x2="25" y2="8"/>
  <line stroke="red" x1="25" y1="1" x2="28" y2="1"/>
  <line stroke="red" x1="3" y1="25" x2="2" y2="26"/>
  <line stroke="red" x1="3" y1="25" x2="9" y2="25"/>
  <line stroke="blue" x1="8" y1="24" x2="9" y2="25"/>
  <line stroke="red" x1="24" y1="8" x2="25" y2="8"/>
  <line stroke="blue" x1="24" y1="8" x2="24" y2="12"/>
  <line stroke="red" x1="18" y1="18" x2="19" y2="18"/>
  <line stroke="blue" x1="18" y1="18" x2="18" y2="20"/>
  <line stroke="red" x1="23" y1="11" x2="24" y2="12"/>
  <line stroke="blue" x1="20" y1="16" x2="21" y2="15"/>
  <line stroke="blue" x1="20" y1="16" x2="20" y2="18"/>
  <line stroke="red" x1="21" y1="15" x2="22" y2="14"/>
  <line stroke="red" x1="21" y1="15" x2="21" y2="18"/>
  <line stroke="blue" x1="2" y1="26" x2="1" y2="27"/>
  <line stroke="blue" x1="2" y1="26" x2="10" y2="26"/>
  <line stroke="red" x1="26" y1="2" x2="27" y2="3"/>
  <line stroke="blue" x1="26" y1="2" x2="26" y2="8"/>
  <line stroke="blue" x1="26" y1="2" x2="28" y2="2"/>
  <line stroke="blue" x1="22" y1="14" x2="23" y2="13"/>
  <line stroke="blue" x1="22" y1="14" x2="22" y2="18"/>
  <line stroke="blue" x1="19" y1="18" x2="20" y2="18"/>
  <line stroke="red" x1="19" y1="18" x2="19" y2="20"/>
  <line stroke="blue" x1="25" y1="8" x2="26" y2="8"/>
  <line stroke="red" x1="25" y1="8" x2="25" y2="11"/>
  <line stroke="red" x1="17" y1="20" x2="18" y2="20"/>
  <line stroke="red" x1="17" y1="20" x2="17" y2="31"/>
  <line stroke="red" x1="23" y1="13" x2="24" y2="12"/>
  <line stroke="red" x1="23" y1="13" x2="23" y2="18"/>
  <line stroke="red" x1="9" y1="25" x2="10" y2="26"/>
  <line stroke="red" x1="24" y1="12" x2="25" y2="11"/>
  <line stroke="red" x1="24" y1="12" x2="25" y2="13"/>
  <line stroke="blue" x1="24" y1="12" x2="24" y2="18"/>
  <line stroke="red" x1="24" y1="12" x2="28" y2="12"/>
  <line stroke="blue" x1="18" y1="20" x2="19" y2="20"/>
  <line stroke="blue" x1="18" y1="20" x2="18" y2="30"/>
  <line stroke="red" x1="20" y1="18" x2="21" y2="18"/>
  <line stroke="blue" x1="20" y1="18" x2="20" y2="20"/>
  <line stroke="red" x1="1" y1="27" x2="0" y2="28"/>
  <line stroke="red" x1="1" y1="27" x2="11" y2="27"/>
  <line stroke="red" x1="27" y1="3" x2="28" y2="3"/>
  <line stroke="red" x1="27" y1="3" x2="27" y2="8"/>
  <line stroke="blue" x1="27" y1="3" x2="28" y2="4"/>
  <line stroke="blue" x1="26" y1="8" x2="26" y2="10"/>
  <line stroke="red" x1="26" y1="8" x2="27" y2="8"/>
  <line stroke="blue" x1="25" y1="11" x2="26" y2="10"/>
  <line stroke="red" x1="25" y1="11" x2="28" y2="11"/>
  <line stroke="red" x1="19" y1="20" x2="20" y2="20"/>
  <line stroke="red" x1="19" y1="20" x2="19" y2="29"/>
  <line stroke="blue" x1="21" y1="18" x2="22" y2="18"/>
  <line stroke="red" x1="21" y1="18" x2="21" y2="20"/>
  <line stroke="blue" x1="10" y1="26" x2="11" y2="27"/>
  <line stroke="red" x1="26" y1="10" x2="27" y2="9"/>
  <line stroke="blue" x1="26" y1="10" x2="28" y2="10"/>
  <line stroke="red" x1="0" y1="28" x2="1" y2="29"/>
  <line stroke="blue" x1="0" y1="28" x2="12" y2="28"/>
  <line stroke="black" x1="0" y1="28" x2="0" y2="32"/>
  <line stroke="blue" x1="28" y1="0" x2="28" y2="1"/>
  <line stroke="black" x1="28" y1="0" x2="32" y2="0"/>
  <line stroke="red" x1="28" y1="1" x2="28" y2="2"/>
  <line stroke="red" x1="28" y1="1" x2="31" y2="1"/>
  <line stroke="blue" x1="28" y1="2" x2="28" y2="3"/>
  <line stroke="blue" x1="28" y1="2" x2="30" y2="2"/>
  <line stroke="red" x1="28" y1="3" x2="28" y2="4"/>
  <line stroke="red" x1="28" y1="3" x2="29" y2="3"/>
  <line stroke="blue" x1="25" y1="13" x2="26" y2="14"/>
  <line stroke="red" x1="25" y1="13" x2="25" y2="18"/>
  <line stroke="red" x1="25" y1="13" x2="28" y2="13"/>
  <line stroke="red" x1="27" y1="8" x2="27" y2="9"/>
  <line stroke="blue" x1="27" y1="8" x2="28" y2="8"/>
  <line stroke="blue" x1="20" y1="20" x2="21" y2="20"/>
  <line stroke="blue" x1="20" y1="20" x2="20" y2="28"/>
  <line stroke="blue" x1="28" y1="4" x2="28" y2="8"/>
  <line stroke="blue" x1="28" y1="4" x2="29" y2="3"/>
  <line stroke="red" x1="22" y1="18" x2="23" y2="18"/>
  <line stroke="blue" x1="22" y1="18" x2="22" y2="20"/>
  <line stroke="blue" x1="27" y1="9" x2="28" y2="8"/>
  <line stroke="red" x1="27" y1="9" x2="28" y2="9"/>
  <line stroke="red" x1="21" y1="20" x2="22" y2="20"/>
  <line stroke="red" x1="21" y1="20" x2="21" y2="27"/>
  <line stroke="blue" x1="1" y1="29" x2="2" y2="30"/>
  <line stroke="red" x1="1" y1="29" x2="13" y2="29"/>
  <line stroke="red" x1="1" y1="29" x2="1" y2="32"/>
  <line stroke="red" x1="28" y1="8" x2="28" y2="9"/>
  <line stroke="red" x1="28" y1="8" x2="29" y2="8"/>
  <line stroke="blue" x1="28" y1="8" x2="29" y2="9"/>
  <line stroke="red" x1="29" y1="3" x2="29" y2="8"/>
  <line stroke="red" x1="29" y1="3" x2="30" y2="2"/>
  <line stroke="red" x1="11" y1="27" x2="12" y2="28"/>
  <line stroke="blue" x1="23" y1="18" x2="24" y2="18"/>
  <line stroke="red" x1="23" y1="18" x2="23" y2="20"/>
  <line stroke="blue" x1="28" y1="9" x2="28" y2="10"/>
  <line stroke="red" x1="28" y1="9" x2="29" y2="9"/>
  <line stroke="red" x1="26" y1="14" x2="27" y2="15"/>
  <line stroke="blue" x1="26" y1="14" x2="28" y2="14"/>
  <line stroke="blue" x1="26" y1="14" x2="26" y2="18"/>
  <line stroke="red" x1="28" y1="10" x2="28" y2="11"/>
  <line stroke="blue" x1="28" y1="10" x2="30" y2="10"/>
  <line stroke="blue" x1="22" y1="20" x2="23" y2="20"/>
  <line stroke="blue" x1="22" y1="20" x2="22" y2="26"/>
  <line stroke="red" x1="24" y1="18" x2="25" y2="18"/>
  <line stroke="blue" x1="24" y1="18" x2="24" y2="20"/>
  <line stroke="red" x1="29" y1="8" x2="29" y2="9"/>
  <line stroke="blue" x1="29" y1="8" x2="30" y2="8"/>
  <line stroke="red" x1="2" y1="30" x2="3" y2="31"/>
  <line stroke="blue" x1="2" y1="30" x2="2" y2="32"/>
  <line stroke="blue" x1="2" y1="30" x2="14" y2="30"/>
  <line stroke="blue" x1="30" y1="2" x2="31" y2="1"/>
  <line stroke="blue" x1="30" y1="2" x2="30" y2="8"/>
  <line stroke="blue" x1="28" y1="11" x2="28" y2="12"/>
  <line stroke="red" x1="28" y1="11" x2="31" y2="11"/>
  <line stroke="red" x1="29" y1="9" x2="30" y2="10"/>
  <line stroke="blue" x1="12" y1="28" x2="13" y2="29"/>
  <line stroke="blue" x1="28" y1="12" x2="28" y2="13"/>
  <line stroke="blue" x1="28" y1="12" x2="32" y2="12"/>
  <line stroke="red" x1="23" y1="20" x2="24" y2="20"/>
  <line stroke="red" x1="23" y1="20" x2="23" y2="25"/>
  <line stroke="blue" x1="25" y1="18" x2="26" y2="18"/>
  <line stroke="red" x1="25" y1="18" x2="25" y2="20"/>
  <line stroke="red" x1="28" y1="13" x2="28" y2="14"/>
  <line stroke="red" x1="28" y1="13" x2="31" y2="13"/>
  <line stroke="red" x1="27" y1="15" x2="28" y2="15"/>
  <line stroke="red" x1="27" y1="15" x2="27" y2="17"/>
  <line stroke="blue" x1="27" y1="15" x2="28" y2="16"/>
  <line stroke="red" x1="31" y1="1" x2="32" y2="0"/>
  <line stroke="red" x1="31" y1="1" x2="31" y2="8"/>
  <line stroke="blue" x1="30" y1="8" x2="30" y2="10"/>
  <line stroke="red" x1="30" y1="8" x2="31" y2="8"/>
  <line stroke="red" x1="3" y1="31" x2="3" y2="32"/>
  <line stroke="blue" x1="3" y1="31" x2="4" y2="32"/>
  <line stroke="red" x1="3" y1="31" x2="15" y2="31"/>
  <line stroke="blue" x1="24" y1="20" x2="25" y2="20"/>
  <line stroke="blue" x1="24" y1="20" x2="24" y2="24"/>
  <line stroke="blue" x1="28" y1="14" x2="28" y2="15"/>
  <line stroke="blue" x1="28" y1="14" x2="30" y2="14"/>
  <line stroke="red" x1="26" y1="18" x2="27" y2="17"/>
  <line stroke="blue" x1="26" y1="18" x2="26" y2="20"/>
  <line stroke="red" x1="26" y1="18" x2="27" y2="19"/>
  <line stroke="blue" x1="26" y1="18" x2="30" y2="18"/>
  <line stroke="blue" x1="30" y1="10" x2="31" y2="11"/>
  <line stroke="red" x1="28" y1="15" x2="28" y2="16"/>
  <line stroke="red" x1="28" y1="15" x2="29" y2="15"/>
  <line stroke="red" x1="13" y1="29" x2="14" y2="30"/>
  <line stroke="blue" x1="27" y1="17" x2="28" y2="16"/>
  <line stroke="red" x1="27" y1="17" x2="29" y2="17"/>
  <line stroke="black" x1="0" y1="32" x2="1" y2="32"/>
  <line stroke="black" x1="32" y1="0" x2="32" y2="8"/>
  <line stroke="red" x1="25" y1="20" x2="26" y2="20"/>
  <line stroke="red" x1="25" y1="20" x2="25" y2="23"/>
  <line stroke="black" x1="1" y1="32" x2="2" y2="32"/>
  <line stroke="red" x1="31" y1="8" x2="31" y2="11"/>
  <line stroke="blue" x1="31" y1="8" x2="32" y2="8"/>
  <line stroke="black" x1="2" y1="32" x2="3" y2="32"/>
  <line stroke="black" x1="3" y1="32" x2="4" y2="32"/>
  <line stroke="black" x1="4" y1="32" x2="16" y2="32"/>
  <line stroke="blue" x1="28" y1="16" x2="29" y2="15"/>
  <line stroke="red" x1="28" y1="16" x2="29" y2="16"/>
  <line stroke="blue" x1="28" y1="16" x2="29" y2="17"/>
  <line stroke="red" x1="29" y1="15" x2="29" y2="16"/>
  <line stroke="red" x1="29" y1="15" x2="30" y2="14"/>
  <line stroke="blue" x1="26" y1="20" x2="27" y2="20"/>
  <line stroke="blue" x1="26" y1="20" x2="26" y2="22"/>
  <line stroke="red" x1="31" y1="11" x2="32" y2="12"/>
  <line stroke="black" x1="32" y1="8" x2="32" y2="12"/>
  <line stroke="red" x1="27" y1="19" x2="27" y2="20"/>
  <line stroke="blue" x1="27" y1="19" x2="28" y2="20"/>
  <line stroke="red" x1="27" y1="19" x2="31" y2="19"/>
  <line stroke="red" x1="29" y1="16" x2="29" y2="17"/>
  <line stroke="blue" x1="29" y1="16" x2="30" y2="16"/>
  <line stroke="blue" x1="14" y1="30" x2="15" y2="31"/>
  <line stroke="blue" x1="30" y1="14" x2="31" y2="13"/>
  <line stroke="blue" x1="30" y1="14" x2="30" y2="16"/>
  <line stroke="red" x1="29" y1="17" x2="30" y2="18"/>
  <line stroke="red" x1="27" y1="20" x2="27" y2="21"/>
  <line stroke="red" x1="27" y1="20" x2="28" y2="20"/>
  <line stroke="red" x1="31" y1="13" x2="32" y2="12"/>
  <line stroke="red" x1="31" y1="13" x2="31" y2="16"/>
  <line stroke="blue" x1="24" y1="24" x2="23" y2="25"/>
  <line stroke="red" x1="24" y1="24" x2="25" y2="23"/>
  <line stroke="blue" x1="24" y1="24" x2="28" y2="24"/>
  <line stroke="red" x1="23" y1="25" x2="22" y2="26"/>
  <line stroke="red" x1="23" y1="25" x2="29" y2="25"/>
  <line stroke="blue" x1="25" y1="23" x2="26" y2="22"/>
  <line stroke="red" x1="25" y1="23" x2="29" y2="23"/>
  <line stroke="red" x1="30" y1="16" x2="31" y2="16"/>
  <line stroke="blue" x1="30" y1="16" x2="30" y2="18"/>
  <line stroke="blue" x1="22" y1="26" x2="21" y2="27"/>
  <line stroke="blue" x1="22" y1="26" x2="30" y2="26"/>
  <line stroke="red" x1="26" y1="22" x2="27" y2="21"/>
  <line stroke="blue" x1="26" y1="22" x2="30" y2="22"/>
  <line stroke="black" x1="32" y1="12" x2="32" y2="16"/>
  <line stroke="red" x1="21" y1="27" x2="20" y2="28"/>
  <line stroke="red" x1="21" y1="27" x2="31" y2="27"/>
  <line stroke="blue" x1="27" y1="21" x2="28" y2="20"/>
  <line stroke="red" x1="27" y1="21" x2="31" y2="21"/>
  <line stroke="blue" x1="20" y1="28" x2="19" y2="29"/>
  <line stroke="blue" x1="20" y1="28" x2="32" y2="28"/>
  <line stroke="blue" x1="28" y1="20" x2="32" y2="20"/>
  <line stroke="red" x1="15" y1="31" x2="16" y2="32"/>
  <line stroke="red" x1="19" y1="29" x2="18" y2="30"/>
  <line stroke="red" x1="19" y1="29" x2="31" y2="29"/>
  <line stroke="blue" x1="31" y1="16" x2="32" y2="16"/>
  <line stroke="red" x1="31" y1="16" x2="31" y2="19"/>
  <line stroke="blue" x1="18" y1="30" x2="17" y2="31"/>
  <line stroke="blue" x1="18" y1="30" x2="30" y2="30"/>
  <line stroke="blue" x1="30" y1="18" x2="31" y2="19"/>
  <line stroke="red" x1="17" y1="31" x2="16" y2="32"/>
  <line stroke="red" x1="17" y1="31" x2="29" y2="31"/>
  <line stroke="black" x1="16" y1="32" x2="28" y2="32"/>
  <line stroke="black" x1="32" y1="16" x2="32" y2="20"/>
  <line stroke="red" x1="31" y1="19" x2="32" y2="20"/>
  <line stroke="blue" x1="28" y1="24" x2="29" y2="23"/>
  <line stroke="red" x1="28" y1="24" x2="29" y2="24"/>
  <line stroke="blue" x1="28" y1="24" x2="29" y2="25"/>
  <line stroke="red" x1="29" y1="23" x2="30" y2="22"/>
  <line stroke="red" x1="29" y1="23" x2="29" y2="24"/>
  <line stroke="blue" x1="30" y1="22" x2="31" y2="21"/>
  <line stroke="blue" x1="30" y1="22" x2="30" y2="24"/>
  <line stroke="red" x1="31" y1="21" x2="32" y2="20"/>
  <line stroke="red" x1="31" y1="21" x2="31" y2="24"/>
  <line stroke="red" x1="29" y1="24" x2="29" y2="25"/>
  <line stroke="blue" x1="29" y1="24" x2="30" y2="24"/>
  <line stroke="black" x1="32" y1="20" x2="32" y2="24"/>
  <line stroke="red" x1="29" y1="25" x2="30" y2="26"/>
  <line stroke="red" x1="30" y1="24" x2="31" y2="24"/>
  <line stroke="blue" x1="30" y1="24" x2="30" y2="26"/>
  <line stroke="blue" x1="31" y1="24" x2="32" y2="24"/>
  <line stroke="red" x1="31" y1="24" x2="31" y2="27"/>
  <line stroke="blue" x1="30" y1="26" x2="31" y2="27"/>
  <line stroke="black" x1="32" y1="24" x2="32" y2="28"/>
  <line stroke="red" x1="31" y1="27" x2="32" y2="28"/>
  <line stroke="red" x1="30" y1="30" x2="29" y2="31"/>
  <line stroke="blue" x1="30" y1="30" x2="31" y2="29"/>
  <line stroke="blue" x1="30" y1="30" x2="30" y2="32"/>
  <line stroke="blue" x1="29" y1="31" x2="28" y2="32"/>
  <line stroke="red" x1="29" y1="31" x2="29" y2="32"/>
  <line stroke="red" x1="31" y1="29" x2="32" y2="28"/>
  <line stroke="red" x1="31" y1="29" x2="31" y2="32"/>
  <line stroke="black" x1="28" y1="32" x2="29" y2="32"/>
  <line stroke="black" x1="32" y1="28" x2="32" y2="32"/>
  <line stroke="black" x1="29" y1="32" x2="30" y2="32"/>
  <line stroke="black" x1="30" y1="32" x2="31" y2="32"/>
  <line stroke="black" x1="31" y1="32" x2="32" y2="32"/>
</svg>""",
    zigzag_shaping_svg_sha=(
        "a96ff7cccf53bcb57f64a6d9cdc8cf792e4c17817bb6022dce7bf7ea469de46e"
    ),
    zigzag_shaping_svg="""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
  <line stroke="red" x1="0" y1="0" x2="31" y2="31"/>
  <line stroke="black" x1="0" y1="0" x2="0" y2="62"/>
  <line stroke="black" x1="0" y1="0" x2="62" y2="0"/>
  <line stroke="blue" x1="31" y1="31" x2="62" y2="62"/>
  <line stroke="red" x1="31" y1="31" x2="31" y2="125"/>
  <line stroke="red" x1="31" y1="31" x2="125" y2="31"/>
  <line stroke="blue" x1="0" y1="62" x2="31" y2="125"/>
  <line stroke="black" x1="0" y1="62" x2="0" y2="219"/>
  <line stroke="black" x1="62" y1="0" x2="125" y2="0"/>
  <line stroke="red" x1="62" y1="0" x2="125" y2="31"/>
  <line stroke="red" x1="62" y1="62" x2="31" y2="125"/>
  <line stroke="blue" x1="62" y1="62" x2="125" y2="31"/>
  <line stroke="red" x1="62" y1="62" x2="94" y2="94"/>
  <line stroke="blue" x1="62" y1="62" x2="125" y2="62"/>
  <line stroke="blue" x1="62" y1="62" x2="94" y2="125"/>
  <line stroke="red" x1="62" y1="62" x2="125" y2="94"/>
  <line stroke="blue" x1="62" y1="62" x2="62" y2="219"/>
  <line stroke="blue" x1="125" y1="0" x2="125" y2="31"/>
  <line stroke="black" x1="125" y1="0" x2="250" y2="0"/>
  <line stroke="red" x1="31" y1="125" x2="31" y2="203"/>
  <line stroke="red" x1="125" y1="31" x2="125" y2="62"/>
  <line stroke="red" x1="125" y1="31" x2="219" y2="31"/>
  <line stroke="blue" x1="94" y1="94" x2="109" y2="109"/>
  <line stroke="red" x1="94" y1="94" x2="94" y2="125"/>
  <line stroke="red" x1="94" y1="94" x2="125" y2="94"/>
  <line stroke="blue" x1="125" y1="62" x2="125" y2="94"/>
  <line stroke="blue" x1="125" y1="62" x2="188" y2="62"/>
  <line stroke="red" x1="109" y1="109" x2="94" y2="125"/>
  <line stroke="blue" x1="109" y1="109" x2="125" y2="94"/>
  <line stroke="blue" x1="109" y1="109" x2="125" y2="125"/>
  <line stroke="red" x1="94" y1="125" x2="94" y2="203"/>
  <line stroke="red" x1="125" y1="94" x2="125" y2="125"/>
  <line stroke="red" x1="125" y1="94" x2="156" y2="94"/>
  <line stroke="blue" x1="125" y1="125" x2="156" y2="94"/>
  <line stroke="blue" x1="125" y1="125" x2="125" y2="219"/>
  <line stroke="red" x1="156" y1="94" x2="188" y2="62"/>
  <line stroke="red" x1="156" y1="94" x2="156" y2="203"/>
  <line stroke="blue" x1="188" y1="62" x2="219" y2="31"/>
  <line stroke="blue" x1="188" y1="62" x2="188" y2="219"/>
  <line stroke="blue" x1="31" y1="203" x2="0" y2="219"/>
  <line stroke="red" x1="31" y1="203" x2="31" y2="219"/>
  <line stroke="red" x1="31" y1="203" x2="62" y2="219"/>
  <line stroke="red" x1="0" y1="219" x2="31" y2="219"/>
  <line stroke="black" x1="0" y1="219" x2="0" y2="250"/>
  <line stroke="blue" x1="31" y1="219" x2="62" y2="219"/>
  <line stroke="red" x1="31" y1="219" x2="31" y2="250"/>
  <line stroke="red" x1="219" y1="31" x2="250" y2="0"/>
  <line stroke="red" x1="219" y1="31" x2="219" y2="203"/>
  <line stroke="blue" x1="94" y1="203" x2="62" y2="219"/>
  <line stroke="red" x1="94" y1="203" x2="94" y2="219"/>
  <line stroke="red" x1="94" y1="203" x2="125" y2="219"/>
  <line stroke="red" x1="62" y1="219" x2="94" y2="219"/>
  <line stroke="blue" x1="62" y1="219" x2="62" y2="250"/>
  <line stroke="blue" x1="94" y1="219" x2="125" y2="219"/>
  <line stroke="red" x1="94" y1="219" x2="94" y2="250"/>
  <line stroke="blue" x1="0" y1="250" x2="31" y2="250"/>
  <line stroke="red" x1="0" y1="250" x2="31" y2="297"/>
  <line stroke="black" x1="0" y1="250" x2="0" y2="375"/>
  <line stroke="red" x1="250" y1="0" x2="281" y2="31"/>
  <line stroke="blue" x1="250" y1="0" x2="250" y2="219"/>
  <line stroke="black" x1="250" y1="0" x2="469" y2="0"/>
  <line stroke="red" x1="31" y1="250" x2="62" y2="250"/>
  <line stroke="red" x1="31" y1="250" x2="31" y2="297"/>
  <line stroke="blue" x1="125" y1="219" x2="156" y2="203"/>
  <line stroke="red" x1="125" y1="219" x2="156" y2="219"/>
  <line stroke="blue" x1="125" y1="219" x2="125" y2="250"/>
  <line stroke="red" x1="156" y1="203" x2="156" y2="219"/>
  <line stroke="red" x1="156" y1="203" x2="188" y2="219"/>
  <line stroke="blue" x1="62" y1="250" x2="94" y2="250"/>
  <line stroke="blue" x1="62" y1="250" x2="31" y2="297"/>
  <line stroke="red" x1="62" y1="250" x2="87" y2="287"/>
  <line stroke="blue" x1="62" y1="250" x2="62" y2="312"/>
  <line stroke="red" x1="94" y1="250" x2="125" y2="250"/>
  <line stroke="red" x1="94" y1="250" x2="94" y2="281"/>
  <line stroke="blue" x1="156" y1="219" x2="188" y2="219"/>
  <line stroke="red" x1="156" y1="219" x2="156" y2="250"/>
  <line stroke="blue" x1="125" y1="250" x2="156" y2="250"/>
  <line stroke="blue" x1="125" y1="250" x2="94" y2="281"/>
  <line stroke="red" x1="125" y1="250" x2="125" y2="281"/>
  <line stroke="blue" x1="125" y1="250" x2="156" y2="281"/>
  <line stroke="blue" x1="281" y1="31" x2="312" y2="62"/>
  <line stroke="red" x1="281" y1="31" x2="281" y2="203"/>
  <line stroke="red" x1="281" y1="31" x2="453" y2="31"/>
  <line stroke="blue" x1="188" y1="219" x2="219" y2="203"/>
  <line stroke="red" x1="188" y1="219" x2="219" y2="219"/>
  <line stroke="blue" x1="188" y1="219" x2="188" y2="250"/>
  <line stroke="red" x1="156" y1="250" x2="188" y2="250"/>
  <line stroke="red" x1="156" y1="250" x2="156" y2="281"/>
  <line stroke="red" x1="94" y1="281" x2="87" y2="287"/>
  <line stroke="red" x1="94" y1="281" x2="125" y2="281"/>
  <line stroke="red" x1="31" y1="297" x2="31" y2="344"/>
  <line stroke="red" x1="219" y1="203" x2="219" y2="219"/>
  <line stroke="red" x1="219" y1="203" x2="250" y2="219"/>
  <line stroke="red" x1="87" y1="287" x2="62" y2="312"/>
  <line stroke="blue" x1="87" y1="287" x2="125" y2="312"/>
  <line stroke="red" x1="125" y1="281" x2="156" y2="281"/>
  <line stroke="blue" x1="125" y1="281" x2="125" y2="312"/>
  <line stroke="red" x1="219" y1="219" x2="219" y2="250"/>
  <line stroke="blue" x1="219" y1="219" x2="250" y2="219"/>
  <line stroke="blue" x1="188" y1="250" x2="156" y2="281"/>
  <line stroke="blue" x1="188" y1="250" x2="219" y2="250"/>
  <line stroke="red" x1="188" y1="250" x2="219" y2="281"/>
  <line stroke="blue" x1="188" y1="250" x2="188" y2="312"/>
  <line stroke="blue" x1="62" y1="312" x2="125" y2="312"/>
  <line stroke="blue" x1="62" y1="312" x2="31" y2="344"/>
  <line stroke="red" x1="312" y1="62" x2="344" y2="94"/>
  <line stroke="blue" x1="312" y1="62" x2="312" y2="219"/>
  <line stroke="blue" x1="312" y1="62" x2="469" y2="62"/>
  <line stroke="red" x1="156" y1="281" x2="125" y2="312"/>
  <line stroke="red" x1="156" y1="281" x2="188" y2="312"/>
  <line stroke="red" x1="219" y1="250" x2="250" y2="250"/>
  <line stroke="red" x1="219" y1="250" x2="219" y2="281"/>
  <line stroke="blue" x1="250" y1="219" x2="281" y2="203"/>
  <line stroke="blue" x1="250" y1="219" x2="250" y2="250"/>
  <line stroke="red" x1="250" y1="219" x2="281" y2="219"/>
  <line stroke="red" x1="125" y1="312" x2="78" y2="344"/>
  <line stroke="blue" x1="125" y1="312" x2="188" y2="312"/>
  <line stroke="red" x1="125" y1="312" x2="125" y2="344"/>
  <line stroke="blue" x1="125" y1="312" x2="156" y2="344"/>
  <line stroke="red" x1="31" y1="344" x2="78" y2="344"/>
  <line stroke="red" x1="31" y1="344" x2="0" y2="375"/>
  <line stroke="red" x1="281" y1="203" x2="281" y2="219"/>
  <line stroke="red" x1="281" y1="203" x2="312" y2="219"/>
  <line stroke="red" x1="78" y1="344" x2="125" y2="344"/>
  <line stroke="blue" x1="78" y1="344" x2="125" y2="375"/>
  <line stroke="blue" x1="250" y1="250" x2="219" y2="281"/>
  <line stroke="blue" x1="250" y1="250" x2="281" y2="250"/>
  <line stroke="red" x1="250" y1="250" x2="281" y2="281"/>
  <line stroke="blue" x1="250" y1="250" x2="250" y2="375"/>
  <line stroke="red" x1="219" y1="281" x2="219" y2="344"/>
  <line stroke="red" x1="281" y1="219" x2="281" y2="250"/>
  <line stroke="blue" x1="281" y1="219" x2="312" y2="219"/>
  <line stroke="blue" x1="344" y1="94" x2="375" y2="125"/>
  <line stroke="red" x1="344" y1="94" x2="344" y2="203"/>
  <line stroke="red" x1="344" y1="94" x2="453" y2="94"/>
  <line stroke="blue" x1="188" y1="312" x2="219" y2="344"/>
  <line stroke="red" x1="125" y1="344" x2="156" y2="344"/>
  <line stroke="blue" x1="125" y1="344" x2="125" y2="375"/>
  <line stroke="blue" x1="0" y1="375" x2="125" y2="375"/>
  <line stroke="red" x1="0" y1="375" x2="31" y2="406"/>
  <line stroke="black" x1="0" y1="375" x2="0" y2="500"/>
  <line stroke="red" x1="281" y1="250" x2="281" y2="281"/>
  <line stroke="red" x1="281" y1="250" x2="312" y2="250"/>
  <line stroke="red" x1="156" y1="344" x2="125" y2="375"/>
  <line stroke="red" x1="156" y1="344" x2="219" y2="344"/>
  <line stroke="blue" x1="312" y1="219" x2="344" y2="203"/>
  <line stroke="blue" x1="312" y1="219" x2="312" y2="250"/>
  <line stroke="red" x1="312" y1="219" x2="344" y2="219"/>
  <line stroke="red" x1="125" y1="375" x2="78" y2="406"/>
  <line stroke="blue" x1="125" y1="375" x2="125" y2="406"/>
  <line stroke="blue" x1="125" y1="375" x2="156" y2="406"/>
  <line stroke="red" x1="125" y1="375" x2="250" y2="375"/>
  <line stroke="blue" x1="375" y1="125" x2="375" y2="219"/>
  <line stroke="red" x1="375" y1="125" x2="406" y2="156"/>
  <line stroke="blue" x1="375" y1="125" x2="469" y2="125"/>
  <line stroke="blue" x1="281" y1="281" x2="312" y2="250"/>
  <line stroke="red" x1="281" y1="281" x2="281" y2="344"/>
  <line stroke="red" x1="344" y1="203" x2="344" y2="219"/>
  <line stroke="red" x1="344" y1="203" x2="375" y2="219"/>
  <line stroke="blue" x1="312" y1="250" x2="344" y2="250"/>
  <line stroke="blue" x1="312" y1="250" x2="312" y2="312"/>
  <line stroke="red" x1="312" y1="250" x2="344" y2="281"/>
  <line stroke="red" x1="31" y1="406" x2="78" y2="406"/>
  <line stroke="blue" x1="31" y1="406" x2="62" y2="438"/>
  <line stroke="red" x1="31" y1="406" x2="31" y2="453"/>
  <line stroke="red" x1="219" y1="344" x2="250" y2="375"/>
  <line stroke="red" x1="344" y1="219" x2="344" y2="250"/>
  <line stroke="blue" x1="344" y1="219" x2="375" y2="219"/>
  <line stroke="red" x1="78" y1="406" x2="125" y2="406"/>
  <line stroke="blue" x1="78" y1="406" x2="125" y2="438"/>
  <line stroke="red" x1="125" y1="406" x2="156" y2="406"/>
  <line stroke="red" x1="125" y1="406" x2="125" y2="438"/>
  <line stroke="red" x1="344" y1="250" x2="344" y2="281"/>
  <line stroke="red" x1="344" y1="250" x2="375" y2="250"/>
  <line stroke="blue" x1="375" y1="219" x2="375" y2="250"/>
  <line stroke="blue" x1="375" y1="219" x2="406" y2="203"/>
  <line stroke="red" x1="375" y1="219" x2="406" y2="219"/>
  <line stroke="red" x1="156" y1="406" x2="125" y2="438"/>
  <line stroke="red" x1="156" y1="406" x2="219" y2="406"/>
  <line stroke="red" x1="406" y1="156" x2="406" y2="203"/>
  <line stroke="blue" x1="406" y1="156" x2="438" y2="188"/>
  <line stroke="red" x1="406" y1="156" x2="453" y2="156"/>
  <line stroke="blue" x1="62" y1="438" x2="125" y2="438"/>
  <line stroke="red" x1="62" y1="438" x2="87" y2="463"/>
  <line stroke="blue" x1="62" y1="438" x2="62" y2="500"/>
  <line stroke="blue" x1="312" y1="312" x2="281" y2="344"/>
  <line stroke="red" x1="312" y1="312" x2="344" y2="281"/>
  <line stroke="blue" x1="312" y1="312" x2="375" y2="312"/>
  <line stroke="red" x1="281" y1="344" x2="250" y2="375"/>
  <line stroke="red" x1="281" y1="344" x2="344" y2="344"/>
  <line stroke="blue" x1="344" y1="281" x2="375" y2="250"/>
  <line stroke="blue" x1="344" y1="281" x2="375" y2="312"/>
  <line stroke="red" x1="344" y1="281" x2="406" y2="281"/>
  <line stroke="red" x1="250" y1="375" x2="219" y2="406"/>
  <line stroke="red" x1="250" y1="375" x2="281" y2="406"/>
  <line stroke="blue" x1="250" y1="375" x2="375" y2="375"/>
  <line stroke="blue" x1="250" y1="375" x2="250" y2="500"/>
  <line stroke="blue" x1="375" y1="250" x2="406" y2="281"/>
  <line stroke="blue" x1="31" y1="453" x2="0" y2="500"/>
  <line stroke="red" x1="31" y1="453" x2="31" y2="500"/>
  <line stroke="red" x1="31" y1="453" x2="62" y2="500"/>
  <line stroke="red" x1="406" y1="203" x2="406" y2="219"/>
  <line stroke="red" x1="406" y1="203" x2="438" y2="219"/>
  <line stroke="red" x1="453" y1="31" x2="469" y2="0"/>
  <line stroke="red" x1="453" y1="31" x2="469" y2="31"/>
  <line stroke="blue" x1="453" y1="31" x2="469" y2="62"/>
  <line stroke="red" x1="125" y1="438" x2="87" y2="463"/>
  <line stroke="blue" x1="125" y1="438" x2="188" y2="438"/>
  <line stroke="blue" x1="125" y1="438" x2="125" y2="469"/>
  <line stroke="blue" x1="125" y1="438" x2="156" y2="469"/>
  <line stroke="blue" x1="219" y1="406" x2="188" y2="438"/>
  <line stroke="red" x1="219" y1="406" x2="219" y2="469"/>
  <line stroke="blue" x1="406" y1="219" x2="438" y2="219"/>
  <line stroke="red" x1="406" y1="219" x2="406" y2="281"/>
  <line stroke="red" x1="453" y1="94" x2="469" y2="62"/>
  <line stroke="red" x1="453" y1="94" x2="469" y2="94"/>
  <line stroke="blue" x1="453" y1="94" x2="469" y2="125"/>
  <line stroke="blue" x1="469" y1="0" x2="469" y2="31"/>
  <line stroke="black" x1="469" y1="0" x2="500" y2="0"/>
  <line stroke="red" x1="469" y1="31" x2="469" y2="62"/>
  <line stroke="blue" x1="469" y1="31" x2="500" y2="0"/>
  <line stroke="red" x1="469" y1="31" x2="500" y2="62"/>
  <line stroke="red" x1="469" y1="31" x2="531" y2="31"/>
  <line stroke="red" x1="87" y1="463" x2="94" y2="469"/>
  <line stroke="blue" x1="87" y1="463" x2="62" y2="500"/>
  <line stroke="blue" x1="469" y1="62" x2="469" y2="94"/>
  <line stroke="blue" x1="469" y1="62" x2="500" y2="62"/>
  <line stroke="red" x1="188" y1="438" x2="156" y2="469"/>
  <line stroke="blue" x1="188" y1="438" x2="188" y2="500"/>
  <line stroke="blue" x1="438" y1="188" x2="438" y2="219"/>
  <line stroke="red" x1="438" y1="188" x2="458" y2="208"/>
  <line stroke="blue" x1="438" y1="188" x2="469" y2="188"/>
  <line stroke="red" x1="94" y1="469" x2="125" y2="469"/>
  <line stroke="red" x1="94" y1="469" x2="94" y2="500"/>
  <line stroke="blue" x1="94" y1="469" x2="125" y2="500"/>
  <line stroke="red" x1="469" y1="94" x2="469" y2="125"/>
  <line stroke="blue" x1="469" y1="94" x2="500" y2="62"/>
  <line stroke="red" x1="469" y1="94" x2="500" y2="125"/>
  <line stroke="red" x1="469" y1="94" x2="531" y2="94"/>
  <line stroke="red" x1="453" y1="156" x2="469" y2="125"/>
  <line stroke="red" x1="453" y1="156" x2="469" y2="156"/>
  <line stroke="blue" x1="453" y1="156" x2="469" y2="188"/>
  <line stroke="red" x1="125" y1="469" x2="156" y2="469"/>
  <line stroke="red" x1="125" y1="469" x2="125" y2="500"/>
  <line stroke="blue" x1="469" y1="125" x2="469" y2="156"/>
  <line stroke="blue" x1="469" y1="125" x2="500" y2="125"/>
  <line stroke="red" x1="344" y1="344" x2="375" y2="312"/>
  <line stroke="blue" x1="344" y1="344" x2="375" y2="375"/>
  <line stroke="red" x1="344" y1="344" x2="406" y2="344"/>
  <line stroke="blue" x1="375" y1="312" x2="406" y2="281"/>
  <line stroke="red" x1="375" y1="312" x2="406" y2="312"/>
  <line stroke="blue" x1="375" y1="312" x2="406" y2="344"/>
  <line stroke="blue" x1="438" y1="219" x2="458" y2="208"/>
  <line stroke="red" x1="438" y1="219" x2="469" y2="219"/>
  <line stroke="blue" x1="438" y1="219" x2="438" y2="281"/>
  <line stroke="blue" x1="156" y1="469" x2="125" y2="500"/>
  <line stroke="red" x1="156" y1="469" x2="188" y2="500"/>
  <line stroke="red" x1="156" y1="469" x2="156" y2="531"/>
  <line stroke="red" x1="281" y1="406" x2="344" y2="406"/>
  <line stroke="blue" x1="281" y1="406" x2="312" y2="438"/>
  <line stroke="red" x1="281" y1="406" x2="281" y2="469"/>
  <line stroke="red" x1="406" y1="281" x2="406" y2="312"/>
  <line stroke="red" x1="406" y1="281" x2="438" y2="281"/>
  <line stroke="red" x1="469" y1="156" x2="469" y2="188"/>
  <line stroke="blue" x1="469" y1="156" x2="500" y2="125"/>
  <line stroke="red" x1="469" y1="156" x2="500" y2="188"/>
  <line stroke="red" x1="469" y1="156" x2="531" y2="156"/>
  <line stroke="blue" x1="0" y1="500" x2="31" y2="500"/>
  <line stroke="red" x1="0" y1="500" x2="31" y2="516"/>
  <line stroke="black" x1="0" y1="500" x2="0" y2="625"/>
  <line stroke="black" x1="500" y1="0" x2="531" y2="0"/>
  <line stroke="blue" x1="500" y1="0" x2="531" y2="31"/>
  <line stroke="red" x1="31" y1="500" x2="62" y2="500"/>
  <line stroke="red" x1="31" y1="500" x2="31" y2="516"/>
  <line stroke="red" x1="458" y1="208" x2="469" y2="188"/>
  <line stroke="red" x1="458" y1="208" x2="469" y2="219"/>
  <line stroke="blue" x1="62" y1="500" x2="94" y2="500"/>
  <line stroke="blue" x1="62" y1="500" x2="31" y2="516"/>
  <line stroke="red" x1="62" y1="500" x2="94" y2="516"/>
  <line stroke="blue" x1="62" y1="500" x2="62" y2="562"/>
  <line stroke="red" x1="500" y1="62" x2="531" y2="31"/>
  <line stroke="blue" x1="500" y1="62" x2="531" y2="62"/>
  <line stroke="blue" x1="500" y1="62" x2="531" y2="94"/>
  <line stroke="blue" x1="469" y1="188" x2="469" y2="219"/>
  <line stroke="blue" x1="469" y1="188" x2="500" y2="188"/>
  <line stroke="red" x1="94" y1="500" x2="125" y2="500"/>
  <line stroke="red" x1="94" y1="500" x2="94" y2="516"/>
  <line stroke="red" x1="406" y1="312" x2="406" y2="344"/>
  <line stroke="blue" x1="406" y1="312" x2="438" y2="312"/>
  <line stroke="blue" x1="125" y1="500" x2="94" y2="516"/>
  <line stroke="blue" x1="125" y1="500" x2="94" y2="531"/>
  <line stroke="red" x1="125" y1="500" x2="109" y2="531"/>
  <line stroke="blue" x1="125" y1="500" x2="156" y2="531"/>
  <line stroke="red" x1="500" y1="125" x2="531" y2="94"/>
  <line stroke="blue" x1="500" y1="125" x2="531" y2="125"/>
  <line stroke="blue" x1="500" y1="125" x2="531" y2="156"/>
  <line stroke="red" x1="31" y1="516" x2="31" y2="594"/>
  <line stroke="blue" x1="219" y1="469" x2="188" y2="500"/>
  <line stroke="red" x1="219" y1="469" x2="250" y2="500"/>
  <line stroke="red" x1="219" y1="469" x2="219" y2="562"/>
  <line stroke="blue" x1="469" y1="219" x2="500" y2="188"/>
  <line stroke="red" x1="469" y1="219" x2="469" y2="281"/>
  <line stroke="red" x1="469" y1="219" x2="531" y2="219"/>
  <line stroke="blue" x1="438" y1="281" x2="438" y2="312"/>
  <line stroke="blue" x1="438" y1="281" x2="469" y2="281"/>
  <line stroke="red" x1="94" y1="516" x2="94" y2="531"/>
  <line stroke="red" x1="375" y1="375" x2="344" y2="406"/>
  <line stroke="blue" x1="375" y1="375" x2="406" y2="344"/>
  <line stroke="red" x1="375" y1="375" x2="406" y2="375"/>
  <line stroke="blue" x1="375" y1="375" x2="406" y2="406"/>
  <line stroke="blue" x1="531" y1="0" x2="531" y2="31"/>
  <line stroke="red" x1="531" y1="0" x2="547" y2="31"/>
  <line stroke="black" x1="531" y1="0" x2="750" y2="0"/>
  <line stroke="red" x1="344" y1="406" x2="406" y2="406"/>
  <line stroke="blue" x1="344" y1="406" x2="375" y2="438"/>
  <line stroke="red" x1="406" y1="344" x2="406" y2="375"/>
  <line stroke="red" x1="406" y1="344" x2="438" y2="344"/>
  <line stroke="red" x1="531" y1="31" x2="531" y2="62"/>
  <line stroke="red" x1="531" y1="31" x2="547" y2="31"/>
  <line stroke="blue" x1="188" y1="500" x2="188" y2="562"/>
  <line stroke="red" x1="500" y1="188" x2="531" y2="156"/>
  <line stroke="blue" x1="500" y1="188" x2="531" y2="188"/>
  <line stroke="blue" x1="500" y1="188" x2="531" y2="219"/>
  <line stroke="blue" x1="531" y1="62" x2="531" y2="94"/>
  <line stroke="blue" x1="531" y1="62" x2="547" y2="31"/>
  <line stroke="red" x1="531" y1="62" x2="547" y2="94"/>
  <line stroke="blue" x1="531" y1="62" x2="688" y2="62"/>
  <line stroke="blue" x1="312" y1="438" x2="375" y2="438"/>
  <line stroke="red" x1="312" y1="438" x2="344" y2="469"/>
  <line stroke="blue" x1="312" y1="438" x2="312" y2="500"/>
  <line stroke="blue" x1="438" y1="312" x2="438" y2="344"/>
  <line stroke="red" x1="438" y1="312" x2="469" y2="312"/>
  <line stroke="red" x1="94" y1="531" x2="109" y2="531"/>
  <line stroke="red" x1="94" y1="531" x2="62" y2="562"/>
  <line stroke="red" x1="531" y1="94" x2="531" y2="125"/>
  <line stroke="red" x1="531" y1="94" x2="547" y2="94"/>
  <line stroke="red" x1="109" y1="531" x2="156" y2="531"/>
  <line stroke="blue" x1="109" y1="531" x2="125" y2="562"/>
  <line stroke="blue" x1="531" y1="125" x2="531" y2="156"/>
  <line stroke="blue" x1="531" y1="125" x2="547" y2="94"/>
  <line stroke="red" x1="531" y1="125" x2="547" y2="156"/>
  <line stroke="blue" x1="531" y1="125" x2="625" y2="125"/>
  <line stroke="blue" x1="281" y1="469" x2="250" y2="500"/>
  <line stroke="red" x1="281" y1="469" x2="312" y2="500"/>
  <line stroke="red" x1="281" y1="469" x2="281" y2="562"/>
  <line stroke="red" x1="469" y1="281" x2="469" y2="312"/>
  <line stroke="red" x1="469" y1="281" x2="531" y2="281"/>
  <line stroke="red" x1="547" y1="31" x2="719" y2="31"/>
  <line stroke="red" x1="406" y1="375" x2="406" y2="406"/>
  <line stroke="blue" x1="406" y1="375" x2="438" y2="375"/>
  <line stroke="red" x1="156" y1="531" x2="188" y2="562"/>
  <line stroke="red" x1="531" y1="156" x2="531" y2="188"/>
  <line stroke="red" x1="531" y1="156" x2="547" y2="156"/>
  <line stroke="red" x1="547" y1="94" x2="656" y2="94"/>
  <line stroke="blue" x1="438" y1="344" x2="438" y2="375"/>
  <line stroke="blue" x1="438" y1="344" x2="469" y2="344"/>
  <line stroke="blue" x1="250" y1="500" x2="250" y2="562"/>
  <line stroke="red" x1="469" y1="312" x2="469" y2="344"/>
  <line stroke="blue" x1="469" y1="312" x2="531" y2="312"/>
  <line stroke="blue" x1="531" y1="188" x2="547" y2="156"/>
  <line stroke="blue" x1="531" y1="188" x2="531" y2="219"/>
  <line stroke="red" x1="531" y1="188" x2="542" y2="208"/>
  <line stroke="blue" x1="531" y1="188" x2="562" y2="188"/>
  <line stroke="blue" x1="62" y1="562" x2="125" y2="562"/>
  <line stroke="blue" x1="62" y1="562" x2="31" y2="594"/>
  <line stroke="red" x1="547" y1="156" x2="594" y2="156"/>
  <line stroke="blue" x1="406" y1="406" x2="375" y2="438"/>
  <line stroke="red" x1="406" y1="406" x2="406" y2="438"/>
  <line stroke="red" x1="406" y1="406" x2="438" y2="406"/>
  <line stroke="red" x1="531" y1="219" x2="542" y2="208"/>
  <line stroke="red" x1="531" y1="219" x2="531" y2="281"/>
  <line stroke="red" x1="531" y1="219" x2="562" y2="219"/>
  <line stroke="blue" x1="125" y1="562" x2="188" y2="562"/>
  <line stroke="red" x1="125" y1="562" x2="109" y2="594"/>
  <line stroke="red" x1="375" y1="438" x2="344" y2="469"/>
  <line stroke="red" x1="375" y1="438" x2="406" y2="438"/>
  <line stroke="blue" x1="375" y1="438" x2="406" y2="469"/>
  <line stroke="blue" x1="438" y1="375" x2="438" y2="406"/>
  <line stroke="red" x1="438" y1="375" x2="469" y2="375"/>
  <line stroke="red" x1="542" y1="208" x2="562" y2="188"/>
  <line stroke="blue" x1="542" y1="208" x2="562" y2="219"/>
  <line stroke="blue" x1="344" y1="469" x2="312" y2="500"/>
  <line stroke="red" x1="344" y1="469" x2="406" y2="469"/>
  <line stroke="blue" x1="344" y1="469" x2="375" y2="500"/>
  <line stroke="red" x1="344" y1="469" x2="344" y2="562"/>
  <line stroke="red" x1="469" y1="344" x2="469" y2="375"/>
  <line stroke="red" x1="469" y1="344" x2="531" y2="344"/>
  <line stroke="blue" x1="312" y1="500" x2="312" y2="562"/>
  <line stroke="blue" x1="188" y1="562" x2="219" y2="562"/>
  <line stroke="red" x1="188" y1="562" x2="156" y2="594"/>
  <line stroke="blue" x1="188" y1="562" x2="188" y2="625"/>
  <line stroke="blue" x1="562" y1="188" x2="562" y2="219"/>
  <line stroke="blue" x1="562" y1="188" x2="594" y2="156"/>
  <line stroke="red" x1="31" y1="594" x2="109" y2="594"/>
  <line stroke="red" x1="31" y1="594" x2="0" y2="625"/>
  <line stroke="blue" x1="406" y1="438" x2="438" y2="438"/>
  <line stroke="red" x1="406" y1="438" x2="406" y2="469"/>
  <line stroke="blue" x1="438" y1="406" x2="438" y2="438"/>
  <line stroke="blue" x1="438" y1="406" x2="469" y2="406"/>
  <line stroke="red" x1="469" y1="375" x2="469" y2="406"/>
  <line stroke="blue" x1="469" y1="375" x2="531" y2="375"/>
  <line stroke="red" x1="531" y1="281" x2="531" y2="312"/>
  <line stroke="blue" x1="531" y1="281" x2="562" y2="281"/>
  <line stroke="red" x1="219" y1="562" x2="250" y2="562"/>
  <line stroke="red" x1="219" y1="562" x2="219" y2="625"/>
  <line stroke="red" x1="562" y1="219" x2="594" y2="203"/>
  <line stroke="blue" x1="562" y1="219" x2="562" y2="281"/>
  <line stroke="blue" x1="562" y1="219" x2="594" y2="219"/>
  <line stroke="red" x1="109" y1="594" x2="156" y2="594"/>
  <line stroke="blue" x1="109" y1="594" x2="125" y2="625"/>
  <line stroke="blue" x1="156" y1="594" x2="125" y2="625"/>
  <line stroke="red" x1="156" y1="594" x2="156" y2="625"/>
  <line stroke="red" x1="594" y1="156" x2="594" y2="203"/>
  <line stroke="red" x1="594" y1="156" x2="625" y2="125"/>
  <line stroke="blue" x1="250" y1="562" x2="281" y2="562"/>
  <line stroke="blue" x1="250" y1="562" x2="250" y2="625"/>
  <line stroke="red" x1="531" y1="312" x2="531" y2="344"/>
  <line stroke="red" x1="531" y1="312" x2="562" y2="312"/>
  <line stroke="blue" x1="438" y1="438" x2="438" y2="469"/>
  <line stroke="red" x1="438" y1="438" x2="469" y2="438"/>
  <line stroke="blue" x1="406" y1="469" x2="375" y2="500"/>
  <line stroke="red" x1="406" y1="469" x2="438" y2="469"/>
  <line stroke="red" x1="406" y1="469" x2="406" y2="500"/>
  <line stroke="red" x1="469" y1="406" x2="469" y2="438"/>
  <line stroke="red" x1="469" y1="406" x2="531" y2="406"/>
  <line stroke="blue" x1="0" y1="625" x2="125" y2="625"/>
  <line stroke="red" x1="0" y1="625" x2="31" y2="656"/>
  <line stroke="black" x1="0" y1="625" x2="0" y2="750"/>
  <line stroke="red" x1="375" y1="500" x2="406" y2="500"/>
  <line stroke="blue" x1="375" y1="500" x2="375" y2="562"/>
  <line stroke="red" x1="594" y1="203" x2="594" y2="219"/>
  <line stroke="blue" x1="594" y1="203" x2="625" y2="219"/>
  <line stroke="red" x1="281" y1="562" x2="312" y2="562"/>
  <line stroke="red" x1="281" y1="562" x2="281" y2="625"/>
  <line stroke="blue" x1="562" y1="281" x2="562" y2="312"/>
  <line stroke="red" x1="562" y1="281" x2="594" y2="281"/>
  <line stroke="red" x1="531" y1="344" x2="531" y2="375"/>
  <line stroke="blue" x1="531" y1="344" x2="562" y2="344"/>
  <line stroke="red" x1="594" y1="219" x2="594" y2="281"/>
  <line stroke="red" x1="594" y1="219" x2="625" y2="219"/>
  <line stroke="red" x1="125" y1="625" x2="156" y2="625"/>
  <line stroke="red" x1="125" y1="625" x2="109" y2="656"/>
  <line stroke="blue" x1="125" y1="625" x2="156" y2="656"/>
  <line stroke="blue" x1="625" y1="125" x2="625" y2="219"/>
  <line stroke="blue" x1="625" y1="125" x2="656" y2="94"/>
  <line stroke="blue" x1="438" y1="469" x2="469" y2="469"/>
  <line stroke="blue" x1="438" y1="469" x2="438" y2="500"/>
  <line stroke="red" x1="469" y1="438" x2="469" y2="469"/>
  <line stroke="blue" x1="469" y1="438" x2="531" y2="438"/>
  <line stroke="blue" x1="312" y1="562" x2="344" y2="562"/>
  <line stroke="blue" x1="312" y1="562" x2="312" y2="625"/>
  <line stroke="blue" x1="562" y1="312" x2="562" y2="344"/>
  <line stroke="blue" x1="562" y1="312" x2="594" y2="312"/>
  <line stroke="blue" x1="156" y1="625" x2="188" y2="625"/>
  <line stroke="red" x1="156" y1="625" x2="156" y2="656"/>
  <line stroke="blue" x1="406" y1="500" x2="438" y2="500"/>
  <line stroke="red" x1="406" y1="500" x2="406" y2="562"/>
  <line stroke="red" x1="531" y1="375" x2="531" y2="406"/>
  <line stroke="red" x1="531" y1="375" x2="562" y2="375"/>
  <line stroke="red" x1="188" y1="625" x2="219" y2="625"/>
  <line stroke="blue" x1="188" y1="625" x2="188" y2="688"/>
  <line stroke="red" x1="31" y1="656" x2="109" y2="656"/>
  <line stroke="blue" x1="31" y1="656" x2="62" y2="688"/>
  <line stroke="red" x1="31" y1="656" x2="31" y2="734"/>
  <line stroke="red" x1="594" y1="281" x2="594" y2="312"/>
  <line stroke="blue" x1="594" y1="281" x2="625" y2="250"/>
  <line stroke="blue" x1="594" y1="281" x2="625" y2="312"/>
  <line stroke="red" x1="594" y1="281" x2="656" y2="281"/>
  <line stroke="red" x1="344" y1="562" x2="375" y2="562"/>
  <line stroke="red" x1="344" y1="562" x2="344" y2="625"/>
  <line stroke="blue" x1="562" y1="344" x2="562" y2="375"/>
  <line stroke="red" x1="562" y1="344" x2="594" y2="344"/>
  <line stroke="blue" x1="219" y1="625" x2="250" y2="625"/>
  <line stroke="red" x1="219" y1="625" x2="219" y2="719"/>
  <line stroke="blue" x1="625" y1="219" x2="625" y2="250"/>
  <line stroke="red" x1="625" y1="219" x2="656" y2="203"/>
  <line stroke="blue" x1="625" y1="219" x2="656" y2="219"/>
  <line stroke="red" x1="469" y1="469" x2="469" y2="500"/>
  <line stroke="red" x1="469" y1="469" x2="531" y2="469"/>
  <line stroke="red" x1="656" y1="94" x2="656" y2="203"/>
  <line stroke="red" x1="656" y1="94" x2="688" y2="62"/>
  <line stroke="red" x1="438" y1="500" x2="469" y2="500"/>
  <line stroke="blue" x1="438" y1="500" x2="438" y2="562"/>
  <line stroke="red" x1="109" y1="656" x2="156" y2="656"/>
  <line stroke="blue" x1="109" y1="656" x2="125" y2="688"/>
  <line stroke="red" x1="531" y1="406" x2="531" y2="438"/>
  <line stroke="blue" x1="531" y1="406" x2="562" y2="406"/>
  <line stroke="red" x1="594" y1="312" x2="594" y2="344"/>
  <line stroke="red" x1="594" y1="312" x2="625" y2="312"/>
  <line stroke="red" x1="250" y1="625" x2="281" y2="625"/>
  <line stroke="blue" x1="250" y1="625" x2="250" y2="750"/>
  <line stroke="red" x1="625" y1="250" x2="656" y2="250"/>
  <line stroke="blue" x1="625" y1="250" x2="656" y2="281"/>
  <line stroke="red" x1="156" y1="656" x2="188" y2="688"/>
  <line stroke="blue" x1="375" y1="562" x2="406" y2="562"/>
  <line stroke="blue" x1="375" y1="562" x2="375" y2="625"/>
  <line stroke="blue" x1="562" y1="375" x2="562" y2="406"/>
  <line stroke="blue" x1="562" y1="375" x2="594" y2="375"/>
  <line stroke="blue" x1="281" y1="625" x2="312" y2="625"/>
  <line stroke="red" x1="281" y1="625" x2="281" y2="781"/>
  <line stroke="blue" x1="469" y1="500" x2="531" y2="500"/>
  <line stroke="red" x1="469" y1="500" x2="469" y2="562"/>
  <line stroke="blue" x1="594" y1="344" x2="625" y2="312"/>
  <line stroke="red" x1="594" y1="344" x2="594" y2="375"/>
  <line stroke="blue" x1="594" y1="344" x2="625" y2="375"/>
  <line stroke="red" x1="594" y1="344" x2="656" y2="344"/>
  <line stroke="red" x1="656" y1="203" x2="656" y2="219"/>
  <line stroke="blue" x1="656" y1="203" x2="688" y2="219"/>
  <line stroke="red" x1="531" y1="438" x2="531" y2="469"/>
  <line stroke="red" x1="531" y1="438" x2="562" y2="438"/>
  <line stroke="blue" x1="62" y1="688" x2="125" y2="688"/>
  <line stroke="red" x1="62" y1="688" x2="94" y2="719"/>
  <line stroke="blue" x1="62" y1="688" x2="62" y2="750"/>
  <line stroke="blue" x1="688" y1="62" x2="719" y2="31"/>
  <line stroke="blue" x1="688" y1="62" x2="688" y2="219"/>
  <line stroke="red" x1="656" y1="219" x2="656" y2="250"/>
  <line stroke="red" x1="656" y1="219" x2="688" y2="219"/>
  <line stroke="red" x1="406" y1="562" x2="438" y2="562"/>
  <line stroke="red" x1="406" y1="562" x2="406" y2="625"/>
  <line stroke="blue" x1="562" y1="406" x2="562" y2="438"/>
  <line stroke="red" x1="562" y1="406" x2="594" y2="406"/>
  <line stroke="blue" x1="125" y1="688" x2="188" y2="688"/>
  <line stroke="red" x1="125" y1="688" x2="109" y2="719"/>
  <line stroke="red" x1="312" y1="625" x2="344" y2="625"/>
  <line stroke="blue" x1="312" y1="625" x2="312" y2="812"/>
  <line stroke="blue" x1="625" y1="312" x2="656" y2="281"/>
  <line stroke="red" x1="625" y1="312" x2="656" y2="344"/>
  <line stroke="blue" x1="625" y1="312" x2="688" y2="312"/>
  <line stroke="red" x1="594" y1="375" x2="594" y2="406"/>
  <line stroke="red" x1="594" y1="375" x2="625" y2="375"/>
  <line stroke="red" x1="656" y1="250" x2="656" y2="281"/>
  <line stroke="blue" x1="656" y1="250" x2="688" y2="250"/>
  <line stroke="red" x1="531" y1="469" x2="531" y2="500"/>
  <line stroke="blue" x1="531" y1="469" x2="562" y2="469"/>
  <line stroke="blue" x1="188" y1="688" x2="219" y2="719"/>
  <line stroke="blue" x1="438" y1="562" x2="469" y2="562"/>
  <line stroke="blue" x1="438" y1="562" x2="438" y2="625"/>
  <line stroke="blue" x1="562" y1="438" x2="562" y2="469"/>
  <line stroke="blue" x1="562" y1="438" x2="594" y2="438"/>
  <line stroke="blue" x1="344" y1="625" x2="375" y2="625"/>
  <line stroke="red" x1="344" y1="625" x2="344" y2="844"/>
  <line stroke="red" x1="656" y1="281" x2="688" y2="250"/>
  <line stroke="red" x1="656" y1="281" x2="688" y2="312"/>
  <line stroke="blue" x1="594" y1="406" x2="625" y2="375"/>
  <line stroke="red" x1="594" y1="406" x2="594" y2="438"/>
  <line stroke="blue" x1="594" y1="406" x2="625" y2="438"/>
  <line stroke="red" x1="594" y1="406" x2="656" y2="406"/>
  <line stroke="red" x1="719" y1="31" x2="719" y2="203"/>
  <line stroke="red" x1="719" y1="31" x2="750" y2="0"/>
  <line stroke="blue" x1="688" y1="219" x2="688" y2="250"/>
  <line stroke="red" x1="688" y1="219" x2="719" y2="203"/>
  <line stroke="blue" x1="688" y1="219" x2="719" y2="219"/>
  <line stroke="red" x1="94" y1="719" x2="109" y2="719"/>
  <line stroke="red" x1="94" y1="719" x2="94" y2="734"/>
  <line stroke="blue" x1="94" y1="719" x2="125" y2="750"/>
  <line stroke="red" x1="109" y1="719" x2="219" y2="719"/>
  <line stroke="blue" x1="109" y1="719" x2="125" y2="750"/>
  <line stroke="red" x1="375" y1="625" x2="406" y2="625"/>
  <line stroke="blue" x1="375" y1="625" x2="375" y2="875"/>
  <line stroke="blue" x1="625" y1="375" x2="656" y2="344"/>
  <line stroke="red" x1="625" y1="375" x2="656" y2="406"/>
  <line stroke="blue" x1="625" y1="375" x2="750" y2="375"/>
  <line stroke="red" x1="531" y1="500" x2="562" y2="500"/>
  <line stroke="red" x1="531" y1="500" x2="531" y2="562"/>
  <line stroke="blue" x1="688" y1="250" x2="688" y2="312"/>
  <line stroke="red" x1="688" y1="250" x2="719" y2="250"/>
  <line stroke="blue" x1="688" y1="250" x2="719" y2="281"/>
  <line stroke="red" x1="469" y1="562" x2="531" y2="562"/>
  <line stroke="red" x1="469" y1="562" x2="469" y2="625"/>
  <line stroke="blue" x1="562" y1="469" x2="562" y2="500"/>
  <line stroke="red" x1="562" y1="469" x2="594" y2="469"/>
  <line stroke="blue" x1="31" y1="734" x2="0" y2="750"/>
  <line stroke="red" x1="31" y1="734" x2="31" y2="750"/>
  <line stroke="red" x1="31" y1="734" x2="62" y2="750"/>
  <line stroke="red" x1="594" y1="438" x2="594" y2="469"/>
  <line stroke="red" x1="594" y1="438" x2="625" y2="438"/>
  <line stroke="blue" x1="94" y1="734" x2="62" y2="750"/>
  <line stroke="red" x1="94" y1="734" x2="94" y2="750"/>
  <line stroke="red" x1="94" y1="734" x2="125" y2="750"/>
  <line stroke="red" x1="656" y1="344" x2="719" y2="344"/>
  <line stroke="blue" x1="406" y1="625" x2="438" y2="625"/>
  <line stroke="red" x1="406" y1="625" x2="406" y2="906"/>
  <line stroke="red" x1="719" y1="203" x2="719" y2="219"/>
  <line stroke="blue" x1="719" y1="203" x2="750" y2="219"/>
  <line stroke="blue" x1="0" y1="750" x2="31" y2="750"/>
  <line stroke="red" x1="0" y1="750" x2="31" y2="758"/>
  <line stroke="black" x1="0" y1="750" x2="0" y2="875"/>
  <line stroke="blue" x1="750" y1="0" x2="750" y2="219"/>
  <line stroke="red" x1="750" y1="0" x2="781" y2="31"/>
  <line stroke="black" x1="750" y1="0" x2="875" y2="0"/>
  <line stroke="red" x1="31" y1="750" x2="62" y2="750"/>
  <line stroke="red" x1="31" y1="750" x2="31" y2="758"/>
  <line stroke="red" x1="219" y1="719" x2="250" y2="750"/>
  <line stroke="red" x1="719" y1="219" x2="719" y2="250"/>
  <line stroke="red" x1="719" y1="219" x2="750" y2="219"/>
  <line stroke="blue" x1="62" y1="750" x2="94" y2="750"/>
  <line stroke="blue" x1="62" y1="750" x2="31" y2="758"/>
  <line stroke="red" x1="62" y1="750" x2="94" y2="758"/>
  <line stroke="blue" x1="62" y1="750" x2="62" y2="812"/>
  <line stroke="blue" x1="562" y1="500" x2="594" y2="500"/>
  <line stroke="blue" x1="562" y1="500" x2="562" y2="562"/>
  <line stroke="blue" x1="688" y1="312" x2="719" y2="344"/>
  <line stroke="red" x1="94" y1="750" x2="125" y2="750"/>
  <line stroke="red" x1="94" y1="750" x2="94" y2="758"/>
  <line stroke="blue" x1="594" y1="469" x2="625" y2="438"/>
  <line stroke="red" x1="594" y1="469" x2="594" y2="500"/>
  <line stroke="blue" x1="594" y1="469" x2="625" y2="500"/>
  <line stroke="red" x1="594" y1="469" x2="656" y2="469"/>
  <line stroke="red" x1="31" y1="758" x2="31" y2="844"/>
  <line stroke="blue" x1="125" y1="750" x2="94" y2="758"/>
  <line stroke="blue" x1="125" y1="750" x2="94" y2="781"/>
  <line stroke="red" x1="125" y1="750" x2="117" y2="781"/>
  <line stroke="blue" x1="125" y1="750" x2="250" y2="750"/>
  <line stroke="red" x1="719" y1="250" x2="719" y2="281"/>
  <line stroke="blue" x1="719" y1="250" x2="750" y2="250"/>
  <line stroke="red" x1="438" y1="625" x2="469" y2="625"/>
  <line stroke="blue" x1="438" y1="625" x2="438" y2="938"/>
  <line stroke="blue" x1="625" y1="438" x2="656" y2="406"/>
  <line stroke="red" x1="625" y1="438" x2="656" y2="469"/>
  <line stroke="blue" x1="625" y1="438" x2="688" y2="438"/>
  <line stroke="red" x1="94" y1="758" x2="94" y2="781"/>
  <line stroke="red" x1="656" y1="406" x2="719" y2="406"/>
  <line stroke="red" x1="719" y1="281" x2="750" y2="250"/>
  <line stroke="red" x1="719" y1="281" x2="719" y2="344"/>
  <line stroke="blue" x1="531" y1="562" x2="562" y2="562"/>
  <line stroke="red" x1="531" y1="562" x2="531" y2="625"/>
  <line stroke="red" x1="594" y1="500" x2="625" y2="500"/>
  <line stroke="red" x1="594" y1="500" x2="594" y2="562"/>
  <line stroke="blue" x1="469" y1="625" x2="531" y2="625"/>
  <line stroke="red" x1="469" y1="625" x2="469" y2="969"/>
  <line stroke="blue" x1="750" y1="219" x2="750" y2="250"/>
  <line stroke="red" x1="750" y1="219" x2="781" y2="203"/>
  <line stroke="blue" x1="750" y1="219" x2="781" y2="219"/>
  <line stroke="red" x1="781" y1="31" x2="781" y2="203"/>
  <line stroke="blue" x1="781" y1="31" x2="812" y2="62"/>
  <line stroke="red" x1="781" y1="31" x2="875" y2="31"/>
  <line stroke="red" x1="94" y1="781" x2="117" y2="781"/>
  <line stroke="red" x1="94" y1="781" x2="62" y2="812"/>
  <line stroke="blue" x1="117" y1="781" x2="125" y2="812"/>
  <line stroke="red" x1="117" y1="781" x2="281" y2="781"/>
  <line stroke="blue" x1="250" y1="750" x2="281" y2="781"/>
  <line stroke="red" x1="750" y1="250" x2="781" y2="250"/>
  <line stroke="blue" x1="750" y1="250" x2="781" y2="281"/>
  <line stroke="blue" x1="750" y1="250" x2="750" y2="375"/>
  <line stroke="red" x1="562" y1="562" x2="594" y2="562"/>
  <line stroke="blue" x1="562" y1="562" x2="562" y2="625"/>
  <line stroke="red" x1="719" y1="344" x2="750" y2="375"/>
  <line stroke="blue" x1="625" y1="500" x2="656" y2="469"/>
  <line stroke="blue" x1="625" y1="500" x2="625" y2="562"/>
  <line stroke="red" x1="656" y1="469" x2="688" y2="438"/>
  <line stroke="blue" x1="656" y1="469" x2="688" y2="500"/>
  <line stroke="red" x1="656" y1="469" x2="656" y2="562"/>
  <line stroke="red" x1="781" y1="203" x2="781" y2="219"/>
  <line stroke="blue" x1="781" y1="203" x2="812" y2="219"/>
  <line stroke="red" x1="781" y1="219" x2="781" y2="250"/>
  <line stroke="red" x1="781" y1="219" x2="812" y2="219"/>
  <line stroke="blue" x1="62" y1="812" x2="125" y2="812"/>
  <line stroke="blue" x1="62" y1="812" x2="31" y2="844"/>
  <line stroke="blue" x1="688" y1="438" x2="719" y2="406"/>
  <line stroke="blue" x1="688" y1="438" x2="688" y2="500"/>
  <line stroke="blue" x1="812" y1="62" x2="812" y2="219"/>
  <line stroke="red" x1="812" y1="62" x2="844" y2="94"/>
  <line stroke="blue" x1="812" y1="62" x2="875" y2="62"/>
  <line stroke="blue" x1="594" y1="562" x2="625" y2="562"/>
  <line stroke="red" x1="594" y1="562" x2="594" y2="625"/>
  <line stroke="red" x1="531" y1="625" x2="562" y2="625"/>
  <line stroke="red" x1="531" y1="625" x2="531" y2="969"/>
  <line stroke="red" x1="781" y1="250" x2="781" y2="281"/>
  <line stroke="blue" x1="781" y1="250" x2="812" y2="250"/>
  <line stroke="red" x1="125" y1="812" x2="117" y2="844"/>
  <line stroke="blue" x1="125" y1="812" x2="312" y2="812"/>
  <line stroke="red" x1="719" y1="406" x2="750" y2="375"/>
  <line stroke="red" x1="719" y1="406" x2="719" y2="469"/>
  <line stroke="red" x1="281" y1="781" x2="312" y2="812"/>
  <line stroke="red" x1="781" y1="281" x2="812" y2="250"/>
  <line stroke="red" x1="781" y1="281" x2="781" y2="344"/>
  <line stroke="red" x1="750" y1="375" x2="781" y2="344"/>
  <line stroke="red" x1="750" y1="375" x2="781" y2="406"/>
  <line stroke="blue" x1="750" y1="375" x2="750" y2="500"/>
  <line stroke="red" x1="750" y1="375" x2="875" y2="375"/>
  <line stroke="blue" x1="562" y1="625" x2="594" y2="625"/>
  <line stroke="blue" x1="562" y1="625" x2="562" y2="938"/>
  <line stroke="red" x1="625" y1="562" x2="656" y2="562"/>
  <line stroke="blue" x1="625" y1="562" x2="625" y2="625"/>
  <line stroke="blue" x1="812" y1="219" x2="812" y2="250"/>
  <line stroke="red" x1="812" y1="219" x2="844" y2="203"/>
  <line stroke="blue" x1="812" y1="219" x2="844" y2="219"/>
  <line stroke="red" x1="31" y1="844" x2="117" y2="844"/>
  <line stroke="red" x1="31" y1="844" x2="0" y2="875"/>
  <line stroke="red" x1="844" y1="94" x2="844" y2="203"/>
  <line stroke="red" x1="844" y1="94" x2="875" y2="94"/>
  <line stroke="blue" x1="844" y1="94" x2="875" y2="125"/>
  <line stroke="red" x1="688" y1="500" x2="719" y2="469"/>
  <line stroke="blue" x1="688" y1="500" x2="688" y2="562"/>
  <line stroke="blue" x1="812" y1="250" x2="812" y2="312"/>
  <line stroke="red" x1="812" y1="250" x2="844" y2="250"/>
  <line stroke="blue" x1="812" y1="250" x2="844" y2="281"/>
  <line stroke="blue" x1="117" y1="844" x2="125" y2="875"/>
  <line stroke="red" x1="117" y1="844" x2="344" y2="844"/>
  <line stroke="blue" x1="781" y1="344" x2="812" y2="312"/>
  <line stroke="red" x1="781" y1="344" x2="844" y2="344"/>
  <line stroke="blue" x1="719" y1="469" x2="750" y2="500"/>
  <line stroke="red" x1="719" y1="469" x2="719" y2="562"/>
  <line stroke="red" x1="594" y1="625" x2="625" y2="625"/>
  <line stroke="red" x1="594" y1="625" x2="594" y2="906"/>
  <line stroke="blue" x1="656" y1="562" x2="688" y2="562"/>
  <line stroke="red" x1="656" y1="562" x2="656" y2="625"/>
  <line stroke="red" x1="844" y1="203" x2="844" y2="219"/>
  <line stroke="blue" x1="844" y1="203" x2="875" y2="219"/>
  <line stroke="blue" x1="312" y1="812" x2="344" y2="844"/>
  <line stroke="red" x1="812" y1="312" x2="844" y2="281"/>
  <line stroke="blue" x1="812" y1="312" x2="875" y2="312"/>
  <line stroke="red" x1="844" y1="219" x2="844" y2="250"/>
  <line stroke="red" x1="844" y1="219" x2="875" y2="219"/>
  <line stroke="blue" x1="0" y1="875" x2="125" y2="875"/>
  <line stroke="red" x1="0" y1="875" x2="31" y2="906"/>
  <line stroke="black" x1="0" y1="875" x2="0" y2="1000"/>
  <line stroke="blue" x1="875" y1="0" x2="875" y2="31"/>
  <line stroke="black" x1="875" y1="0" x2="938" y2="0"/>
  <line stroke="red" x1="875" y1="31" x2="875" y2="62"/>
  <line stroke="red" x1="875" y1="31" x2="938" y2="0"/>
  <line stroke="blue" x1="875" y1="31" x2="938" y2="62"/>
  <line stroke="red" x1="875" y1="31" x2="969" y2="31"/>
  <line stroke="blue" x1="875" y1="62" x2="875" y2="94"/>
  <line stroke="blue" x1="875" y1="62" x2="938" y2="62"/>
  <line stroke="red" x1="844" y1="250" x2="844" y2="281"/>
  <line stroke="blue" x1="844" y1="250" x2="875" y2="250"/>
  <line stroke="red" x1="875" y1="94" x2="875" y2="125"/>
  <line stroke="blue" x1="875" y1="94" x2="891" y2="109"/>
  <line stroke="red" x1="875" y1="94" x2="906" y2="94"/>
  <line stroke="red" x1="875" y1="94" x2="938" y2="62"/>
  <line stroke="red" x1="781" y1="406" x2="781" y2="469"/>
  <line stroke="blue" x1="781" y1="406" x2="812" y2="438"/>
  <line stroke="red" x1="781" y1="406" x2="844" y2="406"/>
  <line stroke="red" x1="125" y1="875" x2="117" y2="906"/>
  <line stroke="blue" x1="125" y1="875" x2="375" y2="875"/>
  <line stroke="blue" x1="625" y1="625" x2="656" y2="625"/>
  <line stroke="blue" x1="625" y1="625" x2="625" y2="875"/>
  <line stroke="blue" x1="875" y1="125" x2="891" y2="109"/>
  <line stroke="blue" x1="875" y1="125" x2="875" y2="219"/>
  <line stroke="red" x1="688" y1="562" x2="719" y2="562"/>
  <line stroke="blue" x1="688" y1="562" x2="688" y2="625"/>
  <line stroke="blue" x1="844" y1="281" x2="875" y2="250"/>
  <line stroke="red" x1="844" y1="281" x2="875" y2="281"/>
  <line stroke="red" x1="844" y1="281" x2="875" y2="312"/>
  <line stroke="blue" x1="891" y1="109" x2="906" y2="94"/>
  <line stroke="red" x1="891" y1="109" x2="906" y2="125"/>
  <line stroke="red" x1="750" y1="500" x2="781" y2="469"/>
  <line stroke="blue" x1="750" y1="500" x2="750" y2="562"/>
  <line stroke="blue" x1="875" y1="219" x2="875" y2="250"/>
  <line stroke="red" x1="875" y1="219" x2="906" y2="203"/>
  <line stroke="blue" x1="875" y1="219" x2="906" y2="219"/>
  <line stroke="red" x1="656" y1="625" x2="688" y2="625"/>
  <line stroke="red" x1="656" y1="625" x2="656" y2="844"/>
  <line stroke="red" x1="31" y1="906" x2="117" y2="906"/>
  <line stroke="blue" x1="31" y1="906" x2="62" y2="938"/>
  <line stroke="red" x1="31" y1="906" x2="31" y2="1000"/>
  <line stroke="red" x1="875" y1="250" x2="875" y2="281"/>
  <line stroke="red" x1="875" y1="250" x2="906" y2="250"/>
  <line stroke="blue" x1="875" y1="250" x2="906" y2="281"/>
  <line stroke="red" x1="344" y1="844" x2="375" y2="875"/>
  <line stroke="blue" x1="781" y1="469" x2="812" y2="500"/>
  <line stroke="red" x1="781" y1="469" x2="781" y2="562"/>
  <line stroke="blue" x1="844" y1="344" x2="875" y2="312"/>
  <line stroke="red" x1="844" y1="344" x2="875" y2="344"/>
  <line stroke="red" x1="844" y1="344" x2="875" y2="375"/>
  <line stroke="red" x1="906" y1="94" x2="906" y2="125"/>
  <line stroke="red" x1="906" y1="94" x2="938" y2="62"/>
  <line stroke="blue" x1="719" y1="562" x2="750" y2="562"/>
  <line stroke="red" x1="719" y1="562" x2="719" y2="625"/>
  <line stroke="blue" x1="117" y1="906" x2="125" y2="938"/>
  <line stroke="red" x1="117" y1="906" x2="406" y2="906"/>
  <line stroke="red" x1="906" y1="125" x2="906" y2="203"/>
  <line stroke="blue" x1="906" y1="125" x2="938" y2="62"/>
  <line stroke="blue" x1="875" y1="281" x2="875" y2="312"/>
  <line stroke="red" x1="875" y1="281" x2="906" y2="281"/>
  <line stroke="blue" x1="812" y1="438" x2="812" y2="500"/>
  <line stroke="red" x1="812" y1="438" x2="844" y2="469"/>
  <line stroke="blue" x1="812" y1="438" x2="875" y2="438"/>
  <line stroke="red" x1="906" y1="203" x2="906" y2="219"/>
  <line stroke="blue" x1="906" y1="203" x2="938" y2="219"/>
  <line stroke="blue" x1="688" y1="625" x2="719" y2="625"/>
  <line stroke="blue" x1="688" y1="625" x2="688" y2="812"/>
  <line stroke="red" x1="875" y1="312" x2="875" y2="344"/>
  <line stroke="blue" x1="875" y1="312" x2="912" y2="287"/>
  <line stroke="red" x1="875" y1="312" x2="922" y2="344"/>
  <line stroke="blue" x1="875" y1="312" x2="938" y2="312"/>
  <line stroke="red" x1="906" y1="219" x2="906" y2="250"/>
  <line stroke="red" x1="906" y1="219" x2="938" y2="219"/>
  <line stroke="blue" x1="844" y1="406" x2="875" y2="375"/>
  <line stroke="red" x1="844" y1="406" x2="875" y2="406"/>
  <line stroke="red" x1="844" y1="406" x2="875" y2="438"/>
  <line stroke="red" x1="750" y1="562" x2="781" y2="562"/>
  <line stroke="blue" x1="750" y1="562" x2="750" y2="625"/>
  <line stroke="black" x1="938" y1="0" x2="1000" y2="0"/>
  <line stroke="blue" x1="62" y1="938" x2="125" y2="938"/>
  <line stroke="red" x1="62" y1="938" x2="94" y2="969"/>
  <line stroke="blue" x1="62" y1="938" x2="62" y2="1000"/>
  <line stroke="blue" x1="938" y1="62" x2="938" y2="219"/>
  <line stroke="blue" x1="938" y1="62" x2="969" y2="31"/>
  <line stroke="red" x1="938" y1="62" x2="969" y2="125"/>
  <line stroke="blue" x1="875" y1="344" x2="875" y2="375"/>
  <line stroke="red" x1="875" y1="344" x2="922" y2="344"/>
  <line stroke="red" x1="906" y1="250" x2="906" y2="281"/>
  <line stroke="blue" x1="906" y1="250" x2="938" y2="250"/>
  <line stroke="red" x1="125" y1="938" x2="117" y2="969"/>
  <line stroke="blue" x1="125" y1="938" x2="438" y2="938"/>
  <line stroke="red" x1="906" y1="281" x2="912" y2="287"/>
  <line stroke="blue" x1="375" y1="875" x2="406" y2="906"/>
  <line stroke="blue" x1="875" y1="375" x2="875" y2="406"/>
  <line stroke="blue" x1="875" y1="375" x2="922" y2="344"/>
  <line stroke="red" x1="875" y1="375" x2="922" y2="406"/>
  <line stroke="blue" x1="875" y1="375" x2="1000" y2="375"/>
  <line stroke="red" x1="719" y1="625" x2="750" y2="625"/>
  <line stroke="red" x1="719" y1="625" x2="719" y2="781"/>
  <line stroke="red" x1="812" y1="500" x2="844" y2="469"/>
  <line stroke="blue" x1="812" y1="500" x2="812" y2="562"/>
  <line stroke="red" x1="912" y1="287" x2="938" y2="250"/>
  <line stroke="red" x1="912" y1="287" x2="938" y2="312"/>
  <line stroke="blue" x1="781" y1="562" x2="812" y2="562"/>
  <line stroke="red" x1="781" y1="562" x2="781" y2="625"/>
  <line stroke="blue" x1="938" y1="219" x2="938" y2="250"/>
  <line stroke="red" x1="938" y1="219" x2="969" y2="203"/>
  <line stroke="blue" x1="938" y1="219" x2="969" y2="219"/>
  <line stroke="red" x1="875" y1="406" x2="875" y2="438"/>
  <line stroke="red" x1="875" y1="406" x2="922" y2="406"/>
  <line stroke="blue" x1="844" y1="469" x2="875" y2="438"/>
  <line stroke="red" x1="844" y1="469" x2="875" y2="469"/>
  <line stroke="red" x1="844" y1="469" x2="844" y2="531"/>
  <line stroke="blue" x1="844" y1="469" x2="875" y2="500"/>
  <line stroke="red" x1="969" y1="31" x2="969" y2="125"/>
  <line stroke="red" x1="969" y1="31" x2="1000" y2="0"/>
  <line stroke="blue" x1="938" y1="250" x2="938" y2="312"/>
  <line stroke="red" x1="938" y1="250" x2="969" y2="250"/>
  <line stroke="blue" x1="938" y1="250" x2="969" y2="297"/>
  <line stroke="red" x1="94" y1="969" x2="117" y2="969"/>
  <line stroke="red" x1="94" y1="969" x2="94" y2="1000"/>
  <line stroke="blue" x1="94" y1="969" x2="125" y2="1000"/>
  <line stroke="blue" x1="117" y1="969" x2="125" y2="1000"/>
  <line stroke="red" x1="117" y1="969" x2="469" y2="969"/>
  <line stroke="blue" x1="750" y1="625" x2="781" y2="625"/>
  <line stroke="blue" x1="750" y1="625" x2="750" y2="750"/>
  <line stroke="red" x1="969" y1="125" x2="969" y2="203"/>
  <line stroke="blue" x1="969" y1="125" x2="1000" y2="62"/>
  <line stroke="blue" x1="875" y1="438" x2="875" y2="469"/>
  <line stroke="blue" x1="875" y1="438" x2="922" y2="406"/>
  <line stroke="red" x1="875" y1="438" x2="912" y2="463"/>
  <line stroke="blue" x1="875" y1="438" x2="938" y2="438"/>
  <line stroke="red" x1="922" y1="344" x2="969" y2="344"/>
  <line stroke="red" x1="812" y1="562" x2="844" y2="531"/>
  <line stroke="blue" x1="812" y1="562" x2="812" y2="625"/>
  <line stroke="red" x1="812" y1="562" x2="844" y2="594"/>
  <line stroke="blue" x1="812" y1="562" x2="875" y2="562"/>
  <line stroke="blue" x1="938" y1="312" x2="969" y2="344"/>
  <line stroke="red" x1="969" y1="203" x2="969" y2="219"/>
  <line stroke="blue" x1="969" y1="203" x2="1000" y2="219"/>
  <line stroke="red" x1="875" y1="469" x2="875" y2="500"/>
  <line stroke="red" x1="875" y1="469" x2="906" y2="469"/>
  <line stroke="red" x1="406" y1="906" x2="438" y2="938"/>
  <line stroke="red" x1="969" y1="219" x2="969" y2="250"/>
  <line stroke="red" x1="969" y1="219" x2="1000" y2="219"/>
  <line stroke="blue" x1="844" y1="531" x2="875" y2="500"/>
  <line stroke="red" x1="844" y1="531" x2="891" y2="531"/>
  <line stroke="black" x1="0" y1="1000" x2="31" y2="1000"/>
  <line stroke="black" x1="1000" y1="0" x2="1000" y2="62"/>
  <line stroke="black" x1="31" y1="1000" x2="62" y2="1000"/>
  <line stroke="red" x1="781" y1="625" x2="812" y2="625"/>
  <line stroke="red" x1="781" y1="625" x2="781" y2="719"/>
  <line stroke="red" x1="969" y1="250" x2="969" y2="297"/>
  <line stroke="blue" x1="969" y1="250" x2="1000" y2="250"/>
  <line stroke="black" x1="62" y1="1000" x2="94" y2="1000"/>
  <line stroke="black" x1="1000" y1="62" x2="1000" y2="219"/>
  <line stroke="black" x1="94" y1="1000" x2="125" y2="1000"/>
  <line stroke="red" x1="922" y1="406" x2="969" y2="406"/>
  <line stroke="black" x1="125" y1="1000" x2="500" y2="1000"/>
  <line stroke="blue" x1="875" y1="500" x2="906" y2="469"/>
  <line stroke="red" x1="875" y1="500" x2="906" y2="500"/>
  <line stroke="red" x1="875" y1="500" x2="891" y2="531"/>
  <line stroke="blue" x1="875" y1="500" x2="906" y2="516"/>
  <line stroke="blue" x1="875" y1="500" x2="906" y2="531"/>
  <line stroke="red" x1="969" y1="297" x2="969" y2="344"/>
  <line stroke="red" x1="969" y1="297" x2="1000" y2="250"/>
  <line stroke="red" x1="906" y1="469" x2="912" y2="463"/>
  <line stroke="red" x1="906" y1="469" x2="906" y2="500"/>
  <line stroke="red" x1="912" y1="463" x2="938" y2="438"/>
  <line stroke="blue" x1="912" y1="463" x2="938" y2="500"/>
  <line stroke="black" x1="1000" y1="219" x2="1000" y2="250"/>
  <line stroke="blue" x1="812" y1="625" x2="844" y2="625"/>
  <line stroke="blue" x1="812" y1="625" x2="812" y2="688"/>
  <line stroke="red" x1="969" y1="344" x2="1000" y2="375"/>
  <line stroke="black" x1="1000" y1="250" x2="1000" y2="375"/>
  <line stroke="red" x1="844" y1="594" x2="844" y2="625"/>
  <line stroke="red" x1="844" y1="594" x2="891" y2="594"/>
  <line stroke="blue" x1="844" y1="594" x2="875" y2="625"/>
  <line stroke="blue" x1="438" y1="938" x2="469" y2="969"/>
  <line stroke="blue" x1="938" y1="438" x2="969" y2="406"/>
  <line stroke="blue" x1="938" y1="438" x2="938" y2="500"/>
  <line stroke="red" x1="906" y1="500" x2="906" y2="516"/>
  <line stroke="blue" x1="906" y1="500" x2="938" y2="500"/>
  <line stroke="blue" x1="891" y1="531" x2="875" y2="562"/>
  <line stroke="red" x1="891" y1="531" x2="906" y2="531"/>
  <line stroke="red" x1="875" y1="562" x2="891" y2="594"/>
  <line stroke="blue" x1="875" y1="562" x2="938" y2="562"/>
  <line stroke="red" x1="906" y1="516" x2="906" y2="531"/>
  <line stroke="red" x1="906" y1="516" x2="938" y2="500"/>
  <line stroke="red" x1="844" y1="625" x2="844" y2="656"/>
  <line stroke="red" x1="844" y1="625" x2="875" y2="625"/>
  <line stroke="red" x1="906" y1="531" x2="938" y2="562"/>
  <line stroke="red" x1="969" y1="406" x2="1000" y2="375"/>
  <line stroke="red" x1="969" y1="406" x2="969" y2="453"/>
  <line stroke="blue" x1="750" y1="750" x2="719" y2="781"/>
  <line stroke="red" x1="750" y1="750" x2="781" y2="719"/>
  <line stroke="blue" x1="750" y1="750" x2="875" y2="750"/>
  <line stroke="red" x1="719" y1="781" x2="688" y2="812"/>
  <line stroke="red" x1="719" y1="781" x2="883" y2="781"/>
  <line stroke="blue" x1="781" y1="719" x2="812" y2="688"/>
  <line stroke="red" x1="781" y1="719" x2="891" y2="719"/>
  <line stroke="red" x1="938" y1="500" x2="969" y2="453"/>
  <line stroke="red" x1="938" y1="500" x2="969" y2="500"/>
  <line stroke="blue" x1="938" y1="500" x2="938" y2="562"/>
  <line stroke="blue" x1="938" y1="500" x2="969" y2="516"/>
  <line stroke="blue" x1="688" y1="812" x2="656" y2="844"/>
  <line stroke="blue" x1="688" y1="812" x2="875" y2="812"/>
  <line stroke="red" x1="812" y1="688" x2="844" y2="656"/>
  <line stroke="blue" x1="812" y1="688" x2="875" y2="688"/>
  <line stroke="black" x1="1000" y1="375" x2="1000" y2="500"/>
  <line stroke="red" x1="656" y1="844" x2="625" y2="875"/>
  <line stroke="red" x1="656" y1="844" x2="883" y2="844"/>
  <line stroke="blue" x1="844" y1="656" x2="875" y2="625"/>
  <line stroke="red" x1="844" y1="656" x2="891" y2="656"/>
  <line stroke="red" x1="969" y1="453" x2="969" y2="500"/>
  <line stroke="blue" x1="969" y1="453" x2="1000" y2="500"/>
  <line stroke="blue" x1="891" y1="594" x2="875" y2="625"/>
  <line stroke="red" x1="891" y1="594" x2="969" y2="594"/>
  <line stroke="blue" x1="625" y1="875" x2="594" y2="906"/>
  <line stroke="blue" x1="625" y1="875" x2="875" y2="875"/>
  <line stroke="red" x1="875" y1="625" x2="891" y2="656"/>
  <line stroke="blue" x1="875" y1="625" x2="1000" y2="625"/>
  <line stroke="red" x1="469" y1="969" x2="500" y2="1000"/>
  <line stroke="red" x1="594" y1="906" x2="562" y2="938"/>
  <line stroke="red" x1="594" y1="906" x2="883" y2="906"/>
  <line stroke="red" x1="969" y1="500" x2="969" y2="516"/>
  <line stroke="blue" x1="969" y1="500" x2="1000" y2="500"/>
  <line stroke="blue" x1="562" y1="938" x2="531" y2="969"/>
  <line stroke="blue" x1="562" y1="938" x2="875" y2="938"/>
  <line stroke="blue" x1="938" y1="562" x2="969" y2="594"/>
  <line stroke="red" x1="969" y1="516" x2="1000" y2="500"/>
  <line stroke="red" x1="969" y1="516" x2="969" y2="594"/>
  <line stroke="red" x1="531" y1="969" x2="500" y2="1000"/>
  <line stroke="red" x1="531" y1="969" x2="883" y2="969"/>
  <line stroke="blue" x1="891" y1="656" x2="875" y2="688"/>
  <line stroke="red" x1="891" y1="656" x2="969" y2="656"/>
  <line stroke="red" x1="875" y1="688" x2="891" y2="719"/>
  <line stroke="blue" x1="875" y1="688" x2="938" y2="688"/>
  <line stroke="black" x1="500" y1="1000" x2="875" y2="1000"/>
  <line stroke="black" x1="1000" y1="500" x2="1000" y2="625"/>
  <line stroke="red" x1="969" y1="594" x2="1000" y2="625"/>
  <line stroke="blue" x1="891" y1="719" x2="875" y2="750"/>
  <line stroke="red" x1="891" y1="719" x2="906" y2="719"/>
  <line stroke="blue" x1="875" y1="750" x2="906" y2="719"/>
  <line stroke="red" x1="875" y1="750" x2="906" y2="734"/>
  <line stroke="red" x1="875" y1="750" x2="906" y2="750"/>
  <line stroke="red" x1="875" y1="750" x2="883" y2="781"/>
  <line stroke="blue" x1="875" y1="750" x2="906" y2="758"/>
  <line stroke="blue" x1="875" y1="750" x2="906" y2="781"/>
  <line stroke="red" x1="906" y1="719" x2="938" y2="688"/>
  <line stroke="red" x1="906" y1="719" x2="906" y2="734"/>
  <line stroke="blue" x1="938" y1="688" x2="969" y2="656"/>
  <line stroke="blue" x1="938" y1="688" x2="938" y2="750"/>
  <line stroke="red" x1="906" y1="734" x2="906" y2="750"/>
  <line stroke="blue" x1="906" y1="734" x2="938" y2="750"/>
  <line stroke="red" x1="969" y1="656" x2="1000" y2="625"/>
  <line stroke="red" x1="969" y1="656" x2="969" y2="734"/>
  <line stroke="red" x1="906" y1="750" x2="906" y2="758"/>
  <line stroke="blue" x1="906" y1="750" x2="938" y2="750"/>
  <line stroke="blue" x1="883" y1="781" x2="875" y2="812"/>
  <line stroke="red" x1="883" y1="781" x2="906" y2="781"/>
  <line stroke="black" x1="1000" y1="625" x2="1000" y2="750"/>
  <line stroke="red" x1="906" y1="758" x2="906" y2="781"/>
  <line stroke="red" x1="906" y1="758" x2="938" y2="750"/>
  <line stroke="red" x1="875" y1="812" x2="883" y2="844"/>
  <line stroke="blue" x1="875" y1="812" x2="938" y2="812"/>
  <line stroke="red" x1="906" y1="781" x2="938" y2="812"/>
  <line stroke="red" x1="938" y1="750" x2="969" y2="734"/>
  <line stroke="red" x1="938" y1="750" x2="969" y2="750"/>
  <line stroke="blue" x1="938" y1="750" x2="969" y2="758"/>
  <line stroke="blue" x1="938" y1="750" x2="938" y2="812"/>
  <line stroke="red" x1="969" y1="734" x2="969" y2="750"/>
  <line stroke="blue" x1="969" y1="734" x2="1000" y2="750"/>
  <line stroke="blue" x1="883" y1="844" x2="875" y2="875"/>
  <line stroke="red" x1="883" y1="844" x2="969" y2="844"/>
  <line stroke="red" x1="969" y1="750" x2="969" y2="758"/>
  <line stroke="blue" x1="969" y1="750" x2="1000" y2="750"/>
  <line stroke="red" x1="969" y1="758" x2="1000" y2="750"/>
  <line stroke="red" x1="969" y1="758" x2="969" y2="844"/>
  <line stroke="red" x1="875" y1="875" x2="883" y2="906"/>
  <line stroke="blue" x1="875" y1="875" x2="1000" y2="875"/>
  <line stroke="blue" x1="938" y1="812" x2="969" y2="844"/>
  <line stroke="black" x1="1000" y1="750" x2="1000" y2="875"/>
  <line stroke="blue" x1="883" y1="906" x2="875" y2="938"/>
  <line stroke="red" x1="883" y1="906" x2="969" y2="906"/>
  <line stroke="red" x1="875" y1="938" x2="883" y2="969"/>
  <line stroke="blue" x1="875" y1="938" x2="938" y2="938"/>
  <line stroke="red" x1="969" y1="844" x2="1000" y2="875"/>
  <line stroke="red" x1="883" y1="969" x2="906" y2="969"/>
  <line stroke="blue" x1="883" y1="969" x2="875" y2="1000"/>
  <line stroke="red" x1="938" y1="938" x2="906" y2="969"/>
  <line stroke="blue" x1="938" y1="938" x2="969" y2="906"/>
  <line stroke="blue" x1="938" y1="938" x2="938" y2="1000"/>
  <line stroke="blue" x1="906" y1="969" x2="875" y2="1000"/>
  <line stroke="red" x1="906" y1="969" x2="906" y2="1000"/>
  <line stroke="red" x1="969" y1="906" x2="1000" y2="875"/>
  <line stroke="red" x1="969" y1="906" x2="969" y2="1000"/>
  <line stroke="black" x1="875" y1="1000" x2="906" y2="1000"/>
  <line stroke="black" x1="1000" y1="875" x2="1000" y2="1000"/>
  <line stroke="black" x1="906" y1="1000" x2="938" y2="1000"/>
  <line stroke="black" x1="938" y1="1000" x2="969" y2="1000"/>
  <line stroke="black" x1="969" y1="1000" x2="1000" y2="1000"/>
</svg>""",
    inscribed_rectangle_shaping_params=[
        inscribed_rectangle_shaping.InscribedRectangleFlapShapingParams(
            flap_name="tail",
            inscribed_rectangles_params=[
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=2,
                    fold_percent=0.2,
                    fold_order=[
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.VALLEY,
                    ],
                ),
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=4,
                    fold_percent=0.4,
                    fold_order=[
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.VALLEY,
                    ],
                ),
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=6,
                    fold_percent=0.2,
                    fold_order=[
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.VALLEY,
                    ],
                ),
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=8,
                    fold_percent=0.2,
                    fold_order=[
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.VALLEY,
                    ],
                ),
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=10,
                    fold_percent=0.5,
                    fold_order=[
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.VALLEY,
                    ],
                ),
            ],
        ),
        inscribed_rectangle_shaping.InscribedRectangleFlapShapingParams(
            flap_name="third left leg",
            inscribed_rectangles_params=[
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=3,
                    fold_percent=0.1,
                    fold_order=[
                        crease_utils.Fold.VALLEY,
                        crease_utils.Fold.MOUNTAIN,
                    ],
                ),
            ],
        ),
        inscribed_rectangle_shaping.InscribedRectangleFlapShapingParams(
            flap_name="fourth left leg",
            inscribed_rectangles_params=[
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=3,
                    fold_percent=0.1,
                    fold_order=[
                        crease_utils.Fold.VALLEY,
                        crease_utils.Fold.MOUNTAIN,
                    ],
                ),
            ],
        ),
        inscribed_rectangle_shaping.InscribedRectangleFlapShapingParams(
            flap_name="third right leg",
            inscribed_rectangles_params=[
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=3,
                    fold_percent=0.1,
                    fold_order=[
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.VALLEY,
                    ],
                ),
            ],
        ),
        inscribed_rectangle_shaping.InscribedRectangleFlapShapingParams(
            flap_name="fourth right leg",
            inscribed_rectangles_params=[
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=3,
                    fold_percent=0.1,
                    fold_order=[
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.VALLEY,
                    ],
                ),
            ],
        ),
        inscribed_rectangle_shaping.InscribedRectangleFlapShapingParams(
            flap_name="fourth right leg",
            inscribed_rectangles_params=[
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=3,
                    fold_percent=0.1,
                    fold_order=[
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.VALLEY,
                    ],
                ),
            ],
        ),
        inscribed_rectangle_shaping.InscribedRectangleFlapShapingParams(
            flap_name="left top claw",
            inscribed_rectangles_params=[
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=0,
                    fold_percent=0.1,
                    fold_order=[
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.VALLEY,
                    ],
                ),
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=2,
                    fold_percent=0.1,
                    fold_order=[
                        crease_utils.Fold.VALLEY,
                        crease_utils.Fold.MOUNTAIN,
                    ],
                ),
            ],
        ),
        inscribed_rectangle_shaping.InscribedRectangleFlapShapingParams(
            flap_name="left bottom claw",
            inscribed_rectangles_params=[
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=0,
                    fold_percent=0.1,
                    fold_order=[
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.VALLEY,
                    ],
                ),
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=2,
                    fold_percent=0.1,
                    fold_order=[
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.VALLEY,
                    ],
                ),
            ],
        ),
        inscribed_rectangle_shaping.InscribedRectangleFlapShapingParams(
            flap_name="right top claw",
            inscribed_rectangles_params=[
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=0,
                    fold_percent=0.1,
                    fold_order=[
                        crease_utils.Fold.VALLEY,
                        crease_utils.Fold.MOUNTAIN,
                    ],
                ),
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=2,
                    fold_percent=0.1,
                    fold_order=[
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.VALLEY,
                    ],
                ),
            ],
        ),
        inscribed_rectangle_shaping.InscribedRectangleFlapShapingParams(
            flap_name="right bottom claw",
            inscribed_rectangles_params=[
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=0,
                    fold_percent=0.1,
                    fold_order=[
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.VALLEY,
                    ],
                ),
                inscribed_rectangle_shaping.InscribedRectangleParams(
                    depth=2,
                    fold_percent=0.1,
                    fold_order=[
                        crease_utils.Fold.MOUNTAIN,
                        crease_utils.Fold.VALLEY,
                    ],
                ),
            ],
        ),
    ],
    inscribed_rectangle_shaping_svg_sha=(
        "52f2a3929f51b6afaf136529362ab6499f57c1419bfcc0a9d85020879fd3860a"
    ),
)
