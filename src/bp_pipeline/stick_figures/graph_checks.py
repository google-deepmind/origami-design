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

"""Quality checks for stick figures."""

import dataclasses
import math
import typing
from typing import Any, Literal

import networkx as nx

from . import converters
from . import shaping_templates
from . import stick as stick_lib
from . import stick_figure


def find_cycle_in_nx_graph(graph: nx.Graph) -> list[Any]:
  try:
    return nx.find_cycle(graph)
  except nx.NetworkXNoCycle:
    return []


def graph_has_cycle(graph: nx.Graph) -> bool:
  return bool(find_cycle_in_nx_graph(graph))


def graph_is_connected(graph: nx.Graph) -> bool:
  if len(graph) == 0:
    return False
  return nx.is_connected(graph)


def explain_symmetry_errors(tree: stick_figure.StickFigure) -> list[str]:
  """Returns actionable diagnostics for sticks lacking a YZ-plane mirror partner."""
  details: list[str] = []
  for s in tree.children:
    az = float(s.azimuth_angle)
    el = float(s.elevation_angle)
    if math.isclose(abs(az), 0.0, abs_tol=1e-3) or math.isclose(
        abs(az), 180.0, abs_tol=1e-3
    ):
      continue
    expected_az = -az
    has_partner = any(
        other.name != s.name
        and other.length == s.length
        and other.width == s.width
        and math.isclose(float(other.azimuth_angle), expected_az, abs_tol=1e-3)
        and math.isclose(float(other.elevation_angle), el, abs_tol=1e-3)
        for other in tree.children
    )
    if not has_partner:
      details.append(
          f"Stick '{s.name}' (start='{s.start}', end='{s.end}',"
          f" length={s.length}, width={s.width}, azimuth_angle={az},"
          f" elevation_angle={el}) has no mirrored partner with"
          f" azimuth_angle={expected_az}, elevation_angle={el},"
          f" length={s.length}, width={s.width}."
      )
  return details


def is_valid_stick_figure(figure: stick_figure.StickFigure) -> bool:
  """Checks if a stick figure is valid."""
  networkx_graph = converters.convert_stick_figure_to_networkx_graph(figure)

  if graph_has_cycle(networkx_graph):
    return False

  if not graph_is_connected(networkx_graph):
    return False

  for stick in figure.children:
    if stick.width not in stick_lib.StickWidthValues:
      return False

  return True


def _field_has_default(field: dataclasses.Field[Any]) -> bool:
  """Checks if a field has a default value."""
  return (
      field.default is not dataclasses.MISSING
      or field.default_factory is not dataclasses.MISSING
  )


def _check_simple_type(value: Any, expected_type: type[Any]) -> bool:
  """Checks if a value has the expected origin type."""
  origin_type = typing.get_origin(expected_type) or expected_type
  if origin_type == int:
    return isinstance(value, (int, float)) and value.is_integer()
  elif origin_type == float:
    return isinstance(value, (int, float))
  elif origin_type == Literal:  # pylint: disable=comparison-with-callable
    return value in typing.get_args(expected_type)
  else:
    return isinstance(value, origin_type)


def _check_float(
    value: Any,
    min_value: float = -float("inf"),
    max_value: float = float("inf"),
) -> bool:
  """Checks if a value is a finite float in the range [min_value, max_value]."""
  is_float = _check_simple_type(value, float)
  return is_float and min_value <= value <= max_value and math.isfinite(value)


def _prevalidate_stick(stick_data: dict[str, Any]) -> list[str]:
  """Validates a stick dictionary and returns a list of errors."""
  stick_name = stick_data.get("name", "Unknown")
  prefix = f"Stick '{stick_name}'"
  errors = []
  for field in dataclasses.fields(stick_lib.Stick):
    if not field.init:
      continue

    if field.name not in stick_data:
      if not _field_has_default(field):
        errors.append(f"{prefix} is missing '{field.name}'.")
    else:
      value = stick_data[field.name]
      expected_type = field.type
      if not _check_simple_type(value, expected_type):  # pyrefly: ignore[bad-argument-type]
        errors.append(
            # pyrefly: ignore[missing-attribute]
            f"{prefix} field '{field.name}' expects type"
            f" '{expected_type.__name__}', got '{type(value).__name__}'."
        )

  length = stick_data.get("length")
  if isinstance(length, int) and length <= 0:
    errors.append(f"{prefix} length must be > 0, got '{length}'.")

  width = stick_data.get("width")
  if width is not None and width not in stick_lib.StickWidthValues:
    errors.append(
        f"{prefix} width must be one of {stick_lib.StickWidthValues}, got"
        f" '{width}'."
    )

  azimuth_angle = stick_data.get("azimuth_angle")
  if azimuth_angle is not None and not _check_float(
      azimuth_angle, min_value=-360, max_value=360
  ):
    errors.append(
        f"{prefix} azimuth_angle must be between [-360, 360], got"
        f" '{azimuth_angle}'."
    )

  elevation_angle = stick_data.get("elevation_angle")
  if elevation_angle is not None and not _check_float(
      elevation_angle, min_value=-90, max_value=90
  ):
    errors.append(
        f"{prefix} elevation_angle must be between [-90, 90], got"
        f" '{elevation_angle}'."
    )

  return errors


def check_symmetry_axis_width_parity(
    tree: stick_figure.StickFigure,
) -> list[str]:
  """Checks that integer-width sticks on the symmetry axis share the same parity."""
  if not tree.is_symmetric_tree:
    return []

  axis_integer_widths = [
      int(tree[name].width)
      for name in tree.symmetry_axis
      if tree[name].width == int(tree[name].width)
  ]

  if not axis_integer_widths:
    return []

  parities = {w % 2 for w in axis_integer_widths}
  if len(parities) > 1:
    return [
        "Sticks on the symmetry axis have mixed width parity (odd and even)."
    ]
  return []


def prevalidate(
    data: dict[str, Any],
    *,
    check_symmetry: bool = False,
    use_shaping_flaps: bool = False,
) -> list[str]:
  """Validates a stick figure dictionary and returns a list of errors."""
  errors = []
  prefix = "Stick figure"

  for field in dataclasses.fields(stick_figure.StickFigure):
    if not field.init:
      continue

    if field.name not in data:
      if not _field_has_default(field):
        errors.append(f"{prefix} is missing field '{field.name}'.")
    else:
      value = data[field.name]
      expected_type = field.type
      if not _check_simple_type(value, expected_type):  # pyrefly: ignore[bad-argument-type]
        errors.append(
            # pyrefly: ignore[missing-attribute]
            f"{prefix} field '{field.name}' expects type"
            f" '{expected_type.__name__}', got '{type(value).__name__}'."
        )

  complexity = data.get("complexity")
  if isinstance(complexity, int) and complexity not in (1, 2, 3, 4, 5):
    errors.append(
        f"{prefix} complexity is not in [1, 2, 3, 4, 5], got '{complexity}'."
    )

  for side_name in ("color_top", "color_bottom"):
    value = data.get(side_name)
    if value is None:
      continue

    if not isinstance(value, list) or len(value) != 3:
      errors.append(
          f"{prefix} field '{side_name}' expects a list of 3 integers,"
          f" got '{value}'."
      )
    elif not all(isinstance(c, int) for c in value):
      errors.append(
          f"{prefix} field '{side_name}' expects a list of 3"
          f" integers, got '{value}'."
      )
    elif not all(0 <= c <= 255 for c in value):
      errors.append(
          f"{prefix} field '{side_name}' expects RGB values in [0,"
          f" 255], got '{value}'."
      )

  children = data.get("children")
  if not isinstance(children, list):
    errors.append(f"{prefix} children must be a list.")
  elif not children:
    errors.append(f"{prefix} must have at least one child.")
  else:
    all_names = set()
    for stick_data in children:
      stick_name = stick_data.get("name")
      if stick_name is not None and stick_name in all_names:
        errors.append(f"{prefix} duplicate stick name '{stick_name}'.")
      all_names.add(stick_name)
      errors.extend(_prevalidate_stick(stick_data))

    root = data.get("root", None)
    if isinstance(root, str):
      if not any(stick.get("name") == root for stick in children):
        errors.append(
            f"{prefix} root stick '{root}' not found in stick figure."
        )

    graph = nx.MultiGraph()
    for stick in children:
      if "start" in stick and "end" in stick:
        graph.add_edge(stick.get("start"), stick.get("end"))

    if graph_has_cycle(graph):
      errors.append(f"{prefix} has a cycle.")

    if not graph_is_connected(graph):
      errors.append(f"{prefix} is not a fully connected tree.")

  if not errors:
    try:
      raw_tree = stick_figure.StickFigure.from_dict(data)
      tree, _ = raw_tree.merge_linear_chains(ignore_shaping_flaps=True)

      if check_symmetry and (
          not raw_tree.is_symmetric_tree or not tree.is_symmetric_tree
      ):
        errors.append("Figure is not symmetric.")
        sym_details = explain_symmetry_errors(raw_tree)
        if not sym_details:
          sym_details = explain_symmetry_errors(tree)
        errors.extend(sym_details)

      if check_symmetry:
        errors.extend(check_symmetry_axis_width_parity(tree))

      if len(tree.children) <= 2:
        errors.append(f"{prefix} is just 1 chain with no branches.")

      shaping_flaps = [flap for flap in tree.children if flap.shaping_flap]
      if use_shaping_flaps:
        if not shaping_flaps:
          errors.append(f"{prefix} should include shaping flaps.")
        else:
          # Enforces the builtin template packer's structural and count limits
          # (<= 2 TailTemplates and <= 4 corner KneeTemplates / <= 2 pairs).
          errors.extend(
              shaping_templates.validate_shaping_flap_templates(raw_tree)
          )
      else:
        if shaping_flaps:
          errors.append(f"{prefix} should not include shaping flaps.")
        if tree.shaping_flaps_limit > 0:
          errors.append(f"{prefix} should set shaping_flaps_limit to 0.")

    except ValueError as e:
      errors.append(f"{prefix} is not a valid stick figure: {e}")

  return errors


def topology_signature(
    tree: stick_figure.StickFigure,
) -> tuple[tuple[int, ...], ...]:
  """Returns a topology signature of the stick figure for isomorphism checks."""
  joint_signatures = [
      tuple(sorted(s.length for s in joint.sticks))
      for joint in tree.joints.values()
  ]
  return tuple(sorted(joint_signatures))
