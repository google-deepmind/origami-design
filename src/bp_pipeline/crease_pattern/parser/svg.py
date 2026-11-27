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

"""Parse SVG strings into CreasePattern objects."""

from collections.abc import Iterator, Mapping
import itertools
import math
import random

from absl import logging
from bp_pipeline.crease_pattern import crease
from bp_pipeline.crease_pattern import crease_pattern
from bp_pipeline.crease_pattern import point
from bp_pipeline.crease_pattern import types
from bp_pipeline.crease_pattern import validate
from bp_pipeline.crease_pattern.parser import parser_utils
import lxml.etree as ET


class SVGParseError(ValueError):
  """Raised when an SVG string fails to parse."""


def _reject_unsafe_xml_declarations(xml: str) -> None:
  """Rejects XML DOCTYPE and ENTITY declarations prior to parsing."""
  lowered = xml.lower()
  if '<!doctype' in lowered or '<!entity' in lowered:
    raise SVGParseError(
        'XML DOCTYPE and ENTITY declarations are not permitted in SVG files.'
    )


def get_fold(color: str) -> crease.Fold:
  match color.strip().lower().replace(' ', ''):
    case 'orange':
      return crease.Fold.MOUNTAIN_ZEBRA
    case 'turquoise':
      return crease.Fold.VALLEY_ZEBRA
    case 'green' | '#00ff00' | '#0f0':
      return crease.Fold.HINGE
    case 'red' | '#ff0000' | '#f00' | 'rgb(255,0,0)' | 'rgb(100%,0%,0%)':
      return crease.Fold.MOUNTAIN
    case 'blue' | '#0000ff' | '#00f' | 'rgb(0,0,255)' | 'rgb(0%,0%,100%)':
      return crease.Fold.VALLEY
    case 'black' | '#000000' | '#000' | 'rgb(0,0,0)' | 'rgb(0%,0%,0%)':
      return crease.Fold.BORDER
    case 'yellow' | '#ffff00' | '#ff0' | 'rgb(255,255,0)' | 'rgb(100%,100%,0%)':
      return crease.Fold.TRIANGLE
    case _:
      raise SVGParseError(f'Color {color} is not in the allowed list.')


def _square_viewbox(viewbox: types.ViewBox) -> bool:
  """Checks if the viewBox is a square."""
  return math.isclose(viewbox[2], viewbox[3], rel_tol=1e-2)


def _get_style_map(style: str) -> dict[str, str]:
  """Parses inline style attributes from an SVG element."""
  style = style.strip()
  if ':' not in style:
    return {}

  styles = {}
  for attr in style.split(';'):
    attr = attr.strip()
    if attr and ':' in attr:
      key, value = attr.split(':', maxsplit=1)
      styles[key.strip()] = value.strip()

  return styles


def parse_line(line: ET.Element, viewbox: types.ViewBox) -> crease.Crease:
  """Creates a Crease object from a line element and a viewbox."""
  if not valid_line(line):
    raise SVGParseError(f'Expected a <line> element, but got <{line.tag}>.')

  styles = _get_style_map(line.attrib.get('style', ''))

  def get_attr(attr: str, default=None):
    return line.attrib.get(attr) or styles.get(attr) or default

  fold = get_fold(get_attr('stroke'))
  fold_percent = float(get_attr('opacity', get_attr('stroke-opacity', 1.0)))
  fold_step = int(get_attr('fold-step', 0))
  raw_elevation = get_attr('elevation', None)
  elevation = int(raw_elevation) if raw_elevation is not None else None

  return crease.Crease(
      point.Point.from_viewbox(
          x=float(get_attr('x1')),
          y=float(get_attr('y1')),
          viewbox=viewbox,
      ),
      point.Point.from_viewbox(
          x=float(get_attr('x2')),
          y=float(get_attr('y2')),
          viewbox=viewbox,
      ),
      fold=fold,
      fold_percent=fold_percent,
      fold_step=fold_step,
      elevation=elevation,
  )


def parse_rect(rect: ET.Element, viewbox: types.ViewBox) -> list[crease.Crease]:
  """Creates a list of 4 Crease objects from a rect element."""
  if not check_tag(rect, 'rect'):
    raise SVGParseError('Not a rect element')

  styles = _get_style_map(rect.attrib.get('style', ''))

  def get_attr(attr: str, default=None):
    return rect.attrib.get(attr) or styles.get(attr) or default

  fold = get_fold(get_attr('stroke'))
  fold_percent = float(get_attr('opacity', 1.0))
  fold_step = int(get_attr('fold-step', 0))
  raw_elevation = get_attr('elevation', None)
  elevation = int(raw_elevation) if raw_elevation is not None else None

  # SVG rect defaults x and y to 0 if not specified
  x = float(get_attr('x', 0.0))
  y = float(get_attr('y', 0.0))

  width = get_attr('width')
  if width is None:
    raise SVGParseError('Rect width must be specified.')

  height = get_attr('height')
  if height is None:
    raise SVGParseError('Rect height must be specified.')

  width = float(width)
  if width < 0:
    raise SVGParseError('Rect width must be non-negative.')

  height = float(height)
  if height < 0:
    raise SVGParseError('Rect height must be non-negative.')

  top = y + height
  right = x + width

  # Create corner points
  p1 = point.Point.from_viewbox(x=x, y=y, viewbox=viewbox)  # Bottom-left
  p2 = point.Point.from_viewbox(x=x, y=top, viewbox=viewbox)  # Top-left
  p3 = point.Point.from_viewbox(x=right, y=top, viewbox=viewbox)  # Top-right
  p4 = point.Point.from_viewbox(x=right, y=y, viewbox=viewbox)  # Bottom-right

  style = dict(
      fold=fold,
      fold_percent=fold_percent,
      fold_step=fold_step,
      elevation=elevation,
  )
  creases_list = [
      crease.Crease(p1, p2, **style),
      crease.Crease(p2, p3, **style),
      crease.Crease(p3, p4, **style),
      crease.Crease(p4, p1, **style),
  ]
  return creases_list


def parse_svg(
    svg: str,
    repair: bool = False,
    new_viewbox: types.ViewBox | None = None,
    light_mode: bool = False,
    merge_trivial: bool = False,
    grid_size: int = -1,
) -> crease_pattern.CreasePattern:
  """Parse an SVG string into a CreasePattern object.

  Args:
    svg: The SVG string to parse.
    repair: Attempt to fix broken XML using lxml parser.
    new_viewbox: If provided, use as the viewBox. If not provided, use the
      viewBox from the SVG and check if the resulting pattern fills the viewBox.
      If the pattern does not fill the viewBox, compute a new viewBox to fill
      the borders of the pattern.
    light_mode: Whether to use light mode.
    merge_trivial: Whether to merge trivial creases.
    grid_size: The size of the grid to use. If -1, no grid is used.

  Returns:
    The CreasePattern object.

  Raises:
    SVGParseError: If the SVG string fails to parse.
  """
  _reject_unsafe_xml_declarations(svg)
  if repair:
    root = ET.fromstring(
        svg,
        parser=ET.XMLParser(
            recover=True, resolve_entities=False, no_network=True
        ),
    )
    svg = ET.tostring(root, encoding='unicode', method='xml')

  if not valid_svg(svg):
    raise SVGParseError('Invalid SVG string.')

  root = ET.fromstring(
      svg, parser=ET.XMLParser(resolve_entities=False, no_network=True)
  )
  viewbox: types.ViewBox = new_viewbox or tuple(  # type: ignore[assignment]
      map(float, root.attrib['viewBox'].split(' '))
  )
  if not _square_viewbox(viewbox):
    raise SVGParseError('ViewBox is not a square.')

  _propagate_group_attributes(root)

  creases: list[crease.Crease] = []
  invalid_creases = 0
  for line in find_elements(root, 'line'):
    try:
      creases.append(parse_line(line, viewbox))
    except ValueError as e:
      logging.warning('Invalid line element: %s %s', line, e)
      invalid_creases += 1

  for rect in find_elements(root, 'rect'):
    try:
      creases.extend(parse_rect(rect, viewbox))
    except ValueError as e:
      logging.warning('Invalid rect element: %s %s', rect, e)
      invalid_creases += 4

  if new_viewbox is not None:
    return crease_pattern.CreasePattern(
        creases,
        light_mode=light_mode,
        merge_trivial=merge_trivial,
        num_invalid_creases=invalid_creases,
        grid_size=grid_size,
    )
  return rescale_viewbox_to_fill_borders(
      creases=creases,
      viewbox=viewbox,
      svg=svg,
      repair=repair,
      light_mode=light_mode,
      invalid_creases=invalid_creases,
      grid_size=grid_size,
      merge_trivial=merge_trivial,
  )


def rescale_viewbox_to_fill_borders(
    creases: list[crease.Crease],
    viewbox: types.ViewBox,
    svg: str,
    repair: bool,
    light_mode: bool,
    merge_trivial: bool,
    invalid_creases: int,
    grid_size: int,
) -> crease_pattern.CreasePattern:
  """Crease patterns may have border edges that do not fill the paper / viewbox.

  This function rescales all the creases so they fill the paper if the borders
  are given as a square. It is doing so by computing a new viewBox and making a
  second call to parse_svg.

  Args:
    creases: The creases to rescale.
    viewbox: The viewBox of the SVG.
    svg: The SVG string.
    repair: Whether to repair the SVG.
    light_mode: Whether to use light mode.
    merge_trivial: Whether to merge trivial creases.
    invalid_creases: The number of creases that failed to parse.
    grid_size: The size of the grid to use. If -1, no grid is used.

  Returns:
    The CreasePattern object.
  """
  merged_border_creases = validate.validate_border_creases(creases)

  # Compute a view box based on the borders of the pattern.
  x_coords = []
  y_coords = []
  for c in merged_border_creases:
    x_coords.append(c.p1.x)
    x_coords.append(c.p2.x)
    y_coords.append(c.p1.y)
    y_coords.append(c.p2.y)

  # revert viewbox
  orig_x_min, orig_y_min, orig_width, orig_height = viewbox
  max_size = max(orig_width, orig_height)
  x_coords = [x * max_size + orig_x_min for x in x_coords]
  y_coords = [y * max_size + orig_y_min for y in y_coords]

  # compute the new viewbox
  x_min = min(x_coords)
  y_min = min(y_coords)
  width = max(x_coords) - x_min
  height = max(y_coords) - y_min

  if (
      math.isclose(width, orig_width, abs_tol=point.DEFAULT_TOLERANCE)
      and math.isclose(height, orig_height, abs_tol=point.DEFAULT_TOLERANCE)
      and math.isclose(x_min, orig_x_min, abs_tol=point.DEFAULT_TOLERANCE)
      and math.isclose(y_min, orig_y_min, abs_tol=point.DEFAULT_TOLERANCE)
  ):
    # The new viewbox is the same as the original viewBox.
    return crease_pattern.CreasePattern(
        creases,
        light_mode=light_mode,
        merge_trivial=merge_trivial,
        num_invalid_creases=invalid_creases,
        grid_size=grid_size,
    )
  new_viewbox = x_min, y_min, width, height
  return parse_svg(
      svg=svg, repair=repair, new_viewbox=new_viewbox, grid_size=grid_size
  )


def _add_line_element(
    parent: ET.Element,
    line: crease.Crease,
    scale: float,
    ndigits: int,
    grid: int,
    omit_stroke: bool = False,
    omit_fold_step: bool = False,
    *,
    colors: Mapping[crease.Fold, str] | None = None,
    stroke_width: float | int | str | None = None,
    dashed_valley: bool = False,
    dash_value: str | None = None,
) -> None:
  """Adds a crease line element to the parent element."""

  line_el = ET.SubElement(parent, 'line')
  attrs = line_el.attrib
  if not omit_stroke:
    attrs['stroke'] = (
        colors.get(line.fold, line.fold.color())
        if colors is not None
        else line.fold.color()
    )
  attrs['x1'] = parser_utils.point_format(
      line.p1.x, scale=scale, ndigits=ndigits, grid=grid
  )
  attrs['y1'] = parser_utils.point_format(
      line.p1.y, scale=scale, ndigits=ndigits, grid=grid
  )
  attrs['x2'] = parser_utils.point_format(
      line.p2.x, scale=scale, ndigits=ndigits, grid=grid
  )
  attrs['y2'] = parser_utils.point_format(
      line.p2.y, scale=scale, ndigits=ndigits, grid=grid
  )
  if stroke_width is not None:
    attrs['stroke-width'] = str(stroke_width)
  if (
      dashed_valley
      and dash_value is not None
      and line.fold in (crease.Fold.VALLEY, crease.Fold.VALLEY_ZEBRA)
  ):
    attrs['stroke-dasharray'] = dash_value
  if line.mountain_or_valley() and line.fold_percent < 1:
    attrs['opacity'] = parser_utils.point_format(
        line.fold_percent, 1, max(ndigits, 6)
    )
  if not omit_fold_step and line.fold_step > 0:
    attrs['fold-step'] = str(line.fold_step)
  if line.elevation is not None:
    attrs['elevation'] = str(line.elevation)


def _propagate_group_attributes(root: ET.Element) -> None:
  """Propagates group attributes to child elements in-place."""

  inheritable = frozenset({'stroke', 'fold-step', 'opacity', 'stroke-opacity'})
  for g in find_elements(root, 'g'):
    for child in g:
      for attr in inheritable:
        if attr in g.attrib and attr not in child.attrib:
          styles = _get_style_map(child.attrib.get('style', ''))
          if attr not in styles:
            child.set(attr, g.attrib[attr])


def to_element_tree(
    pattern: crease_pattern.CreasePattern,
    *,
    scale: float = 1000,
    ndigits: int = 3,
    grid: int = -1,
    replace_border: bool = False,
    shuffle_order: bool = False,
    group: bool = False,
    colors: Mapping[crease.Fold, str] | None = None,
    stroke_width: float | int | str | None = None,
    dashed_valley: bool = False,
    grid_lines: bool = False,
) -> ET.Element:
  """Converts a CreasePattern object to a minimal SVG element tree.

  Args:
    pattern: The CreasePattern object to convert.
    scale: The scale factor to apply to the coordinates.
    ndigits: The number of significant digits to round values to.
    grid: The size of the grid to use. If -1, no grid is used.
    replace_border: Replace the border with a square <rect> element.
    shuffle_order: Shuffle the order of the creases in the SVG.
    group: Group creases by fold-step and fold type using <g> tags.
    colors: Optional custom stroke colors keyed by `crease.Fold`.
    stroke_width: Optional stroke width for crease and border elements.
    dashed_valley: Whether to add `stroke-dasharray` to valley creases.
    grid_lines: Whether to emit background `<path>` grid lines when `grid > 1`.

  Returns:
    The SVG element tree.
  """
  viewbox_scale = scale if grid == -1 else grid
  root = ET.Element(
      'svg',
      xmlns='http://www.w3.org/2000/svg',
      viewBox=f'0 0 {viewbox_scale} {viewbox_scale}',
  )

  if grid_lines and grid > 1:
    grid_group = ET.SubElement(
        root,
        'g',
        attrib={
            'class': 'bp-grid-lines',
            'stroke': '#e0e0e0',
            'stroke-width': '0.05',
            'fill': 'none',
        },
    )
    path_cmds: list[str] = []
    for idx in range(1, grid):
      path_cmds.append(f'M {idx} 0 L {idx} {grid}')
      path_cmds.append(f'M 0 {idx} L {grid} {idx}')
    ET.SubElement(grid_group, 'path', d=' '.join(path_cmds))

  if replace_border:
    size = str(viewbox_scale)
    border_stroke = (
        colors.get(crease.Fold.BORDER, 'black')
        if colors is not None
        else 'black'
    )
    border_attrs = dict(
        stroke=border_stroke,
        x='0',
        y='0',
        width=size,
        height=size,
        fill='none',
    )
    if stroke_width is not None:
      border_attrs['stroke-width'] = str(stroke_width)
    ET.SubElement(root, 'rect', border_attrs)

  if shuffle_order:
    if not replace_border:
      raise ValueError(
          'You almost certainly want to replace the border when shuffling the'
          ' order so that border creases are not shuffled into the pattern.'
      )
    creases = pattern.creases.copy()
    random.shuffle(creases)
  else:
    creases = pattern.creases

  if replace_border:
    creases = [c for c in creases if c.fold != crease.Fold.BORDER]

  dash_value = (
      '0.25,0.15'
      if grid > 0
      else f'{viewbox_scale * 0.01:g},{viewbox_scale * 0.006:g}'
  )

  if group:
    sorted_creases = sorted(creases, key=lambda c: (c.fold_step, c.fold.name))
    for step, step_group in itertools.groupby(
        sorted_creases, key=lambda c: c.fold_step
    ):
      step_attrs = {}
      if step > 0:
        step_attrs['fold-step'] = str(step)

      step_g = ET.SubElement(root, 'g', step_attrs)
      for fold, fold_group in itertools.groupby(
          step_group, key=lambda c: c.fold
      ):
        fold_stroke = (
            colors.get(fold, fold.color())
            if colors is not None
            else fold.color()
        )
        fold_g = ET.SubElement(step_g, 'g', stroke=fold_stroke)
        for line in fold_group:
          _add_line_element(
              fold_g,
              line,
              scale,
              ndigits,
              grid,
              omit_stroke=True,
              omit_fold_step=True,
              colors=colors,
              stroke_width=stroke_width,
              dashed_valley=dashed_valley,
              dash_value=dash_value,
          )

  else:
    for line in creases:
      _add_line_element(
          root,
          line,
          scale,
          ndigits,
          grid,
          colors=colors,
          stroke_width=stroke_width,
          dashed_valley=dashed_valley,
          dash_value=dash_value,
      )

  return root


def to_svg(
    pattern: crease_pattern.CreasePattern,
    *,
    scale: float = 1000,
    ndigits: int | None = None,
    grid: int = -1,
    pretty: bool = False,
    replace_border: bool = False,
    shuffle_order: bool = False,
    group: bool = False,
    colors: Mapping[crease.Fold, str] | None = None,
    stroke_width: float | int | str | None = None,
    dashed_valley: bool = False,
    grid_lines: bool = False,
) -> str:
  """Converts a CreasePattern object to a minimal SVG string.

  Args:
    pattern: The CreasePattern object to convert.
    scale: The scale factor to apply to the coordinates.
    ndigits: The number of decimal digits to round values to. Defaults to 1 when
      `grid > 0` and 3 when `grid == -1`; pass a larger value (e.g. 8) to
      preserve off-grid 3D-shaped vertex coordinates on a grid viewBox.
    grid: The size of the grid to use. When a grid is used, the viewbox is set
      to (0, 0, grid, grid) and scale is set to 1. If -1, no grid is used.
    pretty: Prettify the SVG string.
    replace_border: Replace the border with a square <rect> element.
    shuffle_order: Shuffle the order of the creases in the SVG.
    group: Group creases by fold-step and fold type using <g> tags.
    colors: Optional custom stroke colors keyed by `crease.Fold`.
    stroke_width: Optional stroke width for crease and border elements.
    dashed_valley: Whether to add `stroke-dasharray` to valley creases.
    grid_lines: Whether to emit background `<path>` grid lines when `grid > 1`.

  Returns:
    The SVG string.
  """
  if grid > 0:
    scale = 1
    if ndigits is None:
      ndigits = 1
  elif ndigits is None:
    ndigits = 3
  root = to_element_tree(
      pattern,
      scale=scale,
      ndigits=ndigits,
      grid=grid,
      replace_border=replace_border,
      shuffle_order=shuffle_order,
      group=group,
      colors=colors,
      stroke_width=stroke_width,
      dashed_valley=dashed_valley,
      grid_lines=grid_lines,
  )
  return ET.tostring(
      root, encoding='unicode', method='xml', pretty_print=pretty
  )


def valid_xml(xml: str) -> bool:
  """Checks if the XML string is valid."""
  try:
    _reject_unsafe_xml_declarations(xml)
    ET.fromstring(
        xml, parser=ET.XMLParser(resolve_entities=False, no_network=True)
    )
    return True
  except (ET.ParseError, ValueError):
    return False


def valid_svg(xml: str) -> bool:
  """Checks if the string is a valid SVG element containing at least one line."""
  try:
    _reject_unsafe_xml_declarations(xml)
    root = ET.fromstring(
        xml, parser=ET.XMLParser(resolve_entities=False, no_network=True)
    )
    _propagate_group_attributes(root)
    has_one_crease = False
    for line in find_elements(root, 'line'):
      if valid_line(line):
        has_one_crease = True
        break
    return valid_root(root) and has_one_crease
  except (ET.ParseError, ValueError) as e:
    logging.exception('Invalid SVG string: %s, causes: %s', xml, e)
    return False


def check_tag(el: ET.Element, tag: str) -> bool:
  """Checks if the element has the given tag, ignoring namespaces."""
  return ET.QName(el.tag).localname == tag


def find_elements(root: ET.Element, tag: str) -> Iterator[ET.Element]:
  """Yields all elements with the given tag, ignoring namespaces."""
  yield from root.iter('{*}%s' % tag)


def valid_root(root: ET.Element) -> bool:
  """Checks if the element is a valid SVG root element."""
  return check_tag(root, 'svg') and 'viewBox' in root.attrib.keys()


def valid_line(root: ET.Element) -> bool:
  """Checks if the root element is a valid line element."""
  valid_attributes = ['stroke', 'x1', 'y1', 'x2', 'y2']
  if not check_tag(root, 'line'):
    return False

  styles = _get_style_map(root.attrib.get('style', ''))
  keys = root.attrib.keys() | styles.keys()
  return all(v in keys for v in valid_attributes)
