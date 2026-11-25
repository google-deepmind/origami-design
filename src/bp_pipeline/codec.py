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

"""Versioned, lossless plain-JSON codec for Box Pleating pipeline objects."""

from collections.abc import Callable, Mapping
import dataclasses
import json
import typing
from typing import Any
from bp_pipeline import bus
from bp_pipeline import rectangle as rectangle_lib
from bp_pipeline.crease_pattern import crease
from bp_pipeline.crease_pattern import crease_pattern as crease_pattern_lib
from bp_pipeline.crease_pattern import symmetry
from bp_pipeline.packing import config as packing_config_lib
from bp_pipeline.packing import packing_output as packing_output_lib
from bp_pipeline.packing import river as river_lib
from bp_pipeline.packing import types as packing_types
from bp_pipeline.solving import hinges as hinge_utils
from bp_pipeline.stick_figures import stick as stick_lib
from bp_pipeline.stick_figures import stick_figure as stick_figure_lib
import immutabledict
import numpy as np

SCHEMA_VERSION = '1.0'


def _normalize_json_value(value: Any) -> Any:
  if isinstance(value, np.integer):
    return int(value)
  if isinstance(value, np.floating):
    return float(value)
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, Mapping):
    return {str(k): _normalize_json_value(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_normalize_json_value(v) for v in value]
  return value


def stick_figure_to_dict(
    sf: stick_figure_lib.StickFigure,
) -> dict[str, Any]:
  raw = sf.to_dict()
  for stick_dict, stick_obj in zip(raw['children'], sf.children, strict=True):
    if stick_obj.flap_tip_shape != stick_lib.FlapTipShape.UNSET:
      stick_dict['flap_tip_shape'] = stick_obj.flap_tip_shape.value
  return {
      'type': 'StickFigure',
      'schema_version': SCHEMA_VERSION,
      **_normalize_json_value(raw),
  }


def stick_figure_from_dict(
    data: Mapping[str, Any],
) -> stick_figure_lib.StickFigure:
  sf = stick_figure_lib.StickFigure.from_dict(dict(data))
  for stick_dict in data.get('children', ()):
    tip_shape = stick_dict.get('flap_tip_shape')
    if tip_shape is not None:
      sf[stick_dict['name']].flap_tip_shape = stick_lib.FlapTipShape(tip_shape)
  return sf


def crease_pattern_to_dict(
    cp: crease_pattern_lib.CreasePattern,
) -> dict[str, Any]:
  return {
      'type': 'CreasePattern',
      'schema_version': SCHEMA_VERSION,
      'grid_size': cp.grid_size,
      'light_mode': cp.light_mode,
      'tolerance': cp.tolerance,
      'creases': [c.asdict() for c in cp.creases],
  }


def crease_pattern_from_dict(
    data: Mapping[str, Any],
) -> crease_pattern_lib.CreasePattern:
  if 'creases' not in data or 'grid_size' not in data:
    raise ValueError(
        'CreasePattern dict must include required keys "creases" and'
        ' "grid_size".'
    )
  creases = [crease.Crease.from_dict(c) for c in data['creases']]
  return crease_pattern_lib.CreasePattern(
      creases=creases,
      grid_size=int(data['grid_size']),
      light_mode=bool(data.get('light_mode', False)),
      is_empty=not creases,
      tolerance=float(data.get('tolerance', crease.DEFAULT_TOLERANCE)),
  )


def packing_config_to_dict(
    config: packing_config_lib.PackingConfig,
) -> dict[str, Any]:
  """Serializes a PackingConfig to a dictionary."""
  raw = dict(config.to_dict())
  if isinstance(raw.get('traversal_mode'), packing_types.TraversalMode):
    raw['traversal_mode'] = raw['traversal_mode'].value
  if raw.get('local_tiling_config') is not None and isinstance(
      raw['local_tiling_config'].get('search_type'),
      packing_config_lib.LocalTilingSearchType,
  ):
    raw['local_tiling_config'] = dict(raw['local_tiling_config'])
    raw['local_tiling_config']['search_type'] = raw['local_tiling_config'][
        'search_type'
    ].value
  return {
      'type': 'PackingConfig',
      'schema_version': SCHEMA_VERSION,
      **raw,
  }


def packing_config_from_dict(
    data: Mapping[str, Any],
) -> packing_config_lib.PackingConfig:
  payload = {
      k: v for k, v in data.items() if k not in ('type', 'schema_version')
  }
  return packing_config_lib.PackingConfig.from_dict(payload)


def solver_config_to_dict(
    config: hinge_utils.SolverConfig,
) -> dict[str, Any]:
  return {
      'type': 'SolverConfig',
      'schema_version': SCHEMA_VERSION,
      **config.asdict(),
  }


def solver_config_from_dict(
    data: Mapping[str, Any],
) -> hinge_utils.SolverConfig:
  return hinge_utils.SolverConfig.from_dict(data)


_RIVER_CODEC_REGISTRY: dict[str, Callable[[Mapping[str, Any]], Any]] = {}


def register_river_codec(
    river_type: str,
    from_dict_fn: Callable[[Mapping[str, Any]], Any],
) -> None:
  """Registers a deserializer function for an extended river type."""
  _RIVER_CODEC_REGISTRY[river_type] = from_dict_fn


def packing_river_to_dict(
    river: river_lib.PackingRiver, name: str = ''
) -> dict[str, Any]:
  """Serializes a PackingRiver to a dictionary."""
  if isinstance(river, river_lib.River):
    return {
        'name': name,
        'river_type': 'straight',
        'width': int(river.width),
        'offset': int(river.offset),
        'orientation': river.orientation.value,
        'grid_size': int(river.grid_size),
        'rectangles': [r.asdict() for r in river.rectangles],
    }
  if isinstance(river, river_lib.LShapedRiver):
    return {
        'name': name,
        'river_type': 'l_shaped',
        'width': int(river.width),
        'l_shape_type': river.l_shape_type.value,
        'turn_x': int(river.turn_x),
        'turn_y': int(river.turn_y),
        'grid_size': int(river.grid_size),
        'rectangles': [r.asdict() for r in river.rectangles],
    }
  if isinstance(river, river_lib.HuggingRiver):
    return {
        'name': name,
        'river_type': 'hugging',
        'width': int(river.width),
        'subriver_paths': [
            [[int(pt[0]), int(pt[1])] for pt in path]
            for path in river.subriver_paths
        ],
        'hug_side': river.hug_side.value,
        'is_on_axis': river.is_on_axis,
        'grid_size': int(river.grid_size),
        'variant_type': river.variant_type.value,
        'rectangles': [r.asdict() for r in river.rectangles],
    }
  if hasattr(river, 'asdict') and callable(river.asdict):
    d = dict(river.asdict(name))
    d.setdefault('name', name)
    return d
  raise TypeError(f'Unsupported river type: {type(river)}')


def packing_river_from_dict(
    data: Mapping[str, Any],
) -> river_lib.PackingRiver:
  river_type = data.get('river_type')
  match river_type:
    case 'straight':
      required = ('width', 'offset', 'orientation', 'grid_size')
      missing = [k for k in required if k not in data]
      if missing:
        raise ValueError(f'Missing required straight river keys: {missing}')
      return river_lib.River(
          width=int(data['width']),
          offset=int(data['offset']),
          orientation=river_lib.RiverOrientation(data['orientation']),
          grid_size=int(data['grid_size']),
      )
    case 'l_shaped':
      required = ('width', 'l_shape_type', 'turn_x', 'turn_y', 'grid_size')
      missing = [k for k in required if k not in data]
      if missing:
        raise ValueError(f'Missing required l_shaped river keys: {missing}')
      return river_lib.LShapedRiver(
          width=int(data['width']),
          l_shape_type=river_lib.LShapeType(data['l_shape_type']),
          turn_x=int(data['turn_x']),
          turn_y=int(data['turn_y']),
          grid_size=int(data['grid_size']),
      )
    case 'hugging':
      required = (
          'width',
          'subriver_paths',
          'hug_side',
          'is_on_axis',
          'grid_size',
          'variant_type',
      )
      missing = [k for k in required if k not in data]
      if missing:
        raise ValueError(f'Missing required hugging river keys: {missing}')
      return river_lib.HuggingRiver(
          width=int(data['width']),
          subriver_paths=[
              [(int(pt[0]), int(pt[1])) for pt in path]
              for path in data['subriver_paths']
          ],
          hug_side=river_lib.HugSide(data['hug_side']),
          is_on_axis=bool(data['is_on_axis']),
          grid_size=int(data['grid_size']),
          variant_type=river_lib.PathVariantType(data['variant_type']),
      )
    case _ if river_type in _RIVER_CODEC_REGISTRY:
      return _RIVER_CODEC_REGISTRY[river_type](data)
    case _:
      raise ValueError(
          'Unknown or missing river_type in packing_river_from_dict:'
          f' {river_type!r}'
      )


def design_tree_to_dict(tree: bus.DesignTree) -> dict[str, Any]:
  return {
      'type': 'DesignTree',
      'schema_version': SCHEMA_VERSION,
      'source': stick_figure_to_dict(tree.source),
      'merged': stick_figure_to_dict(tree.merged),
      'merged_from': {k: list(v) for k, v in tree.merged_from.items()},
      'symmetric_partner': dict(tree.symmetric_partner),
      'is_symmetric': tree.is_symmetric,
      'min_grid_size': tree.min_grid_size,
      'max_grid_size': tree.max_grid_size,
  }


def design_tree_from_dict(data: Mapping[str, Any]) -> bus.DesignTree:
  """Deserializes a DesignTree from a dictionary."""
  required = (
      'source',
      'merged',
      'merged_from',
      'symmetric_partner',
      'is_symmetric',
      'min_grid_size',
      'max_grid_size',
  )
  missing = [k for k in required if k not in data]
  if missing:
    raise ValueError(f'Missing required DesignTree keys: {missing}')
  source = stick_figure_from_dict(data['source'])
  merged = stick_figure_from_dict(data['merged'])
  merged_from = immutabledict.immutabledict(
      {str(k): tuple(str(x) for x in v) for k, v in data['merged_from'].items()}
  )
  symmetric_partner = immutabledict.immutabledict(
      {str(k): str(v) for k, v in data['symmetric_partner'].items()}
  )
  return bus.DesignTree(
      source=source,
      merged=merged,
      merged_from=merged_from,
      symmetric_partner=symmetric_partner,
      is_symmetric=bool(data['is_symmetric']),
      min_grid_size=int(data['min_grid_size']),
      max_grid_size=int(data['max_grid_size']),
  )


def _hinge_info_to_dict(h: packing_types.HingeInfo) -> dict[str, Any]:
  coordinate_dict = h.coordinate.asdict() if h.coordinate is not None else None
  return {
      'coordinate': coordinate_dict,
      'assigned': h.assigned,
      'flap_names': list(h.flap_names),
  }


def _hinge_info_from_dict(data: Mapping[str, Any]) -> packing_types.HingeInfo:
  coord_data = data['coordinate']
  return packing_types.HingeInfo(
      coordinate=(
          crease.Crease.from_dict(coord_data)
          if coord_data is not None
          else None
      ),
      assigned=bool(data['assigned']),
      flap_names=typing.cast(tuple[str, str], tuple(data['flap_names'])),
  )


def _sacrificial_flap_to_dict(
    sf: packing_types.SacrificialFlap,
) -> dict[str, Any]:
  """Serializes a SacrificialFlap (supporting flap and river targets)."""
  if isinstance(sf.shifted_element, river_lib.PackingRiver):
    shifted_dict = packing_river_to_dict(
        sf.shifted_element, name=sf.shifted_element_name
    )
  else:
    shifted_dict = _normalize_json_value(dataclasses.asdict(sf.shifted_element))
  return {
      'polygon_name': sf.polygon_name,
      'placement': _normalize_json_value(dataclasses.asdict(sf.placement)),
      'shifted_element_name': sf.shifted_element_name,
      'orientation': sf.orientation,
      'shifted_element': shifted_dict,
      'crimp_direction': sf.crimp_direction,
  }


def _sacrificial_flap_from_dict(
    data: Mapping[str, Any],
) -> packing_types.SacrificialFlap:
  """Deserializes a SacrificialFlap (supporting flap and river targets)."""
  shifted_raw = data.get('shifted_element')
  if isinstance(shifted_raw, Mapping) and 'river_type' in shifted_raw:
    shifted_river = packing_river_from_dict(shifted_raw)
    return packing_types.SacrificialFlap(
        polygon_name=str(data['polygon_name']),
        placement=packing_types.PolygonPlacement.from_dict(
            dict(data['placement'])
        ),
        shifted_element_name=str(data['shifted_element_name']),
        orientation=typing.cast(
            packing_types.Orientation, str(data['orientation'])
        ),
        shifted_element=shifted_river,
        crimp_direction=typing.cast(
            packing_types.MainFlapPosition, str(data['crimp_direction'])
        ),
    )
  return packing_types.SacrificialFlap.from_dict(dict(data))


def pack_to_dict(
    output: packing_output_lib.PackingOutput,
) -> dict[str, Any]:
  """Serializes a PackingOutput to a dictionary."""
  if output.packing_rivers is not None:
    packing_rivers_dict = {
        name: packing_river_to_dict(river, name=name)
        for name, river in output.packing_rivers.items()
    }
  else:
    packing_rivers_dict = None

  if output.sacrificial_flaps is not None:
    sacrificial_flaps_list = [
        _sacrificial_flap_to_dict(sf) for sf in output.sacrificial_flaps
    ]
  else:
    sacrificial_flaps_list = None

  return {
      'type': 'PackingOutput',
      'schema_version': SCHEMA_VERSION,
      'success': output.success,
      'packing_cp': (
          crease_pattern_to_dict(output.packing_cp)
          if output.packing_cp is not None
          else None
      ),
      'flap_rectangles': (
          [r.asdict() for r in output.flap_rectangles]
          if output.flap_rectangles is not None
          else None
      ),
      'packing_rivers': packing_rivers_dict,
      'stick_figure': (
          stick_figure_to_dict(output.stick_figure)
          if output.stick_figure is not None
          else None
      ),
      'hinges': (
          [_hinge_info_to_dict(h) for h in output.hinges]
          if output.hinges is not None
          else None
      ),
      'sacrificial_flaps': sacrificial_flaps_list,
      'details': (
          _normalize_json_value(dict(output.details))
          if output.details is not None
          else None
      ),
      'config': (
          packing_config_to_dict(output.config)
          if output.config is not None
          else None
      ),
      'error_message': output.error_message,
  }


def pack_from_dict(
    data: Mapping[str, Any],
) -> packing_output_lib.PackingOutput:
  """Deserializes a PackingOutput from a dictionary."""
  if 'success' not in data:
    raise ValueError('PackingOutput dict must include required key "success".')
  packing_cp = (
      crease_pattern_from_dict(data['packing_cp'])
      if data.get('packing_cp') is not None
      else None
  )
  flap_rectangles = (
      [rectangle_lib.Rectangle.from_dict(r) for r in data['flap_rectangles']]
      if data.get('flap_rectangles') is not None
      else None
  )
  if data.get('packing_rivers') is not None:
    packing_rivers = {
        name: packing_river_from_dict(r_dict)
        for name, r_dict in data['packing_rivers'].items()
    }
  else:
    packing_rivers = {}
  stick_figure = (
      stick_figure_from_dict(data['stick_figure'])
      if data.get('stick_figure') is not None
      else None
  )
  hinges = (
      [_hinge_info_from_dict(h) for h in data['hinges']]
      if data.get('hinges') is not None
      else None
  )
  if data.get('sacrificial_flaps') is not None:
    sacrificial_flaps = [
        _sacrificial_flap_from_dict(sf) for sf in data['sacrificial_flaps']
    ]
  else:
    sacrificial_flaps = None
  config = (
      packing_config_from_dict(data['config'])
      if data.get('config') is not None
      else None
  )
  raw_details = data.get('details')
  return packing_output_lib.PackingOutput(
      success=bool(data['success']),
      packing_cp=packing_cp,
      flap_rectangles=flap_rectangles,
      packing_rivers=packing_rivers,
      stick_figure=stick_figure,
      hinges=hinges,
      sacrificial_flaps=sacrificial_flaps,
      tracker=None,
      details=typing.cast(
          Any, dict(raw_details) if raw_details is not None else None
      ),
      config=config,
      error_message=data.get('error_message'),
  )


def load_legacy_packing_output(
    data: Mapping[str, Any],
    *,
    packing_cp: crease_pattern_lib.CreasePattern | None = None,
    stick_figure: stick_figure_lib.StickFigure | None = None,
) -> packing_output_lib.PackingOutput:
  """Loads a legacy double-encoded PackingOutput JSON dictionary."""

  def _decode_json_field(val: Any) -> Any:
    return json.loads(val) if isinstance(val, (str, bytes)) else val

  raw_flaps = _decode_json_field(data.get('flap_rectangles'))
  flap_rectangles = (
      [rectangle_lib.Rectangle.from_dict(r) for r in raw_flaps]
      if raw_flaps is not None
      else None
  )
  raw_rivers = _decode_json_field(data.get('packing_rivers')) or {}
  packing_rivers: dict[str, river_lib.PackingRiver] = {}
  for name, r_dict in raw_rivers.items():
    if (
        isinstance(r_dict, Mapping)
        and r_dict.get('river_type') == 'straight'
        and ('offset' not in r_dict or 'orientation' not in r_dict)
    ):
      rects = r_dict.get('rectangles') or ()
      if len(rects) == 1 and 'bounds' in rects[0] and 'width' in r_dict:
        rx, ry, rw, rh = (int(v) for v in rects[0]['bounds'])
        width = int(r_dict['width'])
        if rw == width and rh != width:
          orientation = river_lib.RiverOrientation.VERTICAL.value
          offset = rx
        else:
          orientation = river_lib.RiverOrientation.HORIZONTAL.value
          offset = ry
        r_dict = {
            **r_dict,
            'offset': offset,
            'orientation': orientation,
        }
    try:
      packing_rivers[name] = packing_river_from_dict(r_dict)
    except ValueError:
      factory = river_lib.get_reconstructed_river_factory()
      if factory is not None:
        packing_rivers[name] = factory(
            rectangles=[
                rectangle_lib.Rectangle.from_dict(r)
                for r in r_dict.get('rectangles', ())
            ],
            width=int(r_dict.get('width', -1)),
            grid_size=int(r_dict.get('grid_size', -1)),
        )
      else:
        raise
  raw_hinges = _decode_json_field(data.get('hinges'))
  hinges = (
      [_hinge_info_from_dict(h) for h in raw_hinges]
      if raw_hinges is not None
      else None
  )
  raw_sf = _decode_json_field(data.get('sacrificial_flaps'))
  sacrificial_flaps = (
      [packing_types.SacrificialFlap.from_dict(sf) for sf in raw_sf]
      if raw_sf is not None
      else None
  )
  raw_cfg = _decode_json_field(data.get('config'))
  config = None
  if isinstance(raw_cfg, Mapping):
    valid_cfg_fields = {
        f.name
        for f in dataclasses.fields(packing_config_lib.PackingConfig)
        if f.init
    }
    filtered_cfg = {k: v for k, v in raw_cfg.items() if k in valid_cfg_fields}
    if isinstance(filtered_cfg.get('local_tiling_config'), Mapping):
      ltc = dict(filtered_cfg['local_tiling_config'])
      ltc.pop('use_river_tiling', None)
      filtered_cfg['local_tiling_config'] = ltc
    config = packing_config_lib.PackingConfig.from_dict(filtered_cfg)

  raw_details = _decode_json_field(data.get('details'))
  return packing_output_lib.PackingOutput(
      success=bool(data.get('success', True)),
      packing_cp=packing_cp,
      flap_rectangles=flap_rectangles,
      packing_rivers=packing_rivers,
      stick_figure=stick_figure,
      hinges=hinges,
      sacrificial_flaps=sacrificial_flaps,
      tracker=None,
      details=typing.cast(
          Any, dict(raw_details) if raw_details is not None else None
      ),
      config=config,
      error_message=data.get('error_message'),
  )


def solve_to_dict(result: hinge_utils.SolverResult) -> dict[str, Any]:
  return {
      'type': 'SolverResult',
      'schema_version': SCHEMA_VERSION,
      'crease_pattern': crease_pattern_to_dict(result.crease_pattern),
      'final_state': result.final_state.asdict(),
      'is_globally_flat_foldable': result.is_globally_flat_foldable,
      'time_taken': result.time_taken,
      'base_type': result.base_type.value,
  }


def solve_from_dict(data: Mapping[str, Any]) -> hinge_utils.SolverResult:
  required = ('crease_pattern', 'final_state')
  missing = [k for k in required if k not in data]
  if missing:
    raise ValueError(f'Missing required SolverResult keys: {missing}')
  return hinge_utils.SolverResult(
      crease_pattern=crease_pattern_from_dict(data['crease_pattern']),
      final_state=hinge_utils.State.from_dict(dict(data['final_state'])),
      is_globally_flat_foldable=bool(
          data.get('is_globally_flat_foldable', True)
      ),
      time_taken=float(data.get('time_taken', 0.0)),
      base_type=hinge_utils.BaseType(data.get('base_type', 'normal')),
  )


def _grid_point_to_dict(pt: bus.GridPoint) -> dict[str, int]:
  return {'x': int(pt.x), 'y': int(pt.y)}


def _grid_point_from_dict(data: Mapping[str, Any]) -> bus.GridPoint:
  if 'x' not in data or 'y' not in data:
    raise ValueError('GridPoint dict must include required keys "x" and "y".')
  return bus.GridPoint(x=int(data['x']), y=int(data['y']))


def _grid_segment_to_dict(seg: bus.GridSegment) -> dict[str, Any]:
  return {
      'p1': _grid_point_to_dict(seg.p1),
      'p2': _grid_point_to_dict(seg.p2),
  }


def _grid_segment_from_dict(data: Mapping[str, Any]) -> bus.GridSegment:
  if 'p1' not in data or 'p2' not in data:
    raise ValueError(
        'GridSegment dict must include required keys "p1" and "p2".'
    )
  return bus.GridSegment(
      p1=_grid_point_from_dict(data['p1']),
      p2=_grid_point_from_dict(data['p2']),
  ).canonical()


def _grid_rect_to_dict(rect: bus.GridRect) -> dict[str, int]:
  return {
      'x': int(rect.x),
      'y': int(rect.y),
      'width': int(rect.width),
      'height': int(rect.height),
  }


def _grid_rect_from_dict(data: Mapping[str, Any]) -> bus.GridRect:
  required = ('x', 'y', 'width', 'height')
  missing = [k for k in required if k not in data]
  if missing:
    raise ValueError(f'Missing required GridRect keys: {missing}')
  return bus.GridRect(
      x=int(data['x']),
      y=int(data['y']),
      width=int(data['width']),
      height=int(data['height']),
  )


def _flap_to_dict(flap: bus.Flap) -> dict[str, Any]:
  return {
      'id': flap.id,
      'stick': flap.stick,
      'rect': _grid_rect_to_dict(flap.rect),
      'circle_centers': [_grid_point_to_dict(c) for c in flap.circle_centers],
      'flap_type': flap.flap_type.value,
      'tip_shape': flap.tip_shape.value,
      'mirror_of': flap.mirror_of,
  }


def _flap_from_dict(
    data: Mapping[str, Any],
    *,
    grid_size: int | None = None,
) -> bus.Flap:
  """Deserializes a Flap from a dictionary."""
  required = ('id', 'stick', 'rect', 'circle_centers', 'flap_type')
  missing = [k for k in required if k not in data]
  if missing:
    raise ValueError(f'Missing required Flap keys: {missing}')
  tip_val = data.get('tip_shape', stick_lib.FlapTipShape.UNSET.value)
  rect = _grid_rect_from_dict(data['rect'])
  if grid_size is not None:
    rect = rect.clipped(grid_size)
  return bus.Flap(
      id=str(data['id']),
      stick=str(data['stick']),
      rect=rect,
      circle_centers=tuple(
          _grid_point_from_dict(c) for c in data['circle_centers']
      ),
      flap_type=rectangle_lib.FlapType(data['flap_type']),
      tip_shape=stick_lib.FlapTipShape(tip_val),
      mirror_of=data.get('mirror_of'),
  )


def _bus_river_to_dict(river: bus.River) -> dict[str, Any]:
  """Serializes a bus.River into a JSON-compatible dictionary."""
  d: dict[str, Any] = {
      'id': river.id,
      'stick': river.stick,
      'width': int(river.width),
      'kind': river.kind,
      'rects': [_grid_rect_to_dict(r) for r in river.rects],
      'mirror_of': river.mirror_of,
      'offset': int(river.offset) if river.offset is not None else None,
      'orientation': (
          river.orientation.value if river.orientation is not None else None
      ),
      'l_shape_type': (
          river.l_shape_type.value if river.l_shape_type is not None else None
      ),
      'turn': (
          _grid_point_to_dict(river.turn) if river.turn is not None else None
      ),
      'paths': [
          [_grid_point_to_dict(pt) for pt in path] for path in river.paths
      ],
      'hug_side': river.hug_side.value if river.hug_side is not None else None,
      'is_on_axis': river.is_on_axis,
      'variant_type': (
          river.variant_type.value if river.variant_type is not None else None
      ),
  }
  if river.requires_recovery:
    d['requires_recovery'] = True
  return d


def _bus_river_from_dict(
    data: Mapping[str, Any],
    *,
    grid_size: int | None = None,
) -> bus.River:
  """Deserializes a bus River from a dictionary."""
  required = ('id', 'stick', 'width', 'kind', 'rects')
  missing = [k for k in required if k not in data]
  if missing:
    raise ValueError(f'Missing required River keys: {missing}')
  orientation_val = data.get('orientation')
  l_shape_val = data.get('l_shape_type')
  turn_val = data.get('turn')
  hug_side_val = data.get('hug_side')
  variant_val = data.get('variant_type')
  raw_paths = data.get('paths', ())
  rects = tuple(_grid_rect_from_dict(r) for r in data['rects'])
  if grid_size is not None:
    rects = tuple(
        c
        for r in rects
        if (c := r.clipped(grid_size)).width > 0 and c.height > 0
    )
  if data['kind'] == 'hugging':
    rects = bus.coalesce_grid_rects(rects)
  return bus.River(
      id=str(data['id']),
      stick=str(data['stick']),
      width=int(data['width']),
      kind=data['kind'],
      rects=rects,
      mirror_of=data.get('mirror_of'),
      offset=int(data['offset']) if data.get('offset') is not None else None,
      orientation=(
          river_lib.RiverOrientation(orientation_val)
          if orientation_val is not None
          else None
      ),
      l_shape_type=(
          river_lib.LShapeType(l_shape_val) if l_shape_val is not None else None
      ),
      turn=(_grid_point_from_dict(turn_val) if turn_val is not None else None),
      paths=tuple(
          tuple(_grid_point_from_dict(pt) for pt in path) for path in raw_paths
      ),
      hug_side=(
          river_lib.HugSide(hug_side_val) if hug_side_val is not None else None
      ),
      is_on_axis=bool(data.get('is_on_axis', False)),
      variant_type=(
          river_lib.PathVariantType(variant_val)
          if variant_val is not None
          else river_lib.PathVariantType.STANDARD
      ),
      requires_recovery=bool(data.get('requires_recovery', False)),
  )


def _bus_hinge_to_dict(hinge: bus.Hinge) -> dict[str, Any]:
  return {
      'id': hinge.id,
      'segment': _grid_segment_to_dict(hinge.segment),
      'elements': list(hinge.elements),
      'assigned': hinge.assigned,
  }


def _bus_hinge_from_dict(data: Mapping[str, Any]) -> bus.Hinge:
  required = ('id', 'segment', 'elements', 'assigned')
  missing = [k for k in required if k not in data]
  if missing:
    raise ValueError(f'Missing required Hinge keys: {missing}')
  return bus.Hinge(
      id=int(data['id']),
      segment=_grid_segment_from_dict(data['segment']),
      elements=tuple(str(x) for x in data['elements']),
      assigned=bool(data['assigned']),
  )


def _reserved_region_from_dict(
    data: Mapping[str, Any], *, grid_size: int
) -> bus.ReservedRegion:
  if 'id' not in data or 'rect' not in data:
    raise ValueError('ReservedRegion dict requires keys "id" and "rect".')
  return bus.ReservedRegion(
      id=str(data['id']),
      rect=_grid_rect_from_dict(data['rect']).clipped(grid_size),
  )


def packed_cp_to_dict(packed: bus.PackedCreasePattern) -> dict[str, Any]:
  """Serializes a PackedCreasePattern to a dictionary."""
  out: dict[str, Any] = {
      'type': 'PackedCreasePattern',
      'schema_version': SCHEMA_VERSION,
      'design': (
          design_tree_to_dict(packed.design)
          if packed.design is not None
          else None
      ),
      'packed_tree': (
          stick_figure_to_dict(packed.packed_tree)
          if packed.packed_tree is not None
          else None
      ),
      'grid_size': packed.grid_size,
      'creases': crease_pattern_to_dict(packed.creases),
      'flaps': {k: _flap_to_dict(v) for k, v in packed.flaps.items()},
      'rivers': {k: _bus_river_to_dict(v) for k, v in packed.rivers.items()},
      'hinges': [_bus_hinge_to_dict(h) for h in packed.hinges],
      'config': packing_config_to_dict(packed.config),
      'diagnostics': packed.diagnostics.to_tiling_details(),
  }
  if packed.reservations:
    out['reservations'] = [
        {'id': r.id, 'rect': _grid_rect_to_dict(r.rect)}
        for r in packed.reservations
    ]
  if packed.level_shift_reservations:
    out['level_shift_reservations'] = [
        _level_shift_reservation_to_dict(r)
        for r in packed.level_shift_reservations
    ]
  return out


def _level_shift_reservation_to_dict(
    res: bus.LevelShiftReservation,
) -> dict[str, Any]:
  """Serializes a `LevelShiftReservation` to a JSON-compatible dictionary."""
  return {
      'stick': res.stick,
      'region_ids': list(res.region_ids),
  }


def _level_shift_reservation_from_dict(
    data: Mapping[str, Any],
) -> bus.LevelShiftReservation:
  if 'stick' not in data or 'region_ids' not in data:
    raise ValueError(
        'LevelShiftReservation dict requires keys "stick" and "region_ids".'
    )
  return bus.LevelShiftReservation(
      stick=str(data['stick']),
      region_ids=tuple(str(rid) for rid in data['region_ids']),
  )


def _applied_level_shift_to_dict(
    shift: bus.AppliedLevelShift,
) -> dict[str, Any]:
  return {
      'stick': shift.stick,
      'level': shift.level,
      'axial_creases': [
          _grid_segment_to_dict(seg) for seg in shift.axial_creases
      ],
  }


def _applied_level_shift_from_dict(
    data: Mapping[str, Any],
) -> bus.AppliedLevelShift:
  return bus.AppliedLevelShift(
      stick=str(data['stick']),
      level=int(data['level']),
      axial_creases=tuple(
          _grid_segment_from_dict(seg) for seg in data.get('axial_creases', ())
      ),
  )


def packed_cp_from_dict(data: Mapping[str, Any]) -> bus.PackedCreasePattern:
  """Deserializes a PackedCreasePattern from a dictionary."""
  required = (
      'grid_size',
      'creases',
      'flaps',
      'rivers',
      'hinges',
      'config',
      'diagnostics',
  )
  missing = [k for k in required if k not in data]
  if missing:
    raise ValueError(f'Missing required PackedCreasePattern keys: {missing}')
  grid_size = int(data['grid_size'])
  design = (
      design_tree_from_dict(data['design'])
      if data.get('design') is not None
      else None
  )
  packed_tree = (
      stick_figure_from_dict(data['packed_tree'])
      if data.get('packed_tree') is not None
      else None
  )
  creases = crease_pattern_from_dict(data['creases'])
  flaps = immutabledict.immutabledict({
      str(k): _flap_from_dict(v, grid_size=grid_size)
      for k, v in data['flaps'].items()
  })
  rivers = immutabledict.immutabledict({
      str(k): _bus_river_from_dict(v, grid_size=grid_size)
      for k, v in data['rivers'].items()
  })
  raw_hinges = tuple(_bus_hinge_from_dict(h) for h in data['hinges'])
  hinges = bus.filter_and_merge_hinges(raw_hinges, creases, grid_size)
  reservations = tuple(
      _reserved_region_from_dict(r, grid_size=grid_size)
      for r in data.get('reservations') or ()
  )
  ls_reservations = tuple(
      _level_shift_reservation_from_dict(r)
      for r in data.get('level_shift_reservations') or ()
  )
  return bus.PackedCreasePattern(
      design=design,
      packed_tree=packed_tree,
      grid_size=grid_size,
      creases=creases,
      flaps=flaps,
      rivers=rivers,
      hinges=hinges,
      config=packing_config_from_dict(data['config']),
      diagnostics=bus.PackingDiagnostics.from_tiling_details(
          data['diagnostics'], default_grid_size=grid_size
      ),
      reservations=reservations,
      level_shift_reservations=ls_reservations,
  )


def solved_cp_to_dict(solved: bus.SolvedCreasePattern) -> dict[str, Any]:
  """Serializes a `SolvedCreasePattern` to a JSON-compatible dictionary."""
  out: dict[str, Any] = {
      'type': 'SolvedCreasePattern',
      'schema_version': SCHEMA_VERSION,
      'packed': packed_cp_to_dict(solved.packed),
      'creases': crease_pattern_to_dict(solved.creases),
      'scale_factor': solved.scale_factor,
      'base_type': solved.base_type.value,
      'central_crease_removed': solved.central_crease_removed,
      'symmetry': solved.symmetry.name,
      'config': solver_config_to_dict(solved.config),
      'is_globally_flat_foldable': solved.is_globally_flat_foldable,
      'stats': {
          'time_taken': solved.stats.time_taken,
          'state_score': solved.stats.state_score,
          'partition_progress': solved.stats.partition_progress,
          'symmetry_score': solved.stats.symmetry_score,
      },
      'assigned_hinges': [c.asdict() for c in solved.assigned_hinges],
      'assigned_pleats': [c.asdict() for c in solved.assigned_pleats],
  }
  if solved.applied_level_shifts:
    out['applied_level_shifts'] = [
        _applied_level_shift_to_dict(s) for s in solved.applied_level_shifts
    ]
  return out


def solved_cp_from_dict(data: Mapping[str, Any]) -> bus.SolvedCreasePattern:
  """Deserializes a SolvedCreasePattern from a dictionary."""
  required = (
      'packed',
      'creases',
      'scale_factor',
      'base_type',
      'central_crease_removed',
      'symmetry',
      'config',
      'is_globally_flat_foldable',
      'stats',
  )
  missing = [k for k in required if k not in data]
  if missing:
    raise ValueError(f'Missing required SolvedCreasePattern keys: {missing}')
  raw_stats = data['stats']
  return bus.SolvedCreasePattern(
      packed=packed_cp_from_dict(data['packed']),
      creases=crease_pattern_from_dict(data['creases']),
      scale_factor=int(data['scale_factor']),
      base_type=bus.BaseType(data['base_type']),
      central_crease_removed=bool(data['central_crease_removed']),
      symmetry=symmetry.Transform[data['symmetry']],
      config=solver_config_from_dict(data['config']),
      is_globally_flat_foldable=bool(data['is_globally_flat_foldable']),
      stats=bus.SolveStats(
          time_taken=float(raw_stats.get('time_taken', 0.0)),
          state_score=int(raw_stats.get('state_score', 0)),
          partition_progress=int(raw_stats.get('partition_progress', 0)),
          symmetry_score=float(raw_stats.get('symmetry_score', 0.0)),
      ),
      assigned_hinges=tuple(
          crease.Crease.from_dict(c) for c in data.get('assigned_hinges', ())
      ),
      assigned_pleats=tuple(
          crease.Crease.from_dict(c) for c in data.get('assigned_pleats', ())
      ),
      applied_level_shifts=tuple(
          _applied_level_shift_from_dict(s)
          for s in data.get('applied_level_shifts') or ()
      ),
  )


def shaped_cp_to_dict(shaped: bus.ShapedCreasePattern) -> dict[str, Any]:
  return {
      'type': 'ShapedCreasePattern',
      'schema_version': SCHEMA_VERSION,
      'solved': solved_cp_to_dict(shaped.solved),
      'creases': crease_pattern_to_dict(shaped.creases),
      'diagnostics': dict(shaped.diagnostics),
      'is_locally_flat_foldable': shaped.is_locally_flat_foldable,
      'strategy': shaped.strategy.value,
      'time_taken': shaped.time_taken,
  }


def shaped_cp_from_dict(data: Mapping[str, Any]) -> bus.ShapedCreasePattern:
  required = ('solved', 'creases')
  missing = [k for k in required if k not in data]
  if missing:
    raise ValueError(f'Missing required ShapedCreasePattern keys: {missing}')
  raw_diagnostics = data.get('diagnostics', {})
  return bus.ShapedCreasePattern(
      solved=solved_cp_from_dict(data['solved']),
      creases=crease_pattern_from_dict(data['creases']),
      diagnostics=immutabledict.immutabledict(dict(raw_diagnostics)),
      is_locally_flat_foldable=bool(data.get('is_locally_flat_foldable', True)),
      strategy=bus.ShapingStrategy(
          str(data.get('strategy', bus.ShapingStrategy.DEFAULT.value))
      ),
      time_taken=float(data.get('time_taken', 0.0)),
  )


def to_dict(obj: Any) -> dict[str, Any]:
  """Serializes a domain object into a JSON-serializable dictionary."""
  match obj:
    case bus.DesignTree():
      res = design_tree_to_dict(obj)
    case bus.PackedCreasePattern():
      res = packed_cp_to_dict(obj)
    case bus.SolvedCreasePattern():
      res = solved_cp_to_dict(obj)
    case bus.ShapedCreasePattern():
      res = shaped_cp_to_dict(obj)
    case stick_figure_lib.StickFigure():
      res = stick_figure_to_dict(obj)
    case crease_pattern_lib.CreasePattern():
      res = crease_pattern_to_dict(obj)
    case packing_config_lib.PackingConfig():
      res = packing_config_to_dict(obj)
    case hinge_utils.SolverConfig():
      res = solver_config_to_dict(obj)
    case packing_output_lib.PackingOutput():
      res = pack_to_dict(obj)
    case hinge_utils.SolverResult():
      res = solve_to_dict(obj)
    case _:
      raise TypeError(f'Unsupported object type for codec.to_dict: {type(obj)}')
  return _normalize_json_value(res)


def from_dict(data: Mapping[str, Any]) -> Any:
  if 'schema_version' not in data:
    raise ValueError(
        'Missing required "schema_version" field in codec.from_dict.'
    )
  version = data['schema_version']
  if version != SCHEMA_VERSION:
    raise ValueError(
        f'Unsupported schema_version {version!r}; expected {SCHEMA_VERSION!r}.'
    )
  obj_type = data.get('type')
  match obj_type:
    case 'DesignTree':
      return design_tree_from_dict(data)
    case 'PackedCreasePattern':
      return packed_cp_from_dict(data)
    case 'SolvedCreasePattern':
      return solved_cp_from_dict(data)
    case 'ShapedCreasePattern':
      return shaped_cp_from_dict(data)
    case 'StickFigure':
      return stick_figure_from_dict(data)
    case 'CreasePattern':
      return crease_pattern_from_dict(data)
    case 'PackingConfig':
      return packing_config_from_dict(data)
    case 'SolverConfig':
      return solver_config_from_dict(data)
    case 'PackingOutput':
      return pack_from_dict(data)
    case 'SolverResult':
      return solve_from_dict(data)
    case _:
      raise ValueError(
          f'Unknown or missing type tag in codec.from_dict: {obj_type!r}'
      )


def dumps(obj: Any, *, indent: int | None = None) -> str:
  return json.dumps(to_dict(obj), indent=indent)


def loads(s: str | bytes) -> Any:
  return from_dict(json.loads(s))
