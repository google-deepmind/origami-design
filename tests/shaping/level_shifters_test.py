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

"""Tests for width-2 sacrificial-pocket level shifters and axial elevations."""

import dataclasses
from typing import Any

from absl.testing import absltest
import bp_pipeline as bp
from bp_pipeline import bus
from bp_pipeline import codec
from bp_pipeline import pipeline
from bp_pipeline import plugins
from bp_pipeline.crease_pattern import crease as crease_lib
from bp_pipeline.crease_pattern import crease_pattern as cp_lib
from bp_pipeline.crease_pattern import point as point_lib
from bp_pipeline.packing import level_shifter as packing_ls
from bp_pipeline.packing import utils_packing_flaps
from bp_pipeline.shaping import toolbox
from bp_pipeline.solving import hinges as hinge_utils
from bp_pipeline.solving import level_shifters


def _wide_flap_tree(
    *, flap_width: int = 2, flap_length: int = 4
) -> dict[str, Any]:
  """Symmetric tree with width-`flap_width` wings for flap pocket tests."""
  return {
      'name': f'wide_flap_w{flap_width}',
      'category': 'animals',
      'complexity': 2,
      'color_top': [210, 180, 140],
      'color_bottom': [240, 230, 210],
      'root': 'body',
      'children': [
          {
              'name': 'body',
              'start': 'pelvis',
              'end': 'chest',
              'azimuth_angle': 0.0,
              'elevation_angle': 0.0,
              'length': 2,
              'width': 1,
              'shaping_flap': False,
          },
          {
              'name': 'head',
              'start': 'chest',
              'end': 'head_tip',
              'azimuth_angle': 0.0,
              'elevation_angle': 20.0,
              'length': 2,
              'width': 1,
              'shaping_flap': False,
          },
          {
              'name': 'tail',
              'start': 'pelvis',
              'end': 'tail_tip',
              'azimuth_angle': 180.0,
              'elevation_angle': 0.0,
              'length': 2,
              'width': 1,
              'shaping_flap': False,
          },
          {
              'name': 'left_wing',
              'start': 'chest',
              'end': 'left_wing_tip',
              'azimuth_angle': 90.0,
              'elevation_angle': 0.0,
              'length': flap_length,
              'width': flap_width,
              'shaping_flap': False,
          },
          {
              'name': 'right_wing',
              'start': 'chest',
              'end': 'right_wing_tip',
              'azimuth_angle': -90.0,
              'elevation_angle': 0.0,
              'length': flap_length,
              'width': flap_width,
              'shaping_flap': False,
          },
          {
              'name': 'left_leg',
              'start': 'pelvis',
              'end': 'left_leg_tip',
              'azimuth_angle': 120.0,
              'elevation_angle': -60.0,
              'length': 2,
              'width': 1,
              'shaping_flap': False,
          },
          {
              'name': 'right_leg',
              'start': 'pelvis',
              'end': 'right_leg_tip',
              'azimuth_angle': -120.0,
              'elevation_angle': -60.0,
              'length': 2,
              'width': 1,
              'shaping_flap': False,
          },
      ],
  }


def _wide_river_tree(*, river_width: int = 2) -> dict[str, Any]:
  """Symmetric tree with a width-`river_width` body river for pocket tests."""
  return {
      'name': f'wide_river_w{river_width}',
      'category': 'animals',
      'complexity': 2,
      'color_top': [190, 170, 140],
      'color_bottom': [235, 225, 205],
      'root': 'body',
      'children': [
          {
              'name': 'body',
              'start': 'pelvis',
              'end': 'chest',
              'azimuth_angle': 0.0,
              'elevation_angle': 0.0,
              'length': 5,
              'width': river_width,
              'shaping_flap': False,
          },
          {
              'name': 'head',
              'start': 'chest',
              'end': 'head_tip',
              'azimuth_angle': 0.0,
              'elevation_angle': 25.0,
              'length': 2,
              'width': 2,
              'shaping_flap': False,
          },
          {
              'name': 'tail',
              'start': 'pelvis',
              'end': 'tail_tip',
              'azimuth_angle': 180.0,
              'elevation_angle': 0.0,
              'length': 2,
              'width': 2,
              'shaping_flap': False,
          },
          {
              'name': 'left_arm',
              'start': 'chest',
              'end': 'left_arm_tip',
              'azimuth_angle': 90.0,
              'elevation_angle': -45.0,
              'length': 2,
              'width': 1,
              'shaping_flap': False,
          },
          {
              'name': 'right_arm',
              'start': 'chest',
              'end': 'right_arm_tip',
              'azimuth_angle': -90.0,
              'elevation_angle': -45.0,
              'length': 2,
              'width': 1,
              'shaping_flap': False,
          },
          {
              'name': 'left_leg',
              'start': 'pelvis',
              'end': 'left_leg_tip',
              'azimuth_angle': 90.0,
              'elevation_angle': -60.0,
              'length': 2,
              'width': 1,
              'shaping_flap': False,
          },
          {
              'name': 'right_leg',
              'start': 'pelvis',
              'end': 'right_leg_tip',
              'azimuth_angle': -90.0,
              'elevation_angle': -60.0,
              'length': 2,
              'width': 1,
              'shaping_flap': False,
          },
      ],
  }


class LevelShiftersTest(absltest.TestCase):

  def test_pocket_sizing_unified_and_width3_rejected(self):
    self.assertEqual(packing_ls.pocket_depth(1), 0)
    self.assertEqual(packing_ls.pocket_margin(1), 0)
    self.assertEqual(packing_ls.pocket_depth(2), 2)
    self.assertEqual(packing_ls.pocket_margin(2), 1)
    self.assertIs(
        packing_ls.pocket_depth,
        utils_packing_flaps.level_shifter_pocket_depth,
    )
    self.assertIs(
        packing_ls.pocket_margin,
        utils_packing_flaps.level_shifter_center_margin,
    )
    with self.assertRaisesRegex(ValueError, 'width == 2'):
      packing_ls.pocket_depth(3)
    with self.assertRaisesRegex(ValueError, 'width == 2'):
      packing_ls.pocket_margin(3)

    design_w3 = bp.prepare(_wide_flap_tree(flap_width=3))
    with self.assertRaisesRegex(
        (ValueError, bp.PackingFailedError), 'width == 2'
    ):
      bp.pack(
          design_w3,
          bp.PackingTaskConfig(
              packing_config=bp.PackingConfig(use_level_shifter=True),
              max_retries=2,
          ),
      )

  def test_design_tree_remains_packer_neutral(self):
    sf = bp.StickFigure.from_dict(_wide_flap_tree(flap_width=2))
    d1 = bus.DesignTree.from_stick_figure(sf)
    d2 = bp.prepare(sf)
    self.assertEqual(d1.min_grid_size, d2.min_grid_size)

  def test_compute_axial_elevations_exact_on_s_fold_and_canonical_examples(
      self,
  ):
    # 1. Manual 3-panel S-fold (grid_size=3): x=1 MOUNTAIN, x=2 VALLEY.
    # Panel [0,1] stays at [0,1], [1,2] folds to [1,0], [2,3] folds to [0,1].
    # Axial x=1 folds to 1 -> elevation=1; axial x=2 folds to 0 -> elevation=0.
    g = 3
    creases = [
        crease_lib.Crease(
            point_lib.Point(x=0.0, y=0.0),
            point_lib.Point(x=1.0, y=0.0),
            fold=crease_lib.Fold.BORDER,
        ),
        crease_lib.Crease(
            point_lib.Point(x=1.0, y=0.0),
            point_lib.Point(x=1.0, y=1.0),
            fold=crease_lib.Fold.BORDER,
        ),
        crease_lib.Crease(
            point_lib.Point(x=1.0, y=1.0),
            point_lib.Point(x=0.0, y=1.0),
            fold=crease_lib.Fold.BORDER,
        ),
        crease_lib.Crease(
            point_lib.Point(x=0.0, y=1.0),
            point_lib.Point(x=0.0, y=0.0),
            fold=crease_lib.Fold.BORDER,
        ),
        crease_lib.Crease(
            point_lib.Point(x=1.0 / g, y=0.0),
            point_lib.Point(x=1.0 / g, y=1.0),
            fold=crease_lib.Fold.MOUNTAIN,
        ),
        crease_lib.Crease(
            point_lib.Point(x=2.0 / g, y=0.0),
            point_lib.Point(x=2.0 / g, y=1.0),
            fold=crease_lib.Fold.VALLEY,
        ),
    ]
    s_cp = cp_lib.CreasePattern(creases=creases, grid_size=g, light_mode=True)
    elev_cp = level_shifters.compute_axial_elevations(s_cp)
    by_x = {
        round(c.p1.x * g): c.elevation
        for c in elev_cp.creases
        if c.mountain_or_valley()
    }
    self.assertEqual(by_x, {1: 1, 2: 0})
    for c in elev_cp.creases:
      if c.fold == crease_lib.Fold.BORDER:
        self.assertIsNone(c.elevation)

    # 2. Full uniaxial base (`remove_central_crease=False`) with all elevations
    # stripped beforehand: recovers exact integer elevations on every axial
    # crease and leaves `None` on all hinges, ridges, and borders.
    human_packed = bp.pack(bp.prepare('human'))
    human_solved = bp.solve_best(
        human_packed,
        bp.SolverConfig(
            max_steps=2000,
            max_time_limit=20.0,
            remove_central_crease=False,
        ),
    )
    uniaxial_ref = human_solved.creases
    stripped_human = cp_lib.CreasePattern(
        creases=[
            dataclasses.replace(c, elevation=None)
            for c in uniaxial_ref.creases
        ],
        grid_size=uniaxial_ref.grid_size,
        light_mode=True,
    )
    recomputed_human = level_shifters.compute_axial_elevations(stripped_human)
    for orig, got in zip(
        uniaxial_ref.creases, recomputed_human.creases, strict=True
    ):
      if orig.elevation is None:
        self.assertIsNone(got.elevation)
      else:
        self.assertIn(got.elevation, (0, 1))
        self.assertEqual(got.elevation, orig.elevation)

    # Ordinary packed and solved codec dicts omit empty level-shifter keys.
    encoded_human_packed = codec.to_dict(human_packed)
    self.assertNotIn('reservations', encoded_human_packed)
    self.assertNotIn('level_shift_reservations', encoded_human_packed)
    encoded_human_solved = codec.to_dict(human_solved)
    self.assertNotIn('applied_level_shifts', encoded_human_solved)
    self.assertNotIn('template_creases', encoded_human_solved)

  def test_wide_flap_pocket_solve_codec_and_direction_actuation(self):
    design = bp.prepare(_wide_flap_tree(flap_width=2, flap_length=6))
    packed = bp.pack(
        design,
        bp.PackingTaskConfig(
            packing_config=bp.PackingConfig(
                use_level_shifter=True,
                enforce_proximity_constraints=False,
            ),
            max_retries=6,
            max_cps_per_grid_size=2,
            max_number_of_cps=2,
            sweep_timeout_seconds=30.0,
        ),
        packer='builtin',
    )
    self.assertNotEmpty(packed.reservations)
    self.assertNotEmpty(packed.level_shift_reservations)
    # Pocket polygons live in `packed.reservations`, never in `packed.flaps`.
    self.assertEqual(
        set(packed.flaps.keys()),
        {s.name for s in design.merged.children if s.is_flap},
    )
    res_ids = {r.id for r in packed.reservations}
    for lsr in packed.level_shift_reservations:
      self.assertTrue(set(lsr.region_ids).issubset(res_ids))
    reserved_sticks = {r.stick for r in packed.level_shift_reservations}
    self.assertLessEqual({'left_wing', 'right_wing'}, reserved_sticks)

    # `.bps` export succeeds on pocket packings and emits `reservations`.
    bps_doc = bp.to_bp_studio(packed)
    self.assertNotEmpty(bps_doc['design']['layout'].get('reservations', ()))

    # Round-trip PackedCreasePattern through codec JSON before solving!
    self.assertEqual(
        {s.name for s in packed.packed_tree.children},
        {s.name for s in design.merged.children},
    )
    bps_no_design = bp.to_bp_studio(dataclasses.replace(packed, design=None))
    self.assertLen(
        bps_no_design['design']['tree']['edges'], len(design.merged.children)
    )
    packed_rt = bp.loads(bp.dumps(packed))
    self.assertIsInstance(packed_rt, bp.PackedCreasePattern)
    self.assertEqual(packed_rt.reservations, packed.reservations)
    self.assertEqual(
        packed_rt.level_shift_reservations, packed.level_shift_reservations
    )

    # `remove_central_crease=True` is explicitly rejected with a named error.
    with self.assertRaisesRegex(
        bp.SolvingFailedError, 'remove_central_crease=True'
    ):
      bp.solve_best(
          packed_rt,
          bp.SolverConfig(remove_central_crease=True),
          solver='builtin',
      )

    solved = bp.solve_best(
        packed_rt,
        bp.SolverConfig(
            max_steps=2000,
            max_time_limit=20.0,
            max_number_of_cps=2,
        ),
        record_stages=True,
        solver='builtin',
    )
    self.assertTrue(solved.is_globally_flat_foldable)
    # `assigned_hinges` and `assigned_pleats` describe the widened base.
    solved_exact = {c.exact_match for c in solved.creases.creases}
    self.assertTrue(
        {c.exact_match for c in solved.assigned_hinges}.issubset(solved_exact)
    )
    self.assertTrue(
        {c.exact_match for c in solved.assigned_pleats}.issubset(solved_exact)
    )
    self.assertTrue(
        all(c.elevation is not None for c in solved.assigned_pleats)
    )
    # `record_stages=True` records the 7 solver stages + template stage.
    stage_names = [s.name for s in solved.stages]
    self.assertIn('(1) Packing CP', stage_names)
    self.assertIn('(7) Final Solving CP', stage_names)
    self.assertIn(bus.LS_TEMPLATE_STAGE_NAME, stage_names)
    self.assertLen(solved.stages, 8)
    # 3D mesh maps pocket cells to `'reserved'` regions rather than bogus flaps.
    fold_3d = bp.build_folded_mesh_3d(solved).to_fold_dict('wide_flap')
    for region in fold_3d['file_bp:regions']:
      if region['kind'] == 'reserved':
        self.assertEmpty(region['sticks'])
      else:
        for s_name in region['sticks']:
          self.assertIn(s_name, {s.name for s in design.source.children})
    self.assertTrue(
        hinge_utils.valid_crease_pattern(
            solved.creases,
            solved.grid_size,
            local_flat_foldability=True,
            global_flat_foldability=True,
        )
    )
    flat_dict = solved.flat_fold()
    self.assertIn('vertices_coords', flat_dict)
    self.assertNotEmpty(solved.applied_level_shifts)
    shifted_sticks = {s.stick for s in solved.applied_level_shifts}
    self.assertEqual(shifted_sticks, {'left_wing', 'right_wing'})

    # Verify non-shifted axial creases remain strictly in {0, 1}, and borders /
    # diagonal ridges keep elevation=None.
    shifted_seg_keys: set[tuple[int, int, int, int]] = set()
    for s in solved.applied_level_shifts:
      for seg in s.axial_creases:
        shifted_seg_keys.add((seg.p1.x, seg.p1.y, seg.p2.x, seg.p2.y))
    for c in solved.creases.creases:
      if c.fold == crease_lib.Fold.BORDER:
        self.assertIsNone(c.elevation)
      elif not c.is_horizontal and not c.is_vertical:
        self.assertIsNone(c.elevation)
      elif c.elevation is not None:
        seg = bus.GridSegment.from_normalized_crease(c, solved.grid_size)
        fwd = (seg.p1.x, seg.p1.y, seg.p2.x, seg.p2.y)
        rev = (seg.p2.x, seg.p2.y, seg.p1.x, seg.p1.y)
        if fwd not in shifted_seg_keys and rev not in shifted_seg_keys:
          self.assertIn(c.elevation, (0, 1))

    # Round-trip SolvedCreasePattern through codec JSON (no `template_creases`
    # bloat) and verify re-positioning requires `record_stages=True`.
    solved_dict = codec.to_dict(solved)
    self.assertNotIn('template_creases', solved_dict)
    solved_rt = bp.loads(bp.dumps(solved))
    self.assertIsInstance(solved_rt, bp.SolvedCreasePattern)
    self.assertEqual(
        solved_rt.applied_level_shifts, solved.applied_level_shifts
    )
    state_no_stages = bp.ShapingState.from_solved(solved_rt)
    _, res_need_stages = bp.apply_shaping_tool(
        state_no_stages, 'level_shifter', {'element_name': 'left_wing'}
    )
    self.assertFalse(res_need_stages.ok)
    self.assertIn('record_stages=True', res_need_stages.message)

    # With `record_stages=True`, `from_solved` loads `'LS Template CP'`.
    state = bp.ShapingState.from_solved(solved)
    self.assertIsNotNone(state.ls_template_pattern)
    probe_report = bp.probe_shaping(state, ['left_wing', 'head'])
    self.assertIn('level_shifter', probe_report['left_wing']['available_tools'])
    self.assertNotIn('level_shifter', probe_report['head']['available_tools'])

    def _interior_axial_elevations(
        cp: cp_lib.CreasePattern, stick_name: str
    ) -> set[int]:
      rect = packed.flaps[stick_name].to_rectangle().normalize_rectangle(
          solved.grid_size
      )
      vals: set[int] = set()
      for c in cp.creases:
        if (
            c.mountain_or_valley()
            and c.elevation is not None
            and rect.crease_is_within(c, strict=False)
        ):
          vals.add(int(c.elevation))
      return vals

    for unshifted in ('head', 'tail', 'left_leg', 'right_leg'):
      self.assertLessEqual(
          _interior_axial_elevations(solved.creases, unshifted), {0, 1}
      )

    natural_level = solved.applied_level_shifts[0].level
    self.assertIn(natural_level, (-1, 2))
    valid_dir = 'up' if natural_level == 2 else 'down'
    trapped_dir = 'down' if natural_level == 2 else 'up'
    opposite_level = -1 if natural_level == 2 else 2

    state_a, res_a = bp.apply_shaping_tool(
        state,
        'level_shifter',
        {
            'element_name': 'left_wing',
            'direction': valid_dir,
            'position_percent': 0.0,
        },
    )
    self.assertTrue(res_a.ok, msg=res_a.message)
    self.assertIn(f'levels=[{natural_level}]', res_a.message)
    self.assertTrue(
        hinge_utils.valid_crease_pattern(
            state_a.pattern,
            solved.grid_size,
            local_flat_foldability=True,
            global_flat_foldability=True,
        )
    )
    for wing in ('left_wing', 'right_wing'):
      wing_vals = _interior_axial_elevations(state_a.pattern, wing)
      self.assertIn(natural_level, wing_vals)
      self.assertNotIn(opposite_level, wing_vals)
    for unshifted in ('head', 'tail', 'left_leg', 'right_leg'):
      self.assertLessEqual(
          _interior_axial_elevations(state_a.pattern, unshifted), {0, 1}
      )

    state_b, res_b = bp.apply_shaping_tool(
        state_a,
        'apply_symmetric',
        {
            'tool': 'level_shifter',
            'element_name': 'left_wing',
            'arguments': {'direction': valid_dir, 'position_percent': 1.0},
        },
    )
    self.assertTrue(res_b.ok, msg=res_b.message)
    self.assertTrue(
        hinge_utils.valid_crease_pattern(
            state_b.pattern,
            solved.grid_size,
            local_flat_foldability=True,
            global_flat_foldability=True,
        )
    )

    # Trapped opposite pocket parity must fail global flat-foldability cleanly
    # rather than silently falling back to an untargeted edge shift.
    _, res_trapped = bp.apply_shaping_tool(
        state,
        'level_shifter',
        {'element_name': 'left_wing', 'direction': trapped_dir},
    )
    self.assertFalse(res_trapped.ok)
    self.assertIn('flat-foldability', res_trapped.message)

    # Inverting the LS_TEMPLATE M/V polarity reverses the [0, 1] layer stack
    # and exercises the opposite shift direction (`+2` vs `-1`) end-to-end.
    assert state.ls_template_pattern is not None
    flipped_creases: list[crease_lib.Crease] = []
    for c in state.ls_template_pattern.creases:
      if c.mountain_or_valley():
        new_el = 1 - int(c.elevation) if c.elevation in (0, 1) else None
        flipped_creases.append(
            dataclasses.replace(c, fold=c.fold.flip(), elevation=new_el)
        )
      else:
        flipped_creases.append(c)
    flipped_tpl = cp_lib.CreasePattern(
        creases=flipped_creases,
        grid_size=solved.grid_size,
        light_mode=True,
    )
    flipped_tpl = level_shifters.compute_axial_elevations(flipped_tpl)
    state_inv = state.replace(ls_template_pattern=flipped_tpl)
    state_opp, res_opp = bp.apply_shaping_tool(
        state_inv,
        'level_shifter',
        {
            'element_name': 'left_wing',
            'direction': trapped_dir,
            'position_percent': 0.5,
        },
    )
    self.assertTrue(res_opp.ok, msg=res_opp.message)
    self.assertIn(f'levels=[{opposite_level}]', res_opp.message)
    self.assertTrue(
        hinge_utils.valid_crease_pattern(
            state_opp.pattern,
            solved.grid_size,
            local_flat_foldability=True,
            global_flat_foldability=True,
        )
    )
    for wing in ('left_wing', 'right_wing'):
      wing_opp = _interior_axial_elevations(state_opp.pattern, wing)
      self.assertIn(opposite_level, wing_opp)
      self.assertNotIn(natural_level, wing_opp)
    for unshifted in ('head', 'tail', 'left_leg', 'right_leg'):
      self.assertLessEqual(
          _interior_axial_elevations(state_opp.pattern, unshifted), {0, 1}
      )

  def test_wide_river_pocket_end_to_end_via_pipeline(self):
    sf = bp.StickFigure.from_dict(_wide_river_tree(river_width=2))
    design = bp.prepare(sf)
    pack_res = pipeline.pack_crease_pattern(
        design,
        config=bp.PackingTaskConfig(
            packing_config=bp.PackingConfig(
                use_level_shifter=True,
                enforce_proximity_constraints=False,
            ),
            max_retries=6,
            max_cps_per_grid_size=2,
            max_number_of_cps=2,
            sweep_timeout_seconds=30.0,
        ),
    )
    self.assertNotEmpty(pack_res.packing_outputs)
    solve_res = pipeline.solve_crease_pattern(
        pack_res,
        config=bp.SolvingTaskConfig(
            max_steps=2000,
            max_number_of_cps=2,
            stop_on_first_solve=True,
            sweep_timeout_seconds=30.0,
        ),
    )
    self.assertNotEmpty(solve_res.solver_outputs)
    solved_out = solve_res.solver_outputs[0]
    self.assertTrue(solved_out.is_globally_flat_foldable)
    self.assertEqual(solved_out.base_type, hinge_utils.BaseType.LEVEL_SHIFTED)
    packed_river = bp.PackedCreasePattern.from_packing_output(
        pack_res.packing_outputs[0],
        design=design,
    )
    solved_river = bp.solve_best(
        packed_river,
        bp.SolverConfig(max_steps=2000, max_time_limit=20.0),
        solver='builtin',
    )
    self.assertTrue(solved_river.is_globally_flat_foldable)
    river_exact = {c.exact_match for c in solved_river.creases.creases}
    self.assertTrue(
        {c.exact_match for c in solved_river.assigned_hinges}.issubset(
            river_exact
        )
    )
    # Every axial crease on the widened river base carries an exact elevation.
    bare_river = cp_lib.CreasePattern(
        creases=[
            dataclasses.replace(c, elevation=None)
            for c in solved_river.creases.creases
        ],
        grid_size=solved_river.grid_size,
        light_mode=True,
    )
    recomputed_river = level_shifters.compute_axial_elevations(bare_river)
    for got, ref in zip(
        solved_river.creases.creases, recomputed_river.creases, strict=True
    ):
      self.assertEqual(got.elevation, ref.elevation)
    self.assertTrue(
        hinge_utils.valid_crease_pattern(
            solved_river.creases,
            solved_river.grid_size,
            local_flat_foldability=True,
            global_flat_foldability=True,
        )
    )
    self.assertEqual(
        [s.stick for s in solved_river.applied_level_shifts], ['body']
    )
    self.assertIn('vertices_coords', solved_river.flat_fold())

  def test_non_builtin_packer_and_solver_plugins_rejected_on_pocket_packing(
      self,
  ):
    design = bp.prepare(_wide_flap_tree(flap_width=2, flap_length=4))

    def _dummy_packer(d, **kwargs):
      del d, kwargs
      return None

    plugins.register_packer('dummy_ls_test_packer', _dummy_packer)
    with self.assertRaisesRegex(bp.PackingFailedError, "packer='builtin'"):
      bp.pack(
          design,
          bp.PackingTaskConfig(
              packing_config=bp.PackingConfig(use_level_shifter=True),
          ),
          packer='dummy_ls_test_packer',
      )

    packed = bp.pack(
        design,
        bp.PackingTaskConfig(
            packing_config=bp.PackingConfig(
                use_level_shifter=True,
                enforce_proximity_constraints=False,
            ),
            max_retries=5,
            max_cps_per_grid_size=1,
            max_number_of_cps=1,
        ),
        packer='builtin',
    )

    def _dummy_solver(p, cfg):
      del p, cfg
      return (), None

    plugins.register_solver('dummy_ls_test_solver', _dummy_solver)
    with self.assertRaisesRegex(bp.SolvingFailedError, "solver='builtin'"):
      bp.solve_best(packed, solver='dummy_ls_test_solver')

  def test_level_shifter_tool_rejects_unpacked_state(self):
    normal_res = bp.fold('human', preset='demo', shape=False)
    st = toolbox.ShapingState.from_solved(normal_res.solved)
    _, res = toolbox.apply_tool(
        st, 'level_shifter', {'element_name': 'left_arm'}
    )
    self.assertFalse(res.ok)
    self.assertIn('use_level_shifter=True', res.message)


if __name__ == '__main__':
  absltest.main()
