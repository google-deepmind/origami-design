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

import collections
import copy
import dataclasses
import datetime
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from bp_pipeline import timeout
from bp_pipeline import utils_tree
from bp_pipeline.packing import config
from bp_pipeline.packing import river as river_lib
from bp_pipeline.packing import river_adjacency
from bp_pipeline.packing import scoring
from bp_pipeline.packing import types
from bp_pipeline.packing import utils_packing_flaps
from bp_pipeline.packing import utils_river_solver
from bp_pipeline.packing import utils_tracker
from bp_pipeline.packing.debug import debug_types
from bp_pipeline.stick_figures import packing_utils
from bp_pipeline.stick_figures import stick as stick_lib
from bp_pipeline.stick_figures import stick_figure as stick_figure_lib
from bp_pipeline.stick_figures.examples import example_trees
import numpy as np


def _create_mock_candidate(stick_name):
  """Creates a mock river candidate for testing."""
  mock_river = mock.MagicMock(spec=river_lib.River)
  mock_river.grid_size = 10
  mock_river.width = 1
  mock_river.offset = 0
  mock_river.orientation = river_lib.RiverOrientation.HORIZONTAL
  return {
      'stick_name': stick_name,
      'river': mock_river,
      'packed_flaps': {},
  }


class RiverSolverTest(absltest.TestCase):

  def test_find_possible_straight_river_placements(self):
    rivers = utils_river_solver.find_possible_straight_river_placements(
        grid_size=5, width=2, is_symmetric=False
    )
    self.assertLen(rivers, 4)
    self.assertNotIn(0, [r.offset for r in rivers])
    self.assertCountEqual(
        [r.orientation for r in rivers],
        [
            river_lib.RiverOrientation.HORIZONTAL,
            river_lib.RiverOrientation.HORIZONTAL,
            river_lib.RiverOrientation.VERTICAL,
            river_lib.RiverOrientation.VERTICAL,
        ],
    )

  def test_find_possible_straight_river_placements_symmetric(self):
    rivers = utils_river_solver.find_possible_straight_river_placements(
        grid_size=5, width=2, is_symmetric=True
    )
    self.assertLen(rivers, 2)
    self.assertNotIn(0, [r.offset for r in rivers])
    self.assertNotIn(
        river_lib.RiverOrientation.VERTICAL, [r.orientation for r in rivers]
    )

  def test_find_possible_straight_river_placements_too_big(self):
    rivers = utils_river_solver.find_possible_straight_river_placements(
        grid_size=2, width=10, is_symmetric=False
    )
    self.assertEmpty(rivers)

  def test_validate_river_candidate_valid(self):
    grid = np.zeros((10, 10), dtype=bool)
    river = river_lib.River(
        width=4,
        offset=1,
        orientation=river_lib.RiverOrientation.HORIZONTAL,
        grid_size=10,
    )
    self.assertIsNone(
        utils_river_solver.validate_river_candidate(
            grid, river, area_reqs=(8, 9)
        )
    )
    self.assertIsNone(
        utils_river_solver.validate_river_candidate(
            grid, river, area_reqs=(9, 8)
        )
    )

  def test_validate_river_candidate_invalid(self):
    grid = np.zeros((10, 10), dtype=bool)
    river = river_lib.River(
        width=4,
        offset=0,
        orientation=river_lib.RiverOrientation.HORIZONTAL,
        grid_size=10,
    )
    self.assertIsNotNone(
        utils_river_solver.validate_river_candidate(
            grid, river, area_reqs=(8, 9)
        )
    )


class FindAdjacentStartCellTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    s1 = stick_lib.Stick(
        name='s1', start='j1', end='j2', length=1, azimuth_angle=0.0
    )
    s1.is_river = True
    s2 = stick_lib.Stick(
        name='s2', start='j2', end='j3', length=1, azimuth_angle=90.0
    )
    s2.is_river = True
    s3 = stick_lib.Stick(
        name='s3', start='j4', end='j5', length=1, azimuth_angle=0.0
    )
    s3.is_river = True
    self.sf = stick_figure_lib.StickFigure(
        name='test',
        category='test',
        complexity=1,
        root='s1',
        children=[s1, s2, s3],
    )
    self.grid_size = 5
    self.tracker = utils_tracker.CreasePatternTracker(self.grid_size)
    river1 = river_lib.River(
        1, 0, river_lib.RiverOrientation.HORIZONTAL, self.grid_size
    )
    self.tracker.add_river('s1', river1)
    self.debug_config = debug_types.DebugConfig(live_plot=False)

  def test_find_adjacent_start_cell_one_option(self):
    self.tracker.river_banks['s1'] = [(1, 0)]
    self.tracker.grid[1, 0] = utils_packing_flaps.GridCellState(
        type=utils_packing_flaps.GridCellType.RIVER_ADJACENT, source_id='j2'
    )
    start_cell = utils_river_solver.find_adjacent_start_cell(
        self.sf['s2'],
        self.sf,
        self.tracker,
        self.grid_size,
        debug=debug_types.DebugConfig(),
    )
    self.assertEqual(start_cell, (1, 0))

  def test_find_adjacent_start_cell_no_option_if_not_river_adjacent(self):
    self.tracker.river_banks['s1'] = [(1, 0)]
    self.tracker.grid[1, 0] = utils_packing_flaps.BlockedGridCellState
    start_cell = utils_river_solver.find_adjacent_start_cell(
        self.sf['s2'],
        self.sf,
        self.tracker,
        self.grid_size,
        debug=debug_types.DebugConfig(),
    )
    self.assertIsNone(start_cell)

  def test_find_adjacent_start_cell_no_option_if_no_shared_joint(self):
    self.tracker.river_banks['s1'] = [(1, 0)]
    self.tracker.grid[1, 0] = utils_packing_flaps.GridCellState(
        type=utils_packing_flaps.GridCellType.RIVER_ADJACENT, source_id='j2'
    )
    start_cell = utils_river_solver.find_adjacent_start_cell(
        self.sf['s3'],
        self.sf,
        self.tracker,
        self.grid_size,
        debug=debug_types.DebugConfig(),
    )
    self.assertIsNone(start_cell)

  def test_find_adjacent_start_cell_multiple_options_sorted(self):
    self.tracker.river_banks['s1'] = [(1, 4), (1, 0)]
    self.tracker.grid[1, 0] = utils_packing_flaps.GridCellState(
        type=utils_packing_flaps.GridCellType.RIVER_ADJACENT, source_id='j2'
    )
    self.tracker.grid[1, 4] = utils_packing_flaps.GridCellState(
        type=utils_packing_flaps.GridCellType.RIVER_ADJACENT, source_id='j2'
    )
    start_cell = utils_river_solver.find_adjacent_start_cell(
        self.sf['s2'],
        self.sf,
        self.tracker,
        self.grid_size,
        debug=debug_types.DebugConfig(),
    )
    self.assertEqual(start_cell, (1, 0))

  def test_find_adjacent_start_cell_ccw_sort(self):
    self.tracker.river_banks['s1'] = [(1, 4), (1, 0)]
    self.tracker.grid[1, 0] = utils_packing_flaps.GridCellState(
        type=utils_packing_flaps.GridCellType.RIVER_ADJACENT, source_id='j2'
    )
    self.tracker.grid[1, 4] = utils_packing_flaps.GridCellState(
        type=utils_packing_flaps.GridCellType.RIVER_ADJACENT, source_id='j2'
    )
    river1 = river_lib.River(
        1, 0, river_lib.RiverOrientation.HORIZONTAL, self.grid_size
    )
    self.tracker.packed_rivers['s1'] = river1
    start_cell = utils_river_solver.find_adjacent_start_cell(
        self.sf['s2'],
        self.sf,
        self.tracker,
        self.grid_size,
        debug=debug_types.DebugConfig(),
    )
    self.assertEqual(start_cell, (1, 0))

  def test_find_adjacent_start_cell_non_border_cell_ignored(self):
    self.tracker.river_banks['s1'] = [(1, 1)]
    self.tracker.grid[1, 1] = utils_packing_flaps.GridCellState(
        type=utils_packing_flaps.GridCellType.RIVER_ADJACENT, source_id='j2'
    )
    start_cell = utils_river_solver.find_adjacent_start_cell(
        self.sf['s2'], self.sf, self.tracker, self.grid_size, self.debug_config
    )
    self.assertIsNone(start_cell)


class FindRiverCandidatesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    s1 = stick_lib.Stick(
        name='s1',
        start='j1',
        end='j2',
        length=1,
        azimuth_angle=0.0,
    )
    s1.is_river = True
    s2 = stick_lib.Stick(
        name='s2',
        start='j2',
        end='j3',
        length=1,
        azimuth_angle=90.0,
    )
    s2.is_river = True
    self.sf = stick_figure_lib.StickFigure(
        name='test',
        category='test',
        complexity=1,
        root='s1',
        children=[s1, s2],
    )
    self.grid_size = 5
    self.tracker = utils_tracker.CreasePatternTracker(self.grid_size)
    river1 = river_lib.River(
        1, 0, river_lib.RiverOrientation.HORIZONTAL, self.grid_size
    )
    self.tracker.add_river('s1', river1)
    self.pockets_info = []
    self.traversal_plan = [
        {'stick': s1, 'area_sides': {'j1': 0, 'j2': 0}, 'constraints': {}},
        {'stick': s2, 'area_sides': {'j2': 0, 'j3': 0}, 'constraints': {}},
    ]
    self.debug_config = debug_types.DebugConfig(live_plot=False)

    patcher = mock.patch.object(
        utils_river_solver, 'attempt_flap_packing', return_value={}
    )
    self.mock_pack_flaps = patcher.start()
    self.addCleanup(patcher.stop)
    patcher = mock.patch.object(
        utils_river_solver,
        '_assign_river_sides',
        return_value=({'j1': 'a', 'j2': 'b'}, {'a': 0, 'b': 0}),
    )
    self.mock_assign_sides = patcher.start()
    self.addCleanup(patcher.stop)
    patcher = mock.patch.object(
        utils_river_solver,
        'check_placement_adjacency_constraints',
        return_value=False,
    )
    self.mock_check_adjacency = patcher.start()
    self.addCleanup(patcher.stop)
    patcher = mock.patch.object(
        utils_river_solver, 'validate_river_candidate', return_value=None
    )
    self.mock_validate = patcher.start()
    self.addCleanup(patcher.stop)
    patcher = mock.patch.object(
        utils_river_solver,
        'find_possible_lshape_river_placements',
        return_value=[],
    )
    self.mock_find_lshape = patcher.start()
    self.addCleanup(patcher.stop)
    patcher = mock.patch.object(
        utils_river_solver, 'find_possible_straight_river_placements'
    )
    self.mock_find_straight = patcher.start()
    self.addCleanup(patcher.stop)
    patcher = mock.patch(
        'bp_pipeline.packing.hugging_river_generator.generate_hugging_river_candidates'
    )
    self.mock_generate_hugging = patcher.start()
    self.addCleanup(patcher.stop)
    patcher = mock.patch.object(utils_river_solver, 'find_adjacent_start_cell')
    self.mock_find_adjacent_start_cell = patcher.start()
    self.addCleanup(patcher.stop)

  def test_find_river_candidates_with_start_cell(self):
    self.mock_find_adjacent_start_cell.return_value = (1, 0)
    self.mock_generate_hugging.return_value = []
    self.mock_pack_flaps.return_value = None
    step = self.traversal_plan[1]

    candidates = list(
        utils_river_solver.find_river_candidates(
            step,  # pyrefly: ignore[bad-argument-type]
            self.grid_size,
            self.tracker,
            self.sf,
            self.pockets_info,
            self.traversal_plan,  # pyrefly: ignore[bad-argument-type]
            self.debug_config,
        )
    )
    self.assertEmpty(candidates)
    np.testing.assert_array_equal(
        self.mock_generate_hugging.call_args[0][2],
        self.tracker.occupancy_grid,
    )
    self.mock_generate_hugging.assert_called_once_with(
        5,
        1,
        mock.ANY,
        (1, 0),
        river_lib.HugSide.LEFT,
        hug_boundaries=False,
        area_reqs=step['area_sides'],
        allow_off_grid=False,
        retry_opposite_hug_side=False,
    )

  def test_find_river_candidates_validation_error_records_step(self):
    self.mock_find_adjacent_start_cell.return_value = None
    self.tracker.packed_rivers = {}
    self.mock_validate.return_value = 'some error'
    river = river_lib.River(
        1, 1, river_lib.RiverOrientation.HORIZONTAL, self.grid_size
    )
    self.mock_find_straight.return_value = [river]
    step = self.traversal_plan[1]
    debug_config = debug_types.DebugConfig(live_plot=False, save_history=True)
    with mock.patch.object(debug_config, 'record_step') as mock_record_step:
      candidates = list(
          utils_river_solver.find_river_candidates(
              step,  # pyrefly: ignore[bad-argument-type]
              self.grid_size,
              self.tracker,
              self.sf,
              self.pockets_info,
              self.traversal_plan,  # pyrefly: ignore[bad-argument-type]
              debug_config,
          )
      )
      self.assertEmpty(candidates)
      self.mock_validate.assert_called_once()
      mock_record_step.assert_called_once()
      mock_args, _ = mock_record_step.call_args
      self.assertEqual(mock_args[1], 'FAIL')
      self.assertIn('some error', mock_args[2])

  def test_adjacency_check_skipped_if_start_cell_info_provided(self):
    """Tests adjacency check is skipped if start_cell_info is not None."""
    self.mock_find_adjacent_start_cell.return_value = (1, 0)
    river = river_lib.River(
        1, 1, river_lib.RiverOrientation.HORIZONTAL, self.grid_size
    )
    self.mock_generate_hugging.return_value = [river]
    step = self.traversal_plan[1]

    candidates = list(
        utils_river_solver.find_river_candidates(
            step,  # pyrefly: ignore[bad-argument-type]
            self.grid_size,
            self.tracker,
            self.sf,
            self.pockets_info,
            self.traversal_plan,  # pyrefly: ignore[bad-argument-type]
            self.debug_config,
        )
    )
    self.assertLen(candidates, 1)
    self.mock_check_adjacency.assert_not_called()
    self.mock_find_straight.assert_not_called()
    self.mock_generate_hugging.assert_called_once()

  def test_adjacency_check_performed_if_start_cell_info_is_none(self):
    """Tests adjacency check is performed if start_cell_info is None."""
    self.mock_find_adjacent_start_cell.return_value = None
    self.tracker.packed_rivers = {}
    river = river_lib.River(
        1, 1, river_lib.RiverOrientation.HORIZONTAL, self.grid_size
    )
    self.mock_find_straight.return_value = [river]
    step = self.traversal_plan[1]

    candidates = list(
        utils_river_solver.find_river_candidates(
            step,  # pyrefly: ignore[bad-argument-type]
            self.grid_size,
            self.tracker,
            self.sf,
            self.pockets_info,
            self.traversal_plan,  # pyrefly: ignore[bad-argument-type]
            self.debug_config,
        )
    )
    self.assertEmpty(candidates)
    self.mock_check_adjacency.assert_called_once()
    self.mock_generate_hugging.assert_not_called()
    self.mock_find_straight.assert_called_once()


class RiverSymmetryTest(absltest.TestCase):

  def test_check_river_symmetry_constraints_on_axis(self):
    river1 = river_lib.River(
        width=2,
        offset=4,
        orientation=river_lib.RiverOrientation.VERTICAL,
        grid_size=10,
    )
    self.assertFalse(
        utils_river_solver.check_river_symmetry_constraints(
            river1, 10, is_on_axis=True
        )
    )
    river3 = river_lib.River(
        width=2,
        offset=4,
        orientation=river_lib.RiverOrientation.HORIZONTAL,
        grid_size=10,
    )
    self.assertTrue(
        utils_river_solver.check_river_symmetry_constraints(
            river3, 10, is_on_axis=True
        )
    )
    river4 = river_lib.LShapedRiver(
        width=1,
        grid_size=10,
        l_shape_type=river_lib.LShapeType.BOTTOM_LEFT,
        turn_x=2,
        turn_y=2,
    )
    self.assertFalse(
        utils_river_solver.check_river_symmetry_constraints(
            river4, 10, is_on_axis=True
        )
    )
    river5 = river_lib.LShapedRiver(
        width=2,
        grid_size=10,
        l_shape_type=river_lib.LShapeType.BOTTOM_LEFT,
        turn_x=4,
        turn_y=4,
    )
    self.assertTrue(
        utils_river_solver.check_river_symmetry_constraints(
            river5, 10, is_on_axis=True
        )
    )
    river_br_on_axis_ok = river_lib.LShapedRiver(
        width=1,
        grid_size=10,
        l_shape_type=river_lib.LShapeType.BOTTOM_RIGHT,
        turn_x=4,
        turn_y=4,
    )
    self.assertTrue(
        utils_river_solver.check_river_symmetry_constraints(
            river_br_on_axis_ok, 10, is_on_axis=True
        )
    )
    river_br_on_axis_not_ok = river_lib.LShapedRiver(
        width=1,
        grid_size=10,
        l_shape_type=river_lib.LShapeType.BOTTOM_RIGHT,
        turn_x=5,
        turn_y=5,
    )
    self.assertFalse(
        utils_river_solver.check_river_symmetry_constraints(
            river_br_on_axis_not_ok, 10, is_on_axis=True
        )
    )

  def test_check_river_symmetry_constraints_off_axis(self):
    river1 = river_lib.River(
        width=2,
        offset=0,
        orientation=river_lib.RiverOrientation.VERTICAL,
        grid_size=10,
    )
    self.assertTrue(
        utils_river_solver.check_river_symmetry_constraints(
            river1, 10, is_on_axis=False
        )
    )
    river2 = river_lib.River(
        width=2,
        offset=4,
        orientation=river_lib.RiverOrientation.VERTICAL,
        grid_size=10,
    )
    self.assertFalse(
        utils_river_solver.check_river_symmetry_constraints(
            river2, 10, is_on_axis=False
        )
    )
    river3 = river_lib.River(
        width=2,
        offset=4,
        orientation=river_lib.RiverOrientation.HORIZONTAL,
        grid_size=10,
    )
    self.assertFalse(
        utils_river_solver.check_river_symmetry_constraints(
            river3, 10, is_on_axis=False
        )
    )
    river4 = river_lib.LShapedRiver(
        width=1,
        grid_size=10,
        l_shape_type=river_lib.LShapeType.BOTTOM_LEFT,
        turn_x=1,
        turn_y=1,
    )
    self.assertTrue(
        utils_river_solver.check_river_symmetry_constraints(
            river4, 10, is_on_axis=False
        )
    )
    river5 = river_lib.LShapedRiver(
        width=1,
        grid_size=10,
        l_shape_type=river_lib.LShapeType.BOTTOM_LEFT,
        turn_x=4,
        turn_y=4,
    )
    self.assertTrue(
        utils_river_solver.check_river_symmetry_constraints(
            river5, 10, is_on_axis=False
        )
    )
    river6 = river_lib.LShapedRiver(
        width=1,
        grid_size=10,
        l_shape_type=river_lib.LShapeType.BOTTOM_LEFT,
        turn_x=5,
        turn_y=5,
    )
    self.assertFalse(
        utils_river_solver.check_river_symmetry_constraints(
            river6, 10, is_on_axis=False
        )
    )
    river_br_off_axis = river_lib.LShapedRiver(
        width=1,
        grid_size=10,
        l_shape_type=river_lib.LShapeType.BOTTOM_RIGHT,
        turn_x=1,
        turn_y=1,
    )
    self.assertFalse(
        utils_river_solver.check_river_symmetry_constraints(
            river_br_off_axis, 10, is_on_axis=False
        )
    )


class RiverAxisExclusionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.grid_size = 10
    s1 = stick_lib.Stick(
        name='s1', start='j1', end='j2', length=1, azimuth_angle=0.0
    )
    s1.is_river = True
    self.sf = stick_figure_lib.StickFigure(
        name='test',
        category='test',
        complexity=1,
        root='s1',
        children=[s1],
    )
    self.tracker = utils_tracker.CreasePatternTracker(self.grid_size)
    self.pockets_info = []
    self.traversal_plan = [
        {'stick': s1, 'area_sides': {'j1': 0, 'j2': 0}, 'constraints': {}},
    ]
    self.debug_config = debug_types.DebugConfig(live_plot=False)
    self.rcc_config = config.PackingConfig(remove_central_crease=True)
    self.symmetry_axes = {'s1': False}

    patcher = mock.patch.object(
        utils_river_solver, 'attempt_flap_packing', return_value={}
    )
    self.mock_pack_flaps = patcher.start()
    self.addCleanup(patcher.stop)
    patcher = mock.patch.object(
        utils_river_solver,
        '_assign_river_sides',
        return_value=({'j1': 'a', 'j2': 'b'}, {'a': 0, 'b': 0}),
    )
    patcher.start()
    self.addCleanup(patcher.stop)
    patcher = mock.patch.object(
        utils_river_solver,
        'check_placement_adjacency_constraints',
        return_value=True,
    )
    patcher.start()
    self.addCleanup(patcher.stop)
    patcher = mock.patch.object(
        utils_river_solver, 'validate_river_candidate', return_value=None
    )
    patcher.start()
    self.addCleanup(patcher.stop)
    patcher = mock.patch.object(
        utils_river_solver,
        'find_possible_lshape_river_placements',
        return_value=[],
    )
    self.mock_find_lshape = patcher.start()
    self.addCleanup(patcher.stop)
    patcher = mock.patch.object(
        utils_river_solver, 'find_possible_straight_river_placements'
    )
    self.mock_find_straight = patcher.start()
    self.addCleanup(patcher.stop)
    patcher = mock.patch.object(
        utils_river_solver, 'find_adjacent_start_cell', return_value=None
    )
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_vertical_river_touching_axis_rejected(self):
    river_touching = river_lib.River(
        width=1,
        offset=4,
        orientation=river_lib.RiverOrientation.VERTICAL,
        grid_size=self.grid_size,
    )
    river_safe = river_lib.River(
        width=1,
        offset=2,
        orientation=river_lib.RiverOrientation.VERTICAL,
        grid_size=self.grid_size,
    )
    self.mock_find_straight.return_value = [river_touching, river_safe]
    step = self.traversal_plan[0]

    candidates = list(
        utils_river_solver.find_river_candidates(
            step,  # pyrefly: ignore[bad-argument-type]
            self.grid_size,
            self.tracker,
            self.sf,
            self.pockets_info,
            self.traversal_plan,  # pyrefly: ignore[bad-argument-type]
            self.debug_config,
            is_symmetric=True,
            symmetry_axes=self.symmetry_axes,
            config=self.rcc_config,
        )
    )
    rivers = [c['river'] for c in candidates]
    self.assertNotIn(river_touching, rivers)
    self.assertIn(river_safe, rivers)

    candidates_no_rcc = list(
        utils_river_solver.find_river_candidates(
            step,  # pyrefly: ignore[bad-argument-type]
            self.grid_size,
            self.tracker,
            self.sf,
            self.pockets_info,
            self.traversal_plan,  # pyrefly: ignore[bad-argument-type]
            self.debug_config,
            is_symmetric=True,
            symmetry_axes=self.symmetry_axes,
        )
    )
    rivers_no_rcc = [c['river'] for c in candidates_no_rcc]
    self.assertIn(river_touching, rivers_no_rcc)

  def test_lshape_river_touching_axis_rejected(self):
    river_touching = river_lib.LShapedRiver(
        width=1,
        grid_size=self.grid_size,
        l_shape_type=river_lib.LShapeType.BOTTOM_LEFT,
        turn_x=4,
        turn_y=4,
    )
    river_safe = river_lib.LShapedRiver(
        width=1,
        grid_size=self.grid_size,
        l_shape_type=river_lib.LShapeType.BOTTOM_LEFT,
        turn_x=2,
        turn_y=2,
    )
    self.mock_find_straight.return_value = []
    self.mock_find_lshape.return_value = [river_touching, river_safe]
    step = self.traversal_plan[0]

    candidates = list(
        utils_river_solver.find_river_candidates(
            step,  # pyrefly: ignore[bad-argument-type]
            self.grid_size,
            self.tracker,
            self.sf,
            self.pockets_info,
            self.traversal_plan,  # pyrefly: ignore[bad-argument-type]
            self.debug_config,
            is_symmetric=True,
            symmetry_axes=self.symmetry_axes,
            config=self.rcc_config,
        )
    )
    rivers = [c['river'] for c in candidates]
    self.assertNotIn(river_touching, rivers)
    self.assertIn(river_safe, rivers)

    candidates_no_rcc = list(
        utils_river_solver.find_river_candidates(
            step,  # pyrefly: ignore[bad-argument-type]
            self.grid_size,
            self.tracker,
            self.sf,
            self.pockets_info,
            self.traversal_plan,  # pyrefly: ignore[bad-argument-type]
            self.debug_config,
            is_symmetric=True,
            symmetry_axes=self.symmetry_axes,
        )
    )
    rivers_no_rcc = [c['river'] for c in candidates_no_rcc]
    self.assertIn(river_touching, rivers_no_rcc)
    river_touching = river_lib.HuggingRiver(
        grid_size=self.grid_size,
        width=1,
        subriver_paths=[[(3, 0), (3, 1), (3, 2), (3, 3), (3, 4)]],
        hug_side=river_lib.HugSide.LEFT,
    )
    river_safe = river_lib.HuggingRiver(
        grid_size=self.grid_size,
        width=1,
        subriver_paths=[[(3, 0), (3, 1), (3, 2)]],
        hug_side=river_lib.HugSide.LEFT,
    )
    self.mock_find_straight.return_value = []

    with (
        mock.patch.object(
            utils_river_solver, 'find_adjacent_start_cell', return_value=(3, 0)
        ),
        mock.patch(
            'bp_pipeline.packing.hugging_river_generator.generate_hugging_river_candidates',
            return_value=[river_touching, river_safe],
        ),
    ):
      step = self.traversal_plan[0]
      candidates = list(
          utils_river_solver.find_river_candidates(
              step,  # pyrefly: ignore[bad-argument-type]
              self.grid_size,
              self.tracker,
              self.sf,
              self.pockets_info,
              self.traversal_plan,  # pyrefly: ignore[bad-argument-type]
              self.debug_config,
              is_symmetric=True,
              symmetry_axes=self.symmetry_axes,
              config=self.rcc_config,
          )
      )
    rivers = [c['river'] for c in candidates]
    self.assertNotIn(river_touching, rivers)
    self.assertIn(river_safe, rivers)

    with (
        mock.patch.object(
            utils_river_solver, 'find_adjacent_start_cell', return_value=(3, 0)
        ),
        mock.patch(
            'bp_pipeline.packing.hugging_river_generator.generate_hugging_river_candidates',
            return_value=[river_touching, river_safe],
        ),
    ):
      candidates_no_rcc = list(
          utils_river_solver.find_river_candidates(
              step,  # pyrefly: ignore[bad-argument-type]
              self.grid_size,
              self.tracker,
              self.sf,
              self.pockets_info,
              self.traversal_plan,  # pyrefly: ignore[bad-argument-type]
              self.debug_config,
              is_symmetric=True,
              symmetry_axes=self.symmetry_axes,
          )
      )
    rivers_no_rcc = [c['river'] for c in candidates_no_rcc]
    self.assertIn(river_touching, rivers_no_rcc)


class GeneratePlacementsFromStartCellTest(absltest.TestCase):

  def test_left_border_cell(self):
    placements = utils_river_solver.generate_placements_from_start_cell(
        start_cell_info=(2, 0),
        river_width=2,
        grid_size=5,
        include_straight_rivers=True,
        include_lshape_rivers=True,
    )
    straight_rivers = [p for p in placements if isinstance(p, river_lib.River)]
    l_shape_rivers = [
        p for p in placements if isinstance(p, river_lib.LShapedRiver)
    ]
    self.assertLen(placements, 22)
    self.assertLen(straight_rivers, 2)
    self.assertLen(l_shape_rivers, 20)
    self.assertTrue(
        all(
            p.orientation == river_lib.RiverOrientation.HORIZONTAL
            for p in straight_rivers
        )
    )
    self.assertEqual({p.offset for p in straight_rivers}, {1, 2})
    self.assertTrue(
        all(
            p.l_shape_type in [
                river_lib.LShapeType.BOTTOM_LEFT,
                river_lib.LShapeType.TOP_LEFT,
            ]
            for p in l_shape_rivers
        )
    )
    l_shape_turn_y = collections.Counter([p.turn_y for p in l_shape_rivers])
    self.assertEqual(l_shape_turn_y, {1: 10, 2: 10})

  def test_bottom_border_cell_no_l_shapes(self):
    placements = utils_river_solver.generate_placements_from_start_cell(
        start_cell_info=(4, 2),
        river_width=2,
        grid_size=5,
        include_straight_rivers=True,
        include_lshape_rivers=False,
    )
    straight_rivers = [p for p in placements if isinstance(p, river_lib.River)]
    l_shape_rivers = [
        p for p in placements if isinstance(p, river_lib.LShapedRiver)
    ]
    self.assertLen(placements, 2)
    self.assertLen(straight_rivers, 2)
    self.assertEmpty(l_shape_rivers)
    self.assertTrue(
        all(
            p.orientation == river_lib.RiverOrientation.VERTICAL
            for p in straight_rivers
        )
    )
    self.assertEqual({p.offset for p in straight_rivers}, {1, 2})

  def test_bottom_border_cell_with_l_shapes(self):
    placements = utils_river_solver.generate_placements_from_start_cell(
        start_cell_info=(4, 2),
        river_width=2,
        grid_size=5,
        include_straight_rivers=False,
        include_lshape_rivers=True,
    )
    straight_rivers = [p for p in placements if isinstance(p, river_lib.River)]
    l_shape_rivers = [
        p for p in placements if isinstance(p, river_lib.LShapedRiver)
    ]
    self.assertLen(placements, 20)
    self.assertEmpty(straight_rivers)
    self.assertLen(l_shape_rivers, 20)
    self.assertTrue(
        all(
            p.l_shape_type in [
                river_lib.LShapeType.TOP_LEFT,
                river_lib.LShapeType.TOP_RIGHT,
            ]
            for p in l_shape_rivers
        )
    )
    l_shape_turn_x = collections.Counter([p.turn_x for p in l_shape_rivers])
    self.assertEqual(l_shape_turn_x, {1: 10, 2: 10})

  def test_right_border_cell_no_straight(self):
    placements = utils_river_solver.generate_placements_from_start_cell(
        start_cell_info=(2, 4),
        river_width=2,
        grid_size=5,
        include_straight_rivers=False,
        include_lshape_rivers=True,
    )
    straight_rivers = [p for p in placements if isinstance(p, river_lib.River)]
    l_shape_rivers = [
        p for p in placements if isinstance(p, river_lib.LShapedRiver)
    ]
    self.assertLen(placements, 20)
    self.assertLen(l_shape_rivers, 20)
    self.assertEmpty(straight_rivers)
    self.assertTrue(
        all(
            p.l_shape_type in [
                river_lib.LShapeType.BOTTOM_RIGHT,
                river_lib.LShapeType.TOP_RIGHT,
            ]
            for p in l_shape_rivers
        )
    )
    l_shape_turn_y = collections.Counter([p.turn_y for p in l_shape_rivers])
    self.assertEqual(l_shape_turn_y, {1: 10, 2: 10})

  def test_top_border_cell(self):
    placements = utils_river_solver.generate_placements_from_start_cell(
        start_cell_info=(0, 2),
        river_width=2,
        grid_size=5,
        include_straight_rivers=True,
        include_lshape_rivers=True,
    )
    straight_rivers = [p for p in placements if isinstance(p, river_lib.River)]
    l_shape_rivers = [
        p for p in placements if isinstance(p, river_lib.LShapedRiver)
    ]
    self.assertLen(placements, 22)
    self.assertLen(straight_rivers, 2)
    self.assertEqual({p.offset for p in straight_rivers}, {1, 2})
    self.assertTrue(
        all(
            p.orientation == river_lib.RiverOrientation.VERTICAL
            for p in straight_rivers
        )
    )
    self.assertLen(l_shape_rivers, 20)
    self.assertTrue(
        all(
            p.l_shape_type in [
                river_lib.LShapeType.BOTTOM_LEFT,
                river_lib.LShapeType.BOTTOM_RIGHT,
            ]
            for p in l_shape_rivers
        )
    )
    l_shape_turn_x = collections.Counter([p.turn_x for p in l_shape_rivers])
    self.assertEqual(l_shape_turn_x, {1: 10, 2: 10})

  def test_corner_cell_top_left(self):
    placements = utils_river_solver.generate_placements_from_start_cell(
        start_cell_info=(0, 0),
        river_width=2,
        grid_size=5,
        include_straight_rivers=True,
        include_lshape_rivers=True,
    )
    straight_rivers = [p for p in placements if isinstance(p, river_lib.River)]
    l_shape_rivers = [
        p for p in placements if isinstance(p, river_lib.LShapedRiver)
    ]
    self.assertLen(placements, 21)
    self.assertLen(straight_rivers, 2)
    self.assertLen(l_shape_rivers, 19)


class GetSideAssignmentTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.stick = stick_lib.Stick(
        name='s1', start='j1', end='j2', length=1, azimuth_angle=0.0
    )
    self.area_sides_reqs = {'j1': 10, 'j2': 5}

  def test_get_side_assignment_cand1_valid(self):
    side_areas = {'top': 10, 'bottom': 5}
    assignment = utils_river_solver._get_side_assignment(
        self.stick,
        self.area_sides_reqs,  # pyrefly: ignore[bad-argument-type]
        side_areas,  # pyrefly: ignore[bad-argument-type]
        'top',
        'bottom',
        adjacent_river_constraints={},
    )
    self.assertEqual(assignment, {'j1': 'top', 'j2': 'bottom'})

  def test_get_side_assignment_cand2_valid(self):
    side_areas = {'top': 5, 'bottom': 10}
    assignment = utils_river_solver._get_side_assignment(
        self.stick,
        self.area_sides_reqs,  # pyrefly: ignore[bad-argument-type]
        side_areas,  # pyrefly: ignore[bad-argument-type]
        'top',
        'bottom',
        adjacent_river_constraints={},
    )
    self.assertEqual(assignment, {'j1': 'bottom', 'j2': 'top'})

  def test_get_side_assignment_with_constraints_cand1_invalid(self):
    side_areas = {'top': 10, 'bottom': 5}
    adjacent_river_constraints = {'j1': river_adjacency.Side.TOP}
    assignment = utils_river_solver._get_side_assignment(
        self.stick,
        self.area_sides_reqs,  # pyrefly: ignore[bad-argument-type]
        side_areas,  # pyrefly: ignore[bad-argument-type]
        'top',
        'bottom',
        adjacent_river_constraints=adjacent_river_constraints,
    )
    self.assertIsNone(assignment)

  def test_get_side_assignment_with_constraints_cand1_invalid_cand2_valid(
      self,
  ):
    side_areas = {'top': 10, 'bottom': 10}
    area_sides_reqs = {'j1': 10, 'j2': 10}
    adjacent_river_constraints = {'j1': river_adjacency.Side.TOP}
    assignment = utils_river_solver._get_side_assignment(
        self.stick,
        area_sides_reqs,  # pyrefly: ignore[bad-argument-type]
        side_areas,  # pyrefly: ignore[bad-argument-type]
        'top',
        'bottom',
        adjacent_river_constraints=adjacent_river_constraints,
    )
    self.assertEqual(assignment, {'j1': 'bottom', 'j2': 'top'})

  def test_get_side_assignment_straight_hugging_zero_area_req(self):
    stick = stick_lib.Stick(
        name='s1', start='j1', end='j2', length=1, azimuth_angle=0.0
    )
    stick.is_river = True
    area_sides_reqs = {'j1': 0.0, 'j2': 10.0}
    side_areas = {'side1': 5.0, 'side2': 10.0}
    adjacent_river_constraints = {}

    assignment = utils_river_solver._get_side_assignment(
        stick,
        area_sides_reqs,
        side_areas,
        'side1',
        'side2',
        adjacent_river_constraints,
        variant_type=river_lib.PathVariantType.STRAIGHT,
    )
    self.assertEqual(assignment, {'j1': 'side1', 'j2': 'side2'})

    assignment = utils_river_solver._get_side_assignment(
        stick,
        area_sides_reqs,
        side_areas,
        'side1',
        'side2',
        adjacent_river_constraints,
        variant_type=river_lib.PathVariantType.STANDARD,
    )
    self.assertIsNone(assignment)


class AssignRiverSidesTest(absltest.TestCase):

  def test_assign_river_sides_straight_hugging_zero_area_req(self):
    stick = stick_lib.Stick(
        name='s1', start='j1', end='j2', length=1, azimuth_angle=0.0
    )
    stick.is_river = True
    tree = stick_figure_lib.StickFigure(
        name='test',
        category='test',
        complexity=1,
        root='s1',
        children=[stick],
    )
    step = {
        'stick': stick,
        'area_sides': {'j1': 0.0, 'j2': 1.0},
        'constraints': {},
    }
    tracker = utils_tracker.CreasePatternTracker(3)
    debug = debug_types.DebugConfig(False)

    river_straight = river_lib.HuggingRiver(
        grid_size=3,
        width=1,
        subriver_paths=[[(0, 1), (1, 1), (1, 0)]],
        hug_side=river_lib.HugSide.RIGHT,
        variant_type=river_lib.PathVariantType.STRAIGHT,
    )
    assignment_straight = utils_river_solver._assign_river_sides(
        river_straight,
        step,
        tracker,
        tree,
        debug,
        hugging_joint='j1',
        occupancy_grid=tracker.occupancy_grid,
    )
    self.assertIsNotNone(assignment_straight)
    joint_to_side, _ = assignment_straight
    self.assertEqual(joint_to_side, {'j1': 'hugging', 'j2': 'opposite'})

    river_standard = river_lib.HuggingRiver(
        grid_size=3,
        width=1,
        subriver_paths=[[(0, 1), (1, 1), (1, 0)]],
        hug_side=river_lib.HugSide.RIGHT,
        variant_type=river_lib.PathVariantType.STANDARD,
    )
    assignment_standard = utils_river_solver._assign_river_sides(
        river_standard,
        step,
        tracker,
        tree,
        debug,
        hugging_joint='j1',
        occupancy_grid=tracker.occupancy_grid,
    )
    self.assertIsNone(assignment_standard)

  def test_assign_river_sides_straight_hugging_zero_area_req_and_positive_area(
      self,
  ):
    stick = stick_lib.Stick(
        name='s1', start='j1', end='j2', length=1, azimuth_angle=0.0
    )
    stick.is_river = True
    tree = stick_figure_lib.StickFigure(
        name='test',
        category='test',
        complexity=1,
        root='s1',
        children=[stick],
    )
    step = {
        'stick': stick,
        'area_sides': {'j1': 0.0, 'j2': 0.0},
        'constraints': {},
    }
    tracker = utils_tracker.CreasePatternTracker(2)
    debug = debug_types.DebugConfig(False)

    river_straight = river_lib.HuggingRiver(
        grid_size=2,
        width=1,
        subriver_paths=[[(0, 0), (0, 1)]],
        hug_side=river_lib.HugSide.RIGHT,
        variant_type=river_lib.PathVariantType.STRAIGHT,
    )
    assignment_straight = utils_river_solver._assign_river_sides(
        river_straight,
        step,
        tracker,
        tree,
        debug,
        hugging_joint='j1',
        occupancy_grid=tracker.occupancy_grid,
    )
    self.assertIsNotNone(assignment_straight)

    river_standard = river_lib.HuggingRiver(
        grid_size=2,
        width=1,
        subriver_paths=[[(0, 0), (0, 1)]],
        hug_side=river_lib.HugSide.RIGHT,
        variant_type=river_lib.PathVariantType.STANDARD,
    )
    assignment_standard = utils_river_solver._assign_river_sides(
        river_standard,
        step,
        tracker,
        tree,
        debug,
        hugging_joint='j1',
        occupancy_grid=tracker.occupancy_grid,
    )
    self.assertIsNone(assignment_standard)

  def test_assign_river_sides_standard_hugging_positive_req_fail_mutant(self):
    stick = stick_lib.Stick(
        name='s1', start='j1', end='j2', length=1, azimuth_angle=0.0
    )
    stick.is_river = True
    tree = stick_figure_lib.StickFigure(
        name='test',
        category='test',
        complexity=1,
        root='s1',
        children=[stick],
    )
    step = {
        'stick': stick,
        'area_sides': {'j1': 1.0, 'j2': 1.0},
        'constraints': {},
    }
    tracker = utils_tracker.CreasePatternTracker(3)
    debug = debug_types.DebugConfig(False)
    river_standard = river_lib.HuggingRiver(
        grid_size=3,
        width=1,
        subriver_paths=[[(0, 1), (1, 1), (1, 0)]],
        hug_side=river_lib.HugSide.RIGHT,
        variant_type=river_lib.PathVariantType.STANDARD,
    )
    assignment_standard = utils_river_solver._assign_river_sides(
        river_standard,
        step,
        tracker,
        tree,
        debug,
        hugging_joint='j1',
        occupancy_grid=tracker.occupancy_grid,
    )
    self.assertIsNotNone(assignment_standard)

  def test_assign_river_sides_straight_hugging_opposite_req_zero(self):
    stick = stick_lib.Stick(
        name='s1', start='j1', end='j2', length=1, azimuth_angle=0.0
    )
    stick.is_river = True
    tree = stick_figure_lib.StickFigure(
        name='test',
        category='test',
        complexity=1,
        root='s1',
        children=[stick],
    )
    step = {
        'stick': stick,
        'area_sides': {'j1': 1.0, 'j2': 0.0},
        'constraints': {},
    }
    tracker = utils_tracker.CreasePatternTracker(3)
    debug = debug_types.DebugConfig(False)

    river_straight = river_lib.HuggingRiver(
        grid_size=3,
        width=1,
        subriver_paths=[[(0, 1), (1, 1), (1, 0)]],
        hug_side=river_lib.HugSide.RIGHT,
        variant_type=river_lib.PathVariantType.STRAIGHT,
    )
    assignment_straight = utils_river_solver._assign_river_sides(
        river_straight,
        step,
        tracker,
        tree,
        debug,
        hugging_joint='j1',
        occupancy_grid=tracker.occupancy_grid,
    )
    self.assertIsNotNone(assignment_straight)

  def test_standard_hugging_opposite_req_zero_nonzero_area_is_none(self):
    stick = stick_lib.Stick(
        name='s1', start='j1', end='j2', length=1, azimuth_angle=0.0
    )
    stick.is_river = True
    tree = stick_figure_lib.StickFigure(
        name='test',
        category='test',
        complexity=1,
        root='s1',
        children=[stick],
    )
    step = {
        'stick': stick,
        'area_sides': {'j1': 1.0, 'j2': 0.0},
        'constraints': {},
    }
    tracker = utils_tracker.CreasePatternTracker(3)
    debug = debug_types.DebugConfig(False)
    river_standard = river_lib.HuggingRiver(
        grid_size=3,
        width=1,
        subriver_paths=[[(0, 1), (1, 1), (1, 0)]],
        hug_side=river_lib.HugSide.RIGHT,
        variant_type=river_lib.PathVariantType.STANDARD,
    )
    assignment_standard = utils_river_solver._assign_river_sides(
        river_standard,
        step,
        tracker,
        tree,
        debug,
        hugging_joint='j1',
        occupancy_grid=tracker.occupancy_grid,
    )
    self.assertIsNone(assignment_standard)

  def test_assign_river_sides_standard_hugging_hugging_req_zero_with_positive_area(
      self,
  ):
    stick = stick_lib.Stick(
        name='s1', start='j1', end='j2', length=1, azimuth_angle=0.0
    )
    stick.is_river = True
    tree = stick_figure_lib.StickFigure(
        name='test',
        category='test',
        complexity=1,
        root='s1',
        children=[stick],
    )
    step = {
        'stick': stick,
        'area_sides': {'j1': 0.0, 'j2': 0.0},
        'constraints': {},
    }
    tracker = utils_tracker.CreasePatternTracker(2)
    debug = debug_types.DebugConfig(False)

    river_standard = river_lib.HuggingRiver(
        grid_size=2,
        width=1,
        subriver_paths=[[(0, 0), (0, 1)]],
        hug_side=river_lib.HugSide.RIGHT,
        variant_type=river_lib.PathVariantType.STANDARD,
    )
    assignment_standard = utils_river_solver._assign_river_sides(
        river_standard,
        step,
        tracker,
        tree,
        debug,
        hugging_joint='j1',
        occupancy_grid=tracker.occupancy_grid,
    )
    self.assertIsNone(assignment_standard)


class CheckPlacementAdjacencyConstraintsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    s1 = stick_lib.Stick(
        name='s1', start='j1', end='j2', length=1, azimuth_angle=0.0
    )
    s1.is_river = True
    s2 = stick_lib.Stick(
        name='s2', start='j2', end='j3', length=1, azimuth_angle=90.0
    )
    s2.is_river = True
    s3 = stick_lib.Stick(
        name='s3', start='j4', end='j5', length=1, azimuth_angle=0.0
    )
    s3.is_river = True
    self.sf = stick_figure_lib.StickFigure(
        name='test',
        category='test',
        complexity=1,
        root='s1',
        children=[s1, s2, s3],
    )
    self.grid_size = 5
    self.tracker = utils_tracker.CreasePatternTracker(self.grid_size)
    self.debug_config = debug_types.DebugConfig(live_plot=False)

  def test_adjacent_ok(self):
    placed_river1 = river_lib.River(
        1, 0, river_lib.RiverOrientation.HORIZONTAL, self.grid_size
    )
    self.tracker.add_river('s1', placed_river1)
    river2 = river_lib.River(
        1, 1, river_lib.RiverOrientation.HORIZONTAL, self.grid_size
    )
    step = {'stick': self.sf['s2'], 'area_sides': {}, 'constraints': {}}
    self.assertTrue(
        utils_river_solver.check_placement_adjacency_constraints(
            river2, step, self.tracker, self.sf, self.debug_config
        )
    )

  def test_angular_adjacent_but_not_geom_adjacent_fail(self):
    placed_river1 = river_lib.River(
        1, 0, river_lib.RiverOrientation.HORIZONTAL, self.grid_size
    )
    self.tracker.add_river('s1', placed_river1)
    river2 = river_lib.River(
        1, 2, river_lib.RiverOrientation.HORIZONTAL, self.grid_size
    )
    step = {'stick': self.sf['s2'], 'area_sides': {}, 'constraints': {}}
    self.assertFalse(
        utils_river_solver.check_placement_adjacency_constraints(
            river2, step, self.tracker, self.sf, self.debug_config
        )
    )

  def test_not_angular_adjacent_ok(self):
    placed_river1 = river_lib.River(
        1, 0, river_lib.RiverOrientation.HORIZONTAL, self.grid_size
    )
    self.tracker.add_river('s1', placed_river1)
    river3 = river_lib.River(
        1, 2, river_lib.RiverOrientation.HORIZONTAL, self.grid_size
    )
    step = {'stick': self.sf['s3'], 'area_sides': {}, 'constraints': {}}
    self.assertTrue(
        utils_river_solver.check_placement_adjacency_constraints(
            river3, step, self.tracker, self.sf, self.debug_config
        )
    )


class AddRiverAdjacentCellsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.grid_size = 3
    self.tracker = utils_tracker.CreasePatternTracker(self.grid_size)

  def test_add_river_adjacent_cells_horizontal_river(self):
    river = river_lib.River(
        1, 1, river_lib.RiverOrientation.HORIZONTAL, self.grid_size
    )
    joint_to_side = {'j1': 'top', 'j2': 'bottom'}
    utils_river_solver.add_river_adjacent_cells(
        self.tracker, 'r1', river, joint_to_side
    )
    self.assertEqual(
        self.tracker.grid[0, 0].type,
        utils_packing_flaps.GridCellType.RIVER_ADJACENT,
    )
    self.assertEqual(self.tracker.grid[0, 0].source_id, 'j1')
    self.assertEqual(
        self.tracker.grid[2, 0].type,
        utils_packing_flaps.GridCellType.RIVER_ADJACENT,
    )
    self.assertEqual(self.tracker.grid[2, 0].source_id, 'j2')
    self.assertCountEqual(
        self.tracker.river_banks['r1'],
        [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)],
    )

  def test_add_river_adjacent_cells_lshape_river(self):
    river = river_lib.LShapedRiver(
        1, self.grid_size, river_lib.LShapeType.BOTTOM_LEFT, 1, 1
    )
    joint_to_side = {'j1': 'inner', 'j2': 'outer'}
    utils_river_solver.add_river_adjacent_cells(
        self.tracker, 'r1', river, joint_to_side
    )
    self.assertEqual(
        self.tracker.grid[0, 0].type,
        utils_packing_flaps.GridCellType.RIVER_ADJACENT,
    )
    self.assertEqual(self.tracker.grid[0, 0].source_id, 'j1')
    self.assertEqual(
        self.tracker.grid[0, 2].type,
        utils_packing_flaps.GridCellType.RIVER_ADJACENT,
    )
    self.assertEqual(self.tracker.grid[0, 2].source_id, 'j2')
    self.assertCountEqual(
        self.tracker.river_banks['r1'], [(0, 0), (0, 2), (1, 2), (2, 0), (2, 1)]
    )

  def test_add_river_adjacent_cells_hugging(self):
    grid_size = 4
    tracker = utils_tracker.CreasePatternTracker(grid_size)
    occupancy_grid = np.zeros((grid_size, grid_size), dtype=bool)
    occupancy_grid[0, 0] = True
    tracker.occupancy_grid[0, 0] = True
    tracker.grid[0, 0] = utils_packing_flaps.BlockedGridCellState
    river = river_lib.HuggingRiver.from_contour(
        grid_size, 1, occupancy_grid, (1, 0), river_lib.HugSide.LEFT
    )
    tracker.add_river('hr1', river)
    joint_to_side = {'j1': 'hugging', 'j2': 'opposite'}
    utils_river_solver.add_river_adjacent_cells(
        tracker, 'hr1', river, joint_to_side
    )
    hugging_mask, opposite_mask = river.get_area_masks(tracker.occupancy_grid)
    river_mask, _ = river.get_river_mask_and_forbidden_grid(
        tracker.occupancy_grid
    )
    hugging_adj = set()
    opposite_adj = set()
    river_coords = np.argwhere(river_mask)
    for r, c in river_coords:
      for dr, dc in river_lib.DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
          if nr == 0 and nc == 0:
            continue
          if hugging_mask[nr, nc]:
            hugging_adj.add((nr, nc))
          elif opposite_mask[nr, nc]:
            opposite_adj.add((nr, nc))

    for r, c in hugging_adj:
      self.assertEqual(tracker.grid[r, c].source_id, 'j1')
      self.assertEqual(
          tracker.grid[r, c].type,
          utils_packing_flaps.GridCellType.RIVER_ADJACENT,
      )
    for r, c in opposite_adj:
      self.assertEqual(tracker.grid[r, c].source_id, 'j2')
      self.assertEqual(
          tracker.grid[r, c].type,
          utils_packing_flaps.GridCellType.RIVER_ADJACENT,
      )

    self.assertCountEqual(
        tracker.river_banks['hr1'],
        list(hugging_adj) + list(opposite_adj),
    )

  def test_add_river_adjacent_cells_with_sf_placements(self):
    grid_size = 6
    tracker = utils_tracker.CreasePatternTracker(grid_size)
    river = river_lib.River(
        2, 2, river_lib.RiverOrientation.HORIZONTAL, grid_size
    )
    joint_to_side = {'j1': 'top', 'j2': 'bottom'}
    sf1 = types.PolygonPlacement.from_grid_coords(
        offset=(0, 0), size=(2, 2), grid_size=grid_size
    )
    sf2 = types.PolygonPlacement.from_grid_coords(
        offset=(0, 4), size=(2, 2), grid_size=grid_size
    )
    utils_river_solver.add_river_adjacent_cells(
        tracker,
        'r1',
        river,
        joint_to_side,
        level_shifter_placements=(sf1, sf2),
    )
    river_bank_cells = tracker.river_banks['r1']
    self.assertIn((2, 0), river_bank_cells)
    self.assertIn((2, 1), river_bank_cells)


@mock.patch('bp_pipeline.packing.utils_pack_river_flaps.pack_enclosed_flaps')
class AttemptFlapPackingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    s1 = stick_lib.Stick(
        name='s1', start='j1', end='j2', length=1, azimuth_angle=0.0
    )
    s1.is_river = True
    self.sf = stick_figure_lib.StickFigure(
        name='test',
        category='test',
        complexity=1,
        root='s1',
        children=[s1],
    )
    self.grid_size = 3
    self.tracker = utils_tracker.CreasePatternTracker(self.grid_size)
    self.river = river_lib.River(
        1, 1, river_lib.RiverOrientation.HORIZONTAL, self.grid_size
    )
    self.step = {
        'stick': self.sf['s1'],
        'area_sides': {'j1': 1, 'j2': 1},
        'constraints': {},
    }
    self.pockets_info = []
    self.traversal_plan = []
    self.debug_config = debug_types.DebugConfig(live_plot=False)
    self.joint_to_side = {'j1': 'top', 'j2': 'bottom'}
    self.side_areas = {'top': 3, 'bottom': 3}

  def test_attempt_flap_packing_success(self, mock_pack_flaps):
    mock_pack_flaps.return_value = {'f1': mock.MagicMock()}
    packed_flaps = utils_river_solver.attempt_flap_packing(
        self.river,
        self.step,  # pyrefly: ignore[bad-argument-type]
        self.grid_size,
        self.tracker,
        self.pockets_info,
        self.traversal_plan,
        self.debug_config,
        self.joint_to_side,
        self.side_areas,  # pyrefly: ignore[bad-argument-type]
        fast_packing_debug=False,
        is_symmetric=False,
        symmetry_axes=None,
        scoring_config=scoring.ScoringConfig(),
        config=config.PackingConfig(),
    )
    self.assertIsNotNone(packed_flaps)
    self.assertIn('f1', packed_flaps)
    mock_pack_flaps.assert_called_once()

  def test_attempt_flap_packing_fail_add_river(self, mock_pack_flaps):
    self.tracker.grid[1, 1] = utils_packing_flaps.BlockedGridCellState
    self.tracker.occupancy_grid[1, 1] = True
    packed_flaps = utils_river_solver.attempt_flap_packing(
        self.river,
        self.step,  # pyrefly: ignore[bad-argument-type]
        self.grid_size,
        self.tracker,
        self.pockets_info,
        self.traversal_plan,
        self.debug_config,
        self.joint_to_side,
        self.side_areas,  # pyrefly: ignore[bad-argument-type]
        fast_packing_debug=False,
        is_symmetric=False,
        symmetry_axes=None,
        scoring_config=scoring.ScoringConfig(),
        config=config.PackingConfig(),
    )
    self.assertIsNone(packed_flaps)
    mock_pack_flaps.assert_not_called()

  def test_attempt_flap_packing_fail_pack_flaps(self, mock_pack_flaps):
    mock_pack_flaps.return_value = None
    packed_flaps = utils_river_solver.attempt_flap_packing(
        self.river,
        self.step,  # pyrefly: ignore[bad-argument-type]
        self.grid_size,
        self.tracker,
        self.pockets_info,
        self.traversal_plan,
        self.debug_config,
        self.joint_to_side,
        self.side_areas,  # pyrefly: ignore[bad-argument-type]
        fast_packing_debug=False,
        is_symmetric=False,
        symmetry_axes=None,
        scoring_config=scoring.ScoringConfig(),
        config=config.PackingConfig(),
    )
    self.assertIsNone(packed_flaps)
    mock_pack_flaps.assert_called_once()

  def test_attempt_flap_packing_with_sf_pair(self, mock_pack_flaps):
    mock_pack_flaps.return_value = {'f1': mock.MagicMock()}
    grid_size = 6
    tracker = utils_tracker.CreasePatternTracker(grid_size)
    river = river_lib.River(
        2, 2, river_lib.RiverOrientation.HORIZONTAL, grid_size
    )
    s1 = stick_lib.Stick(
        name='s1', start='j1', end='j2', length=1, azimuth_angle=0.0
    )
    s1.is_river = True
    sf = stick_figure_lib.StickFigure(
        name='test',
        category='test',
        complexity=1,
        root='s1',
        children=[s1],
    )
    step = {
        'stick': sf['s1'],
        'area_sides': {'j1': 1, 'j2': 1},
        'constraints': {},
    }
    sf1 = types.PolygonPlacement.from_grid_coords(
        offset=(0, 0), size=(2, 2), grid_size=grid_size
    )
    sf1 = dataclasses.replace(
        sf1,
        river_sf_info=types.RiverSfInfo(
            river_name='s1',
            river_side='top',
            parent_joint='j1',
        ),
    )
    sf2 = types.PolygonPlacement.from_grid_coords(
        offset=(0, 4), size=(2, 2), grid_size=grid_size
    )
    sf2 = dataclasses.replace(
        sf2,
        river_sf_info=types.RiverSfInfo(
            river_name='s1',
            river_side='bottom',
            parent_joint='j2',
        ),
    )
    packed_flaps = utils_river_solver.attempt_flap_packing(
        river,
        step,  # pyrefly: ignore[bad-argument-type]
        grid_size,
        tracker,
        [],
        [],
        debug_types.DebugConfig(live_plot=False),
        {'j1': 'top', 'j2': 'bottom'},
        {'top': 4, 'bottom': 8},
        fast_packing_debug=False,
        is_symmetric=False,
        symmetry_axes=None,
        scoring_config=scoring.ScoringConfig(),
        config=config.PackingConfig(),
        sf_pair=(sf1, sf2),
    )
    self.assertIsNotNone(packed_flaps)
    sf1_name = f's1{types.RIVER_SF1_SUFFIX}'
    sf2_name = f's1{types.RIVER_SF2_SUFFIX}'
    self.assertIn(sf1_name, packed_flaps)
    self.assertIn(sf2_name, packed_flaps)
    self.assertIn('f1', packed_flaps)
    self.assertIsNotNone(packed_flaps[sf1_name].river_sf_info)
    assert packed_flaps[sf1_name].river_sf_info is not None
    self.assertEqual(packed_flaps[sf1_name].river_sf_info.parent_joint, 'j1')
    self.assertIsNotNone(packed_flaps[sf2_name].river_sf_info)
    assert packed_flaps[sf2_name].river_sf_info is not None
    self.assertEqual(packed_flaps[sf2_name].river_sf_info.parent_joint, 'j2')
    call_kwargs = mock_pack_flaps.call_args[1]
    self.assertCountEqual(
        call_kwargs['sacrificial_polygons'].keys(),
        [sf1_name, sf2_name],
    )

  @mock.patch(
      'bp_pipeline.packing.tiling_solver_local.find_perfect_tiling_from_local_expansions'
  )
  def test_sf_pair_with_local_tiling_merges_tiled_polygons(
      self, mock_tiling, mock_pack_flaps
  ):
    tiled_placement = mock.MagicMock()
    mock_tiled_tracker = mock.MagicMock()
    mock_tiled_tracker.packed_polygons = {'tiled_poly': tiled_placement}
    mock_tiling.return_value = mock_tiled_tracker
    mock_pack_flaps.return_value = {'f1': mock.MagicMock()}
    grid_size = 6
    tracker = utils_tracker.CreasePatternTracker(grid_size)
    river = river_lib.River(
        2, 2, river_lib.RiverOrientation.HORIZONTAL, grid_size
    )
    s1 = stick_lib.Stick(
        name='s1', start='j1', end='j2', length=1, azimuth_angle=0.0
    )
    s1.is_river = True
    sf = stick_figure_lib.StickFigure(
        name='test',
        category='test',
        complexity=1,
        root='s1',
        children=[s1],
    )
    step = {
        'stick': sf['s1'],
        'area_sides': {'j1': 1, 'j2': 1},
        'constraints': {},
    }
    sf1 = types.PolygonPlacement.from_grid_coords(
        offset=(0, 0), size=(2, 2), grid_size=grid_size
    )
    sf1 = dataclasses.replace(
        sf1,
        river_sf_info=types.RiverSfInfo(
            river_name='s1',
            river_side='top',
            parent_joint='j1',
        ),
    )
    sf2 = types.PolygonPlacement.from_grid_coords(
        offset=(0, 4), size=(2, 2), grid_size=grid_size
    )
    sf2 = dataclasses.replace(
        sf2,
        river_sf_info=types.RiverSfInfo(
            river_name='s1',
            river_side='bottom',
            parent_joint='j2',
        ),
    )
    packed_flaps = utils_river_solver.attempt_flap_packing(
        river,
        step,  # pyrefly: ignore[bad-argument-type]
        grid_size,
        tracker,
        [],
        [],
        debug_types.DebugConfig(live_plot=False),
        {'j1': 'top', 'j2': 'bottom'},
        {'top': 4, 'bottom': 8},
        fast_packing_debug=False,
        is_symmetric=False,
        symmetry_axes=None,
        scoring_config=scoring.ScoringConfig(),
        config=config.PackingConfig(
            local_tiling_config=config.LocalTilingConfig()
        ),
        sf_pair=(sf1, sf2),
    )
    self.assertIsNotNone(packed_flaps)
    self.assertIn('tiled_poly', packed_flaps)
    mock_tiling.assert_called_once()

  @mock.patch(
      'bp_pipeline.packing.tiling_solver_local.find_perfect_tiling_from_local_expansions'
  )
  def test_no_sf_pair_skips_local_tiling(self, mock_tiling, mock_pack_flaps):
    mock_pack_flaps.return_value = {'f1': mock.MagicMock()}
    packed_flaps = utils_river_solver.attempt_flap_packing(
        self.river,
        self.step,  # pyrefly: ignore[bad-argument-type]
        self.grid_size,
        self.tracker,
        self.pockets_info,
        self.traversal_plan,
        self.debug_config,
        self.joint_to_side,
        self.side_areas,  # pyrefly: ignore[bad-argument-type]
        fast_packing_debug=False,
        is_symmetric=False,
        symmetry_axes=None,
        scoring_config=scoring.ScoringConfig(),
        config=config.PackingConfig(
            local_tiling_config=config.LocalTilingConfig()
        ),
    )
    self.assertIsNotNone(packed_flaps)
    self.assertNotIn('tiled_poly', packed_flaps)
    mock_tiling.assert_not_called()


class AddRiverCandidateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.grid_size = 5
    self.tracker = utils_tracker.CreasePatternTracker(self.grid_size)

  def test_add_river_candidate_collision_removes_existing(self):
    river = river_lib.River(
        1, 2, river_lib.RiverOrientation.HORIZONTAL, self.grid_size
    )
    existing_placement = types.PolygonPlacement.from_grid_coords(
        offset=(0, 0), size=(2, 2), grid_size=self.grid_size
    )
    self.tracker.add_polygon('f1', existing_placement)
    self.assertIn('f1', self.tracker.packed_polygons)
    new_placement = types.PolygonPlacement.from_grid_coords(
        offset=(0, 3), size=(2, 2), grid_size=self.grid_size
    )
    candidate = {
        'stick_name': 'r1',
        'river': river,
        'packed_flaps': {'f1': new_placement},
        'joint_to_side': {'j1': 'top', 'j2': 'bottom'},
    }
    added_ids = utils_river_solver.add_river_candidate(self.tracker, candidate)
    self.assertIn('f1', self.tracker.packed_polygons)
    self.assertEqual(self.tracker.packed_polygons['f1'], new_placement)
    self.assertEqual(added_ids.count('f1'), 1)


class SolveRiversIterativelyTest(absltest.TestCase):

  @mock.patch(
      'bp_pipeline.packing.utils_river_solver.traversal.packing_traversal'
  )
  @mock.patch.object(packing_utils, 'precompute_flap_pockets')
  @mock.patch.object(utils_tree, 'compute_constraints_along_traversal')
  @mock.patch.object(utils_river_solver, 'find_river_candidates')
  def test_solve_rivers_iteratively_with_backtracking_mocked(
      self,
      mock_find_candidates,
      mock_compute_traversal,
      mock_precompute_pockets,
      mock_packing_traversal,
  ):
    stick1 = mock.MagicMock(spec=stick_lib.Stick)
    stick1.name = 'r1'
    stick1.is_river = True
    stick1.start = 'j1'
    stick1.end = 'j2'
    stick2 = mock.MagicMock(spec=stick_lib.Stick)
    stick2.name = 'r2'
    stick2.is_river = True
    stick2.start = 'j2'
    stick2.end = 'j3'
    stick3 = mock.MagicMock(spec=stick_lib.Stick)
    stick3.name = 'r3'
    stick3.is_river = True
    stick3.start = 'j3'
    stick3.end = 'j4'

    mock_packing_traversal.return_value = [stick1, stick2, stick3]
    mock_compute_traversal.return_value = [
        {'stick': stick1, 'area_sides': {'j1': 0, 'j2': 0}},
        {'stick': stick2, 'area_sides': {'j3': 0, 'j4': 0}},
        {'stick': stick3, 'area_sides': {'j5': 0, 'j6': 0}},
    ]
    mock_precompute_pockets.return_value = []

    cand1a = _create_mock_candidate('r1')
    cand1b = _create_mock_candidate('r1')
    cand2a = _create_mock_candidate('r2')
    cand2b = _create_mock_candidate('r2')
    cand3a = _create_mock_candidate('r3')

    mock_find_candidates.side_effect = [
        iter([cand1a, cand1b]),
        iter([cand2a]),
        iter([]),
        iter([cand2b]),
        iter([cand3a]),
    ]

    tree = mock.MagicMock(spec=stick_figure_lib.StickFigure)
    trackers = utils_river_solver.solve_rivers_iteratively(
        tree, grid_size=10, debug=debug_types.DebugConfig()
    )

    self.assertNotEmpty(trackers)
    tracker = trackers[0]
    self.assertEqual(mock_find_candidates.call_count, 5)
    self.assertIn('r1', tracker.packed_rivers)
    self.assertIn('r2', tracker.packed_rivers)
    self.assertIn('r3', tracker.packed_rivers)
    self.assertEqual(tracker.packed_rivers['r1'], cand1b['river'])
    self.assertEqual(tracker.packed_rivers['r2'], cand2b['river'])
    self.assertEqual(tracker.packed_rivers['r3'], cand3a['river'])

  @mock.patch.object(utils_river_solver, 'find_river_candidates')
  @mock.patch(
      'bp_pipeline.packing.utils_pack_river_flaps.get_pocket_mask_and_cells'
  )
  @mock.patch.object(utils_river_solver, '_pack_pocket_not_enclosed_by_rivers')
  @mock.patch.object(
      utils_river_solver, '_get_next_river_start_cell_from_pocket'
  )
  def test_solve_rivers_iteratively_reduces_area_req_on_intermediate_packing(
      self,
      mock_get_next_river_start_cell,
      mock_pack_pocket,
      mock_get_pocket_mask_and_cells,
      mock_find_candidates,
  ):
    river1 = stick_lib.Stick(
        name='r1', start='j1', end='j2', length=1, azimuth_angle=0
    )
    river1.is_river = True
    flap2 = stick_lib.Stick(
        name='f2', start='j2', end='j3', length=2, azimuth_angle=0
    )
    flap2.is_river = False
    river2 = stick_lib.Stick(
        name='r2', start='j2', end='j4', length=1, azimuth_angle=90
    )
    river2.is_river = True
    tree = stick_figure_lib.StickFigure(
        name='test',
        category='test',
        complexity=1,
        root='r1',
        children=[river1, flap2, river2],
    )

    cand1 = _create_mock_candidate('r1')
    cand1['river'].start_cell = (0, 0)
    cand1['joint_to_side'] = {'j1': 'left', 'j2': 'right'}
    mock_find_candidates.side_effect = [iter([cand1]), iter([])]

    mock_get_pocket_mask_and_cells.return_value = (np.ones((10, 10)), [])

    pocket_tracker = utils_tracker.CreasePatternTracker(10)
    placement = types.PolygonPlacement(
        offset=(5, 0), size=(4, 4), center=mock.MagicMock()
    )
    pocket_tracker.add_polygon('f2', placement)
    mock_pack_pocket.return_value = iter([pocket_tracker])

    mock_get_next_river_start_cell.return_value = (0, 0)

    utils_river_solver.solve_rivers_iteratively(
        tree,
        grid_size=10,
        debug=debug_types.DebugConfig(),
        config=config.PackingConfig(
            traversal_mode=types.TraversalMode.RIVER_PRIORITY
        ),
    )

    self.assertEqual(mock_find_candidates.call_count, 2)
    mock_pack_pocket.assert_called_once()

    step_arg_r2 = mock_find_candidates.call_args_list[1].args[0]
    self.assertEqual(step_arg_r2['stick'].name, 'r2')
    self.assertEqual(step_arg_r2['area_sides'], {'j2': 0.0, 'j4': 0.0})

  def test_solve_rivers_iteratively_human_simple_grid_11(self):
    tree = copy.deepcopy(example_trees.HUMAN_SIMPLE)
    grid_size = 12
    tree['torso'].is_river = True
    trackers_rf = utils_river_solver.solve_rivers_iteratively(
        tree,
        grid_size=grid_size,
        fast_packing_debug=False,
        debug=debug_types.DebugConfig(),
    )
    self.assertNotEmpty(trackers_rf)
    tracker_rf = trackers_rf[0]
    self.assertIn('torso', tracker_rf.packed_rivers)
    river = tracker_rf.packed_rivers['torso']
    self.assertEqual(river.width, 4)
    assert isinstance(river, river_lib.River)
    self.assertIn('head', tracker_rf.packed_polygons)
    self.assertIn('left_arm', tracker_rf.packed_polygons)
    self.assertIn('right_arm', tracker_rf.packed_polygons)
    self.assertIn('left_leg', tracker_rf.packed_polygons)
    self.assertIn('right_leg', tracker_rf.packed_polygons)

  def test_solve_rivers_iteratively_horned_owl(self):
    tree = copy.deepcopy(example_trees.HORNED_OWL)
    tree['body'].is_river = True
    tree['head'].is_river = True

    half_tree = tree.get_symmetric_half()
    symmetry_axes = {
        stick.name: stick.is_on_symmetry_axis for stick in half_tree.children
    }

    trackers_rf = utils_river_solver.solve_rivers_iteratively(
        half_tree,
        grid_size=16,
        is_symmetric=True,
        symmetry_axes=symmetry_axes,
        debug=debug_types.DebugConfig(),
    )
    self.assertNotEmpty(trackers_rf)
    tracker_rf = trackers_rf[0]
    self.assertIn('body', tracker_rf.packed_rivers)
    self.assertIn('head', tracker_rf.packed_rivers)
    self.assertIn('wing_left', tracker_rf.packed_polygons)
    self.assertIn('leg_left', tracker_rf.packed_polygons)
    self.assertIn('horn_tuft_left', tracker_rf.packed_polygons)
    self.assertNotIn('wing_right', tracker_rf.packed_polygons)

  def test_solve_rivers_iteratively_no_solution(self):
    tree = copy.deepcopy(example_trees.HUMAN_SIMPLE)
    tree['torso'].is_river = True
    trackers = utils_river_solver.solve_rivers_iteratively(
        tree, grid_size=4, debug=debug_types.DebugConfig()
    )
    self.assertEmpty(trackers)


class UtilsRiverSolverPrivateHelpersTest(absltest.TestCase):

  @mock.patch('bp_pipeline.packing.utils_backtracking.try_to_backtrack')
  def test_process_solution_and_backtrack_no_first_river_sol_found(
      self, mock_try_to_backtrack
  ):
    mock_try_to_backtrack.return_value = 0
    stack = []
    tracker = mock.MagicMock()
    solutions = []
    debug_config = debug_types.DebugConfig()

    next_step = utils_river_solver.process_solution_and_backtrack(
        stack=stack,
        tracker=tracker,
        solutions=solutions,
        first_river_step_index=-1,
        max_solutions=2,
        debug=debug_config,
    )

    self.assertLen(solutions, 1)
    mock_try_to_backtrack.assert_called_once()
    self.assertEqual(next_step, 0)

  @mock.patch('bp_pipeline.packing.utils_backtracking.try_to_backtrack')
  def test_process_solution_and_backtrack_num_solutions_reached(
      self, mock_try_to_backtrack
  ):
    mock_try_to_backtrack.return_value = 0
    stack = []
    tracker = mock.MagicMock()
    solutions = []
    debug_config = debug_types.DebugConfig()

    next_step = utils_river_solver.process_solution_and_backtrack(
        stack=stack,
        tracker=tracker,
        solutions=solutions,
        first_river_step_index=-1,
        max_solutions=1,
        debug=debug_config,
    )

    self.assertLen(solutions, 1)
    self.assertIsNone(next_step)
    mock_try_to_backtrack.assert_not_called()

  @mock.patch('bp_pipeline.packing.utils_backtracking.try_to_backtrack')
  def test_process_solution_and_backtrack_new_solution_with_first_river(
      self, mock_try_to_backtrack
  ):
    mock_try_to_backtrack.return_value = 0
    river = river_lib.River(1, 1, river_lib.RiverOrientation.HORIZONTAL, 5)
    cand = {'stick_name': 'r1', 'river': river, 'packed_flaps': {}}
    frame = types.RiverStackFrame(
        step_index=0,
        iterator=iter([]),
        applied_candidate=cand,
        object_ids=['r1'],
    )
    stack = [frame]
    tracker = mock.MagicMock()
    solutions = []
    debug_config = debug_types.DebugConfig()

    utils_river_solver.process_solution_and_backtrack(
        stack=stack,
        tracker=tracker,
        solutions=solutions,
        first_river_step_index=0,
        max_solutions=2,
        debug=debug_config,
    )

    self.assertLen(solutions, 1)
    mock_try_to_backtrack.assert_called_once()


class IntermediatePocketUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.grid_size = 10
    self.debug_config = debug_types.DebugConfig(live_plot=False)
    self.flap1 = stick_lib.Stick(
        name='flap1', start='j1', end='j2', length=1, azimuth_angle=0.0
    )
    self.river1 = stick_lib.Stick(
        name='river1', start='j2', end='j3', length=1, azimuth_angle=0.0
    )
    self.river1.is_river = True
    self.tree = stick_figure_lib.StickFigure(
        name='test_tree',
        category='test',
        complexity=1,
        root='river1',
        children=[self.flap1, self.river1],
    )
    self.pocket = {
        'flap_names': {'flap1'},
        'flaps': [self.flap1],
        'boundary_rivers': {'river1'},
    }
    self.traversal_plan = [
        {'stick': self.flap1, 'area_sides': {}, 'constraints': {}},
        {'stick': self.river1, 'area_sides': {}, 'constraints': {}},
    ]
    self.tracker = utils_tracker.CreasePatternTracker(self.grid_size)
    self.placed_river = river_lib.River(
        1, 4, river_lib.RiverOrientation.HORIZONTAL, self.grid_size
    )
    self.tracker.add_river('river1', self.placed_river)
    self.joint_to_side = {'j2': 'top', 'j3': 'bottom'}
    utils_river_solver.add_river_adjacent_cells(
        self.tracker, 'river1', self.placed_river, self.joint_to_side
    )
    self.stack = [
        types.RiverStackFrame(
            step_index=1,
            iterator=iter([]),
            applied_candidate={
                'stick_name': 'river1',
                'river': self.placed_river,
                'joint_to_side': self.joint_to_side,
                'packed_flaps': {},
            },
            object_ids=['river1'],
        )
    ]

  @mock.patch('bp_pipeline.packing.tiling_solver.pack_flaps_with_adjacency')
  def test_pack_pocket_not_enclosed_by_rivers_success(
      self, mock_pack_flaps_with_adjacency
  ):
    pocket_tracker_result = utils_tracker.CreasePatternTracker(self.grid_size)
    pocket_tracker_result.add_polygon(
        'flap1',
        types.PolygonPlacement(
            offset=(0, 0), size=(2, 2), center=mock.MagicMock()
        ),
    )
    mock_pack_flaps_with_adjacency.return_value = iter(
        [(pocket_tracker_result, 0.0, {})]
    )
    pocket_mask = np.ones((self.grid_size, self.grid_size), dtype=bool)
    pocket_river_adj_cells = [(3, 0)]

    pocket_tracker_iterator = (
        utils_river_solver._pack_pocket_not_enclosed_by_rivers(
            self.pocket,
            self.traversal_plan,  # pyrefly: ignore[bad-argument-type]
            self.tracker,
            False,
            None,
            self.debug_config,
            pocket_mask,
            pocket_river_adj_cells,
            config=config.PackingConfig(),
        )
    )
    pocket_tracker = next(pocket_tracker_iterator, None)
    self.assertIsNotNone(pocket_tracker)
    assert pocket_tracker is not None
    self.assertIn('flap1', pocket_tracker.packed_polygons)

  def test_get_adjacent_border_cells(self):
    placement = types.PolygonPlacement(
        offset=(1, 1), size=(2, 2), center=mock.MagicMock()
    )
    occupancy_grid = np.zeros((4, 4), dtype=bool)
    occupancy_grid[1, 1] = occupancy_grid[1, 2] = True
    occupancy_grid[2, 1] = occupancy_grid[2, 2] = True
    border_cells = utils_river_solver._get_adjacent_border_cells(
        placement, 4, occupancy_grid
    )
    expected_border_cells = [
        (0, 1),
        (0, 2),
        (1, 0),
        (2, 0),
        (1, 3),
        (2, 3),
        (3, 1),
        (3, 2),
    ]
    self.assertCountEqual(border_cells, expected_border_cells)

  @parameterized.parameters(False, True)
  def test_get_next_river_start_cell_from_pocket(self, is_symmetric):
    pocket_tracker = utils_tracker.CreasePatternTracker(self.grid_size)
    pocket_tracker.add_polygon(
        'flap1',
        types.PolygonPlacement(
            offset=(0, 0), size=(2, 2), center=mock.MagicMock()
        ),
    )
    start_cell = utils_river_solver._get_next_river_start_cell_from_pocket(
        pocket_tracker,
        self.placed_river,
        self.grid_size,
        is_symmetric=is_symmetric,
    )
    self.assertEqual(start_cell, (0, 2))

  def test_get_next_river_start_cell_from_pocket_symmetric_failure(self):
    pocket_tracker = utils_tracker.CreasePatternTracker(self.grid_size)
    offset_x = self.grid_size // 2 - 1
    pocket_tracker.add_polygon(
        'flap1',
        types.PolygonPlacement(
            offset=(offset_x, 0), size=(2, 2), center=mock.MagicMock()
        ),
    )
    start_cell = utils_river_solver._get_next_river_start_cell_from_pocket(
        pocket_tracker, self.placed_river, self.grid_size, is_symmetric=True
    )
    self.assertEqual(start_cell, (0, offset_x - 1))


class PackIntermediatePocketTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.grid_size = 10
    self.debug_config = debug_types.DebugConfig(live_plot=False)

    flap1 = stick_lib.Stick(
        name='flap1', start='j1', end='j2', length=2, azimuth_angle=0.0
    )
    river1 = stick_lib.Stick(
        name='river1', start='j2', end='j3', length=2, azimuth_angle=0.0
    )
    river1.is_river = True
    self.tree = stick_figure_lib.StickFigure(
        name='test_tree',
        category='test',
        complexity=1,
        root='river1',
        children=[flap1, river1],
    )

    self.pocket = {
        'flap_names': {'flap1'},
        'flaps': [flap1],
        'boundary_rivers': {'river1'},
    }

    self.traversal_plan = [
        {'stick': flap1, 'area_sides': {}, 'constraints': {}},
        {'stick': river1, 'area_sides': {}, 'constraints': {}},
    ]

    self.tracker = utils_tracker.CreasePatternTracker(self.grid_size)
    self.placed_river = river_lib.River(
        2, 4, river_lib.RiverOrientation.HORIZONTAL, self.grid_size
    )
    self.tracker.add_river('river1', self.placed_river)

    self.stack = [
        types.RiverStackFrame(
            step_index=1,
            iterator=iter([]),
            applied_candidate={
                'stick_name': 'river1',
                'river': self.placed_river,
                'joint_to_side': {'j2': 'top', 'j3': 'bottom'},
                'packed_flaps': {},
            },
            object_ids=['river1'],
        )
    ]

  @mock.patch(
      'bp_pipeline.packing.utils_pack_river_flaps.get_pocket_mask_and_cells',
      autospec=True,
  )
  @mock.patch(
      'bp_pipeline.packing.utils_river_solver._pack_pocket_not_enclosed_by_rivers',
      autospec=True,
  )
  @mock.patch(
      'bp_pipeline.packing.utils_river_solver._get_next_river_start_cell_from_pocket',
      autospec=True,
  )
  def test_pack_intermediate_pocket_success(
      self,
      mock_get_next_river_start_cell,
      mock_pack_pocket,
      mock_get_pocket_mask_and_cells,
  ):
    mock_get_pocket_mask_and_cells.return_value = (mock.MagicMock(), [])
    mock_pocket_tracker = mock.MagicMock()
    placement = types.PolygonPlacement(
        offset=(0, 0), size=(2, 2), center=mock.MagicMock()
    )
    mock_pocket_tracker.packed_polygons = {
        'flap1': placement,
    }
    mock_pack_pocket.return_value = iter([mock_pocket_tracker])
    mock_get_next_river_start_cell.return_value = mock.sentinel.start_cell

    result_iterator = utils_river_solver.pack_intermediate_pocket(
        self.pocket,
        self.tree,
        self.traversal_plan,  # pyrefly: ignore[bad-argument-type]
        self.tracker,
        self.stack,
        is_symmetric=False,
        symmetry_axes=None,
        debug=self.debug_config,
        config=config.PackingConfig(),
    )
    result = next(result_iterator, None)

    self.assertIsNotNone(result)
    start_cell, packed_polygons = result
    self.assertEqual(start_cell, mock.sentinel.start_cell)
    self.assertIn('flap1', packed_polygons)
    self.assertEqual(list(packed_polygons.keys()), ['flap1'])
    mock_pack_pocket.assert_called_once()
    mock_get_pocket_mask_and_cells.assert_called_once()

  @mock.patch(
      'bp_pipeline.packing.utils_pack_river_flaps.get_pocket_mask_and_cells'
  )
  @mock.patch(
      'bp_pipeline.packing.utils_river_solver._pack_pocket_not_enclosed_by_rivers'
  )
  def test_pack_intermediate_pocket_packing_fails(
      self, mock_pack_pocket, mock_get_pocket_mask
  ):
    mock_pack_pocket.return_value = iter([])
    mock_get_pocket_mask.return_value = (
        np.ones((self.grid_size, self.grid_size), dtype=bool),
        [],
    )

    initial_packed_polygons = self.tracker.packed_polygons.copy()

    result_iterator = utils_river_solver.pack_intermediate_pocket(
        self.pocket,
        self.tree,
        self.traversal_plan,  # pyrefly: ignore[bad-argument-type]
        self.tracker,
        self.stack,
        is_symmetric=False,
        symmetry_axes=None,
        debug=self.debug_config,
        config=config.PackingConfig(),
    )
    result = next(result_iterator, None)

    self.assertIsNone(result)
    self.assertEqual(self.tracker.packed_polygons, initial_packed_polygons)
    mock_pack_pocket.assert_called_once()
    mock_get_pocket_mask.assert_called_once()

  @mock.patch(
      'bp_pipeline.packing.utils_pack_river_flaps.get_pocket_mask_and_cells'
  )
  @mock.patch(
      'bp_pipeline.packing.utils_river_solver._pack_pocket_not_enclosed_by_rivers'
  )
  def test_pack_intermediate_pocket_no_pocket_mask(
      self, mock_pack_pocket, mock_get_pocket_mask
  ):
    frame = next(
        f for f in self.stack if f.applied_candidate['stick_name'] == 'river1'
    )
    if 'joint_to_side' in frame.applied_candidate:
      del frame.applied_candidate['joint_to_side']

    mock_get_pocket_mask.return_value = (None, [])

    result_iterator = utils_river_solver.pack_intermediate_pocket(
        self.pocket,
        self.tree,
        self.traversal_plan,  # pyrefly: ignore[bad-argument-type]
        self.tracker,
        self.stack,
        is_symmetric=False,
        symmetry_axes=None,
        debug=self.debug_config,
        config=config.PackingConfig(),
    )
    result = next(result_iterator, None)
    self.assertIsNone(result)
    mock_pack_pocket.assert_not_called()
    mock_get_pocket_mask.assert_called_once()

  def test_pack_intermediate_pocket_first_river_start_cell_ccw_from_pocket(
      self,
  ):
    grid_size = 20
    flap_a = stick_lib.Stick(
        name='flap_a', start='j1', end='j2', length=1, azimuth_angle=0.0
    )
    flap_b = stick_lib.Stick(
        name='flap_b', start='j2', end='j3', length=1, azimuth_angle=0.0
    )
    river1 = stick_lib.Stick(
        name='river1', start='j3', end='j4', length=1, azimuth_angle=0.0
    )
    river1.is_river = True
    tree = stick_figure_lib.StickFigure(
        name='test_ccw',
        category='test',
        complexity=1,
        root='river1',
        children=[flap_a, flap_b, river1],
    )
    pocket = {
        'flap_names': {'flap_a', 'flap_b'},
        'flaps': [flap_a, flap_b],
        'boundary_rivers': {'river1'},
    }
    traversal_plan = [
        {'stick': flap_a, 'area_sides': {}, 'constraints': {}},
        {'stick': flap_b, 'area_sides': {}, 'constraints': {}},
        {'stick': river1, 'area_sides': {}, 'constraints': {}},
    ]
    tracker = utils_tracker.CreasePatternTracker(grid_size)
    tracker.add_polygon(
        'flap_a',
        types.PolygonPlacement(
            offset=(9, 0), size=(2, 2), center=mock.MagicMock()
        ),
    )
    tracker.add_polygon(
        'flap_b',
        types.PolygonPlacement(
            offset=(7, 0), size=(2, 2), center=mock.MagicMock()
        ),
    )
    stack = []

    result_iterator = utils_river_solver.pack_intermediate_pocket(
        pocket,
        tree,
        traversal_plan,  # pyrefly: ignore[bad-argument-type]
        tracker,
        stack,
        is_symmetric=False,
        symmetry_axes=None,
        debug=self.debug_config,
        config=config.PackingConfig(),
    )
    result = next(result_iterator, None)
    self.assertIsNotNone(result)
    start_cell, _ = result
    ccw_wrong = (0, 11)
    self.assertNotEqual(
        start_cell,
        ccw_wrong,
    )
    self.assertEqual(start_cell, (0, 6))

  def test_pack_intermediate_pocket_memoization(self):
    pocket_tiling_cache = {}
    cfg = config.PackingConfig(memoize_pocket_tiling=True)
    with (
        mock.patch(
            'bp_pipeline.packing.utils_pack_river_flaps.get_pocket_mask_and_cells'
        ) as mock_get_mask,
        mock.patch(
            'bp_pipeline.packing.utils_river_solver._pack_pocket_not_enclosed_by_rivers'
        ) as mock_pack_pocket,
        mock.patch(
            'bp_pipeline.packing.utils_river_solver._get_next_river_start_cell_from_pocket'
        ) as mock_start,
    ):
      mock_get_mask.return_value = (
          np.ones((self.grid_size, self.grid_size), dtype=bool),
          [],
      )
      mock_tracker = mock.MagicMock()
      mock_tracker.packed_polygons = {
          'flap1': types.PolygonPlacement(
              offset=(0, 0), size=(2, 2), center=mock.MagicMock()
          ),
      }
      mock_pack_pocket.return_value = iter([mock_tracker])
      mock_start.return_value = (0, 1)

      # First call: cache miss, computes and populates cache.
      iter1 = list(
          utils_river_solver.pack_intermediate_pocket(
              self.pocket,
              self.tree,
              self.traversal_plan,  # pyrefly: ignore[bad-argument-type]
              self.tracker,
              self.stack,
              is_symmetric=False,
              symmetry_axes=None,
              debug=self.debug_config,
              config=cfg,
              pocket_tiling_cache=pocket_tiling_cache,
          )
      )
      self.assertLen(iter1, 1)
      self.assertEqual(mock_pack_pocket.call_count, 1)
      self.assertLen(pocket_tiling_cache, 1)

      # Second call: cache hit, returns from cache without calling
      # _pack_pocket_not_enclosed_by_rivers.
      iter2 = list(
          utils_river_solver.pack_intermediate_pocket(
              self.pocket,
              self.tree,
              self.traversal_plan,  # pyrefly: ignore[bad-argument-type]
              self.tracker,
              self.stack,
              is_symmetric=False,
              symmetry_axes=None,
              debug=self.debug_config,
              config=cfg,
              pocket_tiling_cache=pocket_tiling_cache,
          )
      )
      self.assertLen(iter2, 1)
      self.assertEqual(mock_pack_pocket.call_count, 1)

  def test_pack_intermediate_pocket_failure_set_memoization(self):
    pocket_tiling_cache = {}
    cfg = config.PackingConfig(memoize_pocket_tiling=True)
    with (
        mock.patch(
            'bp_pipeline.packing.utils_pack_river_flaps.get_pocket_mask_and_cells'
        ) as mock_get_mask,
        mock.patch(
            'bp_pipeline.packing.utils_river_solver._pack_pocket_not_enclosed_by_rivers'
        ) as mock_pack_pocket,
    ):
      mock_get_mask.return_value = (
          np.ones((self.grid_size, self.grid_size), dtype=bool),
          [],
      )
      mock_pack_pocket.return_value = iter([])

      # First call: cache miss, runs solver and records failure set ([]).
      iter1 = list(
          utils_river_solver.pack_intermediate_pocket(
              self.pocket,
              self.tree,
              self.traversal_plan,  # pyrefly: ignore[bad-argument-type]
              self.tracker,
              self.stack,
              is_symmetric=False,
              symmetry_axes=None,
              debug=self.debug_config,
              config=cfg,
              pocket_tiling_cache=pocket_tiling_cache,
          )
      )
      self.assertEmpty(iter1)
      self.assertEqual(mock_pack_pocket.call_count, 1)
      self.assertLen(pocket_tiling_cache, 1)
      self.assertEqual(list(pocket_tiling_cache.values())[0], [])

      # Second call: cache hit on failure set, returns immediately without
      # calling solver.
      iter2 = list(
          utils_river_solver.pack_intermediate_pocket(
              self.pocket,
              self.tree,
              self.traversal_plan,  # pyrefly: ignore[bad-argument-type]
              self.tracker,
              self.stack,
              is_symmetric=False,
              symmetry_axes=None,
              debug=self.debug_config,
              config=cfg,
              pocket_tiling_cache=pocket_tiling_cache,
          )
      )
      self.assertEmpty(iter2)
      self.assertEqual(mock_pack_pocket.call_count, 1)

  def test_pack_intermediate_pocket_raises_cooperative_timeout(self):
    cfg = config.PackingConfig()
    with (
        mock.patch(
            'bp_pipeline.packing.utils_pack_river_flaps.get_pocket_mask_and_cells'
        ) as mock_get_mask,
        mock.patch(
            'bp_pipeline.packing.utils_river_solver._pack_pocket_not_enclosed_by_rivers'
        ) as mock_pack_pocket,
    ):
      mock_get_mask.return_value = (
          np.ones((self.grid_size, self.grid_size), dtype=bool),
          [],
      )
      mock_pack_pocket.return_value = iter([self.tracker])

      with timeout.cooperative_timeout(datetime.timedelta(microseconds=1)):
        time.sleep(0.002)
        with self.assertRaises(timeout.CooperativeTimeoutError):
          list(
              utils_river_solver.pack_intermediate_pocket(
                  self.pocket,
                  self.tree,
                  self.traversal_plan,  # pyrefly: ignore[bad-argument-type]
                  self.tracker,
                  self.stack,
                  is_symmetric=False,
                  symmetry_axes=None,
                  debug=self.debug_config,
                  config=cfg,
              )
          )


if __name__ == '__main__':
  absltest.main()
