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

import copy
import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from bp_pipeline import bus
from bp_pipeline import codec
from bp_pipeline import pipeline
from bp_pipeline import rectangle as rectangle_lib
from bp_pipeline.crease_pattern import crease
from bp_pipeline.crease_pattern import crease_pattern as cp_lib
from bp_pipeline.crease_pattern import point
from bp_pipeline.crease_pattern import visualize as cp_visualize
from bp_pipeline.crease_pattern.parser import svg as svg_parser
from bp_pipeline.packing import config as packing_config_lib
from bp_pipeline.packing import types as packing_types
from bp_pipeline.packing import utils_packing_svg
from bp_pipeline.shaping import bus_adapter
from bp_pipeline.shaping import geometric_fold_3d_heuristics
from bp_pipeline.shaping import shaping_elements
from bp_pipeline.solving import hinges as hinge_utils
from bp_pipeline.solving import partition as partition_lib
from bp_pipeline.stick_figures import visualize as sf_visualize
from bp_pipeline.stick_figures.examples import example_trees
import matplotlib.pyplot as plt


class BusTest(parameterized.TestCase):

  def test_grid_primitives_and_flap_type(self):
    rect = rectangle_lib.Rectangle(
        name='flap_a', x_start=0, y_start=2, x_len=3, y_len=4
    )
    grid_rect = bus.GridRect.from_rectangle(rect)
    self.assertEqual(grid_rect.x, 0)
    self.assertEqual(grid_rect.y, 2)
    self.assertEqual(grid_rect.width, 3)
    self.assertEqual(grid_rect.height, 4)
    self.assertEqual(grid_rect.w, 3)
    self.assertEqual(grid_rect.h, 4)
    self.assertEqual(grid_rect.to_rectangle(name='flap_a'), rect)
    self.assertEqual(grid_rect.flap_type(grid_size=8), bus.FlapType.EDGE)
    self.assertEqual(
        grid_rect.scaled(2), bus.GridRect(x=0, y=4, width=6, height=8)
    )
    self.assertEqual(
        grid_rect.normalized(grid_size=8),
        rectangle_lib.Rectangle(
            name='', x_start=0.0, y_start=0.25, x_len=0.375, y_len=0.5
        ),
    )

    p = bus.GridPoint.from_normalized_point(
        point.Point(x=0.25, y=0.75), grid_size=8
    )
    self.assertEqual(p, bus.GridPoint(2, 6))
    self.assertEqual(p.scaled(2), bus.GridPoint(4, 12))

    seg = bus.GridSegment(bus.GridPoint(4, 2), bus.GridPoint(1, 2))
    self.assertEqual(
        seg.canonical(),
        bus.GridSegment(bus.GridPoint(1, 2), bus.GridPoint(4, 2)),
    )
    norm_crease = seg.to_normalized_crease(grid_size=8, fold=crease.Fold.HINGE)
    self.assertEqual(
        bus.GridSegment.from_normalized_crease(norm_crease, grid_size=8),
        seg.canonical(),
    )

    corner = bus.GridRect(x=0, y=0, width=2, height=2)
    self.assertEqual(corner.flap_type(grid_size=8), bus.FlapType.CORNER)

    internal = bus.GridRect(x=2, y=2, width=2, height=2)
    self.assertEqual(internal.flap_type(grid_size=8), bus.FlapType.INTERNAL)

    unclipped_border = rectangle_lib.Rectangle(
        name='border_flap', x_start=-3, y_start=6, x_len=6, y_len=9
    )
    clipped_border = bus.GridRect.from_rectangle(unclipped_border, grid_size=12)
    self.assertEqual(clipped_border, bus.GridRect(x=0, y=6, width=3, height=6))
    self.assertEqual(
        bus.coalesce_grid_rects([
            bus.GridRect(x=0, y=2, width=1, height=1),
            bus.GridRect(x=1, y=2, width=1, height=1),
            bus.GridRect(x=0, y=3, width=1, height=1),
            bus.GridRect(x=1, y=3, width=1, height=1),
        ]),
        (bus.GridRect(x=0, y=2, width=2, height=2),),
    )

    with self.assertRaisesRegex(ValueError, 'must be an integer'):
      bus.GridRect.from_rectangle(
          rectangle_lib.Rectangle(
              name='fractional', x_start=0.5, y_start=0, x_len=2, y_len=2
          )
      )

  def test_design_tree_immutability_and_provenance(self):
    base_sf = copy.deepcopy(example_trees.get_example_by_name('human'))
    sf, _ = base_sf.subdivide_sticks({'torso': [2, 2]})
    tree = bus.DesignTree.from_stick_figure(sf)

    self.assertEqual(tree.source, sf)
    self.assertIn('torso_0_torso_1', tree.merged_from)
    self.assertEqual(
        tree.merged_from['torso_0_torso_1'], ('torso_0', 'torso_1')
    )
    for stick in tree.merged.children:
      self.assertIn(stick.name, tree.merged_from)
      if stick.is_on_symmetry_axis:
        self.assertEqual(tree.symmetric_partner[stick.name], stick.name)
      elif stick.symmetric_stick_name is not None:
        self.assertEqual(
            tree.symmetric_partner[stick.name], stick.symmetric_stick_name
        )

    with self.assertRaises(dataclasses.FrozenInstanceError):
      tree.min_grid_size = 99  # type: ignore[misc]
    with self.assertRaises(TypeError):
      tree.merged_from['new'] = ('a',)  # type: ignore[index]

    decoded = codec.loads(codec.dumps(tree))
    self.assertEqual(decoded, tree)

  @parameterized.parameters('human', 'cat', 'spider')
  def test_design_tree_pipeline_characterization_parity(self, tree_name: str):
    sf = copy.deepcopy(example_trees.get_example_by_name(tree_name))
    design_tree = bus.DesignTree.from_stick_figure(sf)

    input_from_sf = pipeline.process_stick_figure(sf)
    input_from_dt = pipeline.process_stick_figure(design_tree)
    self.assertEqual(
        input_from_sf.merged_stick_name_dict,
        input_from_dt.merged_stick_name_dict,
    )
    self.assertEqual(
        input_from_sf.max_grid_size,
        input_from_dt.max_grid_size,
    )
    self.assertEqual(
        input_from_sf.merged_stick_figure.stick_figure,
        input_from_dt.merged_stick_figure.stick_figure,
    )

    pack_cfg = pipeline.PackingTaskConfig(
        max_number_of_cps=1,
        max_cps_per_grid_size=1,
        max_retries=5,
        sweep_timeout_seconds=30,
        packing_config=packing_config_lib.PackingConfig(
            use_level_shifter=False,
            enforce_proximity_constraints=False,
        ),
    )
    pack_sf = pipeline.pack_crease_pattern(copy.deepcopy(sf), config=pack_cfg)
    pack_dt = pipeline.pack_crease_pattern(
        bus.DesignTree.from_stick_figure(copy.deepcopy(sf)), config=pack_cfg
    )
    self.assertEqual(
        pack_sf.packing_cps[0].hash_value,
        pack_dt.packing_cps[0].hash_value,
    )

    solve_cfg = pipeline.SolvingTaskConfig(
        max_number_of_cps=1,
        max_steps=1000,
        sweep_timeout_seconds=30,
    )
    solve_sf = pipeline.solve_crease_pattern(pack_sf, config=solve_cfg)
    solve_dt = pipeline.solve_crease_pattern(pack_dt, config=solve_cfg)
    self.assertEqual(
        solve_sf.flat_foldable_cps[0].hash_value,
        solve_dt.flat_foldable_cps[0].hash_value,
    )

  @parameterized.parameters('human', 'cat')
  def test_direct_cp_builder_and_packed_solved_bus_parity(self, tree_name: str):
    sf = copy.deepcopy(example_trees.get_example_by_name(tree_name))
    design_tree = bus.DesignTree.from_stick_figure(sf)
    pack_cfg = pipeline.PackingTaskConfig(
        max_number_of_cps=1,
        max_cps_per_grid_size=1,
        max_retries=5,
        sweep_timeout_seconds=30,
        packing_config=packing_config_lib.PackingConfig(
            use_level_shifter=False,
            enforce_proximity_constraints=False,
        ),
    )
    pack_res = pipeline.pack_crease_pattern(design_tree, config=pack_cfg)
    pack_out = pack_res.packing_outputs[0]
    grid_size = pack_out.packing_cp.grid_size

    # Step 5 verification: direct CreasePattern builder matches SVG round-trip.
    svg_str, svg_rects = utils_packing_svg.convert_packing_to_svg(
        grid_size,
        pack_out.tracker,
    )
    svg_cp = svg_parser.parse_svg(svg_str)
    svg_cp.grid_size = grid_size
    self.assertEqual(pack_out.packing_cp.hash_value, svg_cp.hash_value)
    self.assertEqual(pack_out.flap_rectangles, svg_rects)

    # Step 4 verification: PackedCreasePattern & HingeIndex.
    packed_cp = bus.PackedCreasePattern.from_packing_output(
        pack_out, design=design_tree
    )
    expected_stroke = round(max(1, grid_size) / 450.0, 4)
    self.assertEqual(packed_cp.grid_size, grid_size)
    packed_svg = packed_cp.to_svg()
    self.assertIn('<svg', packed_svg)
    self.assertIn('fill="none"', packed_svg)
    self.assertIn(f'stroke-width="{expected_stroke}"', packed_svg)
    self.assertIn('stroke-width="0.02"', packed_cp.to_svg(stroke_width=0.02))
    for flap in packed_cp.flaps.values():
      self.assertGreaterEqual(flap.rect.x, 0)
      self.assertGreater(flap.rect.width, 0)
      self.assertLessEqual(flap.rect.x + flap.rect.width, packed_cp.grid_size)
      self.assertGreaterEqual(flap.rect.y, 0)
      self.assertGreater(flap.rect.height, 0)
      self.assertLessEqual(flap.rect.y + flap.rect.height, packed_cp.grid_size)
    for river in packed_cp.rivers.values():
      for r_rect in river.rects:
        self.assertGreaterEqual(r_rect.x, 0)
        self.assertLessEqual(r_rect.x + r_rect.width, packed_cp.grid_size)
        self.assertGreaterEqual(r_rect.y, 0)
        self.assertLessEqual(r_rect.y + r_rect.height, packed_cp.grid_size)
    for h in packed_cp.hinges:
      self.assertEqual(packed_cp.hinge_at[h.id], h)
      self.assertEqual(packed_cp.hinge_at[h.segment].segment, h.segment)
      norm_c = h.segment.to_normalized_crease(
          grid_size=packed_cp.grid_size, fold=crease.Fold.HINGE
      )
      overlapping = packed_cp.hinge_at.find_overlapping(norm_c)
      self.assertIsNotNone(overlapping)
      self.assertIn(overlapping, packed_cp.hinges)

    standalone_packed = bus.PackedCreasePattern.from_crease_pattern(
        packed_cp.creases
    )
    self.assertIsNone(standalone_packed.design)
    self.assertEqual(standalone_packed.grid_size, packed_cp.grid_size)
    self.assertNotEmpty(standalone_packed.hinges)

    # Verify hinge solver and HingeMetaData.has_central_flap work identically
    # on PackedCreasePattern vs legacy PackingOutput.
    solver_cfg = hinge_utils.SolverConfig(
        max_steps=1500,
        max_time_limit=30.0,
        max_number_of_cps=1,
        disallow_trapping_edge_flaps=True,
        hinges_from_stick_figure=True,
        remove_central_crease=True,
        up_down_method='azimuth',
    )
    legacy_solver = solver_cfg.make_from_cp(
        pack_out.packing_cp,
        grid_size=grid_size,
        packing_information=pack_out,
    )
    bus_solver = solver_cfg.make_from_cp(
        packed_cp.creases,
        grid_size=packed_cp.grid_size,
        packing_information=packed_cp,
    )
    for h_c in bus_solver.init_state.unvisited_hinges:
      self.assertEqual(
          bus_solver.init_state.hinge_dict[h_c.hash_value].has_central_flap,
          legacy_solver.init_state.hinge_dict[h_c.hash_value].has_central_flap,
      )
    legacy_solutions = legacy_solver.solve()
    bus_solutions = bus_solver.solve()
    self.assertEqual(
        bus_solutions[0].crease_pattern.hash_value,
        legacy_solutions[0].crease_pattern.hash_value,
    )

    solved_cp = bus.SolvedCreasePattern.from_solver_result(
        bus_solutions[0],
        packed=packed_cp,
        config=solver_cfg,
        scale_factor=1,
    )
    self.assertEqual(solved_cp.grid_size, packed_cp.grid_size)
    solved_svg = solved_cp.to_svg()
    self.assertIn('<svg', solved_svg)
    self.assertIn('fill="none"', solved_svg)
    self.assertIn(f'stroke-width="{expected_stroke}"', solved_svg)
    fold_dict = solved_cp.flat_fold()
    self.assertIn('vertices_coords', fold_dict)
    self.assertIn('edges_vertices', fold_dict)

    fig, _ = cp_visualize.draw_cp(
        packed_cp.creases,
        grid_size=packed_cp.grid_size,
        flaps=list(packed_cp.flaps.values()),
        rivers=list(packed_cp.rivers.values()),
        lw=0.65,
    )
    plt.close(fig)

    shaped_cp = bus.ShapedCreasePattern.from_crease_pattern(
        solved_cp.creases, solved=solved_cp
    )
    shaped_folded = shaped_cp.flat_fold(strict=False)
    self.assertIn('vertices_coords', shaped_folded)
    self.assertIn('edges_vertices', shaped_folded)

  def test_unclipped_flap_rectangle_scorpion_5x9_repro(self):
    # Scorpion 5x9 corner flap repro (grid_size=22): claw_right stretched
    # vertically to 5x9 (x=17..22, y=13..22) with circle center at (20, 18).
    # Internal `_flap_length_from_polygon` uses `min(polygon.size) // 2`
    # (rx=3, ry=5 -> grid_radius=3 -> 6x6 at x=17..23, y=15..21), avoiding
    # horizontal inflation into adjacent leg_2_right (x=11..17).
    claw_right = bus.Flap(
        id='claw_right',
        stick='claw_right',
        rect=bus.GridRect(x=17, y=13, width=5, height=9),
        circle_centers=(bus.GridPoint(20, 18),),
        flap_type=bus.FlapType.CORNER,
    )
    unclipped_claw = bus_adapter.unclipped_flap_rectangle(
        claw_right, grid_size=22
    )
    self.assertEqual(unclipped_claw.x_start, 17.0)
    self.assertEqual(unclipped_claw.y_start, 15.0)
    self.assertEqual(unclipped_claw.x_len, 6.0)
    self.assertEqual(unclipped_claw.y_len, 6.0)

    # Scorpion 5x6 edge flap repro (grid_size=22): leg_3_right stretched
    # horizontally to 5x6 (x=17..22, y=3..9) with circle center at (21, 6)
    # (rx=4, ry=3 -> grid_radius=3 -> 6x6 at x=18..24, y=3..9), avoiding
    # vertical inflation into leg_4_right (y=0..3).
    leg_3_right = bus.Flap(
        id='leg_3_right',
        stick='leg_3_right',
        rect=bus.GridRect(x=17, y=3, width=5, height=6),
        circle_centers=(bus.GridPoint(21, 6),),
        flap_type=bus.FlapType.EDGE,
    )
    unclipped_leg = bus_adapter.unclipped_flap_rectangle(
        leg_3_right, grid_size=22
    )
    self.assertEqual(unclipped_leg.x_start, 18.0)
    self.assertEqual(unclipped_leg.y_start, 3.0)
    self.assertEqual(unclipped_leg.x_len, 6.0)
    self.assertEqual(unclipped_leg.y_len, 6.0)

    # Full-width top-border flap of unclipped size 12x4 (x=0..12, y=0..4 on
    # grid_size=12, circle centers at y=2) has min(12, 4) // 2 = 2 (ry=2).
    full_width_cap_r2 = bus.Flap(
        id='head_r2',
        stick='head_r2',
        rect=bus.GridRect(x=0, y=0, width=12, height=4),
        circle_centers=(bus.GridPoint(0, 2), bus.GridPoint(12, 2)),
        flap_type=bus.FlapType.CORNER,
    )
    unclipped_cap_r2 = bus_adapter.unclipped_flap_rectangle(
        full_width_cap_r2, grid_size=12
    )
    self.assertEqual(unclipped_cap_r2.x_start, -2.0)
    self.assertEqual(unclipped_cap_r2.y_start, 0.0)
    self.assertEqual(unclipped_cap_r2.x_len, 16.0)
    self.assertEqual(unclipped_cap_r2.y_len, 4.0)

    # Full-width top-border flap of unclipped size 12x8 (y=-4..4 clipped to
    # x=0..12, y=0..4 on grid_size=12, circle centers on the top border y=0)
    # has min(12, 8) // 2 = 4 (ry=4).
    full_width_cap_r4 = bus.Flap(
        id='head_r4',
        stick='head_r4',
        rect=bus.GridRect(x=0, y=0, width=12, height=4),
        circle_centers=(bus.GridPoint(0, 0), bus.GridPoint(12, 0)),
        flap_type=bus.FlapType.CORNER,
    )
    unclipped_cap_r4 = bus_adapter.unclipped_flap_rectangle(
        full_width_cap_r4, grid_size=12
    )
    self.assertEqual(unclipped_cap_r4.x_start, -4.0)
    self.assertEqual(unclipped_cap_r4.y_start, -4.0)
    self.assertEqual(unclipped_cap_r4.x_len, 20.0)
    self.assertEqual(unclipped_cap_r4.y_len, 8.0)

  def test_reserved_region_and_packing_output_round_trip(self):
    self.assertEqual(
        [f.name for f in dataclasses.fields(bus.ReservedRegion)],
        ['id', 'rect'],
    )
    region = bus.ReservedRegion(
        id='axis_filler_0', rect=bus.GridRect(x=5, y=7, width=2, height=5)
    )
    with self.assertRaises(dataclasses.FrozenInstanceError):
      region.id = 'other'  # type: ignore[misc]

    sf = copy.deepcopy(example_trees.get_example_by_name('human'))
    design = bus.DesignTree.from_stick_figure(sf)
    pack_cfg = pipeline.PackingTaskConfig(
        max_number_of_cps=1,
        max_cps_per_grid_size=1,
        max_retries=5,
        sweep_timeout_seconds=30,
        packing_config=packing_config_lib.PackingConfig(
            remove_central_crease=True
        ),
    )
    pack_res = pipeline.pack_crease_pattern(design, config=pack_cfg)
    packed = bus.PackedCreasePattern.from_packing_output(
        pack_res.packing_outputs[0], design=design
    )
    self.assertNotEmpty(packed.reservations)
    self.assertTrue(
        all(
            r.id.startswith(packing_types.AXIS_FILLER_PREFIX)
            for r in packed.reservations
        )
    )
    self.assertFalse(
        any(
            fid.startswith(packing_types.AXIS_FILLER_PREFIX)
            for fid in packed.flaps
        )
    )

    reconstructed = bus.PackedCreasePattern.from_packing_output(
        packed.to_packing_output(), design=design
    )
    self.assertEqual(reconstructed.reservations, packed.reservations)
    self.assertEqual(reconstructed.flaps.keys(), packed.flaps.keys())

    # Non-prefixed IDs are prefixed across PackingOutput round-trip.
    custom = dataclasses.replace(
        packed,
        reservations=(
            bus.ReservedRegion(id='strip_0', rect=packed.reservations[0].rect),
        ),
    )
    custom_rt = bus.PackedCreasePattern.from_packing_output(
        custom.to_packing_output(), design=design
    )
    self.assertEqual(
        custom_rt.reservations,
        (
            bus.ReservedRegion(
                id=f'{packing_types.AXIS_FILLER_PREFIX}_strip_0',
                rect=packed.reservations[0].rect,
            ),
        ),
    )

  def test_lazy_visualization_entrypoints(self):
    sf = copy.deepcopy(example_trees.get_example_by_name('cat'))
    design = bus.DesignTree.from_stick_figure(sf)
    pack_cfg = pipeline.PackingTaskConfig(
        max_number_of_cps=1,
        max_cps_per_grid_size=1,
        max_retries=5,
        sweep_timeout_seconds=30,
        packing_config=packing_config_lib.PackingConfig(
            use_level_shifter=False,
            enforce_proximity_constraints=False,
        ),
    )
    pack_res = pipeline.pack_crease_pattern(design, config=pack_cfg)
    pack_out = pack_res.packing_outputs[0]
    grid_size = pack_out.packing_cp.grid_size
    packed = bus.PackedCreasePattern.from_packing_output(
        pack_out, design=design
    )

    parts = partition_lib.partition_hinges_and_ridges(
        hinges=[
            c for c in packed.creases.creases if c.fold == crease.Fold.HINGE
        ],
        ridges=[
            c
            for c in packed.creases.creases
            if c.fold
            in (crease.Fold.MOUNTAIN, crease.Fold.VALLEY, crease.Fold.TRIANGLE)
        ],
        borders=[
            c for c in packed.creases.creases if c.fold == crease.Fold.BORDER
        ],
        vertices=packed.creases.vertices,
    )
    self.assertNotEmpty(parts)
    fig_p, ax_p = parts[0].draw_as_cp()
    self.assertIsNotNone(ax_p)
    plt.close(fig_p)

    invalid_cp = cp_lib.CreasePattern(
        [
            crease.Crease(
                point.Point(x=0.0, y=0.0),
                point.Point(x=1.0, y=0.0),
                fold=crease.Fold.BORDER,
            ),
            crease.Crease(
                point.Point(x=1.0, y=0.0),
                point.Point(x=1.0, y=1.0),
                fold=crease.Fold.BORDER,
            ),
            crease.Crease(
                point.Point(x=1.0, y=1.0),
                point.Point(x=0.0, y=1.0),
                fold=crease.Fold.BORDER,
            ),
            crease.Crease(
                point.Point(x=0.0, y=1.0),
                point.Point(x=0.0, y=0.0),
                fold=crease.Fold.BORDER,
            ),
            crease.Crease(
                point.Point(x=0.0, y=0.5),
                point.Point(x=0.5, y=0.5),
                fold=crease.Fold.MOUNTAIN,
            ),
            crease.Crease(
                point.Point(x=0.5, y=0.5),
                point.Point(x=1.0, y=0.5),
                fold=crease.Fold.MOUNTAIN,
            ),
        ],
        light_mode=False,
        merge_trivial=False,
    )
    fig_inv, ax_inv = cp_visualize.draw_cp(
        invalid_cp,
        show_vertices=True,
        show_only_not_flat_foldable_vertices=False,
        show_invalid=True,
    )
    self.assertNotEmpty(ax_inv.patches)
    plt.close(fig_inv)

    solver_cfg = hinge_utils.SolverConfig(
        max_steps=1500, max_time_limit=30.0, max_number_of_cps=1
    )
    solver = solver_cfg.make_from_cp(
        pack_out.packing_cp,
        grid_size=grid_size,
        packing_information=pack_out,
    )
    sol_res = solver.solve()[0]
    solved_cp = sol_res.crease_pattern
    png_bytes = cp_visualize.render_highlighted_cps_to_png_bytes(
        [solved_cp], [pack_out]
    )
    self.assertTrue(png_bytes.startswith(b'\x89PNG'))

    s_flaps, s_rivers = (
        shaping_elements.convert_packing_elements_to_shaping_elements(
            packing_rectangles=pack_out.flap_rectangles,
            packing_rivers=pack_out.packing_rivers,
            packing_grid_size=grid_size,
            solving_grid_size=solved_cp.grid_size,
            base_cp=solved_cp,
        )
    )
    fig_se, ax_se = shaping_elements.visualize_shaping_elements(
        s_flaps, s_rivers, grid_size=solved_cp.grid_size
    )
    self.assertIsNotNone(ax_se)
    plt.close(fig_se)

    wrapper = (
        geometric_fold_3d_heuristics.shape_cp_with_3d_geometric_folding_heuristics_pipeline_wrapper
    )
    shaped_cp = wrapper(
        base_cp=solved_cp,
        stick_figure=design.merged,
        packing_output=pack_out,
        solving_output=sol_res,
        solving_grid_size=solved_cp.grid_size,
        central_crease_removed=False,
        shape_with_unmerged_tree=False,
        visualize=True,
        shaping_flaps=s_flaps,
        shaping_rivers=s_rivers,
    )
    self.assertGreaterEqual(len(shaped_cp.creases), len(solved_cp.creases))

    with mock.patch.dict(
        'sys.modules', {'plotly': None, 'plotly.graph_objects': None}
    ):
      with self.assertRaisesRegex(ImportError, 'plotly is required'):
        sf_visualize.visualize_stick_figure_3d(sf)


if __name__ == '__main__':
  absltest.main()
