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

from collections.abc import Mapping
import copy
import dataclasses
import json
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import bp_pipeline as bp
from bp_pipeline import bus
from bp_pipeline import codec
from bp_pipeline import pipeline
from bp_pipeline import rectangle as rectangle_lib
from bp_pipeline.packing import config as packing_config_lib
from bp_pipeline.packing import packing_output as packing_output_lib
from bp_pipeline.packing import river as river_lib
from bp_pipeline.packing import types as packing_types
from bp_pipeline.solving import hinges as hinge_utils
from bp_pipeline.stick_figures import stick as stick_lib
from bp_pipeline.stick_figures import stick_figure as stick_figure_lib
from bp_pipeline.stick_figures.examples import example_trees


def _assert_no_double_encoded_json(
    test_case: absltest.TestCase, value: Any, path: str = 'root'
) -> None:
  if isinstance(value, str):
    stripped = value.strip()
    if (stripped.startswith('{') and stripped.endswith('}')) or (
        stripped.startswith('[') and stripped.endswith(']')
    ):
      test_case.fail(
          f'Found double-encoded JSON string at {path}: {value[:80]!r}'
      )
  elif isinstance(value, dict):
    for k, v in value.items():
      _assert_no_double_encoded_json(test_case, v, f'{path}.{k}')
  elif isinstance(value, (list, tuple)):
    for idx, v in enumerate(value):
      _assert_no_double_encoded_json(test_case, v, f'{path}[{idx}]')


class CodecTest(parameterized.TestCase):

  def test_all_example_trees_round_trip_losslessly(self):
    examples = example_trees.list_examples()
    self.assertGreaterEqual(len(examples), 18)
    for example in examples:
      with self.subTest(name=example.name):
        sf = copy.deepcopy(example)
        sf.children[0].flap_tip_shape = stick_lib.FlapTipShape.RECTANGULAR
        if len(sf.children) > 1:
          sf.children[1].flap_tip_shape = stick_lib.FlapTipShape.TRIANGULAR

        encoded = codec.to_dict(sf)
        _assert_no_double_encoded_json(self, encoded)
        self.assertEqual(encoded['schema_version'], codec.SCHEMA_VERSION)
        self.assertEqual(encoded['type'], 'StickFigure')

        decoded = codec.loads(codec.dumps(sf))
        self.assertEqual(codec.to_dict(decoded), encoded)
        self.assertEqual(decoded, sf)

        tree = bus.DesignTree.from_stick_figure(sf)
        encoded_tree = codec.to_dict(tree)
        _assert_no_double_encoded_json(self, encoded_tree)
        decoded_tree = codec.loads(codec.dumps(tree))
        self.assertEqual(codec.to_dict(decoded_tree), encoded_tree)
        self.assertEqual(decoded_tree, tree)

  def test_river_subtypes_round_trip_losslessly(self):
    straight = river_lib.River(
        width=2,
        offset=3,
        orientation=river_lib.RiverOrientation.HORIZONTAL,
        grid_size=12,
    )
    l_shaped = river_lib.LShapedRiver(
        width=1,
        l_shape_type=river_lib.LShapeType.TOP_RIGHT,
        turn_x=4,
        turn_y=5,
        grid_size=12,
    )
    hugging = river_lib.HuggingRiver(
        width=1,
        subriver_paths=[[(0, 2), (1, 2), (2, 2)]],
        hug_side=river_lib.HugSide.LEFT,
        is_on_axis=True,
        grid_size=12,
        variant_type=river_lib.PathVariantType.STANDARD,
    )

    @dataclasses.dataclass(frozen=True)
    class CustomRiver:
      rectangles: list[rectangle_lib.Rectangle]
      width: int
      grid_size: int

      def asdict(self, name: str = '') -> dict[str, Any]:
        return {
            'name': name,
            'rectangles': [r.asdict() for r in self.rectangles],
            'river_type': 'custom',
            'width': self.width,
            'grid_size': self.grid_size,
        }

      @classmethod
      def from_dict(cls, pr_dict: Mapping[str, Any]) -> 'CustomRiver':
        return cls(
            rectangles=[
                rectangle_lib.Rectangle.from_dict(r)
                for r in pr_dict['rectangles']
            ],
            width=int(pr_dict['width']),
            grid_size=int(pr_dict['grid_size']),
        )

    prev_factory = river_lib.get_reconstructed_river_factory()
    self.addCleanup(
        river_lib.register_reconstructed_river_factory, prev_factory
    )
    self.addCleanup(codec._RIVER_CODEC_REGISTRY.pop, 'custom', None)
    self.addCleanup(codec._RIVER_CODEC_REGISTRY.pop, 'reconstructed', None)
    codec.register_river_codec('custom', CustomRiver.from_dict)
    codec.register_river_codec('reconstructed', CustomRiver.from_dict)
    river_lib.register_reconstructed_river_factory(CustomRiver)

    reconstructed = CustomRiver(
        rectangles=[
            rectangle_lib.Rectangle(
                name='', x_start=0, y_start=1, x_len=6, y_len=1
            ),
        ],
        width=1,
        grid_size=6,
    )

    for name, river in [
        ('r_straight', straight),
        ('r_l', l_shaped),
        ('r_hug', hugging),
        ('r_recon', reconstructed),
    ]:
      with self.subTest(name=name):
        encoded = codec.packing_river_to_dict(river, name=name)
        _assert_no_double_encoded_json(self, encoded)
        decoded = codec.packing_river_from_dict(json.loads(json.dumps(encoded)))
        self.assertIs(type(decoded), type(river))
        self.assertEqual(
            codec.packing_river_to_dict(decoded, name=name), encoded
        )

  def test_configs_round_trip_and_solver_config_is_pure_hyperparams(self):
    packing_cfg = packing_config_lib.PackingConfig(
        enforce_proximity_constraints=False,
        easy_mode=True,
        start_with_hugging_river=True,
        traversal_mode=packing_types.TraversalMode.PROXIMITY,
    )
    self.assertEqual(codec.loads(codec.dumps(packing_cfg)), packing_cfg)

    default_solver_cfg = hinge_utils.SolverConfig()
    self.assertIsNone(default_solver_cfg.grid_size)

    dummy_pack = packing_output_lib.PackingOutput(success=True)
    solver_cfg = hinge_utils.SolverConfig(
        max_steps=250,
        remove_central_crease=True,
        hinges_from_stick_figure=True,
        packing_information=dummy_pack,
    )
    encoded = codec.to_dict(solver_cfg)
    _assert_no_double_encoded_json(self, encoded)
    self.assertNotIn('packing_information', encoded)
    self.assertNotIn('preassignments', encoded)
    decoded_solver_cfg = codec.loads(codec.dumps(solver_cfg))
    self.assertEqual(decoded_solver_cfg, solver_cfg)
    self.assertIsNone(decoded_solver_cfg.packing_information)

  @parameterized.parameters('human', 'cat', 'spider')
  def test_packing_output_and_solver_result_lossless_round_trip(
      self, tree_name: str
  ):
    sf = copy.deepcopy(example_trees.get_example_by_name(tree_name))
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
    pack_res = pipeline.pack_crease_pattern(sf, config=pack_cfg)
    pack_out = pack_res.packing_outputs[0]

    encoded_pack = codec.to_dict(pack_out)
    _assert_no_double_encoded_json(self, encoded_pack)
    self.assertEqual(pack_out.asdict(), encoded_pack)
    decoded_pack = codec.loads(codec.dumps(pack_out))
    self.assertEqual(codec.to_dict(decoded_pack), encoded_pack)

    for stick in pack_out.stick_figure.children:
      self.assertEqual(
          decoded_pack.stick_figure[stick.name].flap_tip_shape,
          stick.flap_tip_shape,
      )
    for r_name, river in (pack_out.packing_rivers or {}).items():
      self.assertIs(
          type(decoded_pack.packing_rivers[r_name]),
          type(river),
      )

    solve_cfg = pipeline.SolvingTaskConfig(
        max_number_of_cps=1,
        max_steps=1000,
        sweep_timeout_seconds=30,
    )
    solve_orig = pipeline.solve_crease_pattern(pack_res, config=solve_cfg)

    reconstructed_pack_res = dataclasses.replace(
        pack_res,
        packing_cps=[decoded_pack.packing_cp],
        packing_outputs=[decoded_pack],
    )
    solve_from_decoded = pipeline.solve_crease_pattern(
        reconstructed_pack_res, config=solve_cfg
    )
    self.assertEqual(
        solve_from_decoded.flat_foldable_cps[0].hash_value,
        solve_orig.flat_foldable_cps[0].hash_value,
    )
    self.assertEqual(
        solve_from_decoded.solver_outputs[0].final_state,
        solve_orig.solver_outputs[0].final_state,
    )

    solver_result = solve_orig.solver_outputs[0]
    encoded_solve = codec.to_dict(solver_result)
    _assert_no_double_encoded_json(self, encoded_solve)
    decoded_solve = codec.loads(codec.dumps(solver_result))
    self.assertEqual(codec.to_dict(decoded_solve), encoded_solve)

    # Also test bus PackedCreasePattern and SolvedCreasePattern round-trips.
    design_tree = bus.DesignTree.from_stick_figure(sf)
    packed_cp = bus.PackedCreasePattern.from_packing_output(
        pack_out, design=design_tree
    )
    encoded_packed_cp = codec.to_dict(packed_cp)
    _assert_no_double_encoded_json(self, encoded_packed_cp)
    decoded_packed_cp = codec.loads(codec.dumps(packed_cp))
    self.assertEqual(codec.to_dict(decoded_packed_cp), encoded_packed_cp)

    solved_cp = bus.SolvedCreasePattern.from_solver_result(
        solver_result,
        packed=packed_cp,
        config=hinge_utils.SolverConfig(max_steps=1000),
        scale_factor=1,
    )
    encoded_solved_cp = codec.to_dict(solved_cp)
    _assert_no_double_encoded_json(self, encoded_solved_cp)
    decoded_solved_cp = codec.loads(codec.dumps(solved_cp))
    self.assertEqual(codec.to_dict(decoded_solved_cp), encoded_solved_cp)
    self.assertEqual(
        decoded_solved_cp.creases.hash_value,
        solved_cp.creases.hash_value,
    )

  def test_strict_decoder_and_legacy_loader(self):
    with self.assertRaisesRegex(
        ValueError, 'Missing required "schema_version"'
    ):
      codec.from_dict({'type': 'SolverConfig'})
    with self.assertRaisesRegex(ValueError, 'Unsupported schema_version'):
      codec.from_dict({'type': 'SolverConfig', 'schema_version': '99.0'})
    with self.assertRaisesRegex(ValueError, 'Unknown or missing type tag'):
      codec.from_dict({'schema_version': codec.SCHEMA_VERSION})
    with self.assertRaisesRegex(
        ValueError, 'Missing required straight river keys'
    ):
      codec.packing_river_from_dict({'river_type': 'straight', 'width': 1})
    with self.assertRaisesRegex(ValueError, 'Missing required DesignTree keys'):
      codec.design_tree_from_dict({'source': {}})

    legacy_payload = {
        'success': True,
        'flap_rectangles': json.dumps([{
            'name': 'f1',
            'bounds': [0, 0, 2, 2],
            'circle_centers': [0.0, 0.0],
        }]),
        'packing_rivers': json.dumps({
            'r1': {
                'name': 'r1',
                'river_type': 'straight',
                'width': 1,
                'grid_size': 4,
                'rectangles': [
                    {'name': '', 'bounds': [0, 2, 4, 1], 'circle_centers': []}
                ],
            }
        }),
        'config': {
            'enforce_proximity_constraints': False,
            'local_tiling_config': {
                'search_type': 'backtrack',
                'disable_wide_central_flaps': True,
                'use_river_tiling': True,
            },
        },
    }
    legacy_out = codec.load_legacy_packing_output(legacy_payload)
    self.assertTrue(legacy_out.success)
    self.assertLen(legacy_out.flap_rectangles, 1)
    self.assertIsInstance(legacy_out.packing_rivers['r1'], river_lib.River)
    self.assertEqual(legacy_out.packing_rivers['r1'].offset, 2)
    self.assertEqual(
        legacy_out.packing_rivers['r1'].orientation,
        river_lib.RiverOrientation.HORIZONTAL,
    )
    self.assertLen(legacy_out.packing_rivers['r1'].rectangles, 1)

  @parameterized.named_parameters(
      ('cat_default', example_trees.CAT_SIMPLE, 'default'),
      ('human_default', example_trees.HUMAN_SIMPLE, 'default'),
      ('giraffe_default', example_trees.GIRAFFE_SIMPLE, 'default'),
  )
  def test_json_dumps_packed_output_for_every_packer(
      self, tree: stick_figure_lib.StickFigure, packer: str
  ):
    design = bp.prepare(tree)
    packed = bp.pack(design, packer=packer)
    self.assertIsNotNone(packed)
    encoded = codec.to_dict(packed)
    # Must serialize cleanly with standard json.dumps (no default= lambda).
    raw_json = json.dumps(encoded)
    self.assertIsInstance(raw_json, str)
    decoded = json.loads(raw_json)
    self.assertEqual(decoded['schema_version'], codec.SCHEMA_VERSION)
    self.assertIn('rivers', decoded)
    self.assertIn('flaps', decoded)

  def test_packed_cp_reservations_codec_and_backward_compat(self):
    design = bp.prepare(bp.load_example('human'))
    packed_plain = bp.pack(design)
    self.assertEmpty(packed_plain.reservations)
    encoded_plain = codec.to_dict(packed_plain)
    self.assertNotIn('reservations', encoded_plain)
    decoded_old = codec.from_dict(encoded_plain)
    self.assertEqual(decoded_old.reservations, ())

    packed_rcc = bp.pack(design, bp.PackingConfig(remove_central_crease=True))
    self.assertNotEmpty(packed_rcc.reservations)
    encoded_rcc = codec.to_dict(packed_rcc)
    _assert_no_double_encoded_json(self, encoded_rcc)
    self.assertIn('reservations', encoded_rcc)
    decoded_rcc = codec.loads(codec.dumps(packed_rcc))
    self.assertEqual(decoded_rcc.reservations, packed_rcc.reservations)
    self.assertEqual(codec.to_dict(decoded_rcc), encoded_rcc)


if __name__ == '__main__':
  absltest.main()
