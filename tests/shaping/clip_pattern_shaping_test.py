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

import numpy as np

from bp_pipeline.crease_pattern import crease
from bp_pipeline.shaping.clip_pattern import drawing_machine
from bp_pipeline.shaping.clip_pattern import clip_pattern_shaping
from bp_pipeline.shaping import folder_3d as fold_lib
from bp_pipeline.shaping.clip_pattern import converters
from bp_pipeline.shaping.clip_pattern import folding_grid_graph
from bp_pipeline.shaping.clip_pattern import triangle_geometry
from absl.testing import absltest as googletest

TEST_HINGES_FOLD = """
{
  "file_spec": 1.2,
  "file_creator": "Crease Pattern Editor",
  "file_classes": [
    "singleModel"
  ],
  "frame_classes": [
    "creasePattern",
    "noCuts"
  ],
  "vertices_coords": [
    [0,7],
    [0,0],
    [7,0],
    [7,7],
    [0,3],
    [2,3],
    [2,1],
    [5,1],
    [5,3],
    [7,3],
    [0,4],
    [3,4],
    [3,2],
    [4,2],
    [4,4],
    [7,4]
  ],
  "edges_vertices": [
    [0,10],
    [1,2],
    [2,9],
    [3,0],
    [4,1],
    [4,5],
    [5,6],
    [6,7],
    [7,8],
    [9,15],
    [8,9],
    [10,4],
    [10,11],
    [11,12],
    [12,13],
    [13,14],
    [15,3],
    [14,15]
  ],
  "edges_assignment": [
    "B",
    "B",
    "B",
    "B",
    "B",
    "M",
    "M",
    "M",
    "M",
    "B",
    "M",
    "B",
    "M",
    "M",
    "M",
    "M",
    "B",
    "M"
  ],
  "edges_foldAngle": [
    0,
    0,
    0,
    0,
    0,
    -180,
    -180,
    -180,
    -180,
    0,
    -180,
    0,
    -180,
    -180,
    -180,
    -180,
    0,
    -180
  ],
  "cpedit:page": {"xMin":0,"yMin":0,"xMax":7,"yMax":7},
  "version": 1.2,
  "file_title": "test_hinges",
  "vertices_vertices": [
    [10,3],
    [2,4],
    [9,1],
    [15,0],
    [1,5,10],
    [6,4],
    [7,5],
    [8,6],
    [7,9],
    [2,15,8],
    [4,11,0],
    [12,10],
    [13,11],
    [14,12],
    [13,15],
    [9,3,14]
  ],
  "vertices_edges": [
    [0,3],
    [1,4],
    [2,1],
    [16,3],
    [4,5,11],
    [6,5],
    [7,6],
    [8,7],
    [8,10],
    [2,9,10],
    [11,12,0],
    [13,12],
    [14,13],
    [15,14],
    [15,17],
    [9,16,17]
  ],
  "faces_vertices": [
    [0,10,11,12,13,14,15,3],
    [1,2,9,8,7,6,5,4],
    [4,5,6,7,8,9,15,14,13,12,11,10]
  ],
  "faces_edges": [
    [0,12,13,14,15,17,16,3],
    [1,2,10,8,7,6,5,4],
    [5,6,7,8,10,9,17,15,14,13,12,11]
  ]
}
"""

TEST_BASE_FOLD = """
{
  "file_spec": 1.2,
  "file_creator": "Crease Pattern Editor",
  "file_classes": [
    "singleModel"
  ],
  "frame_classes": [
    "creasePattern",
    "noCuts"
  ],
  "vertices_coords": [
    [0,7],
    [0,0],
    [7,0],
    [7,7],
    [0,3],
    [2,3],
    [2,1],
    [5,1],
    [5,3],
    [7,3],
    [0,4],
    [3,4],
    [3,2],
    [4,2],
    [4,4],
    [7,4],
    [1,3],
    [1,4],
    [2,4],
    [3,3],
    [2,2],
    [3,1],
    [4,1],
    [5,2],
    [4,3],
    [5,4],
    [6,3],
    [6,4]
  ],
  "edges_vertices": [
    [0,10],
    [1,2],
    [2,9],
    [3,0],
    [4,1],
    [4,16],
    [5,20],
    [6,21],
    [7,23],
    [9,15],
    [8,26],
    [10,4],
    [10,17],
    [11,19],
    [12,13],
    [13,24],
    [15,3],
    [14,25],
    [16,5],
    [17,18],
    [16,17],
    [18,11],
    [5,18],
    [5,11],
    [19,12],
    [5,19],
    [20,6],
    [20,12],
    [6,12],
    [21,22],
    [21,12],
    [22,7],
    [22,13],
    [13,7],
    [23,8],
    [13,23],
    [24,14],
    [24,8],
    [8,14],
    [25,27],
    [8,25],
    [26,9],
    [27,15],
    [26,27]
  ],
  "edges_assignment": [
    "B",
    "B",
    "B",
    "B",
    "B",
    "M",
    "M",
    "M",
    "M",
    "B",
    "M",
    "B",
    "M",
    "M",
    "M",
    "M",
    "B",
    "M",
    "M",
    "M",
    "M",
    "M",
    "M",
    "M",
    "M",
    "M",
    "M",
    "M",
    "M",
    "M",
    "M",
    "M",
    "M",
    "M",
    "M",
    "M",
    "M",
    "M",
    "M",
    "M",
    "M",
    "M",
    "M",
    "M"
  ],
  "edges_foldAngle": [
    0,
    0,
    0,
    0,
    0,
    -180,
    -180,
    -180,
    -180,
    0,
    -180,
    0,
    -180,
    -180,
    -180,
    -180,
    0,
    -180,
    -180,
    -180,
    -180,
    -180,
    -180,
    -180,
    -180,
    -180,
    -180,
    -180,
    -180,
    -180,
    -180,
    -180,
    -180,
    -180,
    -180,
    -180,
    -180,
    -180,
    -180,
    -180,
    -180,
    -180,
    -180,
    -180
  ],
  "cpedit:page": {"xMin":0,"yMin":0,"xMax":7,"yMax":7},
  "version": 1.2,
  "file_title": "test_base",
  "vertices_vertices": [
    [10,3],
    [2,4],
    [9,1],
    [15,0],
    [1,16,10],
    [20,19,11,18,16],
    [21,12,20],
    [23,13,22],
    [23,26,25,14,24],
    [2,15,26],
    [4,17,0],
    [5,19,18],
    [6,21,13,19,20],
    [22,7,23,24,12],
    [24,8,25],
    [9,3,27],
    [5,17,4],
    [16,18,10],
    [5,11,17],
    [12,11,5],
    [6,12,5],
    [22,12,6],
    [7,13,21],
    [7,8,13],
    [13,8,14],
    [8,27,14],
    [9,27,8],
    [26,15,25]
  ],
  "vertices_edges": [
    [0,3],
    [1,4],
    [2,1],
    [16,3],
    [4,5,11],
    [6,25,23,22,18],
    [7,28,26],
    [8,33,31],
    [34,10,40,38,37],
    [2,9,41],
    [11,12,0],
    [23,13,21],
    [28,30,14,24,27],
    [32,33,35,15,14],
    [36,38,17],
    [9,16,42],
    [18,20,5],
    [20,19,12],
    [22,21,19],
    [24,13,25],
    [26,27,6],
    [29,30,7],
    [31,32,29],
    [8,34,35],
    [15,37,36],
    [40,39,17],
    [41,43,10],
    [43,42,39]
  ],
  "faces_vertices": [
    [0,10,17,18,11,19,12,13,24,14,25,27,15,3],
    [1,2,9,26,8,23,7,22,21,6,20,5,16,4],
    [4,16,17,10],
    [5,18,17,16],
    [5,11,18],
    [5,19,11],
    [5,20,12,19],
    [6,12,20],
    [6,21,12],
    [7,13,22],
    [7,23,13],
    [8,24,13,23],
    [8,14,24],
    [8,25,14],
    [8,26,27,25],
    [9,15,27,26],
    [12,21,22,13]
  ],
  "faces_edges": [
    [0,12,19,21,13,24,14,15,36,17,39,42,16,3],
    [1,2,41,10,34,8,31,29,7,26,6,18,5,4],
    [5,20,12,11],
    [22,19,20,18],
    [23,21,22],
    [25,13,23],
    [6,27,24,25],
    [28,27,26],
    [7,30,28],
    [33,32,31],
    [8,35,33],
    [37,15,35,34],
    [38,36,37],
    [40,17,38],
    [10,43,39,40],
    [9,42,43,41],
    [30,29,32,14]
  ]
}
"""
# Lines as svg for review: http://gpaste/6413251042738176
EXPECTED_SHAPING_LINES = [
    drawing_machine.Line(
        p1=np.array([11.0, 7.0]),
        p2=np.array([12.0, 6.0]),
        color='blue',
        alpha=0.5,
    ),
    drawing_machine.Line(
        p1=np.array([10.0, 8.0]),
        p2=np.array([11.0, 7.0]),
        color='red',
        alpha=0.5,
    ),
    drawing_machine.Line(
        p1=np.array([12.0, 6.0]),
        p2=np.array([14.0, 8.0]),
        color='red',
        alpha=0.5,
    ),
    drawing_machine.Line(
        p1=np.array([10.0, 8.0]),
        p2=np.array([12.0, 10.0]),
        color='blue',
        alpha=0.5,
    ),
    drawing_machine.Line(
        p1=np.array([16.0, 10.0]),
        p2=np.array([18.0, 12.0]),
        color='blue',
        alpha=0.5,
    ),
    drawing_machine.Line(
        p1=np.array([4.0, 14.0]),
        p2=np.array([2.0, 16.0]),
        color='blue',
        alpha=0.5,
    ),
    drawing_machine.Line(
        p1=np.array([4.0, 14.0]),
        p2=np.array([6.0, 16.0]),
        color='red',
        alpha=0.5,
    ),
    drawing_machine.Line(
        p1=np.array([12.0, 14.0]),
        p2=np.array([11.0, 15.0]),
        color='red',
        alpha=0.5,
    ),
    drawing_machine.Line(
        p1=np.array([11.0, 15.0]),
        p2=np.array([10.0, 16.0]),
        color='blue',
        alpha=0.5,
    ),
    drawing_machine.Line(
        p1=np.array([18.0, 16.0]),
        p2=np.array([20.0, 14.0]),
        color='blue',
        alpha=0.5,
    ),
    drawing_machine.Line(
        p1=np.array([16.0, 14.0]),
        p2=np.array([18.0, 12.0]),
        color='red',
        alpha=0.5,
    ),
    drawing_machine.Line(
        p1=np.array([20.0, 14.0]),
        p2=np.array([22.0, 16.0]),
        color='red',
        alpha=0.5,
    ),
    drawing_machine.Line(
        p1=np.array([26.0, 16.0]),
        p2=np.array([28.0, 14.0]),
        color='blue',
        alpha=0.5,
    ),
]


def simple_clip_pattern(
    machine: drawing_machine.CreasePatternDrawingMachine,
) -> None:
  machine.draw_line((0.5, 0), (1.5, 1), crease.Fold.VALLEY, 90)


class ClipPatternShapingTest(googletest.TestCase):

  def test_get_clip_pattern_shaping_lines(self):
    base_grid = converters.convert_fold_to_grid(
        fold_lib.FOLD.from_json(TEST_BASE_FOLD), grid_size=7
    )
    hinges_grid = converters.convert_fold_to_grid(
        fold_lib.FOLD.from_json(TEST_HINGES_FOLD), grid_size=7
    )
    align_triangle = triangle_geometry.TriangleCellPosition.from_key(
        (3, 0, triangle_geometry.CellTriangle.BOTTOM)
    )
    shaping_lines = clip_pattern_shaping.get_clip_pattern_shaping_lines(
        grid_size=7,
        hinges_grid=hinges_grid,
        grid=base_grid,
        align_triangle=align_triangle,
        draw_pattern=simple_clip_pattern,
        disable_clipping=False,
    )
    scaled_shaping_lines = []
    for line in shaping_lines:
      p1 = np.array([round(line.p1[0] * 28, 8), round(line.p1[1] * 28, 8)])
      p2 = np.array([round(line.p2[0] * 28, 8), round(line.p2[1] * 28, 8)])
      scaled_shaping_lines.append(
          drawing_machine.Line(p1=p1, p2=p2, color=line.color, alpha=line.alpha)
      )

    self.assertCountEqual(
        scaled_shaping_lines,
        EXPECTED_SHAPING_LINES,
    )

  def test_get_polygon_points_fast_equals_slow(self):
    base_grid = converters.convert_fold_to_grid(
        fold_lib.FOLD.from_json(TEST_BASE_FOLD), grid_size=7
    )
    hinges_grid = converters.convert_fold_to_grid(
        fold_lib.FOLD.from_json(TEST_HINGES_FOLD), grid_size=7
    )
    graph = folding_grid_graph.grid_to_nx_graph_disconnect_where_folded(
        base_grid
    )
    graph = folding_grid_graph.disconnect_where_folded(graph, hinges_grid)
    representative_nodes = folding_grid_graph.get_one_node_from_each_component(
        graph
    )
    for node in representative_nodes:
      points_fast = clip_pattern_shaping.get_polygon_points(
          graph, node, grid_size=7, fast=True
      )
      points_slow = clip_pattern_shaping.get_polygon_points(
          graph, node, grid_size=7, fast=False
      )
      # Both implementations must return the same set of hull vertices
      # (order may differ since scipy ConvexHull may index differently).
      self.assertEqual(
          points_fast.shape,
          points_slow.shape,
          msg=f'Shape mismatch for node {node}',
      )
      np.testing.assert_allclose(
          np.sort(points_fast, axis=0),
          np.sort(points_slow, axis=0),
          atol=1e-9,
          err_msg=f'Values mismatch for node {node}',
      )

  def test_degenerate_corner_clipped_segment_is_filtered(self):
    base_grid = converters.convert_fold_to_grid(
        fold_lib.FOLD.from_json(TEST_BASE_FOLD), grid_size=7
    )
    hinges_grid = converters.convert_fold_to_grid(
        fold_lib.FOLD.from_json(TEST_HINGES_FOLD), grid_size=7
    )
    align_triangle = triangle_geometry.TriangleCellPosition.from_key(
        (3, 0, triangle_geometry.CellTriangle.BOTTOM)
    )

    def tiny_clip_pattern(
        machine: drawing_machine.CreasePatternDrawingMachine,
    ) -> None:
      # Sub-tolerance micro-segment plus a normal segment: `to_crease()` must
      # succeed on all returned lines without raising ValueError.
      machine.draw_line(
          (0.5, 0.5), (0.5 + 1e-9, 0.5 + 1e-9), crease.Fold.VALLEY, 90
      )
      machine.draw_line((0.5, 0), (1.5, 1), crease.Fold.VALLEY, 90)

    shaping_lines = clip_pattern_shaping.get_clip_pattern_shaping_lines(
        grid_size=7,
        hinges_grid=hinges_grid,
        grid=base_grid,
        align_triangle=align_triangle,
        draw_pattern=tiny_clip_pattern,
        disable_clipping=False,
    )
    creases = [line.to_crease() for line in shaping_lines]
    self.assertNotEmpty(creases)
    self.assertLen(creases, len(shaping_lines))


if __name__ == '__main__':
  googletest.main()
