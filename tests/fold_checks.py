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

"""Geometric checks on FOLD dicts that hold a flat and a folded frame.

`render_3d.mesh_to_fold_dict` writes this layout and the showcase models under
`docs/models` are stored in it: the key frame is the 3D folded form, and
`file_frames[0]` is the flat crease pattern on the same vertices, edges and
faces.

The checks follow the conventions of the FOLD format: frames are right-handed,
a face is counter-clockwise when seen from the front of the paper, and a
positive fold angle is a valley fold, which raises both faces to the front.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np


def frames(fold: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
  """Returns the flat `(V, 2)` and the folded `(V, 3)` vertex coordinates.

  Args:
    fold: A FOLD dict.

  Raises:
    AssertionError: If `fold` does not have the layout described above.
  """
  (flat_frame,) = fold['file_frames']
  assert fold['frame_classes'] == ['foldedForm']
  assert fold['frame_attributes'] == ['3D']
  assert flat_frame['frame_classes'] == ['creasePattern']
  assert flat_frame['frame_attributes'] == ['2D']
  assert flat_frame['frame_inherit']
  assert flat_frame['frame_parent'] == 0
  flat = np.asarray(flat_frame['vertices_coords'], dtype=float)
  folded = np.asarray(fold['vertices_coords'], dtype=float)
  assert flat.shape == (len(folded), 2)
  assert folded.shape == (len(flat), 3)
  return flat, folded


def face_areas(fold: Mapping[str, Any]) -> np.ndarray:
  """Returns the area of each face in the flat frame.

  Args:
    fold: A FOLD dict.

  Returns:
    One signed area per face: positive if the face is counter-clockwise.
  """
  flat, _ = frames(fold)
  areas = []
  for face in fold['faces_vertices']:
    x, y = flat[face].T
    areas.append(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
  return np.asarray(areas)


def mean_strain(fold: Mapping[str, Any]) -> float:
  """Returns the mean relative change in edge length from flat to folded."""
  flat, folded = frames(fold)
  start, end = np.asarray(fold['edges_vertices']).T
  flat_lengths = np.linalg.norm(flat[start] - flat[end], axis=1)
  folded_lengths = np.linalg.norm(folded[start] - folded[end], axis=1)
  return float(np.mean(np.abs(folded_lengths / flat_lengths - 1.0)))


def partly_folded_creases(fold: Mapping[str, Any]) -> np.ndarray:
  """Returns a mask of the edges folded neither flat nor not at all.

  A crease folded flat looks the same in the mirror image of a model, so only
  these edges show whether a folded frame has the handedness of its crease
  pattern.

  Args:
    fold: A FOLD dict.
  """
  angles = np.abs(fold['edges_foldAngle'])
  return (angles > 5.0) & (angles < 175.0)


def fold_angle_errors(fold: Mapping[str, Any]) -> np.ndarray:
  """Compares `edges_foldAngle` with the angles between the folded faces.

  Args:
    fold: A FOLD dict.

  Returns:
    Per edge, the absolute difference in degrees between the stored fold angle
    and the one measured in the folded frame, or NaN for an edge that is not a
    mountain or valley crease with a face on both sides. Angles are compared
    modulo 360, since -180 and 180 are the same fold.
  """
  _, folded = frames(fold)
  faces = fold['faces_vertices']
  # The normal of a counter-clockwise face points to the front of the paper.
  # Summing cross products (Newell's method) tolerates slightly warped faces.
  normals = []
  for face in faces:
    corners = folded[face]
    normal = np.cross(corners, np.roll(corners, -1, axis=0)).sum(axis=0)
    normals.append(normal / np.linalg.norm(normal))
  # A counter-clockwise face lies to the left of each of its directed edges.
  left_face = {}
  for index, face in enumerate(faces):
    for i in range(len(face)):
      left_face[face[i - 1], face[i]] = index

  errors = np.full(len(fold['edges_vertices']), np.nan)
  for edge, ((start, end), assignment, stored) in enumerate(
      zip(
          fold['edges_vertices'],
          fold['edges_assignment'],
          fold['edges_foldAngle'],
          strict=True,
      )
  ):
    if assignment not in ('M', 'V'):
      continue
    if (start, end) not in left_face or (end, start) not in left_face:
      continue
    left = normals[left_face[start, end]]
    right = normals[left_face[end, start]]
    direction = folded[end] - folded[start]
    direction /= np.linalg.norm(direction)
    # Folding a valley by an angle turns the right face, and with it its
    # normal, by minus that angle about the direction of the edge.
    measured = np.degrees(
        np.arctan2(-np.cross(left, right) @ direction, left @ right)
    )
    errors[edge] = abs((measured - stored + 180.0) % 360.0 - 180.0)
  return errors
