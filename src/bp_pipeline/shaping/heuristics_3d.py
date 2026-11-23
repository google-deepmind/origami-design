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

"""Heuristic functions for generating the parameters for 3D shaping."""

from collections.abc import Collection, Mapping, Sequence
import dataclasses
import enum
import functools
import logging
import math
from typing import Union

import numpy as np

from ..crease_pattern import crease as crease_utils
from ..stick_figures import angle_utils
from ..stick_figures import stick as stick_lib
from ..stick_figures import stick_figure as stick_figure_lib
from ..stick_figures import traversal as tr
from . import heuristics
from . import heuristics_utils
from . import shaping_utils

# The maximum slope of the simple fold direction allowed when shaping.
# NOTE: This will only be used in geometric_simple_fold_3d_heuristics.py and
# rotation_3d.py.
MAX_SLOPE = 10
# Fold offset sets small distance in between multiple folds on the same flap.
FOLD_OFFSET = 0.05


class FoldStrategy3D(enum.Enum):
  """Strategies for 3D shaping."""

  DEFAULT = 0
  SHALLOW_FOLDS = 1
  ROTATE_BEND = 2
  NO_FLIP = 2


@dataclasses.dataclass(kw_only=True, frozen=True)
class PostureMatrix:
  """A 3D posture matrix.

  This class represents a 3D posture matrix as a 3x3 array, with each column
  representing different axes (X, Y, Z).

  Attributes:
    x: The x axis vector, which is the cross product of y and z, orthonormal to
      y and z.
    y: The y axis vector, which is the direction vector of the plane,
      orthonormal to x and z.
    z: The z axis vector, which is the normal vector of the plane, orthonormal
      to x and y.
  """

  y: np.ndarray
  z: np.ndarray

  def __post_init__(self):
    assert np.allclose(
        np.linalg.norm(self.y), 1
    ), f'Invalid y: {self.y} with norm {np.linalg.norm(self.y)}'
    assert np.allclose(
        np.linalg.norm(self.z), 1
    ), f'Invalid z: {self.z} with norm {np.linalg.norm(self.z)}'
    assert math.isclose(
        np.dot(self.y, self.z), 0, abs_tol=1e-8
    ), f'Invalid y: {self.y} and z: {self.z}'

  @classmethod
  def from_matrix(cls, matrix: np.ndarray) -> 'PostureMatrix':
    assert matrix.shape == (3, 3), f'Invalid matrix shape {matrix.shape}!'
    assert np.allclose(
        matrix[:, 0], np.cross(matrix[:, 1], matrix[:, 2])
    ), f'Invalid matrix {matrix}'
    return cls(y=matrix[:, 1], z=matrix[:, 2])

  @functools.cached_property
  def x(self) -> np.ndarray:
    """Returns the cross product of y and z."""
    return np.cross(self.y, self.z)

  @functools.cached_property
  def value(self) -> np.ndarray:
    """Returns the 3D posture matrix as a 3x3 array."""
    return np.stack([self.x, self.y, self.z], axis=-1)

  @functools.cached_property
  def T(self) -> 'PostureMatrix':  # pylint: disable=invalid-name
    """Returns the transpose of the 3D posture matrix."""
    return PostureMatrix.from_matrix(self.value.T)

  def __matmul__(
      self, other: Union['PostureMatrix', np.ndarray]
  ) -> 'PostureMatrix':
    """Returns the 3D posture matrix multiplication."""
    if isinstance(other, PostureMatrix):
      return PostureMatrix.from_matrix(self.value @ other.value)
    elif isinstance(other, np.ndarray):
      return PostureMatrix.from_matrix(self.value @ other)
    else:
      raise NotImplementedError('Unsupported type for multiplication.')

  def __rmatmul__(
      self, other: Union['PostureMatrix', np.ndarray]
  ) -> 'PostureMatrix':
    """Returns the 3D posture matrix multiplication."""
    if isinstance(other, PostureMatrix):
      return PostureMatrix.from_matrix(other.value @ self.value)
    elif isinstance(other, np.ndarray):
      return PostureMatrix.from_matrix(other @ self.value)
    else:
      raise NotImplementedError('Unsupported type for multiplication.')

  def __eq__(self, other: 'PostureMatrix') -> bool:  # pyrefly: ignore[bad-override]
    """Returns True if the two posture matrices are equal."""
    return np.allclose(self.value, other.value, atol=1e-6)

  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    """This method is called when a numpy ufunc is called on a PostureMatrix object."""
    del method  # Unused.
    if ufunc == np.matmul:
      # Extract the numpy array from the PostureMatrix object
      processed_inputs = []
      for input_obj in inputs:
        if isinstance(input_obj, PostureMatrix):
          processed_inputs.append(input_obj.value)
        else:
          processed_inputs.append(input_obj)

      # Perform the original ufunc on the numpy arrays
      result = ufunc(*processed_inputs, **kwargs)

      # If the result is a 3x3 matrix, convert it back to a PostureMatrix
      if isinstance(result, np.ndarray) and result.shape == (3, 3):
        return PostureMatrix.from_matrix(result)
      else:
        return result
    return NotImplemented


@dataclasses.dataclass(kw_only=True, frozen=True)
class SimpleFoldInfo:
  """All information about a flap's simple fold rotation.

  We conduct simple fold to rotate the flap to the target direction. This class
  stores all information of the folding process, including the initial posture
  matrix, the final posture matrix, the rotation axis and angle (local and
  global), and whether a flip is required.

  Attributes:
    posture_matrix_before_shaping: The posture matrix before shaping, when all
      the parent flaps are shaped.
    posture_matrix_after_shaping: The posture matrix after shaping.
    rotation_axis: The global rotation axis of the shaping, shape: (3,).
    rotation_axis_local: The rotation axis of the shaping in the local
      coordinate system, shape: (3,).
    rotation_angle: The rotation angle of the shaping.
    required_flip: Whether the shaping requires a inscribed rectangle flip. This
      is used when shaping angle is acute.
    inscribe_depth: The depth into the flap (away from the hinge) the simple
      simple fold is inscribed.
  """

  posture_matrix_before_shaping: PostureMatrix
  posture_matrix_after_shaping: PostureMatrix
  rotation_axis: np.ndarray
  rotation_axis_local: np.ndarray
  rotation_angle: float | np.float64
  required_flip: bool
  inscribe_depth: float = 0.0

  @functools.cached_property
  def rotation_axis_depth(self) -> float | None:
    """Returns the depth of the rotation axis on the y-axis.

    Returns:
      The depth of the rotation axis on the y-axis. Returns None if the
      rotation axis is parallel to the x-axis (y = 0, e.g. (1, 0, 0)).
    """
    if math.isclose(self.rotation_axis_local[0], 0):
      return None
    depth = self.rotation_axis_local[1] / self.rotation_axis_local[0]
    return depth

  def __eq__(self, other: 'SimpleFoldInfo') -> bool:  # pyrefly: ignore[bad-override]
    """Returns True if the two flap postures are equal."""
    return (
        self.posture_matrix_before_shaping
        == other.posture_matrix_before_shaping
        and self.posture_matrix_after_shaping
        == other.posture_matrix_after_shaping
        and np.allclose(self.rotation_axis, other.rotation_axis)
        and np.allclose(self.rotation_axis_local, other.rotation_axis_local)
        and bool(
            np.isclose(self.rotation_angle, other.rotation_angle, atol=1e-5)
        )
        and self.required_flip == other.required_flip
    )


def _get_initial_posture_matrix_from_root(
    root_stick: stick_lib.Stick,
    child_stick: stick_lib.Stick,
    direction_follow_stick_figure: bool,
    fold_state: heuristics.FoldState,
    init_post_matrix: PostureMatrix,
    make_post_hinge_top: bool,
) -> PostureMatrix:
  """Determines the initial posture matrix of a child flap given the parent.

  Args:
    root_stick: The root stick.
    child_stick: The child stick.
    direction_follow_stick_figure: Whether to follow the stick figure direction.
    fold_state: The fold state of the child flap.
    init_post_matrix: The initial posture matrix of the parent flap.
    make_post_hinge_top: If true, the side of the paper visible after folding
      the hinge will be the top side (positive normal vector).

  Returns:
    The initial posture matrix of the child flap.
  """
  connected_joint_name = (
      {root_stick.start, root_stick.end} & {child_stick.start, child_stick.end}
  ).pop()

  # Determine which joint is the "top" (UP) and "bottom" (DOWN).
  if direction_follow_stick_figure:
    top_joint_name = root_stick.end
    bot_joint_name = root_stick.start
  else:
    top_joint_name = root_stick.start
    bot_joint_name = root_stick.end

  # Determine direction based on connection point and fold state.
  if connected_joint_name == top_joint_name:
    if fold_state == heuristics.FoldState.FOLD_IN:
      z = init_post_matrix.z if make_post_hinge_top else -init_post_matrix.z
      return PostureMatrix(y=-init_post_matrix.y, z=z)
    elif fold_state == heuristics.FoldState.FOLD_OUT:
      return init_post_matrix
    else:
      raise ValueError(
          f'Invalid fold state {fold_state} for the top joint'
          f' {connected_joint_name}!'
      )
  elif connected_joint_name == bot_joint_name:
    if fold_state == heuristics.FoldState.FOLD_IN:
      z = init_post_matrix.z if make_post_hinge_top else -init_post_matrix.z
      return PostureMatrix(y=init_post_matrix.y, z=z)
    elif fold_state == heuristics.FoldState.FOLD_OUT:
      return PostureMatrix(y=-init_post_matrix.y, z=init_post_matrix.z)
    else:
      raise ValueError(
          f'Invalid fold state {fold_state} for the bottom joint'
          f' {connected_joint_name}!'
      )
  else:
    raise ValueError(f'Invalid connected joint {connected_joint_name}!')


def _get_initial_posture_matrix_from_non_root(
    fold_state: heuristics.FoldState,
    parent_pose_matrix: PostureMatrix,
    make_post_hinge_top: bool,
):
  """Determines the initial posture matrix of a child flap given non-root parent.

  If the child flap is FOLD_OUT, the initial posture matrix is the same as the
  parent's posture matrix.
  If the child flap is FOLD_IN, the initial posture matrix is the negative of
  the parent's posture matrix.

  Args:
    fold_state: The fold state of the child flap.
    parent_pose_matrix: The posture matrix of the parent flap.
    make_post_hinge_top: If true, the side of the paper visible after folding
      the hinge will be the top side (positive normal vector).

  Returns:
    The initial posture matrix of the child flap.
  """

  if fold_state == heuristics.FoldState.FOLD_OUT:
    return parent_pose_matrix
  elif fold_state == heuristics.FoldState.FOLD_IN:
    z = parent_pose_matrix.z if make_post_hinge_top else -parent_pose_matrix.z
    return PostureMatrix(y=-parent_pose_matrix.y, z=z)
  else:
    raise ValueError(f'Invalid fold state {fold_state}!')


def get_physical_fold_from_final_direction(
    r_initial: np.ndarray, vec_final_direction: np.ndarray
) -> tuple[np.ndarray | None, float]:
  """Calculates a physically constrained single fold (crease and signed angle).

  This method adheres to the physical constraint that the crease (axis of
  rotation) must lie within the initial plane of the flap.

  Args:
    r_initial: The 3x3 posture matrix of the flap before folding (the specific
      matrix of SimpleFoldInfo).
    vec_final_direction: The desired final direction vector for the flap.

  Returns:
    A tuple containing:
    - A 3D unit vector representing the crease in the global coordinate
      system. Returns None if no fold is needed or possible.
    - The required fold angle in degrees. A positive angle represents a
      "mountain fold" (folding up), and a negative angle represents a
      "valley fold" (folding down).
  """
  # Step 1: Extract the initial direction and normal vectors.
  d_initial = r_initial[:, 1]
  n_initial = r_initial[:, 2]
  if not math.isclose(np.linalg.norm(d_initial), 1.0):
    logging.warning('Direction vector is not normalized: %s', d_initial)
    d_initial = d_initial / np.linalg.norm(d_initial)
  if not math.isclose(np.linalg.norm(n_initial), 1.0):
    logging.warning('Normal vector is not normalized: %s', n_initial)
    n_initial = n_initial / np.linalg.norm(n_initial)

  # Normalize the final direction to ensure it's a unit vector.
  norm_final_direction = np.linalg.norm(vec_final_direction)
  assert not np.isclose(
      norm_final_direction, 0.0
  ), f'Final direction vector is zero: {vec_final_direction}'
  d_final = vec_final_direction / norm_final_direction

  logging.info('d_initial: %s', d_initial)
  logging.info('n_initial: %s', n_initial)
  logging.info('d_final: %s', d_final)

  # Edge case: If directions are already aligned, no fold is needed.
  if np.allclose(d_initial, d_final):
    return None, 0.0

  # Step 2: Calculate the crease vector.
  # The crease must be perpendicular to both the normal and the change in
  # direction.
  delta_d = d_final - d_initial
  crease_vec_unnormalized = np.cross(n_initial, delta_d)
  norm_crease = np.linalg.norm(crease_vec_unnormalized)

  # Edge case: If the crease vector is zero, the target cannot be reached
  # by a single physical fold (as it implies the movement is along the normal).
  if np.isclose(norm_crease, 0.0):
    logging.warning(
        'Target direction cannot be reached via a single physical fold.'
    )
    return None, 0.0

  v_crease = crease_vec_unnormalized / norm_crease

  # Step 3: Calculate the magnitude of the fold angle.
  # This uses a geometric formula derived from the relationships between
  # the initial/final vectors and the rotation axis (crease).
  cos_alpha = np.clip(np.dot(d_initial, d_final), -1.0, 1.0)
  cos_phi = np.clip(np.dot(d_initial, v_crease), -1.0, 1.0)
  sin_phi_sq = 1 - cos_phi * cos_phi

  # If sin_phi_sq is zero, d_initial is parallel to the crease.
  # This can happen if the required fold is 180 degrees.
  if np.isclose(sin_phi_sq, 0.0):
    return v_crease, 180.0

  cos_theta = (cos_alpha - cos_phi * cos_phi) / sin_phi_sq
  angle_rad_magnitude = np.arccos(np.clip(cos_theta, -1.0, 1.0))

  # =======================================================================
  # Step 4: Determine the sign of the angle (Mountain vs. Valley fold).
  # We check if the displacement from the initial to the final direction
  # is "upwards" or "downwards" relative to the flap's initial normal.
  # This is determined by the dot product of the displacement vector
  # (delta_d) and the initial normal vector (n_initial).
  # =======================================================================
  fold_direction_sign = np.sign(np.dot(delta_d, n_initial))

  # If the rotation is 180 degree, the sign will be zero. We set the sign to 1
  # to indicate a mountain fold.
  if fold_direction_sign == 0:
    fold_direction_sign = 1

  signed_angle_deg = math.degrees(angle_rad_magnitude) * fold_direction_sign

  if isinstance(signed_angle_deg, float):
    signed_angle_deg = np.array([signed_angle_deg])
  return v_crease, signed_angle_deg.item()


def get_simple_fold_info_from_final_direction(
    init_post_matrix_global: PostureMatrix,
    direction_vec_global: np.ndarray,
    flip: bool = False,
) -> SimpleFoldInfo:
  """Calculates SimpleFoldInfo from the final direction vector."""
  rotation_axis_global, rotation_angle = get_physical_fold_from_final_direction(
      init_post_matrix_global.value, direction_vec_global
  )

  # If no physical fold is needed, we set the rotation axis to zero
  # and angle to 0.
  if rotation_axis_global is None:
    return SimpleFoldInfo(
        posture_matrix_before_shaping=init_post_matrix_global,
        posture_matrix_after_shaping=init_post_matrix_global,
        rotation_axis=np.zeros(3),
        rotation_axis_local=np.zeros(3),
        rotation_angle=0.0,
        required_flip=False,
    )

  # Calculate the rotation matrix given the rotation axis and angle.
  rotation_matrix = angle_utils.get_rotation_matrix_from_axis_angle(
      axis_vec=rotation_axis_global, angle_deg=rotation_angle
  )
  # Apply the rotation matrix to the initial posture matrix to get the
  # final posture matrix.
  posture_global = rotation_matrix @ init_post_matrix_global

  # Rotate the unshaped flap back to identity to find rotation_axis_local.
  # This is used to identify the zigzag depth.
  rotation_axis_local = init_post_matrix_global.value.T @ rotation_axis_global

  return SimpleFoldInfo(
      posture_matrix_before_shaping=init_post_matrix_global,
      posture_matrix_after_shaping=posture_global,
      rotation_axis=rotation_axis_global,
      rotation_axis_local=rotation_axis_local,
      rotation_angle=rotation_angle,
      required_flip=flip,
  )


def apply_small_angle_flip(
    simple_fold_info: SimpleFoldInfo,
) -> list[SimpleFoldInfo]:
  """Converts an in-plane rotation to two simple folds."""
  zigzag_depth = simple_fold_info.rotation_axis_depth
  if zigzag_depth is None:
    return [simple_fold_info]
  zigzag_depth = abs(zigzag_depth)

  target_flap_angle = shaping_utils.fold_angle_from_zigzag_depth(zigzag_depth)
  split_angles = heuristics_utils.apply_small_angle_flip(target_flap_angle)

  # Return simple fold if multiple are not needed.
  if len(split_angles) == 1:
    return [simple_fold_info]
  assert (
      len(split_angles) == 2
  ), f'Expected 2 split angles, got {len(split_angles)}'

  new_simple_fold_infos = []
  current_posture = simple_fold_info.posture_matrix_before_shaping
  for ind, split_angle in enumerate(split_angles):
    new_depth = shaping_utils.zigzag_depth_from_fold_angle(split_angle)
    logging.info(
        'new_depth: %s',
        np.sign(simple_fold_info.rotation_axis_depth) * new_depth,  # pyrefly: ignore[no-matching-overload]
    )
    rotation_axis_local = np.array(
        [1, np.sign(simple_fold_info.rotation_axis_depth) * new_depth, 0]  # pyrefly: ignore[no-matching-overload]
    )
    rotation_axis_global = current_posture.value @ rotation_axis_local
    rotation_matrix = angle_utils.get_rotation_matrix_from_axis_angle(
        axis_vec=rotation_axis_global, angle_deg=180.0
    )
    posture_matrix_after_shaping = rotation_matrix @ current_posture
    new_simple_fold_info = SimpleFoldInfo(
        posture_matrix_before_shaping=current_posture,
        posture_matrix_after_shaping=posture_matrix_after_shaping,
        rotation_axis=rotation_axis_global,
        rotation_axis_local=rotation_axis_local,
        # flip sign to flip fold-order.
        rotation_angle=180.0 if ind else -180.0,
        required_flip=False,
        inscribe_depth=simple_fold_info.inscribe_depth,
    )
    new_simple_fold_infos.append(new_simple_fold_info)
    current_posture = posture_matrix_after_shaping

  return new_simple_fold_infos


def get_flap_post_dict(
    stickfigure: stick_figure_lib.StickFigure,
    flap_info: dict[str, heuristics.Node],
    root_direction_follow_stick_figure_root: bool = True,
    add_flip_to_acute_angle: bool = False,
    make_post_hinge_top: bool = False,
    default_strategy: FoldStrategy3D = FoldStrategy3D.DEFAULT,
    flap_strategies: dict[str, FoldStrategy3D] | None = None,
    max_slope_thresh: float = MAX_SLOPE,
    unrotated: Collection[str] = (),
    base_signs: Mapping[str, float] | None = None,
    use_tree_root: bool = False,
) -> dict[str, Sequence[SimpleFoldInfo]]:
  """Generates a dictionary of SimpleFoldInfo for each stick in the stick figure.

  This function traverses the stick figure in a breadth-first manner, starting
  from the root. For each stick, it determines the initial and final 3D posture
  matrix based on the fold state and the posture of its parent flap. It also
  calculates the rotation axis and angle required to achieve the final posture.

  Args:
    stickfigure: The stick figure representing the flap connections.
    flap_info: A dictionary mapping stick names to heuristics.Node containing
      fold state information.
    root_direction_follow_stick_figure_root: Whether the direction of the root
      flap follows the stick figure's root direction.
    add_flip_to_acute_angle: Whether to add flip to acute fold angle.
    make_post_hinge_top: If true, the side of the paper visible after folding
      the hinge will be the top side (positive normal vector).
    default_strategy: The default 3D shaping strategy.
    flap_strategies: A dictionary mapping flap names to their specific shaping
      strategy. If not provided, the default strategy will be used.
    max_slope_thresh: The maximum slope threshold for the flap.
    unrotated: Stick names that receive no shaping creases in this pass; each
      keeps its initial posture so descendant sticks start from the unrotated
      parent frame rather than its target stick-figure frame.
    base_signs: Optional mapping of stick names to measured directions on the
      flat-folded base (`> 0` along the root stick, `< 0` against it; `0` is
      ignored). Overrides hinge-inferred initial directions when provided.
    use_tree_root: Whether to anchor posture propagation at `stickfigure.root`.

  Returns:
    A dictionary mapping stick names to their corresponding SimpleFoldInfo.
  """
  first_stick_name = heuristics.get_first_stick_for_shaping(
      stickfigure, use_tree_root=use_tree_root
  ).name
  # Initialize the first root flap posture matrix
  flap_posture_dict = {
      first_stick_name: [
          SimpleFoldInfo(
              posture_matrix_before_shaping=PostureMatrix(
                  y=np.array([0, 1, 0]), z=np.array([0, 0, 1])
              ),
              posture_matrix_after_shaping=PostureMatrix(
                  y=np.array([0, 1, 0]), z=np.array([0, 0, 1])
              ),
              rotation_axis=np.zeros(3),
              rotation_axis_local=np.zeros(3),
              rotation_angle=0.0,
              required_flip=False,
          )
      ]
  }
  # Postures on the flat base (before any shaping), in the same frame.
  base_postures = {
      first_stick_name: PostureMatrix(
          y=np.array([0, 1, 0]), z=np.array([0, 0, 1])
      )
  }
  for stick in tr.breadth_first_traversal(
      stickfigure, start_stick_name=first_stick_name
  ):
    parent_posture_matrix_after_shaping = flap_posture_dict[stick.name][
        -1
    ].posture_matrix_after_shaping
    for joint_name in (stick.start, stick.end):
      joint = stickfigure.joints.get(joint_name)
      if not joint:
        continue
      for neighbor_stick in joint.sticks:
        if neighbor_stick.name in flap_posture_dict:
          continue

        logging.info('Getting posture dict for stick: %s', neighbor_stick.name)

        # Get the target direction vector in the global coordinate system.
        unnormalized_direction_vec_global = neighbor_stick.get_coords(
            joint.name
        )
        direction_vec_global = (
            unnormalized_direction_vec_global
            / np.linalg.norm(unnormalized_direction_vec_global)
        )

        # Get the initial posture matrix of the child flap.
        # Note that we need to choose the top direction for root node
        if stick.name == first_stick_name:
          init_post_matrix_global = _get_initial_posture_matrix_from_root(
              root_stick=stick,
              child_stick=neighbor_stick,
              direction_follow_stick_figure=(
                  root_direction_follow_stick_figure_root
              ),
              fold_state=flap_info[neighbor_stick.name].fold_state,
              init_post_matrix=PostureMatrix(
                  y=np.array([0, 1, 0]), z=np.array([0, 0, 1])
              ),
              make_post_hinge_top=make_post_hinge_top,
          )
        else:
          init_post_matrix_global = _get_initial_posture_matrix_from_non_root(
              fold_state=flap_info[neighbor_stick.name].fold_state,
              parent_pose_matrix=parent_posture_matrix_after_shaping,
              make_post_hinge_top=make_post_hinge_top,
          )

        # The parent's rotation by shaping, applied to the child's posture on
        # the flat base (the child rides on its parent).
        parent_rotation = (
            parent_posture_matrix_after_shaping.value
            @ base_postures[stick.name].value.T
        )
        sign = base_signs.get(neighbor_stick.name) if base_signs else None
        if sign is not None and sign != 0:
          direction = 1.0 if sign > 0 else -1.0
          base_posture = PostureMatrix(
              y=np.array([0.0, direction, 0.0]),
              z=np.array([0.0, 0.0, 1.0]),
          )
          measured = PostureMatrix.from_matrix(
              parent_rotation @ base_posture.value
          )
          if not np.allclose(measured.y, init_post_matrix_global.y, atol=1e-6):
            logging.warning(
                'Base direction of %s measured %s the root; overriding hinge'
                ' fold state %s.',
                neighbor_stick.name,
                'along' if direction > 0 else 'against',
                flap_info[neighbor_stick.name].fold_state,
            )
          init_post_matrix_global = measured
        else:
          base_posture = PostureMatrix.from_matrix(
              parent_rotation.T @ init_post_matrix_global.value
          )
        base_postures[neighbor_stick.name] = base_posture

        if neighbor_stick.name in unrotated:
          flap_posture_dict[neighbor_stick.name] = [
              SimpleFoldInfo(
                  posture_matrix_before_shaping=init_post_matrix_global,
                  posture_matrix_after_shaping=init_post_matrix_global,
                  rotation_axis=np.zeros(3),
                  rotation_axis_local=np.zeros(3),
                  rotation_angle=0.0,
                  required_flip=False,
              )
          ]
          continue

        # If the dot product is positive, we have an acute zigzag rotation,
        # Here we assume we will apply a 180 degree flip to the flap.
        # So we flip the initial posture matrix.
        flip = False
        if add_flip_to_acute_angle:
          if 1 > np.dot(direction_vec_global, init_post_matrix_global.y) > 0:
            flip = True
            init_post_matrix_global = PostureMatrix(
                y=-init_post_matrix_global.y, z=-init_post_matrix_global.z
            )

        strategy = (
            flap_strategies.get(neighbor_stick.name, default_strategy)
            if flap_strategies
            else default_strategy
        )

        if strategy == FoldStrategy3D.DEFAULT:
          flap_posture_dict[neighbor_stick.name] = [
              get_simple_fold_info_from_final_direction(
                  init_post_matrix_global=init_post_matrix_global,
                  direction_vec_global=direction_vec_global,
                  flip=flip,
              )
          ]

        elif strategy in (
            FoldStrategy3D.SHALLOW_FOLDS,
            FoldStrategy3D.ROTATE_BEND,
        ):
          # We will split the single fold into up to two "paper-plane"
          # components with 1.0 fold percent. Then deal with the out
          # of plane component with a 180 degree bend.
          simple_fold_infos = []
          # Project the direction_vec_global onto the local xy plane in global
          # coordinate system.
          x_vec_global, y_vec_global = (
              init_post_matrix_global.x,
              init_post_matrix_global.y,
          )
          dir_xy_proj = (
              np.dot(direction_vec_global, x_vec_global)
          ) * x_vec_global + (
              np.dot(direction_vec_global, y_vec_global)
          ) * y_vec_global

          if np.linalg.norm(dir_xy_proj) > crease_utils.DEFAULT_TOLERANCE:
            new_simple_fold_info = get_simple_fold_info_from_final_direction(
                init_post_matrix_global=init_post_matrix_global,
                direction_vec_global=dir_xy_proj,
            )
            if strategy == FoldStrategy3D.SHALLOW_FOLDS:
              simple_fold_infos.extend(
                  apply_small_angle_flip(new_simple_fold_info)
              )
            else:
              simple_fold_infos.append(new_simple_fold_info)
            init_post_matrix_global = simple_fold_infos[
                -1
            ].posture_matrix_after_shaping
            # Get depth of simple fold on paper.
            depth = simple_fold_infos[-1].rotation_axis_depth
            if depth is None or abs(depth) > max_slope_thresh:
              depth = 0
            else:
              depth = abs(depth) + FOLD_OFFSET
          else:
            depth = 0
          # When there are wide pleats in the flap, the zigzag creases
          # have more length to them and go twice as deep into the flap
          # (when width = 2).
          # This means we need to increase the depth that the rotation fold is
          # inscribed at so that it does not intersect the zigzags.
          depth *= flap_info[neighbor_stick.name].width

          # Handle the orthogonal case where dir_xy_proj is near-zero,
          # meaning the target direction is purely along the flap's normal.
          if np.linalg.norm(dir_xy_proj) < crease_utils.DEFAULT_TOLERANCE:
            logging.warning(
                'dir_xy_proj is close to zero for stick: %s. '
                'Using pure out-of-plane bend.',
                neighbor_stick.name,
            )
            bend_sign = np.sign(
                np.dot(direction_vec_global, init_post_matrix_global.z)
            )
            if bend_sign == 0:
              bend_sign = 1
            dir_global_proj_angle_deg = 90.0 * bend_sign
          else:
            dir_xy_proj_normalized = dir_xy_proj / np.linalg.norm(dir_xy_proj)
            dir_global_proj_angle_rad = np.arccos(
                np.clip(
                    np.dot(direction_vec_global, dir_xy_proj_normalized),
                    -1.0,
                    1.0,
                )
            )
            bend_sign = np.sign(
                np.dot(direction_vec_global, init_post_matrix_global.z)
            )
            if bend_sign == 0:
              bend_sign = 1
            dir_global_proj_angle_deg = (
                math.degrees(dir_global_proj_angle_rad) * bend_sign
            )

          bend_axis_global = init_post_matrix_global.x
          if not np.isclose(dir_global_proj_angle_deg, 0.0):
            bend_rotation_matrix = (
                angle_utils.get_rotation_matrix_from_axis_angle(
                    axis_vec=bend_axis_global,
                    angle_deg=dir_global_proj_angle_deg,
                )
            )
            posture_after_bend = bend_rotation_matrix @ init_post_matrix_global
          else:
            posture_after_bend = init_post_matrix_global

          simple_fold_infos.append(
              SimpleFoldInfo(
                  posture_matrix_before_shaping=init_post_matrix_global,
                  posture_matrix_after_shaping=posture_after_bend,
                  rotation_axis=bend_axis_global,
                  rotation_axis_local=np.array([1, 0, 0]),
                  rotation_angle=dir_global_proj_angle_deg,
                  required_flip=False,
                  inscribe_depth=depth,
              )
          )

          assert simple_fold_infos, (
              'No simple fold info found for stick: %s' % neighbor_stick.name
          )
          flap_posture_dict[neighbor_stick.name] = simple_fold_infos

  return flap_posture_dict
