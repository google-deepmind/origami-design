---
name: stick-figure-design
description: >-
  Design, validate, pack, solve, and 3D-shape origami stick figures (`StickFigure`
  JSON) from natural-language descriptions using `bp_pipeline`. Use when asked to
  create a new origami stick figure or box-pleating crease pattern for an animal,
  character, dragon, insect, plant, or object, or when fixing validation,
  symmetry, packing, or 3D shaping errors on a stick figure tree.
---

# Stick Figure Design Skill (`bp_pipeline`)

This is the canonical workflow skill for designing a 3D `StickFigure` JSON
specification from a natural-language prompt and compiling it into a globally
flat-foldable box-pleated base and 3D-shaped origami model.

Read the two companion references in this skill folder before drafting complex
models:

-   [references/schema_and_archetypes.md](references/schema_and_archetypes.md) —
    Full JSON schema, 3D spherical coordinate math, and 6 verified archetypes
    (quadruped, biped, scorpion, crane, antlered stag, and 2-river-class winged
    dragon).
-   [references/design_rules.md](references/design_rules.md) — Single source of
    truth for river equivalence-class budgets ($\le 2$ classes), end-hub border
    capacity ($\le 3$ flaps), symmetry-axis width parity, 5-flap crown sizing,
    and proximal joint clearance ($L_{\text{upper}} > W
    \cot(\Delta\phi_{\text{base}} / 2)$).

## Workflow Overview

1.  **Plan the 3D Topology & River Budget**:
    -   Pick the closest archetype from
        [references/schema_and_archetypes.md](references/schema_and_archetypes.md)
        and keep the merged tree within **$\le 2$ river equivalence classes** (1
        on-axis spine river = 1 class; 1 mirrored pair of off-axis rivers such
        as `left_wing_arm` / `right_wing_arm` = 1 class).
    -   Use degree-2 joint subdivisions (`merge_linear_chains()`) to add knees,
        elbows, curved necks, horns, and tails at **zero packing cost** (keep
        subdivided segment `length >= 2`).
    -   Respect end-hub border capacity ($\le 3$ flaps on terminal hubs, or
        apply the 5-flap crown rule with `width >= 2` on all symmetry-axis
        sticks — see [references/design_rules.md](references/design_rules.md)).

2.  **Author the `StickFigure` JSON**:
    -   Orient every stick `start` -> `end` outward away from `root` (`+Y` =
        `azimuth_angle: 0`, `-X`/left = `+90`, `+X`/right = `-90`, `+Z`/up =
        `elevation_angle: +90`).
    -   Ensure strict bilateral symmetry across the YZ plane (`+θ` vs `-θ`
        azimuth, identical `elevation_angle`, `length`, and `width`, and uniform
        width parity `width % 2` across all symmetry-axis sticks).
    -   Set `"shaping_flap": false` on regular anatomical sticks, or set
        `"shaping_flap": true` on auxiliary pocket blades belonging to a valid
        `TailTemplate` (alternating `SF - River - SF - ...` chain, or central
        `tip_flap` + `SF` + `River` along the symmetry axis) or `KneeTemplate`
        (degree-3 joint with 1 `shaping_flap` leaf, 1 normal leaf flap, and 1
        proximal river). Shaping-flap templates are realized by the `'builtin'`
        packer only (`easy_mode=True` is not available for figures with shaping
        flaps), and are limited to at most 2 tails (`bottom` and `top`
        symmetry-axis edges) and corner knees (at most 4 total, or 2 mirrored
        pairs when symmetric). Save to e.g. `outputs/candidate.json`.

3.  **Validate, Render, Pack, Solve & Shape (`bp-pipeline` CLI)**:
    -   Run the installed `bp-pipeline` package CLI (or `.venv/bin/bp-pipeline`
        / `python -m bp_pipeline`):

        ```bash
        bp-pipeline outputs/candidate.json \
          --output-dir outputs/candidate \
          --check-symmetry
        ```

        -   `--validate-only` validates topology/symmetry and writes
            `stick_figure_views.png` before running packing.
        -   `--max-retries` (default `15`) controls how many grid sizes above
            $G_{\min}$ are searched; on symmetric trees, grid size steps by $+2$
            per retry ($\Delta G = 2 \times \text{retries}$).
        -   `--pack-timeout` (default `120.0`s) bounds the total multi-grid
            packing sweep so over-constrained 3-river-class trees fail fast
            instead of stalling.
        -   `--use-level-shifter` reserves sacrificial `2x2` corner / `2x4`
            river pockets on `width: 2` sticks (`builtin` packer and solver
            only) so solved bases widen wide flaps and rivers onto shifted
            integer axial-crease elevations. Leaf flaps must have
            `length > width` (`length >= 3` for `width: 2`) to receive a
            pocket; wide flaps with `length <= width` get no pocket and keep
            folded width 1.
        -   `--batch-dir <dir>` processes all `StickFigure` JSONs in `<dir>`,
            writing `<output_dir>/<stem>/report.json` incrementally as each
            design finishes and skipping report/metadata/`.bps.json` files.

4.  **Evaluate `report.json` & Visual Turnarounds**:
    -   Inspect `outputs/candidate/report.json` (written automatically whenever
        `--output-dir` is set):
        -   **Target acceptance criteria**:
            1.  `valid == true` (`validation_errors == []`),
            2.  `packing.success == true`,
            3.  `solving.success == true` and
                `solving.is_globally_flat_foldable == true` (verified uniaxial
                base),
            4.  `shaping.success == true` and **`shaping.discarded_elements ==
                []`** (every limb's 3D shaping folds preserved).
        -   *Expected non-failure*: `shaping.is_globally_flat_foldable: false`
            (e.g., `"taco-tortilla conflict"` or `"Local violations found"`) or
            `global_ff_timed_out: true` is **normal** on complex 3D-shaped
            models because `flat_folder` tests simultaneous $180^\circ$ flat
            folding rather than partial 3D dihedral angles (`fold_percent *
            ±180°`). Conversely, `shaping.is_locally_flat_foldable: true` alone
            is **not** sufficient if limbs were dropped into
            `shaping.discarded_elements`.
    -   Inspect the rendered visual outputs emitted by default into
        `outputs/candidate/` with your image/file viewer tool:
        1.  `outputs/candidate/stick_figure_views.png` (4-view skeleton
            projections; automatically decluttered on $> 16$-stick models),
        2.  `outputs/candidate/pipeline_stages.png` (5-panel summary: Packed CP,
            Solved Base CP, Uniaxial Folded Base, Shaped CP, and 3D Shaped
            Figure),
        3.  `outputs/candidate/shaped_3d_views.png` (**4-angle 3D shaded origami
            turnaround**: `Iso`, `Side`, `Front`, `Top`). Treat
            `shaped_3d_views.png` as ground truth for the folded 3D pose, limb
            separation, and silhouette fidelity.
        4.  `outputs/candidate/contact_sheet.html` (interactive HTML contact
            sheet with layer toggles for flaps, rivers, hinges, and labels).

## Bundled References & Package Entry Points

-   [references/schema_and_archetypes.md](references/schema_and_archetypes.md):
    Complete `StickFigure` JSON schema, coordinate conventions, and 6 canonical
    archetypes.
-   [references/design_rules.md](references/design_rules.md): Single source of
    truth for box-pleating packing and 3D shaping constraints.
-   [examples/](examples/): Verified example `StickFigure` JSON files
    (`cat.json`, `human.json`, `spider.json`, `palm_tree.json`).
-   Package CLI & visual API (`bp_pipeline`): `bp-pipeline` (`python -m
    bp_pipeline` / `bp_pipeline.cli.run_pipeline`),
    `bp.save_stick_figure_views()`, `bp.save_pipeline_summary()`,
    `bp.save_3d_turnaround()`, and `bp.render_packing_grid()`.
