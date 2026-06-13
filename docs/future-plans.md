# Future plans, incomplete features, and notes

## Future plans

- Implement a standalone runtime localisation module that consumes exported marker-corner CSV files.
- Add a robust PnP-with-RANSAC runtime path for live camera pose estimation.
- Provide a documented runtime API for loading maps, detecting markers, estimating pose, and returning diagnostics.
- Improve dense reconstruction quality, usability, and dependency setup.
- Add richer GUI progress reporting and error handling for long-running optimisation and dense reconstruction jobs.
- Expand example datasets and include end-to-end tutorial walkthroughs.

## Incomplete features

| Area | Status |
|---|---|
| Runtime localisation | Planned, not yet a standalone fully implemented module. |
| Dense reconstruction | Experimental and incomplete; optional dependencies are required. |
| Dense GUI integration | Partially integrated; further workflow polish and validation needed. |
| Optional dependency management | Dense reconstruction dependencies are intentionally not installed by the default requirements file. |
| Documentation | Initial documentation now exists, but API-level reference docs and tutorials can be expanded. |

## Critical bug watchlist

No critical confirmed bugs are documented in the repository at this time. Areas that should receive extra scrutiny during development are:

- coordinate-frame convention regressions, especially `T_A_B` composition and inversion;
- marker corner ordering mismatches between detector, optimiser, exporter, and runtime consumers;
- bundle-adjustment gauge freedom if anchor-marker handling changes;
- optional dependency failures in dense reconstruction paths;
- GUI/display assumptions in headless environments;
- persistence migrations or schema changes in `.map_builder/project.sqlite`.

## Additional notes for contributors

- Keep tests focused on geometry conventions and exported data formats when modifying optimisation or export code.
- Prefer clear diagnostics when optional dependencies are missing.
- Preserve the rigid-marker pose parameterisation unless there is a deliberate design change.
- Keep README concise and put detailed architecture, workflows, and module notes in this `docs` folder.
