# Future plans, incomplete features, and notes

## Future plans

- Add optional temporal filtering and tracking around the implemented absolute pose solver.
- Add application-specific pose-quality thresholds and uncertainty estimates.
- Improve dense reconstruction quality, usability, and dependency setup.
- Add richer GUI progress reporting and error handling for long-running optimisation and dense reconstruction jobs.
- Expand example datasets and include end-to-end tutorial walkthroughs.

## Incomplete features

| Area | Status |
|---|---|
| Runtime localisation | Absolute pose solving is implemented; temporal tracking, uncertainty estimates, and live-application integration remain future work. |
| Dense reconstruction | Experimental and incomplete; optional dependencies are required. |
| Dense GUI integration | Partially integrated; further workflow polish and validation needed. |
| Optional dependency management | Dense reconstruction dependencies are intentionally not installed by the default requirements file. |
| Documentation | Core Map Builder and absolute-pose guides exist; more end-to-end tutorials can still be added. |

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
