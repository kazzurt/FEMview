# Initial repository publication — 2026-10-27

Repository: FEMview. Source extracted without changing original installations
or raw engineering measurements. Excludes private CAD, full simulation results,
virtual environments, user profiles and secrets. A local SHA-256 extraction
manifest is retained by the publication task. The public source is an initial
preview; domain-specific numerical acceptance remains separate from software tests.

Validation results will be appended after testing the exact extracted code.

## Verification

Fresh Python 3.12.11 environment with the pinned lockfile and editable FEMview/MechFEA packages; dependency check passed. Portable suite: **51 passed, 46 external-data skips**. Additional read-only RF regressions against retained cylinder, WR340 and copper-cavity studies: **28 passed** in 58.55 seconds. Native workers used the freshly installed EMFEA environment; no solver was launched by viewer tests.

CI pins MechFEA `1f59407fde6fd073cb39de07c859f0fc92225e5f`. The bundled synthetic analytical bar retains original numerical files, with machine-specific run-record paths normalized. Geometry, load ratings and visual appearance on other graphics drivers are not established by these software checks.
