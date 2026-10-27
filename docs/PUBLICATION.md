# Initial repository publication — 2026-10-27

Repository: FEMview. Source extracted without changing original installations
or raw engineering measurements. Excludes private CAD, full simulation results,
virtual environments, user profiles and secrets. A local SHA-256 extraction
manifest is retained by the publication task. The public source is an initial
preview; domain-specific numerical acceptance remains separate from software tests.

## Verification

Fresh Python 3.12.11 environment with the pinned lockfile and editable FEMview/MechFEA packages; dependency check passed. Portable suite: **51 passed, 46 external-data skips**. Additional read-only RF regressions against retained cylinder, WR340 and copper-cavity studies: **28 passed** in 58.55 seconds. Native workers used the freshly installed EMFEA environment; no solver was launched by viewer tests.

CI pins MechFEA `0690840e21bc11a86d94eb365ce35d4c1aed27a3`. The bundled synthetic analytical bar retains original numerical files, with machine-specific run-record paths normalized. Geometry, load ratings and visual appearance on other graphics drivers are not established by these software checks.

## Hosted verification

Code `bb5aab6972a792ee465936da8300b2a9df62dcb2`: [successful GitHub Actions run](https://github.com/kazzurt/FEMview/actions/runs/40593725875). Windows hosted portable suite passed with checksum-pinned Mesa software OpenGL installed only on the disposable runner.
