# FEMview

Read-only structural and electromagnetic finite-element results workbench.
The Python package is `fempost`. Structural and RF results have independent
scientific models and renderers in one tabbed application.

Structural: CalculiX FRD/DAT/INP, deformed fields, stress invariants, contact,
load/restraint display, clipping, probes and convergence comparison.
RF: complex E/H, phase playback, cut-plane vectors, Poynting, spectra/S-parameters,
documented eigenmode Q and conductor surface current where retained data permits it.
Opening results never starts a simulation or alters the saved source files.

## Install on Windows

Clone [MechFEA](https://github.com/kazzurt/MechFEA) beside this repository.
Install Python 3.12 x64 and uv, then:

```powershell
.\setup.ps1 -Python C:\path\to\python.exe
.\launch-femview.ps1 C:\path\to\saved-study
```

Setup installs a dedicated pinned Qt/VTK environment and both local Python packages
editable. MechFEA supplies read-only CalculiX parsers/tensor helpers; its native
solver is not needed. Pass `-MechFEA` if its checkout is elsewhere.

Native `.emerge` loading uses a **separate** EMerge 3.0.0b6 worker environment:

```powershell
$env:FEMPOST_RF_PYTHON = 'C:\path\to\EMFEA\.venv\Scripts\python.exe'
.\launch-femview.ps1 C:\path\to\study
```

The default looks for sibling `EMFEA/.venv/Scripts/python.exe`. Sampled NPZ RF
results do not require EMerge. `FEMPOST_CACHE` and `FEMPOST_RF_CACHE` override
disposable per-user caches. The RF and GUI environments must stay separate because
their verified VTK versions differ. See [EMFEA](https://github.com/kazzurt/EMFEA).

## Verification

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv\Scripts\python.exe -m pytest -q
```

Synthetic tests run without private studies. Retained-data tests skip when absent;
`FEMVIEW_RF_FIXTURES` may point to a folder holding `emerge-cylindrical-cavity` and
`emerge-wr340-waveguide`, and `FEMVIEW_STRUCTURAL_FIXTURE` to a retained structural
run. A read-only guard checks source files. Headless tests do not establish visual
quality on every graphics driver. Latest phase/vector display still benefits from
user visual review. See `docs/PUBLICATION.md` for validation scope.
