"""
split_notebook.py
-----------------
Splits visualsv3.ipynb into three focused notebooks:
  01_ode_models.ipynb   -- Sections 1-2  (cells  0- 7)  ODE models
  02_1d_pde.ipynb       -- Sections 3-4  (cells  8-37)  1D PDE single & two-species
  03_2d_ocean.ipynb     -- Sections 5-6  (cells 38-48)  2D ocean (self-contained)

The original visualsv3.ipynb is NOT modified.
"""

import nbformat
import copy

SRC = "visualsv3.ipynb"

SPLITS = [
    ("01_ode_models.ipynb",  0,  8,  "§1–2  Logistic ODE & ODE+Fishing"),
    ("02_1d_pde.ipynb",      8, 38,  "§3–4  1D PDE: single-species & two-species"),
    ("03_2d_ocean.ipynb",   38, 49,  "§5–6  2D Ocean: single-species & competing"),
]

nb_src = nbformat.read(SRC, as_version=4)
total = len(nb_src.cells)
print(f"Source: {SRC}  ({total} cells)")

for fname, start, end, label in SPLITS:
    nb_new = nbformat.v4.new_notebook()
    nb_new.metadata = copy.deepcopy(nb_src.metadata)   # copy kernelspec etc.
    nb_new.cells = copy.deepcopy(nb_src.cells[start:end])
    n = len(nb_new.cells)
    nbformat.write(nb_new, fname)
    print(f"  Written {fname}  ({n} cells, [{start}:{end}])  -- {label}")

print("\nDone. visualsv3.ipynb is unchanged.")
