#!/usr/bin/env bash
set -e

# Usage: ./scripts/make_mesh.sh cases/poisson/mesh.geo
GEO="$1"
OUTDIR="$(dirname "$GEO")"
BASENAME="$(basename "$GEO" .geo)"

mkdir -p "$OUTDIR"

# Generate 3D (or 2D) mesh with Gmsh
gmsh -3 "$GEO" -format msh2 -o "$OUTDIR/${BASENAME}.msh"

python3 - <<PY
import meshio
import numpy as np

m = meshio.read("$OUTDIR/${BASENAME}.msh")

# --- Volume mesh (tetra in 3D / triangle in 2D) ---
volume_types = ["tetra", "triangle"]
volume_cells = []
volume_cell_data = []

for cell_block in m.cells:
    if cell_block.type in volume_types:
        volume_cells.append(cell_block)
        # Corresponding physical tags
        if "gmsh:physical" in m.cell_data_dict:
            data_array = m.cell_data_dict["gmsh:physical"][cell_block.type]
            volume_cell_data.append(np.array(data_array, dtype=int))
        else:
            volume_cell_data.append(np.zeros(len(cell_block.data), dtype=int))

volume_mesh = meshio.Mesh(
    points=m.points,
    cells=volume_cells,
    cell_data={"gmsh:physical": volume_cell_data}
)
meshio.write("$OUTDIR/${BASENAME}_domain.xdmf", volume_mesh)
print("Written volume mesh:", "$OUTDIR/${BASENAME}_domain.xdmf")

# --- Facet mesh (lines in 2D, triangles in 3D that are not volume) ---
facet_types = ["line", "triangle"]
facet_cells = []
facet_cell_data = []

for cell_block in m.cells:
    if cell_block.type in facet_types and cell_block.type not in volume_types:
        facet_cells.append(cell_block)
        if "gmsh:physical" in m.cell_data_dict:
            data_array = m.cell_data_dict["gmsh:physical"][cell_block.type]
            facet_cell_data.append(np.array(data_array, dtype=int))
        else:
            facet_cell_data.append(np.zeros(len(cell_block.data), dtype=int))

if facet_cells:
    facet_mesh = meshio.Mesh(
        points=m.points,
        cells=facet_cells,
        cell_data={"gmsh:physical": facet_cell_data}
    )
    meshio.write("$OUTDIR/${BASENAME}_facets.xdmf", facet_mesh)
    print("Written facet mesh:", "$OUTDIR/${BASENAME}_facets.xdmf")

PY

echo "Mesh generation completed: $OUTDIR/${BASENAME}.msh, _domain.xdmf, _facets.xdmf"
