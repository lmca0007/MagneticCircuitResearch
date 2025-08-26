#!/usr/bin/env python3
"""mesh_utils.py
Load a DolfinX mesh from a .msh file and output an XDMF for reuse.
"""

import os
from mpi4py import MPI
from dolfinx import io, mesh
from dolfinx.io import gmshio

def generate_or_load_mesh(msh_file: str, xdmf_file: str):
    """Load a .msh file and write it as XDMF for DolfinX.
    
    Returns:
        msh: dolfinx.mesh.Mesh
        cell_tags: dolfinx.mesh.MeshTags
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if os.path.exists(xdmf_file):
        if rank == 0:
            print(f"Loading mesh from {xdmf_file}")
        with io.XDMFFile(comm, xdmf_file, "r") as xdmf:
            msh = xdmf.read_mesh(name="mesh")
            cell_tags = xdmf.read_meshtags(msh, name="Cell tags")
        return msh, cell_tags

    # -----------------------------
    # Load mesh directly from .msh
    # -----------------------------
    if rank == 0:
        print(f"Loading mesh from {msh_file}")
    msh, cell_tags, facet_tags = gmshio.read_from_msh(msh_file, comm=comm, gdim=3)

    # -----------------------------
    # Write XDMF for reuse
    # -----------------------------
    with io.XDMFFile(comm, xdmf_file, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_meshtags(cell_tags, msh.geometry)

    if rank == 0:
        print(f"Mesh written to {xdmf_file} with {msh.topology.index_map(msh.topology.dim).size_global} cells")
    return msh, cell_tags

# Example usage:
if __name__ == "__main__":
    msh_file = "magnet.msh"
    xdmf_file = "magnet.xdmf"
    load_msh_to_xdmf(msh_file, xdmf_file)
