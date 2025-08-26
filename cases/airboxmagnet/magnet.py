#!/usr/bin/env python3
"""
magnet.py
Solve vector potential A for a permanent magnet in an airbox and output A and B.

Key fixes vs previous version:
- Use magnetization in RHS as  ∫_{magnet} M · curl(v) dx  (not curl(M)·v).
  This properly injects the bound surface current K_m = n × M.
- Compute B via L2 projection: find B∈CG1^3 s.t. ∫ B·w dx = ∫ curl(A)·w dx ∀w.
  This gives a clean CG1 vector field for ParaView.
- Assemble with PETSc backends to avoid MatrixCSR issues across dolfinx versions.
"""
import os
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, mesh
from dolfinx.fem import Function, functionspace, Expression
from dolfinx.fem.petsc import (
    assemble_matrix as p_assemble_matrix,
    assemble_vector as p_assemble_vector,
    apply_lifting as p_apply_lifting,
    set_bc as p_set_bc,
)
from mesh_utils import generate_or_load_mesh

comm = MPI.COMM_WORLD
rank = comm.rank

# -----------------------------
# Load mesh + cell tags
# -----------------------------
msh, domain_tags = generate_or_load_mesh("magnet.msh", "magnet.xdmf")

msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim)


# -----------------------------
# Function spaces
# -----------------------------
V = functionspace(msh, ("N1curl", 1))                   # H(curl) for A
V_cg1_vec = functionspace(msh, ("CG", 1, (msh.geometry.dim,)))  # CG1^3 for vis/projection
W_mat = functionspace(msh, ("DG", 0))                   # DG0 for material field

# -----------------------------
# Material constants
# -----------------------------
mu0 = 4 * np.pi * 1e-7
nu = fem.Constant(msh, PETSc.ScalarType(1.0 / mu0))     # assume air everywhere

# -----------------------------
# Magnetization: uniform +z inside magnet subdomain
# -----------------------------
magnet_tag = 2

# Use a physically meaningful magnitude (A/m). For Br≈1.1T, M≈Br/μ0 ≈ 8.75e5 A/m
Br = 1.1  # Tesla (adjust to your geometry)
Mmag = Br / mu0

# DG0 magnetization
M_space = functionspace(msh, ("CG", 1, (msh.geometry.dim,)))
M_func = fem.Function(M_space)
M_func.x.array[:] = 0.0

magnet_cells = np.where(domain_tags.values == magnet_tag)[0]
for cell in magnet_cells:
    dofs = fem.locate_dofs_topological(M_space, msh.topology.dim, [cell])
    for dof in dofs:
        M_func.x.array[dof * msh.geometry.dim + 2] = Mmag  # z
M_func.x.scatter_forward()

# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
dx = ufl.Measure("dx", domain=msh, subdomain_data=domain_tags)

a = ufl.inner(nu * ufl.curl(u), ufl.curl(v)) * dx
L = ufl.inner(M_func, ufl.curl(v)) * dx


# -----------------------------
# Boundary condition: A×n = 0 (implemented as A=0 on outer box facets)
# -----------------------------
facet_dim = msh.topology.dim - 1
box_half = 0.5  # must match your .geo

def outer_boundary(x):
    return np.any(np.isclose(np.abs(x), box_half), axis=0)

outer_facets = mesh.locate_entities_boundary(msh, facet_dim, outer_boundary)
bc_dofs = fem.locate_dofs_topological(V, facet_dim, outer_facets)
bc_func = Function(V)
bc_func.x.array[:] = 0.0
#bc = fem.dirichletbc(bc_func, bc_dofs)

# -----------------------------
# Helpers
# -----------------------------
def assemble_system(a_ufl, L_ufl, bcs=None, name="System"):
    """Assemble PETSc matrix and vector with BCs (dolfinx.fem.petsc backend)."""
    if rank == 0:
        print(f"{name}: Assembling system")

    a_form = fem.form(a_ufl)
    L_form = fem.form(L_ufl)

    A = p_assemble_matrix(a_form, bcs=bcs)
    A.assemble()

    b = p_assemble_vector(L_form)
    if bcs:
        p_apply_lifting(b, [a_form], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        p_set_bc(b, bcs)
    return A, b

def save_function(fnc, name, filename):
    fnc.name = name
    with io.XDMFFile(msh.comm, filename, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(fnc)
    if rank == 0:
        arr = fnc.x.array
        print(f"{name}: min={arr.min():.3e}, max={arr.max():.3e}")

def l2_project_to(V_proj, expr, bcs=None, label="proj"):
    """L2 projection of an expression into a target space V_proj."""
    u = ufl.TrialFunction(V_proj)
    v = ufl.TestFunction(V_proj)

    aP = ufl.inner(u, v) * ufl.dx
    LP = ufl.inner(expr, v) * ufl.dx

    a_form = fem.form(aP)
    L_form = fem.form(LP)

    A = p_assemble_matrix(a_form, bcs=bcs if bcs else [])
    A.assemble()
    b = p_assemble_vector(L_form)
    if bcs:
        p_apply_lifting(b, [a_form], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        p_set_bc(b, bcs)

    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("jacobi")

    uh = fem.Function(V_proj)
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    return uh



# -----------------------------
# Assemble and solve for A
# -----------------------------
A_mat, b_vec = assemble_system(a, L, bcs=[], name="A_system")

solver = PETSc.KSP().create(comm)
solver.setOperators(A_mat)
solver.setType("minres")              
solver.getPC().setType("hypre")       
solver.getPC().setHYPREType("boomeramg")
solver.setTolerances(rtol=1e-8, max_it=5000)
solver.setFromOptions()


A_sol = Function(V)
solver.solve(b_vec, A_sol.x.petsc_vec)
A_sol.x.scatter_forward()

if rank == 0:
    print("Solved A. H(curl) DOFs:", V.dofmap.index_map.size_global)

# -----------------------------
# Interpolate A to CG1^3 for visualization (optional)
# -----------------------------
A_vis = Function(V_cg1_vec)
A_expr = Expression(A_sol, V_cg1_vec.element.interpolation_points())
A_vis.interpolate(A_expr)
A_vis.x.scatter_forward()

# -----------------------------
# Compute B = curl(A) and L2-project to CG1^3 for ParaView
# -----------------------------
B_vis = l2_project_to(V_cg1_vec, ufl.curl(A_sol), label="B_from_A")

# -----------------------------
# Material field as DG0 (cell tags)
# -----------------------------
mat = Function(W_mat)
mat.x.array[:] = domain_tags.values
mat.x.scatter_forward()

# -----------------------------
# Write outputs
# -----------------------------
for f in (
    "A_field.xdmf",
    "A_field.xdmf.h5",
    "B_from_A.xdmf",
    "B_from_A.xdmf.h5",
    "material_field.xdmf",
    "material_field.xdmf.h5",
):
    try:
        os.remove(f)
    except FileNotFoundError:
        pass

save_function(A_vis, "A_field", "A_field.xdmf")
save_function(B_vis, "B_field", "B_from_A.xdmf")
save_function(mat, "Material_field", "material_field.xdmf")

if rank == 0:
    print("All outputs written for ParaView.")
