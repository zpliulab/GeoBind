import pymesh
import numpy
from compute_surface.compute_normal import compute_normal
"""
read_ply.py: Save a ply file to disk using pymesh and load the attributes used by MaSIF. 
Pablo Gainza - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""


def save_ply(
    filename,
    vertices,
    faces=[],
    normals=None,
    u=None,
        v=None,
    charges=None,
    vertex_cb=None,
    hbond=None,
    hphob=None,
    iface_residue=None,
        iface_atom=None,
        iface_vertex=None,
    normalize_charges=False,
        features =None,
        label=None,
        curvature=None,
        patch=None
):
    """ Save vertices, mesh in ply format.
        vertices: coordinates of vertices
        faces: mesh
    """
    # filename = '/home/aoli/Documents/GeoBind/dataset/ply/'+filename+ '.ply'
    mesh = pymesh.form_mesh(vertices, faces)
    normals = compute_normal(vertices,faces)
    if normals is not None:
        n1 = normals[:, 0]
        n2 = normals[:, 1]
        n3 = normals[:, 2]
        mesh.add_attribute("vertex_nx")
        mesh.set_attribute("vertex_nx", n1)
        mesh.add_attribute("vertex_ny")
        mesh.set_attribute("vertex_ny", n2)
        mesh.add_attribute("vertex_nz")
        mesh.set_attribute("vertex_nz", n3)
    if u is not None:
        u1 = u[:, 0]
        u2 = u[:, 1]
        u3 = u[:, 2]
        mesh.add_attribute("vertex_ux")
        mesh.set_attribute("vertex_ux", u1)
        mesh.add_attribute("vertex_uy")
        mesh.set_attribute("vertex_uy", u2)
        mesh.add_attribute("vertex_uz")
        mesh.set_attribute("vertex_uz", u3)
    if v is not None:
        v1 = v[:, 0]
        v2 = v[:, 1]
        v3 = v[:, 2]
        mesh.add_attribute("vertex_vx")
        mesh.set_attribute("vertex_vx", v1)
        mesh.add_attribute("vertex_vy")
        mesh.set_attribute("vertex_vy", v2)
        mesh.add_attribute("vertex_vz")
        mesh.set_attribute("vertex_vz", v3)
    if charges is not None:
        mesh.add_attribute("charge")
        if normalize_charges:
            charges = charges / 10
        mesh.set_attribute("charge", charges)
    if hbond is not None:
        mesh.add_attribute("hbond")
        mesh.set_attribute("hbond", hbond)
    if vertex_cb is not None:
        mesh.add_attribute("vertex_cb")
        mesh.set_attribute("vertex_cb", vertex_cb)
    if hphob is not None:
        mesh.add_attribute("vertex_hphob")
        mesh.set_attribute("vertex_hphob", hphob)
    if iface_residue is not None:
        mesh.add_attribute("vertex_iface_residue")
        mesh.set_attribute("vertex_iface_residue", iface_residue)
    if iface_atom is not None:
        mesh.add_attribute("vertex_iface_atom")
        mesh.set_attribute("vertex_iface_atom", iface_atom)
    if iface_vertex is not None:
        mesh.add_attribute("vertex_iface_vertex")
        mesh.set_attribute("vertex_iface_vertex", iface_vertex)
    if features is not None:
        mesh.add_attribute("features")
        mesh.set_attribute("features", features)
    if label is not None:
        mesh.add_attribute("label")
        mesh.set_attribute("label", label)
    if curvature is not None:
        mesh.add_attribute('curvature')
        mesh.set_attribute("curvature", curvature)
    if patch is not None:
        mesh.add_attribute('patch')
        mesh.set_attribute("patch", patch)
    pymesh.save_mesh(
        filename, mesh, *mesh.get_attribute_names(), use_float=True, ascii=True
    )

