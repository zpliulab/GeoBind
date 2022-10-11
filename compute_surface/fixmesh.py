from numpy.linalg import norm
import pymesh
from sklearn.neighbors import KDTree

"""
fixmesh.py: Regularize a protein surface mesh. 
- based on code from the PyMESH documentation. 
"""


def fix_mesh(mesh, vertice_info, resolution, detail="normal"):
    bbox_min, bbox_max=mesh.bbox;
    diag_len=norm(bbox_max - bbox_min);
    if detail == "normal":
        target_len=diag_len * 5e-3;
    elif detail == "high":
        target_len=diag_len * 2.5e-3;
    elif detail == "low":
        target_len=diag_len * 1e-2;

    target_len=resolution
    # print("Target resolution: {} mm".format(target_len));
    # PGC 2017: Remove duplicated vertices first
    mesh_new, _=pymesh.remove_duplicated_vertices(mesh, 0.001)

    count=0;
    # print("Removing degenerated triangles")
    mesh_new, __=pymesh.remove_degenerated_triangles(mesh_new, 100);
    mesh_new, __=pymesh.split_long_edges(mesh_new, target_len);
    num_vertices=mesh.num_vertices;
    while True:
        mesh_new, __=pymesh.collapse_short_edges(mesh_new, 1e-6);
        mesh_new, __=pymesh.collapse_short_edges(mesh_new, target_len,
                                             preserve_feature=True);
        mesh_new, __=pymesh.remove_obtuse_triangles(mesh_new, 150.0, 100);
        if mesh.num_vertices == num_vertices:
            break;

        num_vertices=mesh.num_vertices;
        # print("#v: {}".format(num_vertices));
        count+=1;
        if count > 10: break;

    mesh_new=pymesh.resolve_self_intersection(mesh_new);
    mesh_new, __=pymesh.remove_duplicated_faces(mesh_new);
    mesh_new =pymesh.compute_outer_hull(mesh_new);
    mesh_new, __=pymesh.remove_duplicated_faces(mesh_new);
    mesh_new, __=pymesh.remove_obtuse_triangles(mesh_new, 179.0, 5);
    mesh_new, __=pymesh.remove_isolated_vertices(mesh_new);
    mesh_new, _=pymesh.remove_duplicated_vertices(mesh_new, 0.001)
    # mesh_new, __=pymesh.remove_degenerated_triangles_raw(mesh_new.vertices, mesh_new.faces, 100);

    new_vertice_info = []
    kdtree = KDTree(mesh.vertices)
    for vertex in mesh_new.vertices:
        dis, pos = kdtree.query(vertex[None,:])
        new_vertice_info.append(vertice_info[pos[0,0]])

    return mesh_new.vertices, mesh_new.faces, new_vertice_info
