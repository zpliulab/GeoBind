import os
import _pickle as cPickle
import numpy as np
from sklearn.neighbors import KDTree
from subprocess import Popen, PIPE

from Bio.PDB import *
from Bio.Seq import Seq
from Bio import SeqIO

from default_config.bin_path import bin_path

from compute_surface.protonate import protonate
from compute_surface.extract_multimer import extractPDB
from compute_surface.extract_xyzrn import extract_xyzrn, atom_radii
from compute_surface.apply_msms import computeMSMS
from compute_surface.fixmesh import fix_mesh
from compute_surface.compute_normal import compute_normal

from compute_surface.get_protein_seq import get_seq
import pymesh

class GeoBindProcessor():
    # pair: '2b3j:A_B:E_F' where 2b3j is PDB_id. A_B refers a dimer consists of two protein chains. E_F refers
    # to the nucleic chains binding to dimer.
    # anno: binding residues annotated by BioLip.
    # ligand_type: is RNA or DNA binding protein.
    def __init__(self, pair, anno, ligand_type='', dir_opts={}, Gaussian_window=12.0):
        self.ligand_type = ligand_type
        self.dir_opts = dir_opts
        self.pair = pair
        self.Gaussian_window= Gaussian_window
        self.pdb_id, self.protein_chains, self.nucleic_chains = pair.split(':') #complex, protein, rna
        #download pdb, call Reduce, MSMS and downsampling
        # self._download_pdb()
        self.mesh, self.normalv, self.vert_info = self._get_surface()

        #Extract the sequence of protien
        self.seq, self.index2resid = get_seq(self.pdb_id,  self.protein_chains, dir_opts=self.dir_opts)

        # Get the coordinates of atom of all RNAs in complex
        self.nucleic_space = self._get_nucleic_acid_atom()

        #Get res_id on the surface
        self.surface_res = self._get_surface_res()

        #Compute interface and site
        if ligand_type in ['DNA', 'RNA', 'ATP', 'HEM']:
            self.interface = self._get_interface()
        else:
            self.interface=self._get_interface_mental(anno)

        self.site, self.site2point = self._get_site_biolip(anno)
        pass

    def _download_pdb(self):
        if not os.path.exists(self.dir_opts['raw_pdb_dir']):
            os.makedirs(self.dir_opts['raw_pdb_dir'])

        pdbl = PDBList()
        pdbl.retrieve_pdb_file(self.pdb_id, pdir=self.dir_opts['raw_pdb_dir'], file_format='mmCif')

    def _get_surface(self):
        extractPDB(self.pair, dir_opts=self.dir_opts)
        protonate(self.pair, dir_opts=self.dir_opts)
        extract_xyzrn(self.pair, dir_opts=self.dir_opts)
        if self.ligand_type in ['DNA', 'RNA', 'ATP', 'HEM']:
            probe_radius = 1.5
        else:
            probe_radius = 0.5
        vertices, faces, normalv, vert_info = computeMSMS(self.pair,dir_opts=self.dir_opts, probe_radius=probe_radius)
        mesh = pymesh.form_mesh(vertices, faces)
        vertices_new, faces_new, vert_info_new = fix_mesh(mesh, vert_info, 1.2)
        new_normalv = compute_normal(vertices_new, faces_new)
        mesh_new = pymesh.form_mesh(vertices_new, faces_new)
        mesh_new.add_attribute("vertex_nx")
        mesh_new.set_attribute("vertex_nx", new_normalv[:, 0])
        mesh_new.add_attribute("vertex_ny")
        mesh_new.set_attribute("vertex_ny", new_normalv[:, 1])
        mesh_new.add_attribute("vertex_nz")
        mesh_new.set_attribute("vertex_nz", new_normalv[:, 2])
        #return vertices_new, vert_info_new, faces_new, new_normalv
        return mesh_new, new_normalv, vert_info_new
        # return mesh, normalv, vert_info

    def _load_pdb(self):
        pdb_file = os.path.join(self.dir_opts['raw_pdb_dir'], self.pdb_id+'.pdb')
        parser = MMCIFParser(QUIET=True)
        struct = parser.get_structure(self.pdb_id, pdb_file)
        model = Selection.unfold_entities(struct, "M")[0]
        return model

    def _get_nucleic_acid_atom(self):
        if self.ligand_type == 'RNA':
            ligand_residue_names = ["RA", "RC", "RG", "RU"]
            ligand_symbol = 'NUC'
        elif self.ligand_type == 'DNA':
            ligand_residue_names = ["DA", "DC", "DG", "DT"]
            ligand_symbol = 'NUC'
        else:
            ligand_residue_names = [self.ligand_type,]
            ligand_symbol = self.ligand_type
        # print(self.ligand_type)
        atom_list = []
        for chain_name in self.nucleic_chains.split('_'): #chain_name: e.g. A1. A is chain name, while 1 is kept for finding ligand pdb
            if len(chain_name)>1:
                pdb_file=os.path.join(self.dir_opts['ligand_dir'], self.pdb_id + '_' + ligand_symbol + '_' + chain_name[0] + '_' + chain_name[1] + '.pdb')
            else:
                pdb_file=os.path.join(self.dir_opts['ligand_dir'], self.pdb_id + '_' + ligand_symbol + '_' + chain_name[0] + '_1' + '.pdb')
            parser = PDBParser(QUIET=True)
            struct = parser.get_structure(self.pdb_id, pdb_file)
            model = Selection.unfold_entities(struct, "M")[0]
            chain = model.child_dict[chain_name[0]]
            for res in chain.child_list:
                res_type = res.resname.strip()
                if res_type not in ligand_residue_names:
                    continue
                for atom in res:
                    atom_list.append((atom.element, atom.coord, chain.id))
        return atom_list

    def _get_interface(self):
        NA_coords = np.array([atom[1] for atom in self.nucleic_space])
        # if self.aug:
        #     NA_coords =  np.dot(NA_coords, self.rot_matrix)
        kdtree = KDTree(NA_coords)
        interface = np.zeros([self.mesh.vertices.shape[0]])
        for index, vertex in enumerate(self.mesh.vertices):
            dis, indice = kdtree.query(vertex[None, :], k=1)
            if dis[0][0] < 3.0:
                interface[index] = 1
        return interface

    def _get_interface_mental(self, anno):
        anno = anno.split(':')
        site_by_biolip = set()
        for chain in anno:
            chain, binding_residues = chain.split(' ')[0], chain.split(' ')[1:]
            chain = chain[0]
            for residue in binding_residues:
                site_by_biolip.add(chain+'_'+ residue[1:])
        interface = []
        for item in self.vert_info:
            chain_res_id = item.split('_')[0] + '_' + item.split('_')[1]
            if chain_res_id in site_by_biolip:
                interface.append(1)
            else:
                interface.append(0)
        return interface

    def _get_site_biolip(self, anno):
        anno = anno.split(':')
        site_by_biolip = set()
        for chain in anno:
            chain, binding_residues = chain.split(' ')[0], chain.split(' ')[1:]
            chain = chain[0]
            for residue in binding_residues:
                site_by_biolip.add(chain+'_'+ residue[1:])
        self.site_by_biolip=site_by_biolip
        index2surface_res = {}
        surface_res2index = {}
        for index, res_id in enumerate(self.surface_res):
            index2surface_res[index] = res_id
            surface_res2index[res_id] = index
        residue2label = {surface_res2index[res_id]:[] for res_id in self.surface_res}
        point_res_id = np.zeros([len(self.vert_info)],dtype=np.long)
        for index, point_info in enumerate(self.vert_info):
            vert_resid = point_info.split('_')[0] +'_'+point_info.split('_')[1]
            point_res_id[index] = surface_res2index[vert_resid]
            if vert_resid in site_by_biolip:
                residue2label[surface_res2index[vert_resid]].append(1)
            else:
                residue2label[surface_res2index[vert_resid]].append(0)
        site_label = np.zeros([len(self.surface_res)],dtype=np.long)
        for index in range(len(self.surface_res)):
            site_label[index] = np.max(residue2label[index])
        return site_label, point_res_id

    def _get_surface_res(self):
        surface_res = set()
        for item in self.vert_info:
            chain = item.split('_')[0]
            resid = item.split('_')[1]
            surface_res.add(chain+'_'+resid)
        return list(surface_res)

    def _call_hhm(self, chain):
        HHblits = bin_path['HHblits']
        HHblits_DB = bin_path['HHblits_DB']

        if not os.path.exists(self.dir_opts['fasta_dir']):
            os.mkdir(self.dir_opts['fasta_dir'])
        if not os.path.exists(self.dir_opts['hhm_dir']):
            os.mkdir(self.dir_opts['hhm_dir'])

        #save fasta file
        seq = Seq(self.seq[chain])
        seq_io = SeqIO.SeqRecord(seq, name = self.pdb_id + '_' + chain)
        fasta_dir = os.path.join(self.dir_opts['fasta_dir'], self.pdb_id + '_' + chain + '.fasta')
        if not os.path.exists(fasta_dir):
        # if True:
            SeqIO.write(seq_io, fasta_dir, "fasta")

        hhm_dir = os.path.join(self.dir_opts['hhm_dir'], self.pdb_id+'_'+chain+'.hhm')

        args = [HHblits, '-d', HHblits_DB, '-i', fasta_dir, '-ohhm', hhm_dir]
        if not os.path.exists(hhm_dir):
        # if True:
            p2 = Popen(args, stdout=PIPE, stderr=PIPE)
            print('calling hhblits start')
            stdout, stderr = p2.communicate()
            print('calling hhblits over')
        hhm_feature = {}

        with open(hhm_dir, 'r') as f:
            text=f.readlines()
        hhm_begin_line=0
        hhm_end_line=0
        for i in range(len(text)):
            if '#' in text[i]:
                hhm_begin_line=i + 5
            elif '//' in text[i]:
                hhm_end_line=i
        hhm=np.zeros([int((hhm_end_line - hhm_begin_line) / 3), 30])

        axis_x=0
        for i in range(hhm_begin_line, hhm_end_line, 3):
            line1=text[i].split()[2:-1]
            line2=text[i + 1].split()
            axis_y=0
            for j in line1:
                if j == '*':
                    hhm[axis_x][axis_y]=9999 / 10000.0
                else:
                    hhm[axis_x][axis_y]=float(j) / 10000.0
                axis_y+=1
            for j in line2:
                if j == '*':
                    hhm[axis_x][axis_y]=9999 / 10000.0
                else:
                    hhm[axis_x][axis_y]=float(j) / 10000.0
                axis_y+=1
            axis_x+=1
        hhm=(hhm - np.min(hhm)) / (np.max(hhm) - np.min(hhm))
        for index in range(int((hhm_end_line - hhm_begin_line) / 3)):
            resid = self.index2resid[chain][index]
            hhm_feature[resid] = hhm[index,:].tolist()

        return hhm_feature

    def _get_curvature(self):
        #Referring to masif https://github.com/LPDI-EPFL/masif/blob/master/source/masif_modules/read_data_from_surface.py
        self.mesh.add_attribute("vertex_mean_curvature")
        H = self.mesh.get_attribute("vertex_mean_curvature")
        self.mesh.add_attribute("vertex_gaussian_curvature")
        K = self.mesh.get_attribute("vertex_gaussian_curvature")
        elem = np.square(H) - K
        elem[elem <= 0] = 1e-8
        k1 = H + np.sqrt(elem)
        k2 = H - np.sqrt(elem)
        si = (k1 + k2) / (k1 - k2)
        si = np.arctan(si) * (2 / np.pi)
        return si

    def _get_STED(self):
        #sum of total Euclidean distance
        STED = []
        for point in self.mesh.vertices:
            dis = np.sum(np.linalg.norm(self.mesh.vertices - point[None,:], axis=1))
            STED.append(dis)
        STED = np.array(STED)
        STED = (STED-np.min(STED))/(np.max(STED)-np.min(STED))
        self.STED_graph = STED[:,None]

    def _get_BOARD(self):
        tree = KDTree(self.mesh.vertices)
        indices = tree.query_radius(self.mesh.vertices, r=12)
        BOARD = []
        for index in range(self.mesh.vertices.shape[0]):
            points = indices[index]
            board_coord = self.mesh.vertices[points]
            signed_dis = np.mean((board_coord - self.mesh.vertices[index])*self.normalv[index])
            BOARD.append(signed_dis)
        BOARD = np.array(BOARD)
        BOARD = (BOARD-np.min(BOARD))/(np.max(BOARD)-np.min(BOARD))
        self.BOARD_graph = BOARD[:, None]

    def _get_FLARE(self):
        tree = KDTree(self.mesh.vertices)
        indices_outer_ring = tree.query_radius(self.mesh.vertices,r=12)
        indices_inter_ring = tree.query_radius(self.mesh.vertices,r=9)
        FLARE = []
        for index in range(self.mesh.vertices.shape[0]):
            points_outer = set(indices_outer_ring[index])
            points_inter = set(indices_inter_ring[index])
            points = list(points_outer-points_inter)
            board_coord = self.mesh.vertices[points]
            signed_dis = np.mean((board_coord - self.mesh.vertices[index])*self.normalv[index])
            FLARE.append(signed_dis)
        FLARE = np.array(FLARE)
        FLARE = (FLARE-np.min(FLARE))/(np.max(FLARE)-np.min(FLARE))
        self.FLARE_graph = FLARE[:,None]

    def _get_graph(self):
        atom_onehot = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4} #defulat is 5.
        self.feature_hhm = {}
        for chain in self.protein_chains.split('_'):
            self.feature_hhm[chain] = self._call_hhm(chain)
        self.feature_curvature = self._get_curvature()
        hmm_graph = []
        atom_type_graph = []
        curvature_graph = []
        for index, vertex in enumerate(self.mesh.vertices):
            vert = self.vert_info[index]
            chain = vert.split('_')[0]
            atom_type = vert.split('_')[5]
            res_id = int(vert.split('_')[1])
            try:
                hmm_graph.append(self.feature_hhm[chain][res_id])#30 hmm
            except:
                pass
            if atom_type in atom_onehot.keys():
                atom_type_graph.append([atom_onehot[atom_type],]) #1 atom_type
            else:
                atom_type_graph.append([5,])
            curvature_graph.append([self.feature_curvature[index],]) #1 curvature

        return np.array(hmm_graph, dtype=float), \
               np.array(atom_type_graph, dtype=int),\
                np.array(curvature_graph, dtype=float)\

    def _tangent_vectors(self, scaler_function='curvature_graph', normals=None, tangent_bases=None):

        """Returns a pair of vector fields u and v to complete the orthonormal basis [n,u,v].

              normals        ->             uv
        (N, 3) or (N, S, 3)  ->  (N, 2, 3) or (N, S, 2, 3)

        This routine assumes that the 3D "normal" vectors are normalized.
        It is based on the 2017 paper from Pixar, "Building an orthonormal basis, revisited".

        Args:
            normals (Tensor): (N,3) or (N,S,3) normals `n_i`, i.e. unit-norm 3D vectors.

        Returns:
            (Tensor): (N,2,3) or (N,S,2,3) unit vectors `u_i` and `v_i` to complete
                the tangent coordinate systems `[n_i,u_i,v_i].
        Referring to dMaSIF (https://github.com/FreyrS/dMaSIF/blob/945196bebfe0298afe8bc6327a39378b3ad31dc1/geometry_processing.py)
        """
        from pykeops.numpy import LazyTensor

        if normals is None:
            normals = self.normalv
        x, y, z= normals[..., 0], normals[..., 1], normals[..., 2]
        s=(2 * (z >= 0)) - 1.0  # = z.sign(), but =1. if z=0.
        a=-1 / (s + z)
        b=x * y * a
        uv=np.stack((1 + s * x * x * a, s * b, -s * x, b, s + y * y * a, -y), axis=-1)
        if tangent_bases is None:
            tangent_bases = uv.reshape(uv.shape[:1] + (2, 3))
        points = self.mesh.vertices / self.Gaussian_window
        # Normals and local areas:


        # 3. Steer the tangent bases according to the gradient of "weights" ----

        # 3.a) Encoding as KeOps LazyTensors:
        # Orientation scores:
        weights_j = LazyTensor(getattr(self, scaler_function)[:,0].reshape(1, -1, 1))  # (1, N, 1)

        # weights_j = LazyTensor(self.curvature_graph[:,0].reshape(1, -1, 1))  # (1, N, 1)
        # Vertices:
        x_i = LazyTensor(points[:, None, :])  # (N, 1, 3)
        x_j = LazyTensor(points[None, :, :])  # (1, N, 3)
        # Normals:
        n_i = LazyTensor(normals[:, None, :])  # (N, 1, 3)
        n_j = LazyTensor(normals[None, :, :])  # (1, N, 3)
        # Tangent basis:
        uv_i = LazyTensor(tangent_bases.reshape([-1, 1, 6]))  # (N, 1, 6)

        # 3.b) Pseudo-geodesic window:
        # Pseudo-geodesic squared distance:
        rho2_ij = ((x_j - x_i) ** 2).sum(-1) * ((2 - (n_i | n_j)) ** 2)  # (N, N, 1)
        # Gaussian window:
        window_ij = (-rho2_ij).exp()  # (N, N, 1)

        # 3.c) Coordinates in the (u, v) basis - not oriented yet:
        X_ij = uv_i.matvecmult(x_j - x_i)  # (N, N, 2)

        # 3.d) Local average in the tangent plane:
        orientation_weight_ij = window_ij * weights_j  # (N, N, 1)
        orientation_vector_ij = orientation_weight_ij * X_ij  # (N, N, 2)


        orientation_vector_i = orientation_vector_ij.sum(dim=1)  # (N, 2)
        orientation_vector_i = (
            orientation_vector_i + 1e-5
        )  # Just in case someone's alone...

        # 3.e) Normalize stuff:
        # orientation_vector_i = F.normalize(orientation_vector_i, p=2, dim=-1)  #  (N, 2)
        orientation_vector_i = orientation_vector_i / np.linalg.norm(orientation_vector_i,axis=1,keepdims=True)
        ex_i, ey_i = (
            orientation_vector_i[:, 0][:, None],
            orientation_vector_i[:, 1][:, None],
        )  # (N,1)

        # 3.f) Re-orient the (u,v) basis:
        uv_i = tangent_bases  # (N, 2, 3)
        u_i, v_i = uv_i[:, 0, :], uv_i[:, 1, :]  # (N, 3)
        tangent_bases = np.stack(
            (ex_i * u_i + ey_i * v_i, -ey_i * u_i + ex_i * v_i), axis=1
        )  # (N, 6)

        # 4. Store the local 3D frame as an attribute --------------------------
        nuv = np.concatenate(
            (normals.reshape(-1, 1, 3), tangent_bases), axis=1
        )
        return nuv

    def get_data(self):
        self.hmm_graph, self.atom_type_graph, self.curvature_graph = self._get_graph()

        self._get_STED()
        self._get_BOARD()
        self._get_FLARE()

        # self.nuv_curvature = self._tangent_vectors(scaler_function='curvature_graph') #curvature_graph, STED_graph, BOARD_graph, FLARE_graph

        # self.nuv_STED = self._tangent_vectors(scaler_function='STED_graph') #curvature_graph, STED_graph, BOARD_graph, FLARE_graph
        self.nuv_BOARD = self._tangent_vectors(scaler_function='BOARD_graph') #curvature_graph, STED_graph, BOARD_graph, FLARE_graph
        # self.nuv_FLARE = self._tangent_vectors(scaler_function='FLARE_graph') #curvature_graph, STED_graph, BOARD_graph, FLARE_graph
        data_label = {'xyz':self.mesh.vertices,
                      # 'nuv_curvature':self.nuv_curvature,
                      # 'nuv_STED':self.nuv_STED,
                      'nuv_BOARD':self.nuv_BOARD,
                      # 'nuv_FLARE':self.nuv_FLARE,
                      'y':self.interface,
                      'site':self.site,
                      'site_point':self.site2point,
                      'hmm': self.hmm_graph,
                      'geo':self.curvature_graph,
                      'xyz_type': self.atom_type_graph}
        print('hello')
        # if np.sum(np.isnan(data_label['nuv_curvature'])) > 0 or np.sum(np.isnan(data_label['geo'])) > 0:
        #     print('Do not satisfy condition')
        #     return

        save_path = os.path.join(self.dir_opts['data_label'], self.pair)
        with open(save_path, 'wb') as pid:
            cPickle.dump(data_label, pid)


if __name__ == '__main__':
    from default_config.dir_options import dir_opts
    dir_opts = dir_opts('Dataset/RNA/')
    rbp=GeoBindProcessor('4kzz:b:i1',
                         anno='b_i N32 I34 G35 F38 S39 P40 T41:B_C F10 G18 S19 E20 Q21 G22 V24 R25 P26 I47 T48 A49 Q50 K53 L56 P57 T58 H59 F69 E70 S73 E78 Q79 D90',
                         ligand_type='RNA',
                         dir_opts=dir_opts)
    # item='6pif:A_B_C_D_E_F_G_H_J:1'
    item='4rb6:8:A'
    # rbp=GeoBindProcessor(item,
    #                  anno='A_1 A8 Y9 E10 R11 L39 L40 E44 H71 Y101 K102 W143 K144 R222 S224 Q225 V226 F227 K263 R283 R291 G344 G345 M346'+
    #                      ':B_1 A8 Y9 E10 R11 L39 L40 G41 H71 V73 Y101 K102 W143 K144 R222 S224 Q225 V226 F227 F262 K263 A266 R283 R291 K343 G344 G345 M346'+
    #                      ':C_1 Y9 E10 R11 L39 L40 G41 H71 V73 Y101 K102 W143 R222 S224 Q225 V226 F227 S243 F262 K263 A266 R283 R291 K343 G344 G345 M346'+
    #                      ':D_1 Y9 E10 R11 L39 L40 G41 Q42 V73 Y101 K102 W143 K144 R222 Q225 V226 F227 F262 K263 A266 R283 R291 G344 G345 M346'+
    #                      ':E_1 Y9 E10 R11 L39 L40 G41 H71 V73 Y101 W143 K144 R222 Q225 V226 F227 T228 F262 K263 A266 R283 R291 K343 G344 G345 M346'+
    #                      ':F_1 A8 Y9 E10 R11 H77 Y101 K102 W143 R147 K148 M220 R222 S224 Q225 V226 F227 R244 K263 R283 R291 K343 G344 G345 M346'+
    #                      ':G_1 F89 L198 T199 Q200 I201 S202 L203 Y210 P215 V216 A217 S403 P404 S411 A414 G417 R424 L452 T453 E455 P501 R503 L504 A505 R506 Y570 L583 R584 Y586'+
    #                      ':H_1 H29 Y33 R103 Q105 D108 K109 R117 R120 R121 L122 K124 R125 F138 S151 R161 N162 F163 R164 N188 S189 E190'+
    #                      ':J_1 D45 N47',
    #                  ligand_type='RNA',
    #                  dir_opts=dir_opts)
    # rbp=GeoBindProcessor(item,
    #                  anno='8_A P2 K3 M4 K5 T6 H7 K8 K11 K12 R13 K15 T17 A18 S19 K21 K26 T27 G28 K29 R30 H31 L32 N33 W34 Q35 S37 G38 K39 I41 R42 Q43 K44 R46 K47 F48 V49 L50 A51 K52 P53 E54 R57 Y64',
    #                  ligand_type='RNA',
    #                  dir_opts=dir_opts)
    rbp.get_data()
    print('over')