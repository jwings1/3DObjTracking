from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import glob
import time
import pickle
from sklearn.neighbors import NearestNeighbors
import torch
from psbody.mesh import Mesh
from collections import defaultdict
import numpy as np
from scipy.sparse import load_npz
import argparse
import trimesh


class JointRegressor:
    def __init__(self, device='cuda:0'):
        # Loading the sparse matrix
        J_np = load_npz('/scratch/lgermano/3DTrackEval_copy/J_regressor.npz').todense()
        self.J = torch.tensor(J_np, dtype=torch.float32).to(device)
        
    def multiply_by_J(self, mesh, device='cuda:0'):
        mesh_v = torch.tensor(mesh.v, dtype=torch.float32).to(device)
        return self.J.matmul(mesh_v)

def mpjpe(pred, gt, device='cuda:0'):
    """ 
    Compute the Mean Per Joint Position Error (MPJPE) between the predicted and ground truth 3D poses.
    
    Parameters:
    pred -- Predicted joints, torch tensor of shape (24, 3) on the specified device
    gt -- Ground truth joints, torch tensor of shape (24, 3) on the specified device

    Returns:
    mpjpe -- Mean Per Joint Position Error
    """
    # Check shapes
    assert pred.shape == gt.shape
    assert pred.shape[1] == 3  # Ensure 3D joints

    # Compute per-joint position error: Euclidean distance between predicted and true joints
    joint_errors = torch.norm(pred - gt, dim=1)

    # Compute mean per-joint position error
    mpjpe = torch.mean(joint_errors)

    return mpjpe.item()

class ProcrusteAlign:
    'procrustes align'
    def __init__(self, smpl_only=False, device='cuda:0'):
        self.warned = False
        self.smpl_only = smpl_only
        self.device = device

    def to_tensor(self, data):
        return torch.tensor(data).to(self.device)

    def to_numpy(self, tensor):
        return tensor.cpu().numpy()

    def align_meshes(self, ref_meshes, recon_meshes):
        ref_v, recon_v = [], []
        v_lens = []
        R, recon_v, scale, t = self.get_transform(recon_meshes, recon_v, ref_meshes, ref_v, v_lens)

        recon_v = self.to_tensor(recon_v)
        recon_hat = (scale * R.mm(recon_v.t()) + t).t()

        ret_meshes = []
        offset = 0
        for m in recon_meshes:
            newm = Mesh(v=self.to_numpy(recon_hat[offset:offset + len(m.v)].clone()), f=m.f.copy())
            ret_meshes.append(newm)
            offset += len(m.v)

        return ret_meshes

    def get_transform(self, recon_meshes, recon_v, ref_meshes, ref_v, v_lens):
        offset = 0
        recon_v, ref_v = self.comb_meshes(offset, recon_meshes, recon_v, ref_meshes, ref_v, v_lens)
        recon_v, ref_v = self.to_tensor(recon_v), self.to_tensor(ref_v)

        if ref_v.shape == recon_v.shape and not self.smpl_only:
            R, t, scale = compute_transform(recon_v, ref_v)
            return R, recon_v, scale, t
        else:
            if not self.warned:
                print("Warning: align using only smpl meshes!")
                self.warned = True
            smpl_recon_v = self.to_tensor(recon_meshes[0].v)
            smpl_ref_v = self.to_tensor(ref_meshes[0].v)
            R, t, scale = compute_transform(smpl_recon_v, smpl_ref_v)
            return R, recon_v, scale, t

    def comb_meshes(self, offset, recon_meshes, recon_v, ref_meshes, ref_v, v_lens):
        for fm, rm in zip(ref_meshes, recon_meshes):
            ref_v.append(fm.v)
            recon_v.append(rm.v)
            offset += fm.v.shape[0]
            v_lens.append(offset)

        ref_v = np.concatenate(ref_v, 0)
        recon_v = np.concatenate(recon_v, 0)
        return recon_v, ref_v

    def align_neural_recon(self, ref_meshes, recon_meshes, neural_recons):
        ref_v, recon_v = [], []
        v_lens = []
        R, recon_v, scale, t = self.get_transform(recon_meshes, recon_v, ref_meshes, ref_v, v_lens)

        points_all = self.to_tensor(np.concatenate([x.v for x in neural_recons], 0))
        recon_hat = (scale * R.mm(points_all.t()) + t).t()

        ret_meshes = []
        last_idx = 0
        for i, L in enumerate(v_lens):
            m = Mesh(v=self.to_numpy(recon_hat[last_idx:L].clone()), f=recon_meshes[i].f.copy())
            ret_meshes.append(m)
            last_idx = L

        return ret_meshes

def compute_transform(S1, S2):
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.t()
        S2 = S2.t()
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    mu1 = S1.mean(dim=1, keepdim=True)
    mu2 = S2.mean(dim=1, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    var1 = torch.sum(X1**2)

    K = X1.mm(X2.t())

    U, s, V = torch.svd(K)
    Z = torch.eye(U.shape[0]).to('cuda:0')
    Z[-1, -1] *= torch.sign(torch.det(U.mm(V.t())))
    R = V.mm(Z.mm(U.t()))

    scale = torch.trace(R.mm(K)) / var1

    t = mu2 - scale*(R.mm(mu1))

    return R, t, scale


class GPUChamferDistance:
    def __init__(self):
        self.x_nn = None
        self.y_nn = None

    def _nearest_neighbors(self, x, y):
        x = x.unsqueeze(0).contiguous()
        y = y.unsqueeze(0).contiguous()
        dists = torch.cdist(x, y)
        min_y_to_x = dists.min(dim=2)[0]
        min_x_to_y = dists.min(dim=1)[0]
        return min_y_to_x, min_x_to_y

    def compute(self, x, y, direction='bi'):
        x = torch.tensor(x).cuda()
        y = torch.tensor(y).cuda()

        if direction == 'y_to_x':
            min_y_to_x, _ = self._nearest_neighbors(x, y)
            chamfer_dist = min_y_to_x.mean().item()
        elif direction == 'x_to_y':
            _, min_x_to_y = self._nearest_neighbors(x, y)
            chamfer_dist = min_x_to_y.mean().item()
        elif direction == 'bi':
            min_y_to_x, min_x_to_y = self._nearest_neighbors(x, y)
            chamfer_dist = min_y_to_x.mean().item() + min_x_to_y.mean().item()
        else:
            raise ValueError("Invalid direction type. Supported types: 'y_x', 'x_y', 'bi'")

        return chamfer_dist

def compute_similarity_transform_batch(S1, S2):
    S1_hat = torch.zeros_like(S1).cuda()
    for i in range(S1.shape[0]):
        S1_hat[i] = torch.tensor(compute_similarity_transform(S1[i], S2[i])).cuda()
    return S1_hat.cpu().numpy()

def reconstruction_error(S1, S2, reduction='mean'):
    S1 = torch.tensor(S1).cuda()
    S2 = torch.tensor(S2).cuda()

    S1_hat = compute_similarity_transform_batch(S1, S2)
    S1_hat = torch.tensor(S1_hat).cuda()
    re = torch.sqrt(((S1_hat - S2) ** 2).sum(dim=-1)).mean(dim=-1)
    if reduction == 'mean':
        re = re.mean().item()
    elif reduction == 'sum':
        re = re.sum().item()

    return re

def chamfer_distance(x, y, metric='l2', direction='bi'):
    chamfer = GPUChamferDistance()
    return chamfer.compute(x, y, direction)


def read_obj_file(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v'):
                # Splits the line into its components
                split_line = line.split()
                
                # We're only interested in the x, y, z coordinates which are the 2nd, 3rd, and 4th elements in the split_line list
                vertex_coordinates = [float(split_line[i]) for i in range(1, 4)]
                
                vertices.append(vertex_coordinates)
                
    # Convert the list of vertices to a NumPy array for easier manipulation
    vertices = np.array(vertices)

    return vertices

def v2v_err(p1, p2):
    "vertex to vertex error, p1, p2: (N, 3)"
    return np.sqrt(((p1 - p2) ** 2).sum(axis=-1)).mean(axis=-1)

def compute_metrics(trace_mesh_list, ground_mesh_list, aligned_trace_meshes):
    """
    Compute various evaluation metrics for pairs of trace meshes and ground truth meshes.

    Parameters:
    - trace_mesh_list: List of trace meshes.
    - ground_mesh_list: List of ground truth meshes.
    - aligned_trace_meshes: List of aligned trace meshes.

    Returns:
    - A dictionary containing the following metrics for each pair of trace mesh and ground truth mesh:
      - chamfer: L2-Chamfer distance.
      - pa_chamfer: PA-L2-Chamfer distance for aligned trace meshes.
      - v2v_err: Vertex to vertex error.
      - MPJPE: Mean per-joint position error.
      - PA_MPJPE: PA mean per-joint position error for aligned trace meshes.
    """
           
    result = {}
    CHAMFER = {}
    PA_CHAMFER = {}
    V2V_ERR = {}
    PA_MPJPE = {}
    MPJPE = {}

    # Compute L2-CHAMFER
    for i, (trace_mesh, ground_mesh) in enumerate(zip(trace_mesh_list, ground_mesh_list)):
        dist = chamfer_distance(trace_mesh.v, ground_mesh.v)
        mean_dist = np.mean(dist)
        CHAMFER[i] = mean_dist
        #print(CHAMFER[i])

    # Compute PA-L2-CHAMFER
    for i, (aligned_trace_mesh, ground_mesh) in enumerate(zip(aligned_trace_meshes, ground_mesh_list)):
        dist = chamfer_distance(aligned_trace_mesh.v, ground_mesh.v)
        mean_dist = np.mean(dist)
        PA_CHAMFER[i] = mean_dist
        #print(PA_CHAMFER[i])

    # Compute V2V
    for i, (aligned_trace_mesh, ground_mesh) in enumerate(zip(aligned_trace_meshes, ground_mesh_list)):
        v2v_error = v2v_err(aligned_trace_mesh.v, ground_mesh.v)
        V2V_ERR[i] = v2v_error
        #print(V2V_ERR[i])

    regressor = JointRegressor()

    # Compute PA-MPJPE
    for i, (aligned_trace_mesh, ground_mesh) in enumerate(zip(aligned_trace_meshes, ground_mesh_list)):
        aligned_trace_joints = regressor.multiply_by_J(aligned_trace_mesh)
        ground_joints = regressor.multiply_by_J(ground_mesh)
        pa_mpjpe_error = mpjpe(aligned_trace_joints, ground_joints)
        PA_MPJPE[i] = pa_mpjpe_error
        #print(PA_MJPE[i])

    # Compute MPJPE
    for i, (trace_mesh, ground_mesh) in enumerate(zip(trace_mesh_list, ground_mesh_list)):
        trace_joints = regressor.multiply_by_J(trace_mesh)
        ground_joints = regressor.multiply_by_J(ground_mesh)
        mpjpe_error = mpjpe(trace_joints, ground_joints)
        MPJPE[i] = mpjpe_error
        #print(MPJPE[i])

    result['chamfer'] = CHAMFER
    result['pa_chamfer'] = PA_CHAMFER
    result['v2v_err'] = V2V_ERR
    result['MPJPE'] = MPJPE
    result['PA_MPJPE'] = PA_MPJPE

    return result

def align_meshes(trace_mesh_list, ground_mesh_list):
    """
    Align trace meshes to ground truth meshes using Procruste alignment.

    Parameters:
    - trace_mesh_list: List of trace meshes.
    - ground_mesh_list: List of ground truth meshes.

    Returns:
    - aligned_trace_meshes: List of aligned trace meshes.
    """
    
    # Initialize aligner
    aligner = ProcrusteAlign(smpl_only=True)

    # Determine the length of the shorter list
    min_length = min(len(trace_mesh_list), len(ground_mesh_list))

    # Cut off the excess elements from the beginning of each list
    trace_mesh_list = trace_mesh_list[-min_length:]
    ground_mesh_list = ground_mesh_list[-min_length:]

    # Alignment list wise
    aligned_trace_meshes = aligner.align_meshes(trace_mesh_list, ground_mesh_list)

    return trace_mesh_list, ground_mesh_list, aligned_trace_meshes


def calculate_distances(candidate_path, ground_path, method):

    #add caching? 
    ground_mesh_list = load_ground_truth_meshes(ground_path)

    if method == 'romp':

        print("Method: ROMP")
        # Print the name of the sequence currently being processed
        print("Processing sequence:", os.path.basename(candidate_path))

        # Initialize lists to store bev meshes, and aligned bev meshes
        romp_mesh_list = []
        aligned_romp_meshes = []

        for root, dirs, files in os.walk(candidate_path):
            for filename in files:
                if filename.startswith('frame_') and filename.endswith('.npz'):
                    full_path = os.path.join(root, filename)
                    loaded_mesh = np.loadtxt(full_path)
                    mesh = Mesh()
                    mesh.v = loaded_mesh[:6890,:3]
                    mesh.f = []
                    romp_mesh_list.append(mesh)

        # Align the trace meshes with the ground truth meshes
        romp_mesh_list, ground_mesh_list, aligned_romp_meshes = align_meshes(romp_mesh_list, ground_mesh_list)

        # Compute various evaluation metrics for the meshes
        return compute_metrics(romp_mesh_list, ground_mesh_list, aligned_romp_meshes)

    elif method == 'bev':

        print("Method: BEV")
        # Print the name of the sequence currently being processed
        print("Processing sequence:", os.path.basename(candidate_path))

        # Initialize lists to store bev meshes, and aligned bev meshes
        bev_mesh_list = []
        aligned_bev_meshes = []

        for root, dirs, files in os.walk(candidate_path):
            for filename in files:
                if filename.startswith('frame_') and filename.endswith('.npz'):
                    full_path = os.path.join(root, filename)
                    loaded_mesh = np.loadtxt(full_path)
                    mesh = Mesh()
                    mesh.v = loaded_mesh[:6890,:3]
                    mesh.f = []
                    bev_mesh_list.append(mesh)

        # Align the trace meshes with the ground truth meshes
        bev_mesh_list, ground_mesh_list, aligned_bev_meshes = align_meshes(bev_mesh_list, ground_mesh_list)

        # Compute various evaluation metrics for the meshes
        return compute_metrics(bev_mesh_list, ground_mesh_list, aligned_bev_meshes)
    
    elif method == 'trace':
        
        print("Method: TRACE")
        # Print the name of the sequence currently being processed
        print("Processing sequence:", os.path.basename(candidate_path))

        # Initialize lists to store trace meshes, and aligned trace meshes
        trace_mesh_list = []
        aligned_trace_meshes = []

        # TRACE mesh
        # Load all mesh files (with .ply extension) from the candidate path
        for item in os.listdir(candidate_path):
            full_path = os.path.join(candidate_path, item)
            
            # Check if the current item is a file with the .ply extension
            if os.path.isfile(full_path) and item.endswith('.ply'):
                # Load the mesh from the file and append to the trace mesh list
                mesh_candidate = Mesh(filename=full_path)
                trace_mesh_list.append(mesh_candidate)

        # Align the trace meshes with the ground truth meshes
        trace_mesh_list, ground_mesh_list, aligned_trace_meshes = align_meshes(trace_mesh_list, ground_mesh_list)

        # Compute various evaluation metrics for the meshes
        return compute_metrics(trace_mesh_list, ground_mesh_list, aligned_trace_meshes)

    else:
        
        # Indicate the processing method being used
        print("Method: 4DH")
        
        # Print the name of the sequence currently being processed
        print("Processing sequence:", os.path.basename(candidate_path))

        # Initialize lists to store 4DH meshes, and aligned 4DH meshes
        fdh_mesh_list = []
        aligned_fdh_meshes = []

        # Load all 4DH mesh files from the candidate path with a specific naming pattern (frame_*.obj)
        for filename in glob.glob(os.path.join(candidate_path, 'frame_*.obj')):
            mesh = Mesh(filename = filename)
            # Retain the first 6890 vertices and their x, y, z coordinates
            mesh.v = mesh.v[:6890,:3]
            # Remove all face information
            mesh.f = []
            fdh_mesh_list.append(mesh)

        # Align the FDH meshes with the ground truth meshes
        fdh_mesh_list, ground_mesh_list, aligned_fhd_meshes = align_meshes(fdh_mesh_list, ground_mesh_list)

        # Compute various evaluation metrics for the meshes
        return compute_metrics(fdh_mesh_list, ground_mesh_list, aligned_fdh_meshes)

    
def generate_directories_dict(method):

    if method == 'romp':

        sequences_dir = '/scratch_net/biwidl307_second/lgermano/behave/sequences'
        romp_ev_dir = '/scratch_net/biwidl307_second/lgermano/ROMP_ev/behave_date_03_1fps_python2'

        directories = {}

        # Iterate through directories in /scratch_net/biwidl307_second/lgermano/behave/sequences
        for root, dirs, _ in os.walk(sequences_dir):
            for dir_name in dirs:
                directories[os.path.join(root, dir_name)] = []
            # prevent os.walk from going into subdirectories
            break

        # Iterate through directories in /scratch_net/biwidl307_second/lgermano/ROMP_ev/behave_date_03_1fps
        for root, dirs, _ in os.walk(romp_ev_dir):
            for dir_name in dirs:
                #remove from dir name anything after .
                dir_name_root = dir_name.split('.')[0]
                for key in directories.keys():
                    # If the key ends with dir_name, append the full path to the value list
                    if key.endswith(dir_name_root):
                        directories[key].append(os.path.join(root, dir_name))
            # prevent os.walk from going into subdirectories
            break
        
        return directories

    elif method == 'bev':
        
        sequences_dir = '/scratch_net/biwidl307_second/lgermano/behave/sequences'
        bev_ev_dir = '/scratch-second/lgermano/BEV_ev/behave_date_03_1fps_python2'

        directories = {}

        # Iterate through directories in /scratch_net/biwidl307_second/lgermano/behave/sequences
        for root, dirs, _ in os.walk(sequences_dir):
            for dir_name in dirs:
                directories[os.path.join(root, dir_name)] = []
            # prevent os.walk from going into subdirectories
            break

        # Iterate through directories in /scratch_net/biwidl307_second/lgermano/BEV_ev/behave_date_03_1fps
        for root, dirs, _ in os.walk(bev_ev_dir):
            for dir_name in dirs:
                #remove from dir name anything after .
                dir_name_root = dir_name.split('.')[0]
                for key in directories.keys():
                    # If the key ends with dir_name, append the full path to the value list
                    if key.endswith(dir_name_root):
                        directories[key].append(os.path.join(root, dir_name))
            # prevent os.walk from going into subdirectories
            break  

        return directories
    
    elif method == 'trace':
        
        sequences_dir = '/scratch-second/lgermano/behave/sequences'
        trace_ev_dir = '/scratch/lgermano/TRACE_results/meshes_ply'

        directories = {}

        # Iterate through directories in /scratch_net/biwidl307_second/lgermano/behave/sequences
        for root, dirs, _ in os.walk(sequences_dir):
            for dir_name in dirs:
                directories[os.path.join(root, dir_name)] = []
            # prevent os.walk from going into subdirectories
            break

        for root, dirs, files in os.walk(trace_ev_dir):
            for file_name in dirs:
                # Remove from file name anything after .
                file_name_root = file_name.split('.')[0]
                for key in directories.keys():
                    # If the key ends with file_name_root, append the full path to the value list
                    if key.endswith(file_name_root):
                        directories[key].append(os.path.join(root, file_name))

        return directories    

    else :
        #4DH
        
        sequences_dir = '/scratch_net/biwidl307_second/lgermano/behave/sequences'
        dh_ev_dir = '/scratch_net/biwidl307_second/lgermano/4DH_ev/behave_date_03_1fps'

        directories = {}

        # Iterate through directories in /scratch_net/biwidl307_second/lgermano/behave/sequences
        for root, dirs, _ in os.walk(sequences_dir):
            for dir_name in dirs:
                directories[os.path.join(root, dir_name)] = []
            # prevent os.walk from going into subdirectories
            break

        # Iterate through directories in /scratch_net/biwidl307_second/lgermano/BEV_ev/behave_date_03_1fps
        for root, dirs, _ in os.walk(dh_ev_dir):
            for dir_name in dirs:
                #remove from dir name anything after .
                dir_name_root = dir_name.split('.')[0]
                for key in directories.keys():
                    # If the key ends with dir_name, append the full path to the value list
                    if key.endswith(dir_name_root):
                        directories[key].append(os.path.join(root, dir_name))
            # prevent os.walk from going into subdirectories
            break    

        return directories


def process_dict(directories, calculate_distances, method):
    new_dict = {}
    for key, values in directories.items():
        for value in values:
            # If values are also paths and we need to calculate distance for each pair
            result = calculate_distances(value, key, method)
            new_key = value.split('/')[-1]  # Extract the last part of the key after '/'
            new_dict[new_key] = result  # Store the result in the new dictionary
    return new_dict
    

def compute_average_chamfer_distances(data):
    # Create dictionaries to store the sum and count of the Chamfer distances for each scene-camera and scene.
    sums_per_scene_camera_chamfer = defaultdict(float)
    counts_per_scene_camera_chamfer = defaultdict(int)

    sums_per_scene_chamfer = defaultdict(float)
    counts_per_scene_chamfer = defaultdict(int)

    sums_per_scene_camera_pa_chamfer = defaultdict(float)
    counts_per_scene_camera_pa_chamfer = defaultdict(int)

    sums_per_scene_pa_chamfer = defaultdict(float)
    counts_per_scene_pa_chamfer = defaultdict(int)

    # overall sum and count
    overall_sum_chamfer = 0
    overall_count_chamfer = 0

    overall_sum_pa_chamfer = 0
    overall_count_pa_chamfer = 0

    # Iterate over the data
    for scene_camera, chamfer_types in data.items():
        # Split the scene and camera
        scene, camera = scene_camera.rsplit(".", 1)

        # Compute the sum and count of Chamfer distances for the current scene-camera
        for chamfer_type, chamfer_distances in chamfer_types.items():
            if chamfer_type == 'chamfer':
                for chamfer_distance in chamfer_distances.values():
                    sums_per_scene_camera_chamfer[scene_camera] += chamfer_distance
                    counts_per_scene_camera_chamfer[scene_camera] += 1

                    # Also add to the sum and count for the scene
                    sums_per_scene_chamfer[scene] += chamfer_distance
                    counts_per_scene_chamfer[scene] += 1

                    # Also add to the overall sum and count
                    overall_sum_chamfer += chamfer_distance
                    overall_count_chamfer += 1
            elif chamfer_type == 'pa_chamfer':
                for chamfer_distance in chamfer_distances.values():
                    sums_per_scene_camera_pa_chamfer[scene_camera] += chamfer_distance
                    counts_per_scene_camera_pa_chamfer[scene_camera] += 1

                    # Also add to the sum and count for the scene
                    sums_per_scene_pa_chamfer[scene] += chamfer_distance
                    counts_per_scene_pa_chamfer[scene] += 1

                    # Also add to the overall sum and count
                    overall_sum_pa_chamfer += chamfer_distance
                    overall_count_pa_chamfer += 1

    # Now compute the averages
    # Function to format dictionary values
    def format_dict_values(data_dict):
        return {k: "{:.2f}".format(v) for k, v in data_dict.items()}

    # Now compute the averages
    avg_per_scene_camera_chamfer = {scene_camera: total for scene_camera, total in zip(sums_per_scene_camera_chamfer.keys(), sums_per_scene_camera_chamfer.values())}
    avg_per_scene_chamfer = {scene: total / count for scene, (total, count) in zip(sums_per_scene_chamfer.keys(), zip(sums_per_scene_chamfer.values(), counts_per_scene_chamfer.values()))}
    overall_avg_chamfer = overall_sum_chamfer / overall_count_chamfer

    avg_per_scene_camera_pa_chamfer = {scene_camera: total for scene_camera, total in zip(sums_per_scene_camera_pa_chamfer.keys(), sums_per_scene_camera_pa_chamfer.values())}
    avg_per_scene_pa_chamfer = {scene: total / count for scene, (total, count) in zip(sums_per_scene_pa_chamfer.keys(), zip(sums_per_scene_pa_chamfer.values(), counts_per_scene_pa_chamfer.values()))}
    overall_avg_pa_chamfer = overall_sum_pa_chamfer / overall_count_pa_chamfer

    # Format results
    avg_per_scene_camera_chamfer = format_dict_values(avg_per_scene_camera_chamfer)
    avg_per_scene_chamfer = format_dict_values(avg_per_scene_chamfer)
    avg_per_scene_camera_pa_chamfer = format_dict_values(avg_per_scene_camera_pa_chamfer)
    avg_per_scene_pa_chamfer = format_dict_values(avg_per_scene_pa_chamfer)
    overall_avg_chamfer = "{:.2f}".format(overall_avg_chamfer)
    overall_avg_pa_chamfer = "{:.2f}".format(overall_avg_pa_chamfer)

    print("Average Chamfer distance per scene-camera: ", avg_per_scene_camera_chamfer)
    print("Average Chamfer distance per scene: ", avg_per_scene_chamfer)
    print("Overall average Chamfer distance: ", overall_avg_chamfer)

    print("Average PA Chamfer distance per scene-camera: ", avg_per_scene_camera_pa_chamfer)
    print("Average PA Chamfer distance per scene: ", avg_per_scene_pa_chamfer)
    print("Overall average PA Chamfer distance: ", overall_avg_pa_chamfer)

    return avg_per_scene_camera_chamfer, avg_per_scene_chamfer, overall_avg_chamfer, avg_per_scene_camera_pa_chamfer, avg_per_scene_pa_chamfer, overall_avg_pa_chamfer

def compute_average_v2v_errors(data):
    # Create dictionaries to store the sum and count of the v2v errors for each scene-camera and scene.
    sums_per_scene_camera_v2v = defaultdict(float)
    counts_per_scene_camera_v2v = defaultdict(int)

    sums_per_scene_v2v = defaultdict(float)
    counts_per_scene_v2v = defaultdict(int)

    # overall sum and count
    overall_sum_v2v = 0
    overall_count_v2v = 0

    # Iterate over the data
    for scene_camera, v2v_errors in data.items():
        # Split the scene and camera
        scene, camera = scene_camera.rsplit(".", 1)

        # Compute the sum and count of v2v errors for the current scene-camera
        for v2v_error in v2v_errors['v2v_err'].values():
            sums_per_scene_camera_v2v[scene_camera] += v2v_error
            counts_per_scene_camera_v2v[scene_camera] += 1

            # Also add to the sum and count for the scene
            sums_per_scene_v2v[scene] += v2v_error
            counts_per_scene_v2v[scene] += 1

            # Also add to the overall sum and count
            overall_sum_v2v += v2v_error
            overall_count_v2v += 1

    # Now compute the averages
    avg_per_scene_camera_v2v = {scene_camera: total / count for scene_camera, (total, count) in zip(sums_per_scene_camera_v2v.keys(), zip(sums_per_scene_camera_v2v.values(), counts_per_scene_camera_v2v.values()))}
    avg_per_scene_v2v = {scene: total / count for scene, (total, count) in zip(sums_per_scene_v2v.keys(), zip(sums_per_scene_v2v.values(), counts_per_scene_v2v.values()))}
    overall_avg_v2v = overall_sum_v2v / overall_count_v2v

    print("Average v2v error per scene-camera:")
    for scene_camera, avg in avg_per_scene_camera_v2v.items():
        print(f"  {scene_camera}: {avg:.2f}")

    print("\nAverage v2v error per scene:")
    for scene, avg in avg_per_scene_v2v.items():
        print(f"  {scene}: {avg:.2f}")

    print(f"\nOverall average v2v error: {overall_avg_v2v:.2f}")

    return avg_per_scene_camera_v2v, avg_per_scene_v2v, overall_avg_v2v

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
    - A namespace containing the parsed arguments.
    """
    
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Choose a method to run. Available methods are 'bev', 'romp', '4dh', 'trace', and 'all'.")
    
    # Add the method argument
    parser.add_argument("method", 
                        choices=['bev', '4dh', 'trace', 'all', 'romp'], 
                        help="Choose the method to run. 'all' will run all methods.")

    # Parse the arguments and return
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Parse the arguments
    args = parse_arguments()
    
    # Process based on the method chosen
    if args.method == 'bev':
        # Run the BEV method
        method = ['bev']
    elif args.method == 'romp':
        # Run the ROMP method
        method = ['romp']
    elif args.method == '4dh':
        # Run the 4DH method
        method = ['4dh']
    elif args.method == 'trace':
        # Run the TRACE method
        method = ['trace']
    elif args.method == 'all':
        # Run all the methods
        method = ['bev', 'romp', '4dh', 'trace']

    for method in method:
        #Link gt and prediciton from the method sought
        directories = generate_directories_dict(method)

        #Compute distances
        result = process_dict(directories, calculate_distances, method)

        compute_average_chamfer_distances(result)
        compute_average_v2v_errors(result)
