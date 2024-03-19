import numpy as np
import open3d as o3d
import os, struct



def load_point_cloud(file_path):
    return o3d.io.read_point_cloud(file_path)

def downsample_point_cloud(pcd, voxel_size):
    return pcd.voxel_down_sample(voxel_size)


def scale_point_cloud_by_std_dev(pcd_target, pcd_to_scale):
    """
    Scale 'pcd_to_scale' to have the same spread (standard deviation from the mean center) as 'pcd_target'.
    
    :param pcd_target: The target Open3D point cloud to match the scale.
    :param pcd_to_scale: The Open3D point cloud that needs to be scaled.
    """
    # Convert Open3D point clouds to numpy arrays
    target_points = np.asarray(pcd_target.points)
    to_scale_points = np.asarray(pcd_to_scale.points)
    
    # Calculate the mean of each point cloud
    mean_target = np.mean(target_points, axis=0)
    mean_to_scale = np.mean(to_scale_points, axis=0)
    
    # Calculate the distances from each point to the mean point
    distances_target = np.linalg.norm(target_points - mean_target, axis=1)
    distances_to_scale = np.linalg.norm(to_scale_points - mean_to_scale, axis=1)
    
    # Calculate the standard deviation of these distances
    std_dev_target = np.std(distances_target)
    std_dev_to_scale = np.std(distances_to_scale)
    
    # Calculate the overall scale factor
    overall_scale_factor = std_dev_target / std_dev_to_scale *6.402/7.36
    
    # Apply the overall scale factor uniformly to the point cloud to scale
    scaled_points = (to_scale_points - mean_to_scale) * overall_scale_factor + mean_target
    
    # Update the point cloud with scaled points
    pcd_to_scale.points = o3d.utility.Vector3dVector(scaled_points)

    return pcd_to_scale, overall_scale_factor, np.array([mean_to_scale, mean_target])

def filter_point_cloud(pcd, nb_neighbors = 30, std_ratio = 1.5):
    # verified by visualization
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                         std_ratio=std_ratio)
    
    pcd_filtered = pcd.select_by_index(ind)


    return pcd_filtered

def remove_useless_info(plydata):
    # visualization varified
    vertex_data = plydata['vertex']

    new_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    new_vertex_data = np.empty(vertex_data.count, dtype=new_dtype)

    # Copy the data from the original array
    for prop in new_dtype:
        new_vertex_data[prop[0]] = vertex_data[prop[0]]

    # Create a new PlyElement from the modified data
    xyz = np.vstack((new_vertex_data['x'], new_vertex_data['y'], new_vertex_data['z'])).T
    rgb = np.vstack((new_vertex_data['red'], new_vertex_data['green'], new_vertex_data['blue'])).T / 255.0  # Normalize to 0-1

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    return pcd 


def pcd_to_ply(filename:str):
    pcd = o3d.io.read_point_cloud(filename)
    return pcd


def adding_normals(pcd):
    num_points = np.asarray(pcd.points).shape[0]
    zeros = np.zeros((num_points, 1))

    # Add these as new attributes to the point cloud
    pcd.normals = o3d.utility.Vector3dVector(np.hstack((zeros, zeros, zeros)))
    return pcd


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' was created.")
    else:
        print(f"Folder '{folder_path}' already exists. No action taken.")


def read_transformation_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = np.array([list(map(float, line.split())) for line in lines])
    return matrix


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors