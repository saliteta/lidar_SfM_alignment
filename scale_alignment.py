import argparse
from utils import *
from plyfile import PlyData


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='align the scale together')

    # Add arguments
    parser.add_argument('--lidar', type=str, help='File paths for lidar point cloud', required=True)
    parser.add_argument('--sparse', type=str, help='File paths for sparse SfM point cloud', required=True)
    parser.add_argument('--result_folder', type=str, help='File paths for sparse SfM point cloud', required=True)

    # Parse the arguments
    args = parser.parse_args()

    # You can now use args.lidar and args.sparse in your program
    print('Lidar files:', args.lidar)
    print('Sparse files:', args.sparse)

    try:
        plydata = PlyData.read(args.sparse)
        plydata = remove_useless_info(plydata)
    except: 
        xyz, rgb, _ = read_points3D_binary(args.sparse)
        plydata = o3d.geometry.PointCloud()
        plydata.points = o3d.utility.Vector3dVector(xyz)
        plydata.colors = o3d.utility.Vector3dVector(rgb/255.0)

    filtered_sparse = filter_point_cloud(plydata)

    lidar_data = load_point_cloud(args.lidar)
    lidar_down = downsample_point_cloud(lidar_data, 1)
    
    lidar_down, scale_factors, mean = scale_point_cloud_by_std_dev(filtered_sparse, lidar_down)
    # mean_lidar, mean sfm
    # when loading, substrut mean_lidar, rescale + mean_sfm

    create_folder_if_not_exists(args.result_folder)
    print(f'Storage file {args.result_folder}')
    np.save(os.path.join(args.result_folder, 'scales.npy'), scale_factors)
    np.save(os.path.join(args.result_folder, 'mean.npy'), mean)
    o3d.io.write_point_cloud(os.path.join(args.result_folder, 'lidar.ply'), lidar_down)
    o3d.io.write_point_cloud(os.path.join(args.result_folder, 'sparse.ply'), filtered_sparse)
        
if __name__ == '__main__':
    main()