from utils import *
import open3d as o3d
import argparse


def main():
    # visualized verified
    parser = argparse.ArgumentParser(description='loading_lidar')

    parser.add_argument('--lidar', type=str, help='File path for lidar point cloud', required=True)
    parser.add_argument('--transform', type=str, help='Folder path for transformation txt and scale npy', required=True)

    args = parser.parse_args()
    scales = np.load(os.path.join(args.transform, 'scales.npy'))
    means = np.load(os.path.join(args.transform, 'mean.npy'))
    transformation = read_transformation_matrix(os.path.join(args.transform, 'transform.txt'))
    print('scale loading done!')

    lidar_data = load_point_cloud(args.lidar)
    to_scale_points = np.asarray(lidar_data.points)
    scaled_points = (to_scale_points - means[0]) * scales + means[1]
    lidar_data.points = o3d.utility.Vector3dVector(scaled_points)
    lidar_data.transform(transformation)
    print("transformation done!")
    o3d.io.write_point_cloud(os.path.join(args.transform, 'lidar_full.ply'), lidar_data)
    print(f"saving accomplsihed, file path{os.path.join(args.transform, 'lidar_full.ply')}")

if __name__ == '__main__':
    main()



