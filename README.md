# Dataset And Transformation Matrix
Notice that this code repo is only used for coarse alignment, and the dataset SHOULD BE AQUIRED THROUGH EMAIL (check our project page for details). If you just want to find the guidance to use our dataset, you can find the guidance here: [Guidance](https://lacy-backbone-098.notion.site/GauUScene-Instruction-2d6ce2bf41634cadb422c4e0fb640122?pvs=4)




## Lidar and Camera Positon Alignment
This repository accompany with CouldCompare can align lidar data that we collected together with the Sparse point cloud generated by SfM

### Instruction
- need to have open3d, plyfile installed first
```
python scale_alignment.py --lidar lidar_path.ply --sparse (SfM_path.ply or bin) --result_folder folder_name
```

Then one can start to align them together using Cloud Compare, after that, one needs to copy transformation matrix back to result folder named as transform.txt
And one can simply run the following command:
```
python lidar_transform.py --lidar lidar_path.ply --transform result_folder_path
```
