#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <string>
#include <pcl/conversions.h>


extern "C"{
    void pcl_downSample(double *array_points, double *array_color, size_t n_points, float leaf_size, double *downsample_points, double *downsample_colors){
        /* The function `pcl_downSample` is downsampling a point cloud represented by the input arrays `array_points` and `array_color`.
            It takes in the following parameters:
        - `array_points`: An array containing the x, y, z coordinates of the points in the point cloud.
        - `array_color`: An array containing the RGB color values of the points in the point cloud.
        - `n_points`: The number of points in the point cloud.
        - `downsample_points`: An array to store the downsampled x, y, z coordinates of the points.
        - `downsample_colors`: An array to store the downsampled RGB color values of the points. */

        pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2());
        pcl::PCLPointCloud2::Ptr cloud_filtered (new pcl::PCLPointCloud2());
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_XYZRGB(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        int cols = 3;  // Dimensions (x, y, z) and (r, g, b)


        // Fill the PointCloud with the points and colors from the input arrays
        for (int i = 0; i < n_points; i++) {
            pcl::PointXYZRGB point;

            point.x = array_points[i * cols + 0];   // x coordinate
            point.y = array_points[i * cols + 1];   // y coordinate
            point.z = array_points[i * cols + 2];   // z coordinate
            point.r = array_color[i * cols + 0];    // r color
            point.g = array_color[i * cols + 1];    // g color
            point.b = array_color[i * cols + 2];    // b color

            cloud_XYZRGB->points.push_back(point);
        }

        // Convert to PCLPointCloud2 to apply the VoxelGrid filter
        pcl::toPCLPointCloud2(*cloud_XYZRGB, *cloud);

        std::cout << "\nPointCloud before filtering: " << cloud->width * cloud->height << " data points (" << pcl::getFieldsList (*cloud) << ")." << std::endl;

        std::cout << "Downsampling with leaf size of " << leaf_size << "... \n";

        // Downsample the point cloud
        pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
        sor.setInputCloud (cloud);
        sor.setLeafSize (leaf_size, leaf_size, leaf_size);
        sor.filter (*cloud_filtered);

        std::cout << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height << " data points (" << pcl::getFieldsList (*cloud_filtered) << ")." << std::endl;

        // Convert to PointCloud<pcl::PointXYZRGB> to get the points and colors of the downsampled point cloud
        pcl::fromPCLPointCloud2(*cloud_filtered, *downsampled_cloud);

        int n_points_downsampled = downsampled_cloud->width * downsampled_cloud->height;
        // Fill the downsampled points and colors into the output arrays
        for (int i = 0; i < n_points_downsampled; i++) {
            const pcl::PointXYZRGB point = downsampled_cloud->points[i];
            downsample_points[i * cols + 0] = point.x;
            downsample_points[i * cols + 1] = point.y;
            downsample_points[i * cols + 2] = point.z;
            downsample_colors[i * cols + 0] = point.r;
            downsample_colors[i * cols + 1] = point.g;
            downsample_colors[i * cols + 2] = point.b;
        }

    }
}
