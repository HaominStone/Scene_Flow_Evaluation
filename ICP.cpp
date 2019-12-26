#include "read_depth_file.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Eigenvalues>
#include <boost/algorithm/string.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Eigen::Matrix3d K;
Eigen::Matrix3d Ki;
const int PNGwidth = 640;
const int PNGheight = 480;
bool plane;

// Vectors below stores all the corresponding rgb and dept file names
vector<string> vector_of_rgb_files; 
vector<string> vector_of_depth_files;

void image2point(int i, int j, float depth, Eigen::Vector3d & rtn){
    // Transform a point from the image plane to the world frame.
    Eigen::Vector3d tmp(i,j,1.0);
    rtn = depth * Ki * tmp;//formula given
}

void transform(std::vector<Eigen::Vector3d> & cloud, Eigen::Matrix3d rotation,Eigen::Vector3d translation){
    for (int i = 0; i < cloud.size(); ++i)
    {
        if (cloud[i] == Eigen::Vector3d(0,0,0)){continue;}
        cloud[i] = rotation * cloud[i] + translation;
    }
}


void recover_by_q_and_pose(Eigen::Vector3d& t, Eigen::Quaterniond q, Eigen::Vector3d& pose, Eigen::Vector3d& output)
{
    q.normalize();
    Eigen::Vector3d dst;
    dst = t - pose;
    Eigen::Quaterniond p;
    p.w() = 0;
    p.vec() = dst;
    Eigen::Quaterniond rotatedP = q.conjugate() * p * q;
    output = rotatedP.vec();
}

void rotate_by_q_and_pose(Eigen::Vector3d& dst, Eigen::Quaterniond q, Eigen::Vector3d& pose, Eigen::Vector3d& output)
{
    q.normalize();
    // Eigen::Vector3d dst;
    Eigen::Quaterniond p;
    p.w() = 0;
    p.vec() = dst;
    Eigen::Quaterniond rotatedP = q * p * q.conjugate();
    output = rotatedP.vec();
    output += pose;
}

void find_corr(Eigen::ArrayXXf& depth, Eigen::Quaterniond& r1, Eigen::Quaterniond& r2,Eigen::Vector3d& pose1,
    Eigen::Vector3d& pose2, Eigen::Matrix<Eigen::Vector2d,Eigen::Dynamic,Eigen::Dynamic>& output)
{
    int width = depth.cols();
    int height = depth.rows();
    Eigen::Vector3d dst;
    output.resize(height,width);
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            // Iterate each point
            if (depth(i,j) == -1)
            {
                // cout << "1";
            }
            image2point(i,j,depth(i,j),dst);// TODO: Maybe issue of coordinates!!!!!!!!!!!!!!!!!!!!!!!!!

            Eigen::Vector3d recoveredV;
            Eigen::Vector3d rotatedV;
            recover_by_q_and_pose(dst,r1,pose1,recoveredV);
            rotate_by_q_and_pose(recoveredV,r2,pose2,rotatedV);
            Eigen::Vector3d new_point_in_image = K * rotatedV;
            // cout << "here?"<<endl;
            output(i,j)(0) = new_point_in_image[0] / new_point_in_image[2];
            output(i,j)(1) = new_point_in_image[1] / new_point_in_image[2];
            // cout << i<<" " << j <<" : "<<  output(i,j).transpose() <<endl;

        }
    }
    cout << "done one pair" << endl;
}

void output_to_file(string depth_0,string depth_1, string rgb_0, string rgb_1, Eigen::Matrix<Eigen::Vector2d,Eigen::Dynamic,Eigen::Dynamic>& corr)
{
    ofstream fout(".."+depth_0+".txt");
    fout << depth_0 << " " << depth_1 <<" " << rgb_0 << " " << rgb_1 << endl;
    for (int i = 0; i < PNGheight; ++i)
    {
        for (int j = 0; j < PNGwidth; ++j)
        {
            // Iterate each point
            fout << corr(i,j).transpose() <<endl;

        }
    }

}

void visualize_correspondence(Mat img_0,Mat img_1, Eigen::Matrix<Eigen::Vector2d,Eigen::Dynamic,Eigen::Dynamic>& corr)
{
    // int count = 0;
    Mat dst;
    int rows = img_0.rows;
    int cols = img_0.cols+img_1.cols;
    CV_Assert(img_0.type () == img_1.type ());
    dst.create (rows,cols,img_0.type ());
    img_0.copyTo (dst(Rect(0,0,img_0.cols,img_0.rows)));
    img_1.copyTo (dst(Rect(img_0.cols,0,img_1.cols,img_1.rows)));

    for (int i = 0; i < PNGheight; ++i)
    {
        for (int j = 0; j < PNGwidth; ++j)
        {
            // Eigen::Vector3d tmp(K*in_o[i]);
            if (rand()%100000 != 0)
            {
                continue;
            }
            circle(dst,Point(j,i),1,Scalar(0),2,8,0);
            line(dst,Point(j,i),Point(img_0.cols+corr(i,j)(1),corr(i,j)(0)),Scalar(0,255,255),1);
        }
    }   
    // circle(img_1,Point(639,479),1,Scalar(0),2,8,0);
    // line(img_1,Point(639,479),Point(corr(479,639)(1),corr(479,639)(0)),Scalar(0,255,255),1);     
    imshow("in",dst);
    waitKey();
}

int main(int argc, char const *argv[])
{
    std::ifstream infile("../rgbd_dataset_freiburg2_desk/matches.txt");
    // pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    K << 525,0,319.5,0,525,239.5,0,0,1;
    Ki = K.inverse();
    // cout << K ;
    int count = 0;

    std::string line;
    std::string line2;

    std::ifstream infile2("../rgbd_dataset_freiburg2_desk/tra_3.txt");
    string depthfile_0;
    string rgbfile_0;
    Eigen::Vector3d pose0;
    Eigen::Quaterniond q0;
    // getline(infile2,line);
    // Eigen::ArrayXXf depth_read;
    // string file_name = "../rgbd_dataset_freiburg2_desk/depth/1311868164.477039.png";
    // cout<< file_name << endl;
    // read_depth_file(file_name, depth_read);
    while (getline(infile2,line))
    {
        if (count++ % 5 != 0)
        {
            continue;
        }
        vector<string> result;
        boost::split(result,line, boost::is_any_of(" "));
        string depthfile_1 = result[1];
        string rgbfile_1 = result[3];
        Eigen::Vector3d pose1;
        pose1 << stod(result[5]) ,stod(result[6]) ,stod(result[7]);
        Eigen::Quaterniond q1(stod(result[8]), stod(result[9]), stod(result[10]), stod(result[11]));
        if (count != 1)
        {
            // cout << "inwhile2?" << endl;
            Eigen::ArrayXXf depth_read;
            string file_name = "../rgbd_dataset_freiburg2_desk/"+depthfile_0;
            cout<< file_name << endl;
            read_depth_file(file_name, depth_read);
            Eigen::Matrix<Eigen::Vector2d,Eigen::Dynamic,Eigen::Dynamic> output;
            find_corr(depth_read,q0,q1,pose0,pose1,output);
            // while (std::getline(infile, line2))
            // {

            //     std::istringstream iss(line2);
            //     string a, b;
            //     if (!(iss >> a >> b)) { break; cout << "Wrong!" << endl;} // error
            //     cout << a << endl;
            //     cout << depthfile_0 << endl;
            //     if (a == depthfile_0)
            //     {
            output_to_file(depthfile_0,depthfile_1,rgbfile_0,rgbfile_1,output);
            Mat img_0 = imread("../rgbd_dataset_freiburg2_desk/"+rgbfile_0);
            Mat img_1 = imread("../rgbd_dataset_freiburg2_desk/"+rgbfile_1);
            visualize_correspondence(img_0, img_1,output);
            //         break;
            //     }
            //     // process pair (a,b)
            //     // cout << "inwhile1?" << endl;
            // }
            // print(output);
        }
        depthfile_0 = depthfile_1;
        rgbfile_0 = rgbfile_1;
        pose0 = pose1;
        q0 = q1;
        
    }

    // vector<Eigen::ArrayXXf> DepthArray(count / 20+1);// stores all the png files' depth info.
    // int tmp = 0;
    // for (vector<string>::iterator it = vector_of_files.begin(); it != vector_of_files.end(); it++){
    //     read_depth_file(*it, DepthArray[tmp]);
    //     tmp++;
    // }


    // Eigen::ArrayXXf InitialPose = DepthArray[0];
    // cout<<InitialPose.rows()<<endl;
    // PNGheight = InitialPose.rows();
    // cout<<InitialPose.cols()<<endl;
    // PNGwidth = InitialPose.cols();
    // std::vector<Eigen::Matrix3d> Rs;
    // std::vector<Eigen::Vector3d> ts;
    // Eigen::Vector3d origin(0.1170,-1.1503,1.4005);

    // for (int i = 0; i < tmp - 1; ++i)
    // {
    //     // cout<<"in for"<<endl;
    //     cout << vector_of_files[i]<<endl;
    //     cout <<"origin"<< i*180+130<<" "<< origin[0] <<" "<< origin[1] <<" "<< origin[2] << endl;
    //     Eigen::ArrayXXf depth1 = DepthArray[i];
    //     Eigen::ArrayXXf depth2 = DepthArray[i+1];
        
    //     std::vector<Eigen::Vector3d> cloud1(DepthArray[0].size());
    //     std::vector<Eigen::Vector3d> cloud2(DepthArray[0].size());
    //     depth2cloud(depth1,cloud1);
    //     // cout << cloud1.size() << endl;
    //     depth2cloud(depth2,cloud2);


    //     boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> pointCloud2(new pcl::PointCloud<pcl::PointXYZ>());
        

    //     for (int j = 0; j < cloud2.size(); j++) {
    //             if (cloud2[j]!=Eigen::Vector3d(0,0,0)) {
    //                 pcl::PointXYZ point;
    //                 point.x = cloud2[j][0];
    //                 point.y = cloud2[j][1];
    //                 point.z = cloud2[j][2];
    //                 pointCloud2->points.push_back(point);
    //             }
                
                
    //         }
    //         pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(pointCloud2, 0, 255, 0);
    //         viewer->removePointCloud("sample cloud");
    //         viewer->addPointCloud<pcl::PointXYZ> (pointCloud2, single_color, "sample cloud");
    //         viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    //         viewer->addCoordinateSystem (0.1);
    //         viewer->initCameraParameters ();



    //     Rs.push_back(Eigen::Matrix3d::Identity());
    //     ts.push_back(Eigen::Vector3d::Zero());
    //     float norm = 10000;
    //     for (int iter =0;iter < 20 ; iter++){
    //         // cout << "in deeper for"<<endl;
    //         boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> pointCloud1(new pcl::PointCloud<pcl::PointXYZ>());
    //         for (int b = 0; b < cloud1.size(); b++) {
    //             if (cloud1[b]!=Eigen::Vector3d(0,0,0)) {
    //                 pcl::PointXYZ point;
    //                 point.x = cloud1[b][0];
    //                 point.y = cloud1[b][1];
    //                 point.z = cloud1[b][2];
    //                 pointCloud1->points.push_back(point);
    //             }
    //         }
    //         pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color2(pointCloud1, 0, 0, 255);
    //         viewer->removePointCloud("sample cloud2");
    //         viewer->addPointCloud<pcl::PointXYZ> (pointCloud1, single_color2, "sample cloud2");
    //         viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud2");
    //         Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    //         Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    //         norm = matching(cloud2,cloud1,rotation,translation);
    //         cout << norm << endl;
    //         Rs[i] = rotation * Rs[i] ;
    //         ts[i] = rotation * ts[i] + translation;
    //         transform(cloud1,rotation,translation);
    //         viewer->spinOnce (100);
    //     }
        
    //     Eigen::Vector3d tmp = Rs[i]*origin + ts[i];
    //     origin = tmp;
        
    // }




    return 0;
}