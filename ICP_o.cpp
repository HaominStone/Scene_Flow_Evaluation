#include "read_depth_file.hpp"
#include <iostream>
#include <vector>
#include <boost/filesystem.hpp>
#include "pcl/common/common_headers.h"
#include "pcl/features/normal_3d.h"
#include "pcl/io/pcd_io.h"
#include "pcl/visualization/pcl_visualizer.h"
#include "pcl/console/parse.h"
#include <pcl/visualization/cloud_viewer.h>
#include <eigen3/Eigen/Eigenvalues>
using namespace std;

Eigen::Matrix3d K;
Eigen::Matrix3d Ki;
int PNGwidth;
int PNGheight;
bool plane;

// Eigen::MatrixXd pseudoInverse(Eigen::MatrixXd& a,double epsilon = std::numeric_limits<typename Eigen::MatrixXd::Scalar>::epsilon())
// {
//     Eigen::MatrixXd result;
//     constexpr auto m = Eigen::JacobiSVD< Eigen::MatrixXd >::DiagSizeAtCompileTime;
//   Eigen::JacobiSVD< Eigen::Matrix<typename Eigen::MatrixXd::Scalar, Eigen::Dynamic, Eigen::Dynamic> > svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);

//   typename Eigen::MatrixXd::Scalar tolerance =
//     epsilon * std::max(a.cols(), a.rows()) * svd.singularValues().array().abs().maxCoeff();

//   // Sigma for ThinU and ThinV
//   Eigen::Matrix<typename Eigen::MatrixXd::Scalar, m, m> sigmaThin = Eigen::Matrix<typename Eigen::MatrixXd::Scalar, m, 1>(
//     (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0)).asDiagonal();

//   result = svd.matrixV() * sigmaThin * svd.matrixU().transpose();
//   return result;
// }
Eigen::MatrixXd pseudoInverse(Eigen::MatrixXd  A)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);//M=USV*
    double  pinvtoler = 1.e-8; //tolerance
    int row = A.rows();
    int col = A.cols();
    int k = min(row,col);
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(col,row);
    Eigen::MatrixXd singularValues_inv = svd.singularValues();
    Eigen::MatrixXd singularValues_inv_mat = Eigen::MatrixXd::Zero(col, row);
    for (long i = 0; i<k; ++i) {
        if (singularValues_inv(i) > pinvtoler)
            singularValues_inv(i) = 1.0 / singularValues_inv(i);
        else singularValues_inv(i) = 0;
    }
    for (long i = 0; i < k; ++i) 
    {
        singularValues_inv_mat(i, i) = singularValues_inv(i);
    }
    X=(svd.matrixV())*(singularValues_inv_mat)*(svd.matrixU().transpose());//X=VS+U*
 
    return X;
 
}


void norm_extract(vector<Eigen::Vector3d> &cloud,vector<Eigen::Vector3d> &norm)
{
    for (int i = 0; i < cloud.size(); ++i)
    {
        if (cloud[i] == Eigen::Vector3d(0,0,0))// for each point in cloud2 not equal to 000:
        {
            continue; // Measurement error point won't be taken into account
        }
        int count = 0;
        Eigen::Vector3d tmp(0,0,0);
        for(int h = i/PNGwidth - 100; h < i/PNGwidth + 100; h++) // parameter can be adjusted
        {
            for(int w = i%PNGwidth - 100; w < i%PNGwidth + 100; w++)
            {
                int j = h * PNGwidth + w;
                if (j < 0 || j>cloud.size()){
                    continue;
                }
                if (cloud[j] == Eigen::Vector3d(0,0,0))
                {
                    continue; // Measurement error point won't be taken into account
                }
                tmp = tmp + cloud[j];
                count++;
            }
        }
        if (count == 0)
        {
            continue;
        }
        tmp = tmp/count;
        int countforsvd = 0;
        
        Eigen::MatrixXd cov(count,3);
        for(int h = i/PNGwidth - 100; h < i/PNGwidth + 100; h++) // parameter can be adjusted
        {
            for(int w = i%PNGwidth - 100; w < i%PNGwidth + 100; w++)
            {
                int j = h * PNGwidth + w;
                if (j < 0 || j>cloud.size()){
                    continue;
                }
                if (cloud[j] == Eigen::Vector3d(0,0,0))
                {
                    continue; // Measurement error point won't be taken into account
                }
                cov(countforsvd,0) = cov(countforsvd,0) + (cloud[j] - tmp)[0];
                cov(countforsvd,1) = cov(countforsvd,1) + (cloud[j] - tmp)[1];
                cov(countforsvd,2) = cov(countforsvd,2) + (cloud[j] - tmp)[2];
                countforsvd++;
            }
        }
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXd U = svd.matrixU();
        Eigen::MatrixXd V = svd.matrixV();
        norm[i] += V.col(V.cols()-1);
    }
}

float dist(Eigen::Vector3d cloud1,Eigen::Vector3d cloud2){
    return (cloud2-cloud1).squaredNorm();//Euclidean distance
}

void image2point(int i, int j, float depth, Eigen::Vector3d & rtn){
    Eigen::Vector3d tmp(i,j,1.0);
    rtn = depth * Ki * tmp;//formula given
}

void depth2cloud(const Eigen::ArrayXXf depth,std::vector<Eigen::Vector3d> &cloud){
    int width = depth.cols();
    int height = depth.rows();
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            if (depth(i,j)<0||rand()%50!=1)
            {
                cloud[j+i*width] = Eigen::Vector3d(0,0,0);//if the point's depth is -1, means measure error
            }
            //if we need random reduction maybe we can reduce it to Eigen::Vector3d(0,0,0)
            else
            {
                image2point(i,j,depth(i,j),cloud[j+i*width]);// TODO: Maybe issue of coordinates!!!!!!!!!!!!!!!!!!!!!!!!!
            }
        }
    }
}

float matching ( std::vector<Eigen::Vector3d> cloud1,
                std::vector<Eigen::Vector3d> cloud2, 
                Eigen::Matrix3d & rotation, 
                Eigen::Vector3d & translation ) {
    // cv::Mat source(cloud2.size(),3, CV_32F);
    // for ( int i = 0; i < cloud2.size(); i++ )
    // {
    //     source.at<float>(i,0) = cloud2[i][0];
    //     source.at<float>(i,1) = cloud2[i][1];
    //     source.at<float>(i,2) = cloud2[i][2];
    // }
    // cv::flann::KDTreeIndexParams indexParams(2);
    // cv::flann::Index kdtree(source,indexParams);
    std::vector<float> init{-1.0,999.0};
    vector<std::vector<float>> matches(cloud1.size());
    for(int i = 0; i < cloud1.size(); i++)
    {
        matches[i] = init;//initialize all into init
    }
    
    for (int i = 0; i < cloud2.size(); ++i)
    {
        if (cloud2[i] == Eigen::Vector3d(0,0,0))// for each point in cloud2 not equal to 000:
        {
            continue; // Measurement error point won't be taken into account
        }
        float distance = 999.0;
        int closest = -1;
        for(int h = i/PNGwidth - 80; h < i/PNGwidth + 80; h++) // parameter can be adjusted
        {
            for(int w = i%PNGwidth - 80; w < i%PNGwidth + 80; w++)
            {
                int j = h * PNGwidth + w;
                if (j < 0 || j>cloud1.size()){
                    continue;
                }
                if (cloud1[j] == Eigen::Vector3d(0,0,0))
                {
                    continue; // Measurement error point won't be taken into account
                }
                float tdistance = dist(cloud2[i] , cloud1[j]);// find min distance
                // cout <<"cloud1 "<< cloud1[j][0] <<" "<< cloud1[j][1] <<" "<< cloud1[j][2] << endl;
                if (tdistance < distance)
                {
                    distance = tdistance;
                    closest = j;
                }
            }
        }
        if (closest < 0)
        {
            continue;
        }
        
        if (distance < matches[closest][1])
        {
            matches[closest][0] = i;
            matches[closest][1] = distance;
            // cout << matches[closest][1] << endl;
        }
    }
    int NumberOfPoints = 0;
    vector<Eigen::Vector3d> norm_v(cloud1.size());
    if (plane) {
        cout << plane << endl;
        for (int n1 = 0; n1 < cloud1.size(); ++n1)
        {
            norm_v[n1] = Eigen::Vector3d::Zero();
        }
        norm_extract(cloud1,norm_v);
        cout << "after extraction" << endl;
        
        for (int i = 0; i < cloud1.size(); ++i)
        {
            if (matches[i][0]>=0)
            {
                NumberOfPoints++;
            }
        }
        Eigen::VectorXd b(NumberOfPoints);
        Eigen::MatrixXd A(NumberOfPoints,6);
        NumberOfPoints = 0;
        for (int i = 0; i < cloud1.size(); ++i)
        {
            if (matches[i][0]>=0)
            {
                Eigen::Vector3d n = norm_v[i];
                Eigen::Vector3d d = cloud1[i];
                Eigen::Vector3d s = cloud2[matches[i][0]];
                b[NumberOfPoints] = n[0]*d[0]+n[1]*d[1]+n[2]*d[2]-n[0]*s[0]-n[1]*s[1]-n[2]*s[2];
                A(NumberOfPoints,0) = n[2]*s[1]-n[1]*s[2];
                A(NumberOfPoints,1) = n[0]*s[2]-n[2]*s[0];
                A(NumberOfPoints,2) = n[1]*s[0]-n[0]*s[1];
                A(NumberOfPoints,3) = n[0];
                A(NumberOfPoints,4) = n[1];
                A(NumberOfPoints,5) = n[2];
                NumberOfPoints++;
            }
        }
        // cout << pseudoInverse(A) << endl;
        Eigen::VectorXd x = pseudoInverse(A)*b;
        // cout << x << endl;
        rotation << 1, -x[2], x[1],
                    x[2], 1, -x[0],
                    -x[1], x[0], 1;
        translation << x[3],x[4],x[5];
    }
    else
    {
        // cout << matches.size() <<endl;
        Eigen::Vector3d p = Eigen::Vector3d::Zero();
        Eigen::Vector3d q = Eigen::Vector3d::Zero();
        for (int i = 0; i < cloud1.size(); ++i)
        {
            if (matches[i][0]>=0)
            {
                p = p + cloud1[i];
                q = q + cloud2[matches[i][0]];
                // cout << q << endl;
                NumberOfPoints++;
            }
        }
        // cout << NumberOfPoints << endl;
        p = p/NumberOfPoints;//p_bar
        q = q/NumberOfPoints;//q_bar

        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
        for (int i = 0; i < cloud1.size(); ++i)
        {
            if (matches[i][0]>=0)
            {
                Eigen::Vector3d p_t = cloud1[i] - p;
                Eigen::Vector3d q_t = cloud2[matches[i][0]] - q;
                cov = cov + q_t*p_t.transpose();
            }
        }
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();
        
        if (U.determinant() * V.determinant() < 0)
        {
            for (int x = 0; x < 3; ++x)
            {
                U(x, 2) *= -1;
            }
        }
        rotation = V* U.transpose();
        translation = p - rotation*q;
    }
    float norms = 0;
    for(int i = 0; i < cloud1.size(); i++)
    {
        if (matches[i][0]>=0)
        {
            norms+=(cloud1[i] - rotation*cloud2[matches[i][0]] -translation).norm();
        }
    }
    cout<<"NumberOfPoints " << NumberOfPoints << endl;
    return norms/NumberOfPoints;
}

void transform(std::vector<Eigen::Vector3d> & cloud, Eigen::Matrix3d rotation,Eigen::Vector3d translation){
    for (int i = 0; i < cloud.size(); ++i)
    {
        if (cloud[i] == Eigen::Vector3d(0,0,0)){continue;}
        cloud[i] = rotation * cloud[i] + translation;
    }
}

int main(int argc, char const *argv[])
{
    if (argc >= 2) {
        plane = true;
        if (argv[1] == "-p") {
            cout << argv[1] << endl;
            
        }
    }
    
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    K << 525,0,319.5,0,525,239.5,0,0,1;
    Ki = K.inverse();
    // cout << K ;
    boost::filesystem::path my_path( "/home/shihm/Desktop/Slam_hw1/rgbd_dataset_freiburg2_pioneer_slam2/rgbd_dataset_freiburg2_xyz/depth" );
    boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end
    int count = 0;
    std::vector<string> vector_of_files;
    for ( boost::filesystem::directory_iterator itr( my_path );
        itr != end_itr;
        ++itr )
    {

        if (count++%20 != 0)// accept every 20 frame
        {
            continue;
        }
        // cout<<count%20<<endl;
        vector_of_files.push_back(itr->path().string());
        // cout<< itr->path().string()<<endl;
        // cout<<count<<endl;
    }
    sort(vector_of_files.begin(), vector_of_files.end());
    vector<Eigen::ArrayXXf> DepthArray(count / 20+1);// stores all the png files' depth info.
    int tmp = 0;
    for (vector<string>::iterator it = vector_of_files.begin(); it != vector_of_files.end(); it++){
        read_depth_file(*it, DepthArray[tmp]);
        tmp++;
    }


    Eigen::ArrayXXf InitialPose = DepthArray[0];
    cout<<InitialPose.rows()<<endl;
    PNGheight = InitialPose.rows();
    cout<<InitialPose.cols()<<endl;
    PNGwidth = InitialPose.cols();
    std::vector<Eigen::Matrix3d> Rs;
    std::vector<Eigen::Vector3d> ts;
    Eigen::Vector3d origin(0.1170,-1.1503,1.4005);

    for (int i = 0; i < tmp - 1; ++i)
    {
        // cout<<"in for"<<endl;
        cout << vector_of_files[i]<<endl;
        cout <<"origin"<< i*180+130<<" "<< origin[0] <<" "<< origin[1] <<" "<< origin[2] << endl;
        Eigen::ArrayXXf depth1 = DepthArray[i];
        Eigen::ArrayXXf depth2 = DepthArray[i+1];
        
        std::vector<Eigen::Vector3d> cloud1(DepthArray[0].size());
        std::vector<Eigen::Vector3d> cloud2(DepthArray[0].size());
        depth2cloud(depth1,cloud1);
        // cout << cloud1.size() << endl;
        depth2cloud(depth2,cloud2);


        boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> pointCloud2(new pcl::PointCloud<pcl::PointXYZ>());
        

        for (int j = 0; j < cloud2.size(); j++) {
                if (cloud2[j]!=Eigen::Vector3d(0,0,0)) {
                    pcl::PointXYZ point;
                    point.x = cloud2[j][0];
                    point.y = cloud2[j][1];
                    point.z = cloud2[j][2];
                    pointCloud2->points.push_back(point);
                }
                
                
            }
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(pointCloud2, 0, 255, 0);
            viewer->removePointCloud("sample cloud");
            viewer->addPointCloud<pcl::PointXYZ> (pointCloud2, single_color, "sample cloud");
            viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
            viewer->addCoordinateSystem (0.1);
            viewer->initCameraParameters ();



        Rs.push_back(Eigen::Matrix3d::Identity());
        ts.push_back(Eigen::Vector3d::Zero());
        float norm = 10000;
        for (int iter =0;iter < 20 ; iter++){
            // cout << "in deeper for"<<endl;
            boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> pointCloud1(new pcl::PointCloud<pcl::PointXYZ>());
            for (int b = 0; b < cloud1.size(); b++) {
                if (cloud1[b]!=Eigen::Vector3d(0,0,0)) {
                    pcl::PointXYZ point;
                    point.x = cloud1[b][0];
                    point.y = cloud1[b][1];
                    point.z = cloud1[b][2];
                    pointCloud1->points.push_back(point);
                }
            }
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color2(pointCloud1, 0, 0, 255);
            viewer->removePointCloud("sample cloud2");
            viewer->addPointCloud<pcl::PointXYZ> (pointCloud1, single_color2, "sample cloud2");
            viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud2");
            Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
            Eigen::Vector3d translation = Eigen::Vector3d::Zero();
            norm = matching(cloud2,cloud1,rotation,translation);
            cout << norm << endl;
            Rs[i] = rotation * Rs[i] ;
            ts[i] = rotation * ts[i] + translation;
            transform(cloud1,rotation,translation);
            viewer->spinOnce (100);
        }
        
        Eigen::Vector3d tmp = Rs[i]*origin + ts[i];
        origin = tmp;
        
    }




    return 0;
}