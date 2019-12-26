#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <boost/filesystem.hpp>
#include <algorithm>
#include <eigen3/Eigen/Eigenvalues>
// #include "opencv2/core/core.hpp"
// #include"opencv2/imgproc/imgproc.hpp"
// #include"opencv2/core/core.hpp"
// #include"opencv2/highgui/highgui.hpp"
// #include <opencv2/nonfree/features2d.hpp>
// #include <opencv2/nonfree/nonfree.hpp>
// #include "opencv2/features2d/features2d.hpp"
#include <opencv2/opencv.hpp>  //头文件
#include <opencv2/xfeatures2d.hpp>
using namespace cv;  //包含cv命名空间
using namespace std;
Eigen::Matrix3d K;
Eigen::Matrix3d Ki;
int PNGwidth;
int PNGheight;

bool triangulation( Eigen::Matrix3d R1, 
                    Eigen::Vector3d t1, 
                    Eigen::Matrix3d R2, 
                    Eigen::Vector3d t2, 
                    Eigen::Vector3d &point1, 
                    Eigen::Vector3d &point2, 
                    Eigen::Vector4d &rtn)
{
    Eigen::MatrixXd P(3,4);
    P <<  R1(0,0), R1(0,1), R1(0,2), t1[0],
          R1(1,0), R1(1,1), R1(1,2), t1[1],
          R1(2,0), R1(2,1), R1(2,2), t1[2];
    Eigen::MatrixXd P2(3,4);
    P2 << R2(0,0), R2(0,1), R2(0,2), t2[0],
          R2(1,0), R2(1,1), R2(1,2), t2[1],
          R2(2,0), R2(2,1), R2(2,2), t2[2];
    Eigen::MatrixXd P_T(P.transpose());
    Eigen::MatrixXd P2_T(P2.transpose());
    Eigen::Vector4d p1 = P_T.col(0).transpose();
    Eigen::Vector4d p2 = P_T.col(1).transpose();
    Eigen::Vector4d p3 = P_T.col(2).transpose();

    Eigen::Vector4d p1_ = P2_T.col(0).transpose();
    Eigen::Vector4d p2_ = P2_T.col(1).transpose();
    Eigen::Vector4d p3_ = P2_T.col(2).transpose();

    Eigen::Matrix4d A;
    Eigen::Vector4d r1(point1[0]*p3 - point1[2]*p1);
    Eigen::Vector4d r2(point1[1]*p3 - point1[2]*p2);
    Eigen::Vector4d r3(point2[0]*p3_ - point2[2]*p1_);
    Eigen::Vector4d r4(point2[1]*p3_ - point2[2]*p2_);
    A << r1[0], r1[1], r1[2], r1[3],
         r2[0], r2[1], r2[2], r2[3],
         r3[0], r3[1], r3[2], r3[3],
         r4[0], r4[1], r4[2], r4[3];
    Eigen::JacobiSVD<Eigen::Matrix4d> svd_A(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix4d U_A = svd_A.matrixU();
    Eigen::Matrix4d V_A = svd_A.matrixV();
    rtn = V_A.col(V_A.cols()-1);
    // cout << rtn.transpose() << endl;
    double c1 = rtn[2]*rtn[3];
    double c2 = (P2*rtn)[2] * rtn[3];
    if (c1 > 0 && c2 > 0)
    {
        return true;
    }
    return false;
    // Eigen::Vector3d p1(R1());
}

float reprojection_err(Eigen::MatrixXd P, Eigen::Vector4d Space_point, Eigen::Vector3d camera_point)
{
    // cout << "-------------space point ------------------------ ";
    // cout << P * Space_point << endl;
    // cout << camera_point << endl;
    // cout << "-------------space point ------------------------ "
    return fabs((P * Space_point - camera_point)[0])+fabs((P * Space_point - camera_point)[1]);
}

void F2Rt(Eigen::Matrix3d F, Eigen::Matrix3d& R, Eigen::Vector3d &t,vector<Eigen::Vector3d>& x, vector<Eigen::Vector3d>& x_p)
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_F(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U_f = svd_F.matrixU();
    Eigen::Matrix3d V_f = svd_F.matrixV();
    Eigen::Matrix3d Z;
    Z << 0, 1, 0,
        -1, 0, 0,
         0, 0, 0;
    Eigen::Matrix3d W;
    W << 0, -1, 0,
         1, 0, 0,
         0, 0, 1;
    Eigen::Matrix3d R1 = U_f*W*V_f.transpose();
    if (R1.determinant() < -0.9 && R1.determinant() > -1.1)
    {

        // cout <<"____________________!!!!!!!!!!!!!!!!!!!neg!!!!!!!!!!!!!!!!!!!!!!!!________________________________"<<endl;
        V_f(0,2) = -V_f(0,2);
        V_f(1,2) = -V_f(1,2);
        V_f(2,2) = -V_f(2,2);
        R1 = U_f*W*V_f.transpose();
    }
    Eigen::Matrix3d R2 = U_f*W.transpose()*V_f.transpose();
    if (R1.determinant() < -0.9 && R1.determinant() > -1.1)
    {

        cout <<"____________________!!!!!!!!!!!!!!!!!!!neg!!!!!!!!!!!!!!!!!!!!!!!!________________________________"<<endl;
        R1 = -R1;
    }
    else if (R1.determinant() > 0.9 && R1.determinant() < 1.1)
    {
        
    }
    else
    {
        cout <<"____________________!!!!!!!!!!!!!!!!!!!WRONG!!!!!!!!!!!!!!!!!!!!!!!!________________________________"<<endl;
    }
    if (R2.determinant() < -0.9 && R2.determinant() > -1.1)
    {
        cout <<"____________________!!!!!!!!!!!!!!!!!!!neg!!!!!!!!!!!!!!!!!!!!!!!!________________________________"<<endl;
        R2 = -R2;
    }
    else if (R2.determinant() > 0.9 && R2.determinant() < 1.1)
    {
        
    }
    else
    {
        cout <<"____________________!!!!!!!!!!!!!!!!!!!WRONG!!!!!!!!!!!!!!!!!!!!!!!!________________________________"<<endl;
    }
    Eigen::Matrix3d t_x = U_f * Z * U_f.transpose();
    // t << t_x(2,1), t_x(0,2), t_x(1,0);
    t << U_f.col(U_f.cols()-1)[0],U_f.col(U_f.cols()-1)[1],U_f.col(U_f.cols()-1)[2];
    // cout << t << endl;
    // cout << R2 << endl;
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Vector4d space;
    std::vector<int> choose(4);
    for (int i = 0; i < 50; ++i)
    {
        if (triangulation(I ,Eigen::Vector3d::Zero(),R2,t,x[0],x_p[0],space))
        {
            // cout << "111111111111111111" << endl;
            choose[0]++;
        }
        else if (triangulation(I,Eigen::Vector3d::Zero(),R1,t,x[0],x_p[0],space))
        {
            // cout << "222222222222222222222" << endl;
            choose[1]++;
        }
        else if (triangulation(I,Eigen::Vector3d::Zero(),R2,-t,x[0],x_p[0],space))
        {
            // cout << "33333333333333333333" << endl;
            choose[2]++;
        }
        else if (triangulation(I,Eigen::Vector3d::Zero(),R1,-t,x[0],x_p[0],space))
        {
            // cout << "44444444444444444444" << endl;
            choose[3]++;
        }
    }
    int maxPosition = max_element(choose.begin(),choose.end()) - choose.begin();
    // cout << maxPosition << endl;
    switch (maxPosition)
    {
        case 0:
            R = R2;
            break;
        case 1:
            R = R1;
            break;
        case 2:
            R = R2;
            t = -t;
            break;
        case 3:
            R = R1;
            t = -t;
            break;
    }
}

int main(int argc, char const *argv[])
{
    ofstream f1("trajectory.txt"); 
    Eigen::Vector3d InitialPosition(0.1163,-1.1498,1.4015);
    K << 525,0,319.5,0,525,239.5,0,0,1;
    Ki = K.inverse();
    // cout << K ;
    boost::filesystem::path my_path( "/home/shihm/Desktop/slam_hw2/rgbd_dataset_freiburg2_pioneer_slam2/rgbd_dataset_freiburg2_xyz/rgb" );
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
    // vector<Mat> DepthArray(count / 20+1);// stores all the png files' depth info.
    // int tmp = 0;
    // for (vector<string>::iterator it = vector_of_files.begin(); it != vector_of_files.end(); it++){
    //     DepthArray[tmp] = imread(*it, CV_LOAD_IMAGE_GRAYSCALE);
    //     tmp++;
    // }
    // Mat InitialPose = DepthArray[0];
    // cout<<InitialPose.rows()<<endl;
    // PNGheight = InitialPose.rows();
    // cout<<InitialPose.cols()<<endl;
    // PNGwidth = InitialPose.cols();
    std::vector<Eigen::Matrix3d> Rs;
    std::vector<Eigen::Vector3d> ts;
    Ptr<Feature2D> f2d = xfeatures2d::SIFT::create(0,3,0.04,10,1.6);
    std::vector<Eigen::Vector4d> in_space_out;
    std::vector<Eigen::Vector3d> in_prev;
    int d = 1;
    // unordered_map<Eigen::Vector3d, Eigen::Vector4d> dict;
    for (int files = 0; files < 30;//vector_of_files.size() - 1;
     ++files)
    {
        // unordered_map<Eigen::Vector3d, Eigen::Vector4d> dict_each;

        //Create SIFT class pointer
        //读入图片
        Mat img_1 = imread(vector_of_files[files]);
        Mat img_2 = imread(vector_of_files[files+1]);
        // imshow("img1",img_1);
        // cout<<"???????????"<<endl;
        // imshow("img2",img_2);



        //Detect the keypoints




        vector<KeyPoint> keypoints_1, keypoints_2;
        f2d->detect(img_1, keypoints_1);
        f2d->detect(img_2, keypoints_2);
        //Calculate descriptors (feature vectors)
        Mat descriptors_1, descriptors_2;
        f2d->compute(img_1, keypoints_1, descriptors_1);
        f2d->compute(img_2, keypoints_2, descriptors_2);    
        //Matching descriptor vector using BFMatcher
        FlannBasedMatcher matcher;
        vector<DMatch> matches_o;
        matcher.match(descriptors_1, descriptors_2, matches_o);
        // cout<<matches<<endl;
        //绘制匹配出的关键点
        // Mat img_matches;
          double max_dist = 0; double min_dist = 100;
  //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < descriptors_1.rows; i++ )
          { double dist = matches_o[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
          }
          // printf("-- Max dist : %f \n", max_dist );
          // printf("-- Min dist : %f \n", min_dist );
          //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
          //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
          //-- small)
          //-- PS.- radiusMatch can also be used here.
          std::vector< DMatch > matches;
          for( int i = 0; i < descriptors_1.rows; i++ )
          { 
            double diffx = abs(keypoints_1[matches_o[i].queryIdx].pt.x - keypoints_2[matches_o[i].trainIdx].pt.x);
            double diffy = abs(keypoints_1[matches_o[i].queryIdx].pt.y - keypoints_2[matches_o[i].trainIdx].pt.y);
            if(diffx + diffy < 10000000)
            { matches.push_back( matches_o[i]); }
          }
          //-- Draw only "good" matches
          // cout << matches.size() << endl;


//**********************************plotting!*****************************************
          Mat img_matches;
          drawMatches( img_1, keypoints_1, img_2, keypoints_2,
                       matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                       vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
          //-- Show detected matches
          imshow( "Good Matches", img_matches );
          waitKey();
//**********************************end plotting!*****************************************



          /*store the feature points*/
        std::vector<Eigen::Vector3d>obj;
        std::vector<Eigen::Vector3d>scene;
        int match_num = matches.size();
        for(size_t i=0;i<matches.size();i++)
        {
            Eigen::Vector3d tmp(0,0,0);
            tmp[0] = keypoints_1[matches[i].queryIdx].pt.x;
            tmp[1] = keypoints_1[matches[i].queryIdx].pt.y;
            tmp[2] = 1;
            tmp = Ki * tmp;
            // tmp /= tmp[2];
            Eigen::Vector3d tmp2(0,0,0);
            tmp2[0] = keypoints_2[matches[i].trainIdx].pt.x;
            tmp2[1] = keypoints_2[matches[i].trainIdx].pt.y;
            tmp2[2] = 1;
            tmp2 = Ki * tmp2;
            // tmp2 /= tmp2[2];
            obj.push_back(tmp);
            // cout << tmp << "-----";
            scene.push_back(tmp2);
            // cout << tmp2 << endl;
        }
        // drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);
        // imshow("match", img_matches);
        // waitKey();
        
        
        //Computing Fundamental matrix
        double max = 0;
        std::vector<Eigen::Vector3d> in_o;
        std::vector<Eigen::Vector3d> in_s;
        std::vector<Eigen::Vector4d> in_space;
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        for (int ite = 0; ite < 2000; ++ite)
        {
            // unordered_map<Eigen::Vector3d, Eigen::Vector4d> dict_tmp;
            vector<Eigen::Vector3d> x;
            vector<Eigen::Vector3d> x_p;
            Eigen::MatrixXd F(7,9);
            for (int i = 0; i < 7; ++i)
            {
                int ind = rand() % matches.size();
                x.push_back(obj[ind]);
                x_p.push_back(scene[ind]);
                F(i,0) = obj[ind][0] * scene[ind][0];
                F(i,1) = scene[ind][0] * obj[ind][1];
                F(i,2) = scene[ind][0];
                F(i,3) = scene[ind][1] * obj[ind][0];
                F(i,4) = scene[ind][1] * obj[ind][1];
                F(i,5) = scene[ind][1];
                F(i,6) = obj[ind][0];
                F(i,7) = obj[ind][1];
                F(i,8) = 1;
            }
            // cout << F << endl;
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::MatrixXd U = svd.matrixU();
            Eigen::MatrixXd V = svd.matrixV();
            Eigen::VectorXd V8(9);
            Eigen::VectorXd V9(9);
            V8 = V.col(V.cols()-2);
            V9 = V.col(V.cols()-1);
            // cout<< V8 << endl;
            Eigen::Matrix3d F1;
            F1 << V8[0], V8[1], V8[2],
                  V8[3], V8[4], V8[5],
                  V8[6], V8[7], V8[8];
            Eigen::Matrix3d F2;
            F2 << V9[0], V9[1], V9[2],
                  V9[3], V9[4], V9[5],
                  V9[6], V9[7], V9[8];
            Eigen::Matrix3d A = F2.inverse() * F1;
            // cout << A << endl;
            Eigen::EigenSolver<Eigen::Matrix3d> es(A);
            double lambda = 0;
            if(!es.eigenvalues()[0].imag())
            {lambda = -es.eigenvalues()[0].real();}
            else if (!es.eigenvalues()[1].imag())
            {
                lambda = -es.eigenvalues()[1].real();
            }
            else if (!es.eigenvalues()[2].imag())
            {
                lambda = -es.eigenvalues()[2].real();
            }
            else
            {
                cout << "WRONG!!!!!!!!!!!!!!!!!!" << endl;
                continue;
            }
            Eigen::Matrix3d FundamentalMatrix = F1+lambda*F2;
            // cout << FundamentalMatrix << endl;
            F2Rt(FundamentalMatrix,R,t,x,x_p);
            
            //JUDGING INLIERS
            Eigen::MatrixXd P(3,4);
            P <<  R(0,0), R(0,1), R(0,2), t[0],
                  R(1,0), R(1,1), R(1,2), t[1],
                  R(2,0), R(2,1), R(2,2), t[2];
            int count = 0;
            std::vector<Eigen::Vector3d> inlier_o;
            std::vector<Eigen::Vector3d> inlier_s;
            std::vector<Eigen::Vector4d> inlier_space;
            for (int corr = 0; corr < match_num; ++corr)
            {
                Eigen::Vector3d p_tmp_1 = obj[corr];
                Eigen::Vector3d p_tmp_2 = scene[corr];
                Eigen::Vector4d sp_tmp;
                triangulation(Eigen::Matrix3d::Identity(),Eigen::Vector3d::Zero(),R,t,p_tmp_1,p_tmp_2,sp_tmp);
                float error = reprojection_err(P,sp_tmp,p_tmp_2);
                // cout << corr << " : " << error << endl;
                if (fabs(error) < 0.01)
                {
                    // dict_tmp.insert(make_pair(p_tmp_2,sp_tmp)); // a map for scale
                    count ++;
                    // cout << sp_tmp.transpose()<<endl;
                    inlier_space.push_back(sp_tmp);
                    inlier_o.push_back(p_tmp_1);
                    inlier_s.push_back(p_tmp_2);
                }
            }

            if (((float)count)/match_num > max)
            {
                max = ((float)count)/match_num;
                in_o = inlier_o;
                in_s = inlier_s;
                in_space = inlier_space;
            }
            // if (count > 0.8 * match_num)
            // {
            //     cout << "!!!!!!!!!!!!!!!!" << endl;
            //     cout << count << endl;
            //     break;
            // }
        }
        Eigen::MatrixXd F_re((int)(max*match_num),9);
        for (int i = 0; i < (int)(max*match_num); ++i)
        {
            // cout << in_o[i][0] << in_s[i][0] << in_o[i][0] * in_s[i][0] << endl;
            F_re(i,0) = in_o[i][0] * in_s[i][0];
            F_re(i,1) = in_s[i][0] * in_o[i][1];
            F_re(i,2) = in_s[i][0];
            F_re(i,3) = in_s[i][1] * in_o[i][0];
            F_re(i,4) = in_s[i][1] * in_o[i][1];
            F_re(i,5) = in_s[i][1];
            F_re(i,6) = in_o[i][0];
            F_re(i,7) = in_o[i][1];
            F_re(i,8) = 1;
        }
        // cout <<  max << "%" << endl; 
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_re(F_re, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXd U_re = svd_re.matrixU();
        Eigen::MatrixXd V_re = svd_re.matrixV();
        Eigen::VectorXd v9 = V_re.col(V_re.cols()-1);
        Eigen::Matrix3d fundamental_re;
        fundamental_re << v9[0], v9[1], v9[2],
                            v9[3], v9[4], v9[5],
                        v9[6], v9[7], v9[8];

        Eigen::JacobiSVD<Eigen::Matrix3d> svd_f_re(fundamental_re, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U_f_re = svd_f_re.matrixU();
        Eigen::Matrix3d V_f_re = svd_f_re.matrixV();
        Eigen::Vector3d A = svd_f_re.singularValues();
        // A[A.size()-1] = 0;
        Eigen::Matrix3d sigma;
        sigma << 1,0,0,
                 0,1,0,
                 0,0,0;
        // cout << U_f_re * sigma * V_f_re.transpose() << endl;
        Mat matched_img = img_1;
        Eigen::Matrix3d R_re;
        Eigen::Vector3d t_re;
        F2Rt(U_f_re * sigma * V_f_re.transpose(),R_re,t_re,in_o,in_s);
        // cout << R_re << endl;
        // cout << t_re << endl;


        // scale propagation
        std::vector<float> mid_num;
        bool in_loop = false;
        // cout << "R_re: " << R_re << endl;
        // cout << "t_re: " << t_re.transpose() << endl;
        cout << "size:" << in_o.size() << in_space.size()<<endl;
        cout << "size:" <<in_prev.size() << in_space_out.size()<<endl;
        for (int points = 0; points < in_o.size(); ++points)
        {
            for (int point_prev = 0; point_prev < in_prev.size(); ++point_prev)
            {
                if ((double)in_o[points][0] == (double)in_prev[point_prev][0] && (double)in_o[points][1] == (double)in_prev[point_prev][1] && (double)in_o[points][2] == (double)in_prev[point_prev][2])
                {
                    in_loop = true;
                    // cout << "in_o[points]" << in_o[points] << endl;
                    // cout << "in_prev[points]" << in_prev[point_prev] << endl;
                    // cout << in_o[points] == in_prev[points]<< endl;
                    Eigen::Vector3d threed_tmp;
                    Eigen::Vector3d threed_tmp_prev;
                    // cout << threed_tmp << endl;
                    threed_tmp << in_space[points][0], in_space[points][1],in_space[points][2];
                    threed_tmp_prev << in_space_out[point_prev][0], in_space_out[point_prev][1],in_space_out[point_prev][2];
                    // cout << endl;
                    threed_tmp = (R_re.transpose()*(threed_tmp - t_re));
                    cout << "threed_tmp: " << threed_tmp.transpose() << endl;
                    cout << "threed_tmp: " << threed_tmp.transpose().squaredNorm() << endl;
                    cout << "threed_tmp_prev: " << threed_tmp_prev.transpose() << endl;
                    cout << "threed_tmp_prev: " << threed_tmp_prev.transpose().squaredNorm() << endl;
                    cout << "ratio: " << threed_tmp_prev.squaredNorm()/threed_tmp.squaredNorm() << endl;
                    cout << endl;
                    mid_num.push_back(threed_tmp_prev.squaredNorm()/threed_tmp.squaredNorm());
                }
            }
        }
        in_space_out = in_space;
        in_prev = in_s;
        if (in_loop)
        {
            cout<<"size is " << mid_num.size() << endl;
            nth_element(mid_num.begin(),mid_num.begin()+mid_num.size()/2, mid_num.end());
            cout <<"mid is "<< mid_num[mid_num.size()/2] << endl;
            d*= sqrt(mid_num[mid_num.size()/2]);
        }

        // t_re/=sqrt(t_re[0]*t_re[0] + t_re[1]*t_re[1]);
        f1 << InitialPosition.transpose() << endl;
        InitialPosition = R_re * InitialPosition + t_re;

//**********************************plotting!*******************************************

        for (int i = 0; i < (int)(max*match_num); ++i)
        {
            Eigen::Vector3d tmp(K*in_o[i]);
            circle(matched_img,Point(tmp[0],tmp[1]),3,Scalar(0),2,8,0);
            line(matched_img,Point(tmp[0],tmp[1]),Point((K*in_s[i])[0],(K*in_s[i])[1]),Scalar(0,255,255),1);
        }
        imshow("in",matched_img);
        waitKey();
//**********************************end plotting!*****************************************
    }
    return 0 ;
}