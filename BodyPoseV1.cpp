#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include "text/csv/istream.hpp"
using namespace std;
using namespace cv;
namespace csv = ::text::csv;

Point2f shoulder_2(Mat fs,string rl);
bool findface(Mat face);
Point shoulder(Mat img, string side );


//string eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
//string mouth_cascade_name = "haarcascade_mcs_mouth.xml";
//string nose_cascade_name = "haarcascade_mcs_nose.xml";
string face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
//CascadeClassifier eyes_cascade;
//CascadeClassifier mouth_cascade;
//CascadeClassifier nose_cascade;
vector<Rect> faces,nose,eyes,mouth;

int main()
{
    VideoCapture cap("./images/Image_%04d.jpg");
    face_cascade.load(face_cascade_name);
    Mat img, clone,clone1, gray,clone_gray,face_roi,shoulder_roi;
    Rect ROI,TEST;
    Point p;
    size_t i;
    bool face_present = 0;
    int name = 0,face_height = 0,face_width = 0,frame_width = 0, frame_height = 0,index;
    Point face_bottom,left,right,shifted_left,shifted_right;
    shifted_left.x = shifted_left.y = 0;
    shifted_right.x = shifted_right.y = 0;
    RNG rng(12345);
    while(1){
    cap >> img;
    if(img.empty())
        break;
    face_present = false;
    frame_width = img.cols;
    frame_height = img.rows;
    cvtColor(img,gray,CV_BGR2GRAY);
    face_cascade.detectMultiScale( gray, faces, 1.1,7, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    for ( i = 0; i < faces.size(); i++ )
    {
        if(faces.size()==0 || faces.size()>1)
            break;
        else{
            face_present = true;
            clone = img.clone();
            clone1 = img.clone();
            face_height = faces[i].height;
            face_bottom.y = faces[i].y + faces[i].height + 1;
            face_bottom.x = faces[i].x;
            face_width = faces[i].width;
            rectangle(clone,faces[i],Scalar(0,255,0),2);
            break;
        }
    }
    if(face_present==true){
        cvtColor(clone1,clone_gray,CV_BGR2GRAY);
        threshold(clone_gray,clone_gray,10,255,THRESH_BINARY);
        vector<vector<Point> > contours;
        findContours( clone_gray, contours, RETR_TREE, CHAIN_APPROX_SIMPLE );
        vector<vector<Point> > contours_poly( contours.size() );
        vector<Rect> boundRect( contours.size() );
        for( size_t i = 0; i < contours.size(); i++ ){
            approxPolyDP( contours[i], contours_poly[i], 3, true );
            boundRect[i] = boundingRect( contours_poly[i] );
            if(boundRect[i].width>face_width*1.5 && boundRect[i].height>face_height*4){
                ROI = Rect(Point(boundRect[i].tl().x,face_bottom.y),Point(boundRect[i].br().x,face_bottom.y+face_width*1.4));
                rectangle( clone, Point(boundRect[i].tl().x,face_bottom.y), boundRect[i].br(), Scalar(0,0,255), 2 );
                shoulder_roi = clone1(ROI);
                index = i;
                break;
            }
        }
        //cout<<name<<endl;
        // FIND SHOULDER POINTS
        left = shoulder(shoulder_roi,"left");
        right = shoulder(shoulder_roi,"right");
        // SHIFT POINTS TO ORIGINAL IMAGE
        shifted_left.x = left.x + boundRect[index].tl().x;
        shifted_left.y = left.y+face_bottom.y;
        shifted_right.x = right.x + boundRect[index].tl().x;
        shifted_right.y = right.y+face_bottom.y;
        circle(clone,shifted_left,4,Scalar(255,0,0),-1);
        circle(clone,shifted_right,4,Scalar(0,0,255),-1);
        line(clone,shifted_left,shifted_right,Scalar(0,255,0),2);
        // FIND PARAMETERS
        float SLS = (float)(shifted_right.y - shifted_left.y) / (shifted_right.x - shifted_left.x);
        int shoulder_width = shifted_right.x - shifted_left.x;
        int body_height = boundRect[index].br().y - boundRect[index].tl().y;
        cout<<"SLS = "<<SLS<<endl;
        cout<<"SW / BH = "<<(float)shoulder_width/body_height<<endl;
        cout<<"SW / IW = "<<(float)shoulder_width/clone.cols<<endl;
        cout<<"FH / BH = "<<(float)face_height/body_height<<endl;
        cout<<"FH / SW = "<<(float)face_height/shoulder_width<<endl<<endl<<endl;
        namedWindow("Output",WINDOW_NORMAL);
        resizeWindow("Output",Size(640,480));
        imshow("Output",clone);
        if(waitKey(1)==27)
            break;
        name++;
    }
    else
        name++;
}
    return 0;
}

Point shoulder(Mat img, string side )
{
  Mat src_gray,canny_output,left,right;
  Rect left_roi,right_roi;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  cvtColor(img,src_gray,CV_BGR2GRAY);
  threshold(src_gray,src_gray,15,255,THRESH_BINARY);
  Canny( src_gray, canny_output, 200, 255, 3 );
  if(side == "right"){
      left_roi = Rect(canny_output.cols/2,0,canny_output.cols/2,canny_output.rows);
      left = canny_output(left_roi);
      canny_output = left.clone();
  }
  else{
      right_roi = Rect(0,0,canny_output.cols/2,canny_output.rows);
      right = canny_output(right_roi);
      canny_output = right.clone();
  }
  findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
  vector<Moments> mu(contours.size() );
  for( size_t i = 0; i < contours.size(); i++ )
     { mu[i] = moments( contours[i], false );}
  vector<Point2f> mc( contours.size() );
  for( size_t i = 0; i < contours.size(); i++ )
     { mc[i] = Point2f( static_cast<float>(mu[i].m10/mu[i].m00) , static_cast<float>(mu[i].m01/mu[i].m00) ); }
  int largest_area1 = 0,largest_contour_index1 = 0;
  for( int i = 0; i< contours.size(); i++ )
      {
          int a=arcLength(contours[i],true);
          if(a>largest_area1){
              largest_area1=a;
              largest_contour_index1=i;
          }
      }
  if(side == "right"){
      mc[largest_contour_index1].x = mc[largest_contour_index1].x + canny_output.cols;
      return mc[largest_contour_index1];
  }
  else
      return mc[largest_contour_index1];
}

Point2f shoulder_2(Mat fs,string rl)
{
    Point2f shoulder;
    int name = 1,prev = 0;
   // namedWindow("face",WINDOW_NORMAL);
    //resizeWindow("face",Size(640,480));
    int nn =0;
    Mat src = fs.clone();
    //Mat src = imread("./shoulder/"+to_string(name)+".jpg",-1);
    Mat cdst = Mat::zeros(src.rows,src.cols, src.type());
    name++;
    //Mat src = imread("./shoulder/1.jpg",-1);
    Mat gray,bw;
    cvtColor(src, gray, CV_BGR2GRAY);
    threshold (gray, bw, 15, 255, CV_THRESH_BINARY);
    //adaptiveThreshold(~gray, bw, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, -2);
    int arr[src.rows];
    int arrx[src.rows];
    int arr1[src.rows];
    int arrx1[src.rows];
    int i = 0,ii=0;
    for(int x = 0;x<bw.rows;x++){
    for(int start = 0;start<bw.cols;start++){
        int val = (int)bw.at<uchar>(x,start);
        if(val > 15){
            if(start == prev){
                circle(cdst,Point(start,x),3,Scalar(0,0,255),1);
                arr[i] = start;
                arrx[i] = x;
                i++;
            }
            nn++;
                           bw.at<uchar>(x,start) = 255;
                           //cout<<"Point_1 "<<"( "<<start<<" , "<<x<<" )"<<endl;
                           prev = start;
                           //circle(src,Point(start,x),1,Scalar(0,0,255),1);
                           //imshow("face",src);
                           //waitKey(1);
                           break;
                       }
                       //else{
                         //  img.at<uchar>(x,start) = 100;
                       //}
                       //imshow("face",img);
                       //waitKey(1);
                          //color.val[2] = 255;
                     //image_roi.at<Vec3b>(Point(ii,j)) = color;


    }}
    //right
    prev = 0;
    for(int x = 0;x<bw.rows;x++){
    for(int start = bw.cols;start>1;start--){
        int val = (int)bw.at<uchar>(x,start);
        if(val > 15){
            if(start == prev){
                circle(cdst,Point(start,x),3,Scalar(0,0,255),1);
                arr1[ii] = start;
                arrx1[ii] = x;
                ii++;
            }
            nn++;
                           bw.at<uchar>(x,start) = 255;
                          // cout<<"Point_1 "<<"( "<<start<<" , "<<x<<" )"<<endl;
                           prev = start;
                           //circle(src,Point(start,x),1,Scalar(0,0,255),1);
                           //imshow("face",src);
                           //waitKey(1);
                           break;
                       }
                       //else{
                         //  img.at<uchar>(x,start) = 100;
                       //}
                       //imshow("face",img);
                       //waitKey(1);
                          //color.val[2] = 255;
                     //image_roi.at<Vec3b>(Point(ii,j)) = color;


    }}
    //imshow("face",cdst);
   // waitKey(0);
   //exited both loops
    if(rl == "left"){
    for(int z = 0; z<i;z++){
        if(arr[z]-arr[z+1]>10){
           // cout<<arr[z+1]<<"___"<<arrx[z+1]<<endl;
            circle(src,Point(arr[z+1],arrx[z+1]),3,Scalar(0,0,255),3);
            shoulder.x = arr[z+1];
            shoulder.y = arrx[z+1];
            return shoulder;
            //imshow("face",src);
            //waitKey(0);

        }
    }
    }
    if(rl == "right"){
    for(int z = 0; z<ii;z++){
        if(arr1[z]-arr1[z+1]<-10){
           // cout<<arr[z+1]<<"___"<<arrx[z+1]<<endl;
            circle(src,Point(arr1[z+1],arrx1[z+1]),3,Scalar(0,0,255),3);
            shoulder.x = arr1[z+1];
            shoulder.y = arrx1[z+1];
            //imshow("face",src);
            //waitKey(0);
            return shoulder;
        }
    }
    }


}

bool findface(Mat face)
{
    //eyes_cascade.load (eyes_cascade_name);
    //mouth_cascade.load (mouth_cascade_name);
    //nose_cascade.load (nose_cascade_name);
    //nose_cascade.detectMultiScale (face , nose , 1.1 , 2 , 0 | CV_HAAR_SCALE_IMAGE );
    //eyes_cascade.detectMultiScale (face , eyes , 1.1 , 2 , 0 | CV_HAAR_SCALE_IMAGE );
    //mouth_cascade.detectMultiScale (face, mouth, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE);
    //nose_cascade.detectMultiScale (face , nose  );
    //eyes_cascade.detectMultiScale (face , eyes  );
    //mouth_cascade.detectMultiScale (face, mouth );
    //if(nose.size()!=0 || mouth.size()!= 0 || eyes.size()!=0){
      //      cout<<nose.size()<<endl;
        //    cout<<mouth.size()<<endl;
          //  cout<<eyes.size()<<endl<<endl;
            return true;

    //}
    //else
      //  return false;

}
