#include <iostream>

#include <cv.h>
#include <highgui.h>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    ros::init(argc, argv, "image_publisher");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("camera/image", 1);
    Mat opencv_img;

    opencv_img = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv_bridge::CvImage cv_img;
    cv_img.header.stamp = ros::Time::now();
    cv_img.header.frame_id = "Display window";
    cv_img.encoding = "bgr8";
    cv_img.image = opencv_img;

    ros::Rate loop_rate(5);
    while (nh.ok()) {
        pub.publish(cv_img.toImageMsg());
        ros::spinOnce();
        loop_rate.sleep();
    }
    return 0;
}
