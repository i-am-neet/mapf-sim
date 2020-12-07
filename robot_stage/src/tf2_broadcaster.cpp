#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>

void odomCallback(const nav_msgs::OdometryConstPtr& msg){
  static tf2_ros::TransformBroadcaster br;
  geometry_msgs::TransformStamped transformStamped;

  transformStamped.header.stamp = ros::Time::now();
  transformStamped.header.frame_id = "map";
  transformStamped.child_frame_id = msg->header.frame_id;
  transformStamped.transform.translation.x = msg->pose.pose.position.x;
  transformStamped.transform.translation.y = msg->pose.pose.position.y;
  transformStamped.transform.translation.z = 0.0;
  transformStamped.transform.rotation.x = msg->pose.pose.orientation.x;
  transformStamped.transform.rotation.y = msg->pose.pose.orientation.y;
  transformStamped.transform.rotation.z = msg->pose.pose.orientation.z;
  transformStamped.transform.rotation.w = msg->pose.pose.orientation.w;

  br.sendTransform(transformStamped);
}

int main(int argc, char** argv){
  ros::init(argc, argv, "my_tf2_broadcaster");

  std::cout<<"argc: "<<argc<<std::endl;
  for (int i = 0; i < argc; ++i) {
    printf("[%d] %s\n", i, argv[i]);
  }

  if (argc != 2) {
    std::cerr<<"Arugments Error"<<std::endl;
    return 0;
  }

  ros::NodeHandle node;
  ros::Subscriber sub = node.subscribe("/"+std::string(argv[1])+"/odom", 10, &odomCallback);

  ros::spin();
  return 0;
};
