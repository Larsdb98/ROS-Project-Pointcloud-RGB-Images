#!/usr/bin/env python

#################################################################################
# Pointcloud to RGB image overlay
#
# This ROS node is subscribed to the following topics: 
#   Pointcloud topic (PointCloud2 messages)
#   Image topic (Image messages)
#   Camera intrinsic parameters (CameraInfo messages)
# All topic names can be adjusted with the private parameter server
# See launchfile for an example.
# 
# This node projects the pointcloud (which needs to be defined in
# the camera optical frame coordinates) onto the pixel image coordinate frame.
# Additionally, it uses the height component of the pointcloud (component Z)
# To produce a colormap. Where the color represents the height of the projected
# point in 3D.
# This colormap is overlayed onto the Images recieved under the Image topic. 
# It is important to remember that the image messages recieved and the cameraInfo
# messages corespond to the same camera. Projection errors may occur otherwise.
# The produced image is then published as an Image messsage.

import rospy
import cv2
import cv_bridge
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import os


class PointCloudProcessor:
    def __init__(self):
        rospy.init_node("pointcloud_rgb_image_overlay_from_untransformed_pointclouds")

        # private parameters need to be used to avoid accidental use with other
        # nodes. use the "~" sign in front of the name of the param.
        self.__point_cloud_in = rospy.get_param("~point_cloud_in_topic", "/points") # /grid_map_visualization/elevation_points_rgb_frame
        self.__image_in = rospy.get_param("~image_in_topic", "/camera/color/image_raw") # /wrist_camera/color/image_raw
        self.__camera_info_in = rospy.get_param("~camera_info_topic", "/camera/color/camera_info") # /wrist_camera/color/camera_info
        self.__image_out = rospy.get_param("~image_out_topic", "/camera/color/elevation_map_overlayed_image_raw") # /wrist_camera/color/elevation_map_overlayed_image_raw
        self.__biggerPoints = rospy.get_param("~biggerPoints", "True")
        self.__opacity = rospy.get_param("~opacity", 0.0)

        # Setup subscribers
        rospy.Subscriber(self.__point_cloud_in, PointCloud2, self.point_cloud_callback)
        rospy.Subscriber(self.__image_in, Image, self.image_callback)
        rospy.Subscriber(self.__camera_info_in, CameraInfo, self.camera_info_callback)

        # Setup publisher
        self.image_pub = rospy.Publisher(self.__image_out, Image, queue_size=10)

        # Initialize message variables as attributes:
        self.camera_info = None
        self.rgb_image = None

        # initialize all other attributes:

        self.__VERBOSE = False
        self.__EXPORT_CURRENT_FRAME = False
        self.__EXPORT_ARRAY_TO_FILE = False

        # Inform ROS startup completed
        rospy.loginfo("Pointcloud to Colormap node is up !")

    ############################################
    ################ Callbacks #################
    ############################################

    def point_cloud_callback(self, pc_msg):
        if self.camera_info is not None and self.rgb_image is not None:
            # Project point cloud onto camera pixel coordinates
            points = pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans = True)
            points = np.array(list(points))
            projected_points = self.project_point_cloud(points, self.camera_info)

            # camera image dimensions
            img_h = self.camera_info.height
            img_w = self.camera_info.width
            camera_image_shape = [img_h, img_w]
            
            # Generate colormap based on Z component
            colormap = self.generate_colormap(projected_points, points, camera_image_shape, BiggerPoints = self.__biggerPoints)

            # Overlay colormap onto RGB image with adjustable opacity
            final_image = self.overlay_colormap(self.rgb_image, colormap, opacity=0.8)

            # Debug: 
            if self.__EXPORT_ARRAY_TO_FILE: 
                this_dir = os.path.dirname(os.path.abspath(__file__)) 
                txt_filename = os.path.join(this_dir, "pixel_coordinates.txt") 
                np.savetxt(txt_filename, projected_points[:, 0, :]) 


            if self.__EXPORT_CURRENT_FRAME:
                this_dir = os.path.dirname(os.path.abspath(__file__))
                img_filename = os.path.join(this_dir, "test_img.png")
                cv2.imwrite(img_filename, final_image)


            # Publish the final image
            self.publish_image(final_image)
            rospy.loginfo("Published image under the topic {}".format(self.__image_out))
        else:
            rospy.logwarn("Cannot produce height colormap ! CameraInfo or Image message still missing. Waiting...")


    def image_callback(self, image_msg):
        bridge = cv_bridge.CvBridge()
        self.rgb_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")


    def camera_info_callback(self, camera_info_msg):
        self.camera_info = camera_info_msg





    ############################################
    ############### Other methods ##############
    ############################################

    def project_point_cloud(self, points, camera_info_msg):
        # print("Shape of point cloud array: {}".format(points.shape))

        # Create intrinsic matrix from latest camera_info message
        camera_matrix = np.array(camera_info_msg.K).reshape((3,3))

        if self.__VERBOSE:
            print("Camera matrix:")
            print(camera_matrix)    

        # Project points onto the image plane:
        image_points, _ = cv2.projectPoints(points, rvec = np.eye(3), tvec = np.zeros(3), cameraMatrix=camera_matrix, distCoeffs=None)
        # print("Projected points array: {}".format(image_points.shape))
        

        return image_points
    



    def generate_colormap(self, projected_points, original_points, rgb_image_shape, BiggerPoints = True):
        # Initial declarations of variables that may be used:
        pixel_coordinates = None
        pixel_coordinates_2 = None
        pixel_coordinates_3 = None
        pixel_coordinates_4 = None
        pixel_coordinates_5 = None
        pixel_coordinates_6 = None
        pixel_coordinates_7 = None
        pixel_coordinates_8 = None
        pixel_coordinates_9 = None

        # Map normalized Z values to pixel coordinates
        h, w = rgb_image_shape
        # Extract Z values from the original 3D points
        z_values = original_points[:, 2]

        # Normalize Z values
        normalized_z = ((z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values)) * 255).astype(int)

        jet_colormap = cv2.applyColorMap(normalized_z[:].astype(np.uint8), cv2.COLORMAP_JET)

        colormap = np.zeros((h, w, 3), dtype = np.uint8)
        
        

        
        # raw_projected_points_x = (projected_points[:, 0, 0] * (w - 1)).astype(int)
        # raw_projected_points_y = (projected_points[:, 0, 1] * (h - 1)).astype(int)
        # raw_projected_points = [raw_projected_points_x, raw_projected_points_y]
        # raw_projected_points = np.transpose(raw_projected_points)

        if not BiggerPoints:
            pixel_coordinates_x = np.clip((projected_points[:, 0, 0]).astype(int), 0, w - 1)
            pixel_coordinates_y = np.clip((projected_points[:, 0, 1]).astype(int), 0, h - 1)

            pixel_coordinates = np.vstack([pixel_coordinates_x, pixel_coordinates_y])
            pixel_coordinates = np.transpose(pixel_coordinates)

            if self.__VERBOSE:
                # Just more debugging shit
                print("jet colormap shape: {}".format(jet_colormap.shape))
                print("colormap shape: {}".format(colormap.shape))
                print("normalized z shape: {}".format(normalized_z.shape))
                print("pixel coordinates shape: {}".format(pixel_coordinates.shape))
                print(pixel_coordinates)

            # Assign color values to the corresponding pixel coordinates using the jet colormap 
            colormap[pixel_coordinates[:, 1], pixel_coordinates[:, 0], :] = jet_colormap[: , 0, :]

            return colormap
        
        else:
            pixel_coordinates_x = np.clip((projected_points[:, 0, 0]).astype(int), 0, w - 1)
            pixel_coordinates_x_plus = np.clip((projected_points[:, 0, 0] + np.ones_like(projected_points[:, 0, 0])).astype(int), 0, w - 1)
            pixel_coordinates_x_min = np.clip((projected_points[:, 0, 0] - np.ones_like(projected_points[:, 0, 0])).astype(int), 0, w - 1)

            pixel_coordinates_y = np.clip((projected_points[:, 0, 1]).astype(int), 0, h - 1)
            pixel_coordinates_y_plus = np.clip((projected_points[:, 0, 1] + np.ones_like(projected_points[:, 0, 1])).astype(int), 0, h - 1)
            pixel_coordinates_y_min = np.clip((projected_points[:, 0, 1] - np.ones_like(projected_points[:, 0, 1])).astype(int), 0, h - 1)


            pixel_coordinates = np.vstack([pixel_coordinates_x, pixel_coordinates_y])
            pixel_coordinates_2 = np.vstack([pixel_coordinates_x, pixel_coordinates_y_plus])
            pixel_coordinates_3 = np.vstack([pixel_coordinates_x_plus, pixel_coordinates_y])
            pixel_coordinates_4 = np.vstack([pixel_coordinates_x_plus, pixel_coordinates_y_plus])


            pixel_coordinates = np.transpose(pixel_coordinates)
            pixel_coordinates_2 = np.transpose(pixel_coordinates_2)
            pixel_coordinates_3 = np.transpose(pixel_coordinates_3)
            pixel_coordinates_4 = np.transpose(pixel_coordinates_4)     

            colormap[pixel_coordinates[:, 1], pixel_coordinates[:, 0], :] = jet_colormap[: , 0, :]
            colormap[pixel_coordinates_2[:, 1], pixel_coordinates_2[:, 0], :] = jet_colormap[: , 0, :]
            colormap[pixel_coordinates_3[:, 1], pixel_coordinates_3[:, 0], :] = jet_colormap[: , 0, :]
            colormap[pixel_coordinates_4[:, 1], pixel_coordinates_4[:, 0], :] = jet_colormap[: , 0, :]

            return colormap


    def overlay_colormap(self, rgb_image, colormap, opacity = 0.8):
        # TODO: 
        # Implement this function con overlay the colormap on the rgb image.
        # Then use CvBridge to convert the OpenCV image into a ROS Image
        # message and publish it using the declared publisher.

        gamma = 0.0

        # Convert colormap to 3 channels to match RGB image
        colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
        # Blend RGB image and colormap together based on opacity
        final_image = cv2.addWeighted(rgb_image, 1 - self.__opacity, colormap_rgb, self.__opacity, gamma)

        return final_image
    



    def publish_image(self, final_image):
        # Convert OpenCV image to message
        bridge = cv_bridge.CvBridge()
        image_msg = bridge.cv2_to_imgmsg(final_image, encoding = "bgr8")

        # Publish the final image
        self.image_pub.publish(image_msg)

    


    def run(self):
        rospy.spin()




### Main function
def main():
    process = PointCloudProcessor()
    process.run()






#############################################
if __name__ == "__main__":
    main()
