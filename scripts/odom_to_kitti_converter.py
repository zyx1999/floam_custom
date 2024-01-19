#!/usr/bin/env python
import rospy
import tf
import tf.transformations as transform_lib

kitti_data = []
count=0
def listen_to_transform():
    global count
    listener = tf.TransformListener()
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            now = rospy.Time.now()
            listener.waitForTransform("map", "base_link", now, rospy.Duration(4.0))
            (trans, rot) = listener.lookupTransform('map', 'base_link', now)
            
            # 将四元数转换为旋转矩阵
            rotation_matrix = transform_lib.quaternion_matrix(rot)[:3,:3]

            # 转换为KITTI格式
            kitti_format = " ".join(["%.6e" % number for row in rotation_matrix for number in row] +
                                    ["%.6e" % number for number in trans])
            kitti_data.append(kitti_format)
            count += 1

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.loginfo("TF Exception: %s", str(e))
            continue

        rate.sleep()

def save_to_file(file_path):
    with open(file_path, 'w') as file:
        for data in kitti_data:
            file.write(data + '\n')

if __name__ == '__main__':
    rospy.init_node('odom_to_kitti_converter')
    listen_to_transform()
    rospy.loginfo(f"Num of pose: {count}")
    save_to_file('/home/yuxuanzhao/ros_workspace/floam_ws/src/floam/result/transform_kitti.txt')
