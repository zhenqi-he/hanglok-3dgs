import numpy as np
import cv2
import os
import argparse
import glob
import json

# This is for transferring NGP-style camera position to COLMAP version for 3D Gaussian Splatting

def quart_to_rpy(x, y, z, w):
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(2 * (w * y - x * z))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return np.array([roll, pitch, yaw])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_cam_dir', default='/app/hzq/data/NeRF/processed_07_06_18_40/base_cam.json') 
    parser.add_argument('--target_dir', default='/app/hzq/data/NeRF/processed_07_06_18_40/') 
    args = parser.parse_args()
    base_cam_dir = args.base_cam_dir
    target_dir = args.target_dir
    target_images_dir = os.path.join(target_dir, 'images')
    target_json_path = os.path.join(target_dir,'transforms_test.json')

    json_base_cam = json.load(open(base_cam_dir))

    frames = []
    for i in range(len(json_base_cam['path'])):

        x,y,z,w = json_base_cam['path'][i]['R']
        rotation_matrix = cv2.Rodrigues(quart_to_rpy(x, y, z, w))[0]
        t = np.array(json_base_cam['path'][i]['T'])
        transform_matrix = np.row_stack((np.column_stack((rotation_matrix,t)),np.array([0,0,0,1])))
        id_dict = {'file_path':os.path.join(target_images_dir,str(i)), 'transform_matrix':transform_matrix.tolist()}
        frames.append(id_dict)

    # print(out_dict)
    out_dict = {
			"camera_angle_x": 1.2230789030277986,
            "camera_angle_y": 0.7517771630122048,
            "fl_x": 912.7044060585573,
            "fl_y": 912.1932986043852,
            "k1": 0.050439925261315646,
            "k2": -0.08024922667487974,
            "k3": 0,
            "k4": 0,
            "p1": -0.0009262625277640219,
            "p2": -0.00474904843281617,
            "is_fisheye": False,
            "cx": 621.5531921326699,
            "cy": 371.53830322216083,
            "w": 1280.0,
            "h": 720.0,
			"frames": frames
		  }
    json_object = json.dumps(out_dict, indent=2)
    with open(target_json_path,'w') as outfile:
        outfile.write(json_object)

if __name__ == '__main__':
    main()