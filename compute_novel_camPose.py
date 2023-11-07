import torch
import numpy as np
import cv2
import os
import argparse
import glob
import json

def comp_ray_dir_cam_fxfy(H, W, fx, fy):
    """Compute ray directions in the camera coordinate, which only depends on intrinsics.
    This could be further transformed to world coordinate later, using camera poses.
    :return: (H, W, 3) torch.float32
    """
    y, x = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=fx.device),
                          torch.arange(W, dtype=torch.float32, device=fx.device))  # (H, W)

    # Use OpenGL coordinate in 3D:
    #   x points to right
    #   y points to up
    #   z points to backward
    #
    # The coordinate of the top left corner of an image should be (-0.5W, 0.5H, -1.0).
    dirs_x = (x - 0.5*W) / fx  # (H, W)
    dirs_y = -(y - 0.5*H) / fy  # (H, W)
    dirs_z = -torch.ones(H, W, dtype=torch.float32, device=fx.device)  # (H, W)
    rays_dir = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)  # (H, W, 3)
    return rays_dir


t_vals = torch.linspace(0, 1, args.num_sample, device=my_devices)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', default=20) 
    parser.add_argument('--target_dir', default='/app/hzq/data/NeRF/processed_07_06_18_40/') 
    args = parser.parse_args()

    target_dir = args.target_dir
    num_samples = args.num_samples
    target_images_dir = os.path.join(target_dir, 'images')
    target_json_path = os.path.join(target_dir,'transforms_test_selfGenerated.json')

    frames = []
    rx = np.zeros(num_samples)
    ry = np.zeros(num_samples)
    rz = np.zeros(num_samples)

    # Calculate the angle increment for each point
    theta = np.linspace(0, 2*np.pi, n_points)

    # Generate the spiral trajectory

    for i in range(n_points):
        rx[i] = radius * np.cos(theta[i])
        ry[i] = radius * np.sin(theta[i])
        rz[i] = pitch * theta[i] / (2*np.pi) + height
        
        t =  np.array([rx[-1], ry[-1], rz[-1]])

        rotation_matrix = cv2.Rodrigues([rx[i],ry[i],rz[i]])[0]
        transform_matrix = np.row_stack((np.column_stack((rotation_matrix,t)),np.array([0,0,0,1])))

        id_dict = {'file_path':os.path.join(target_images_dir,str(i)), 'transform_matrix':transform_matrix.tolist()}
        frames.append(id_dict)

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