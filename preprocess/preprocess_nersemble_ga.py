import os
import numpy as np
import cv2
import glob
import json



def extract_frames(subject_id):
    camera_path = os.path.join(DATA_SOURCE, 'camera_params', subject_id, 'camera_params.json')
    with open(camera_path, 'r') as f:
        camera = json.load(f)
    
    background_paths=dict()
    for camera_id in camera['world_2_cam'].keys():
        # background_path = os.path.join(DATA_SOURCE, 'sequence_BACKGROUND_part-1', id, 'BACKGROUND', 'image_%s.jpg' % camera_id)
        background_paths[camera_id] = os.path.join(DATA_SOURCE, f'participant_{subject_id}', subject_id, 'BACKGROUND', 'image_%s.jpg' % camera_id)

    # 对每个文件夹单独处理
    for trial in os.listdir(os.path.join(DATA_SOURCE, f'participant_{subject_id}', subject_id)):
        if trial == 'BACKGROUND':
            continue
        trial_path = os.path.join(DATA_SOURCE, f'participant_{subject_id}', subject_id, trial)
        
        # 写bg
        bg_path=os.path.join(DATA_OUTPUT, subject_id, trial, 'background')
        os.makedirs(bg_path, exist_ok=True)

        for camera_id in camera['world_2_cam'].keys():
            background = cv2.imread(background_paths[camera_id])
            cv2.imwrite(os.path.join(bg_path, 'bg_' + camera_id + '.jpg'), background)
        

        # mkdir
        img_path=os.path.join(DATA_OUTPUT, subject_id, trial, 'images')
        os.makedirs(img_path, exist_ok=True)
        cam_path=os.path.join(DATA_OUTPUT, subject_id, trial, 'cameras')
        os.makedirs(cam_path, exist_ok=True)

        video_paths = glob.glob(os.path.join(trial, 'cam_*'))
        for video_path in video_paths:
            camera_id = video_path[-13:-4]
            extrinsic = np.array(camera['world_2_cam'][camera_id][:3])
            intrinsic = np.array(camera['intrinsics'])
            

            # 原视频73fps，取每3帧，约24-25fps
            # 这里最好先降采样再处理
            timestep_index = 0
            cap = cv2.VideoCapture(video_path)
            count = -1
            while(1): 
                _, image = cap.read()
                if image is None:
                    break
                count += 1
                if count % 3 != 0:
                    continue
                
                timestep_index+=1
                # cam_id
                cam_id = {v:k for k,v in cam2id.items()}['cam_'+camera_id]
                img_name = '{:05d}_{:02d}'.format(timestep_index,cam_id)
                cv2.imwrite(os.path.join(img_path, img_name + '.png'), image)
                cam_name = '{:05d}_{:02d}'.format(timestep_index,cam_id)
                np.savez(os.path.join(cam_path, cam_name + '.npz'), extrinsic=extrinsic, intrinsic=intrinsic)
                



if __name__ == "__main__":
    DATA_SOURCE = '/data/chenziang/codes/Gaussian-Head-Avatar/data'
    DATA_OUTPUT = '../NeRSemble_ga'

    cam2id={
        0:'cam_222200042',
        1:'cam_222200044',
        2:'cam_222200046',
        3:'cam_222200040',
        4:'cam_222200036',
        5:'cam_222200048',
        6:'cam_220700191',
        7:'cam_222200041',
        8:'cam_222200037',
        9:'cam_222200038',
        10:'cam_222200047',
        11:'cam_222200043',
        12:'cam_222200049',
        13:'cam_222200039',
        14:'cam_222200045',
        15:'cam_221501007',
    }

    id_list=['074']
    for id in id_list:
        extract_frames(id)