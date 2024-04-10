import os
import numpy as np
import cv2
import glob
import json


from collections import defaultdict

def CropImage(left_up, crop_size, image=None, K=None):
    """
    对图像进行裁剪操作，并且调整内参矩阵
    """
    crop_size = np.array(crop_size).astype(np.int32)
    left_up = np.array(left_up).astype(np.int32)

    if not K is None:
        K[0:2,2] = K[0:2,2] - np.array(left_up)

    if not image is None:
        if left_up[0] < 0:
            image_left = np.zeros([image.shape[0], -left_up[0], image.shape[2]], dtype=np.uint8)
            image = np.hstack([image_left, image])
            left_up[0] = 0
        if left_up[1] < 0:
            image_up = np.zeros([-left_up[1], image.shape[1], image.shape[2]], dtype=np.uint8)
            image = np.vstack([image_up, image])
            left_up[1] = 0
        if crop_size[0] + left_up[0] > image.shape[1]:
            image_right = np.zeros([image.shape[0], crop_size[0] + left_up[0] - image.shape[1], image.shape[2]], dtype=np.uint8)
            image = np.hstack([image, image_right])
        if crop_size[1] + left_up[1] > image.shape[0]:
            image_down = np.zeros([crop_size[1] + left_up[1] - image.shape[0], image.shape[1], image.shape[2]], dtype=np.uint8)
            image = np.vstack([image, image_down])

        image = image[left_up[1]:left_up[1]+crop_size[1], left_up[0]:left_up[0]+crop_size[0], :]

    return image, K


def ResizeImage(target_size, source_size, image=None, K=None):
    """
    对图像进行resize操作，并且调整内参矩阵
    """
    if not K is None:
        K[0,:] = (target_size[0] / source_size[0]) * K[0,:]
        K[1,:] = (target_size[1] / source_size[1]) * K[1,:]

    if not image is None:
        image = cv2.resize(image, dsize=target_size)
    return image, K


def extract_frames(id_list,emos=['neutral']):
    # camera_path = os.path.join(DATA_SOURCE, 'cam_params.json')
    for id in id_list:
        # camera_path="/data/chenziang/codes/Gaussian-Head-Avatar/data_mead/cam_params.json"
        camera_path = os.path.join(DATA_SOURCE, id, 'cam_params.json')
        with open(camera_path, 'r') as f:
            camera = json.load(f)

        fids = defaultdict(int)
        

        # no bg image

        # TODO: all emos
        for emo in emos:
            if not os.path.exists(os.path.join(DATA_SOURCE, id, emo)):
                continue
            video_folders = os.listdir(os.path.join(DATA_SOURCE, id, emo))
            video_folders.sort()
            for video_folder in video_folders:
                video_paths = os.listdir(os.path.join(DATA_SOURCE, id, emo, video_folder))
                video_paths.sort()

                for video_path in video_paths:
                    camera_id = video_path[4]
                    extrinsic = np.array(camera['world_2_cam_RT'][camera_id][:3])
                    intrinsic = camera['intrinsics'][camera_id]

                    # 预处理intrinsics
                    fx,fy,cx,cy=intrinsic['fx'],intrinsic['fy'],intrinsic['cx'],intrinsic['cy']
                    # intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    _, intrinsic = CropImage(LEFT_UP, CROP_SIZE, None, intrinsic)
                    _, intrinsic = ResizeImage(SIZE, CROP_SIZE, None, intrinsic)
                    
                    cap = cv2.VideoCapture(os.path.join(DATA_SOURCE, id, video_folder, video_path))
                    count = -1
                    while(1): 
                        _, image = cap.read()
                        if image is None:
                            break
                        count += 1
                        ### important
                        ### drop frames
                        if count % 3 != 0:
                            continue
                        visible = (np.ones_like(image) * 255).astype(np.uint8)
                        image, _ = CropImage(LEFT_UP, CROP_SIZE, image, None)
                        image, _ = ResizeImage(SIZE, CROP_SIZE, image, None)
                        visible, _ = CropImage(LEFT_UP, CROP_SIZE, visible, None)
                        visible, _ = ResizeImage(SIZE, CROP_SIZE, visible, None)
                        image_lowres = cv2.resize(image, SIZE_LOWRES)
                        visible_lowres = cv2.resize(visible, SIZE_LOWRES)
                        os.makedirs(os.path.join(DATA_OUTPUT, id, emo, 'images', '%04d' % fids[camera_id]), exist_ok=True)
                        cv2.imwrite(os.path.join(DATA_OUTPUT, id, emo, 'images', '%04d' % fids[camera_id], 'image_' + camera_id + '.jpg'), image)
                        cv2.imwrite(os.path.join(DATA_OUTPUT, id, emo, 'images', '%04d' % fids[camera_id], 'image_lowres_' + camera_id + '.jpg'), image_lowres)
                        cv2.imwrite(os.path.join(DATA_OUTPUT, id, emo, 'images', '%04d' % fids[camera_id], 'visible_' + camera_id + '.jpg'), visible)
                        cv2.imwrite(os.path.join(DATA_OUTPUT, id, emo, 'images', '%04d' % fids[camera_id], 'visible_lowres_' + camera_id + '.jpg'), visible_lowres)
                        os.makedirs(os.path.join(DATA_OUTPUT, id, emo, 'cameras', '%04d' % fids[camera_id]), exist_ok=True)
                        np.savez(os.path.join(DATA_OUTPUT, id, emo, 'cameras', '%04d' % fids[camera_id], 'camera_' + camera_id + '.npz'), extrinsic=extrinsic, intrinsic=intrinsic)
                        
                        fids[camera_id] += 1





if __name__ == "__main__":
    # 确保裁切后画面中心不动
    LEFT_UP = [(1920-1080)//2, 0]
    CROP_SIZE = [1080, 1080]

    # 超参数不改
    SIZE = [2048, 2048]
    SIZE_LOWRES = [256, 256]
    # 

    DATA_SOURCE = '/data/chenziang/codes/Gaussian-Head-Avatar/data_mead/'
    DATA_OUTPUT = '../Mead'

    import ipdb

    # ipdb.set_trace()
    extract_frames(['M003'])

