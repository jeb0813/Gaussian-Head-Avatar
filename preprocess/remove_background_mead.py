import os
import numpy as np
import cv2
import glob
import json


from PIL import Image
from collections import defaultdict


# source deactivate
# source activate splatting



def convert_png_to_jpg_with_alpha(png_path, jpg_path):
    # 打开PNG图像
    png_img = Image.open(png_path)

    png_array = np.array(png_img)

    jpg_array = np.full(png_array.shape[:2] + (3,), 255, dtype=np.uint8)
    
    # 使用NumPy的广播功能，根据不透明度设置颜色
    jpg_array[...,:3][png_array[...,3] != 0] = [0, 0, 0]

    # 创建PIL图像对象
    jpg_image = Image.fromarray(jpg_array)

    # 保存结果为JPG图像
    jpg_image.save(jpg_path)

    # # 创建一个新的白色背景图像
    # new_img = Image.new("RGB", img.size, (255, 255, 255))
    
    # # 遍历每个像素并根据不透明度设置颜色
    # for y in range(height):
    #     for x in range(width):
    #         # 获取当前像素的RGBA值
    #         r, g, b, alpha = png_image.getpixel((x, y))
            
    #         # 如果不透明度不是0，将颜色设置为黑色
    #         if alpha != 0:
    #             jpg_image.putpixel((x, y), (0, 0, 0))
    
    # # 将PNG图像的alpha通道作为蒙版，将不透明部分填充为黑色
    # new_img.paste(img, mask=img.split()[3])
    
    # # 保存为JPEG格式
    # new_img.convert("RGB").save(output_path, "JPEG")
        # 打开PNG图像
    pass

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

def _extract_frames(vid_path,root_folder_path):
    """
    背景提取，抽帧，重命名
    """
    temp_path = os.path.join(root_folder_path, 'temp')
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    
    # model应该被持久化到显存，不要反复加载
    cmd = "python /data/chenziang/codes/SplattingAvatar/preprocess/IMavatar/preprocess/submodules/RobustVideoMatting/segmentation_api.py --input {video_path} --output {img_path}".format(video_path=os.path.join(root_folder_path,vid_path), img_path=temp_path)
    os.system(cmd)

    # 抽帧重命名
    cnt=-1
    imgs=os.listdir(temp_path)
    imgs.sort()
    for img in imgs:
        cnt+=1
        if cnt%3!=0:
            # 删除这张图
            # print("删除:",img)
            os.remove(os.path.join(temp_path,img))
        else:
            # 重命名
            # print("重命名:",img)
            new_name = "{:04d}.png".format(cnt//3)
            os.rename(os.path.join(temp_path,img),os.path.join(temp_path,new_name))

            # 变换为mask
            convert_png_to_jpg_with_alpha(os.path.join(temp_path,new_name),os.path.join(temp_path,new_name.replace('.png','.jpg')))



def extract_masks(id_list,emos=['neutral']):
    for id in id_list:
        camera_path = os.path.join(DATA_SOURCE, id, 'cam_params.json')
        with open(camera_path, 'r') as f:
            camera = json.load(f)

        fids = defaultdict(int)
        for emo in emos:
            if not os.path.exists(os.path.join(DATA_SOURCE, id, emo)):
                continue
            video_folders = os.listdir(os.path.join(DATA_SOURCE, id, emo))
            video_folders.sort()

            for video_folder in video_folders:
                video_paths = os.listdir(os.path.join(DATA_SOURCE, id, emo, video_folder))
                video_paths.sort()

                for video_path in video_paths:
                    # print(video_path)
                    print(os.path.join(video_folder, video_path))
                    camera_id = video_path[4]
                    _extract_frames(video_path,os.path.join(DATA_SOURCE, id, emo, video_folder))
                    # exit()

                    # resize、重命名和复制

                    masks = os.listdir(os.path.join(DATA_SOURCE, id, emo, video_folder, 'temp'))
                    masks = [mask for mask in masks if mask.endswith('.jpg')]
                    masks.sort()

                    for mask in masks:
                        mask_path = os.path.join(DATA_SOURCE, id, emo, video_folder, 'temp', mask)
                        mask_img = cv2.imread(mask_path)
                        mask_img, _ = CropImage(LEFT_UP, CROP_SIZE, mask_img)
                        mask_img, _ = ResizeImage(SIZE, CROP_SIZE, mask_img)
                        
                        mask_img_lower = cv2.resize(mask_img, SIZE_LOWRES)

                        mask_path = os.path.join(DATA_OUTPUT, id, emo, 'images', '%04d' % fids[camera_id], 'mask_' + camera_id + '.jpg')
                        mask_lowres_path = os.path.join(DATA_OUTPUT, id, emo, 'images', '%04d' % fids[camera_id], 'mask_lowres_' + camera_id + '.jpg')

                        cv2.imwrite(mask_path, mask_img)
                        cv2.imwrite(mask_lowres_path, mask_img_lower)

                        fids[camera_id] += 1
                    
                    # 删除临时文件夹
                    temp_path = os.path.join(DATA_SOURCE, id, emo, video_folder, 'temp')
                    os.system('rm -r ' + temp_path)
                    # exit()


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

    extract_masks(['M003'])
    
