source activate gha

CUDA_VISIBLE_DEVICES=3 python train_meshhead.py --config config/train_meshhead_N031.yaml

CUDA_VISIBLE_DEVICES=3 python train_gaussianhead.py --config config/train_gaussianhead_N031.yaml

CUDA_VISIBLE_DEVICES=3 python reenactment.py --config config/reenactment_N031.yaml

CUDA_VISIBLE_DEVICES=3 python preprocess_nersemble.py

CUDA_VISIBLE_DEVICES=3 python remove_background_nersemble.py

CUDA_VISIBLE_DEVICES=3 python train_meshhead.py --config config/train_meshhead_N074.yaml

CUDA_VISIBLE_DEVICES=3 python train_gaussianhead.py --config config/train_gaussianhead_N074.yaml

CUDA_VISIBLE_DEVICES=3 python reenactment.py --config config/reenactment_N074.yaml



CUDA_VISIBLE_DEVICES=1 python train_meshhead.py --config config/dummy_train_meshhead_N031.yaml


# 2024/04/18
重新训练一次,先mini_dataset_N031
作者说3000帧只需要5epoch,但是实际训练时候保存不了ckpt,所以使用50epoch
我感觉不恰当啊应该说iter更合适吧
CUDA_VISIBLE_DEVICES=3 python train_meshhead.py --config config/train_meshhead_N031.yaml
这里完全遵循原仓库要求
CUDA_VISIBLE_DEVICES=3 python train_gaussianhead.py --config config/train_gaussianhead_N031.yaml


# 修改了不要sr的配置文件
CUDA_VISIBLE_DEVICES=1 python my_train_gaussianhead.py --config config/train_gaussianhead_N031_nosr.yaml


这里严格遵循了5epoch
CUDA_VISIBLE_DEVICES=2 python train_meshhead.py --config config/train_meshhead_N074.yaml
CUDA_VISIBLE_DEVICES=2 python train_gaussianhead.py --config config/train_gaussianhead_N074.yaml