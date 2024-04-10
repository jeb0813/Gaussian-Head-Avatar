source activate gha

CUDA_VISIBLE_DEVICES=3 python train_meshhead.py --config config/train_meshhead_N031.yaml

CUDA_VISIBLE_DEVICES=3 python train_gaussianhead.py --config config/train_gaussianhead_N031.yaml

CUDA_VISIBLE_DEVICES=3 python reenactment.py --config config/reenactment_N031.yaml

CUDA_VISIBLE_DEVICES=3 python preprocess_nersemble.py

CUDA_VISIBLE_DEVICES=3 python remove_background_nersemble.py

CUDA_VISIBLE_DEVICES=3 python train_meshhead.py --config config/train_meshhead_N074.yaml

CUDA_VISIBLE_DEVICES=3 python train_gaussianhead.py --config config/train_gaussianhead_N074.yaml
