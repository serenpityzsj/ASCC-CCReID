# ASCC-CCReID

ASCC: Semantic Cross-branch Collaboration with Chebyshev’s Theorem-guided Graph Re-ranking for Cloth-Changing Person Re-Identification

## Getting Started

### Environment

- Python == 3.8
- PyTorch == 1.12.1

### Prepare Data

Please download cloth-changing person re-identification datasets and place them in any path `DATASET_ROOT`:

    DATASET_ROOT
    	└─ LTCC-reID or PRCC or Celeb-reID
    		├── train
    		├── query
    		└── gallery


### Training

```sh
# LTCC
python main.py --gpu_devices 0 --dataset ltcc --dataset_root DATASET_ROOT --dataset_filename LTCC-reID --save_dir SAVE_DIR --save_checkpoint

# PRCC
python main.py --gpu_devices 0 --dataset prcc --dataset_root DATASET_ROOT --dataset_filename PRCC --save_dir SAVE_DIR --save_checkpoint

# Celeb-reID
python main.py --gpu_devices 0 --dataset celeb --dataset_root DATASET_ROOT --dataset_filename Celeb-reID --num_instances 4 --save_dir SAVE_DIR --save_checkpoint
```

`--dataset_root` : replace `DATASET_ROOT` with your dataset root path

`--save_dir`: replace `SAVE_DIR` with the path to save log file and checkpoints

To facilitate reproduction and comparison, we release the **trained model weights** and complete training logs for the datasets. This includes our proposed method's SOTA performance models. All resources can be accessed via: [Download Link](https://pan.baidu.com/s/1dEVDaCyXCNMO0MQsabUcbA?pwd=eve5 )


### Evaluation

```sh
python main.py --gpu_devices 0 --dataset DATASET --dataset_root DATASET_ROOT --dataset_filename DATASET_FILENAME --resume RESUME_PATH --save_dir SAVE_DIR --evaluate
```

`--dataset`: replace `DATASET` with the dataset name

`--dataset_filename`: replace `DATASET_FILENAME` with the folder name of the dataset

`--resume`: replace `RESUME_PATH` with the path of the saved checkpoint

The above three arguments are set corresponding to Training.

The code is based on CSSC-CCReID

```sh
@inproceedings{wang2025content,
  title={Content and salient semantics collaboration for cloth-changing person re-identification},
  author={Wang, Qizao and Qian, Xuelin and Li, Bin and Chen, Lifeng and Fu, Yanwei and Xue, Xiangyang},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1-5},
  year={2025},
  organization={IEEE}
}
```
