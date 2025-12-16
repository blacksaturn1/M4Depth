# M4Depth

## Setup Notes
### Check that CUDA version is 11.2
```shell
nvcc --version
```

### Create Conda setup
```shell
conda create -n m4depth python=3.9

conda activate m4depth 

conda install -c conda-forge tensorflow-gpu=2.7 numpy pandas
```
### Unzip files
```shell
unzip '*.zip'
```

## Download MidAir Dataset
```shell
bash  scripts/0a-get_midair.sh /home/mind/dev/rbe/M4Depth/datasets/midair /home/mind/dev/rbe/M4Depth/datasets/midair/config/download_config.txt
```
```shell
bash  scripts/0a-get_midair.sh /home/mind/dev/rbe/M4Depth/datasets/midair /home/mind/dev/rbe/M4Depth/datasets/midair/config/download_config_2.txt
```

```shell
bash  scripts/0a-get_midair.sh /media/mind/MeBackup/wpi/RBE-MachineLearning/M4Depth/datasets/midair /media/mind/MeBackup/wpi/RBE-MachineLearning/M4Depth/datasets/midair/config/download_config_3.txt
```


## Extract MidAir Dataset
```shell
bash  scripts/0a-get_midair-expand.sh /media/mind/MeBackup/datasets/midair /media/mind/MeBackup/datasets/midair/config/download_config_all.txt
```


## Test Evaluation
```shell
bash  scripts/2-evaluate.sh midair /media/mind/MeBackup/wpi/RBE-MachineLearning/M4Depth/pretrained_weights/midair
```
## Use debugger to use 'predict' dataset

## Train
```shell
bash  scripts/1a-train-midair.sh weights/midair
```

## Tensorboard
'''
tensorboard --logdir weights/midair
'''