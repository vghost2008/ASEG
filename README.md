# Introduction

## prepare dataset

- download fg mask from [Download link](https://drive.google.com/file/d/1Qdpuwx5NO2rgBzQD1Eh3dU3AgYEI6INP/view?usp=drive_link)
- download dtd data from [Download link](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
- make ***datadir_root*** directory
- unpack dtd to ***datadir_root*** directory
- unpack MVTEC AD2 dataset to ***datadir_root/datasets*** directory
- unpack fg_mask to ***datadir_root/datasets*** directory
- make sure the directory structure is:

```
tree datadir_root
.
├── datasets
│   ├── fg_mask
│   │   ├── can
│   │   ├── fruit_jelly
│   │   ├── vial
│   │   ├── wallplugs
│   │   └── walnuts
│   ├── can
│   │   ├── test_private
│   │   ├── test_private_mixed
│   │   ├── test_public
│   │   ├── train
│   │   └── validation
│   ├── fabric
│   ├── fruit_jelly
│   ├── rice
│   ├── sheet_metal
│   ├── vial
│   ├── wallplugs
│   └── walnuts
│       ├── test_private
│       ├── test_private_mixed
│       ├── test_public
│       ├── train
│       └── validation
├── dtd
│   ├── images
│   ├── imdb
│   └── labels

```

## train

usage python ./tools/train.py work_dir datadir_root -d dataset_name --gpu gpu_idx

- work_dir: directory to save checkpoint, log and other temporary files.
- datadir_root: the dataset directory.
- dataset_name: subdatasets name, for example can, fruit_jelly, et.
- gpu: gpu idx.

Example:
```
python ./tools/train.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d can --gpu 0
python ./tools/train.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d fabric --gpu 1
python ./tools/train.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d fruit_jelly --gpu 2
python ./tools/train.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d rice --gpu 3
python ./tools/train.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d sheet_metal --gpu 4
python ./tools/train.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d vial --gpu 5
python ./tools/train.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d wallplugs --gpu 6
python ./tools/train.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d walnuts --gpu 7
```

## evaluate

usage python ./tools/eval.py work_dir datadir_root -d dataset_name --gpu gpu_idx

Example:
```
python ./tools/eval.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d can --gpu 0
python ./tools/eval.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d fabric --gpu 0
python ./tools/eval.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d fruit_jelly --gpu 0
python ./tools/eval.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d rice --gpu 0
python ./tools/eval.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d sheet_metal --gpu 0
python ./tools/eval.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d vial --gpu 0
python ./tools/eval.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d wallplugs --gpu 0
python ./tools/eval.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d walnuts --gpu 0
```

evaluate will generate a threshold file which will be used in predict.

## predict

usage python ./tools/predict.py work_dir datadir_root -d dataset_name --gpu gpu_idx

Example:
```
python ./tools/predict.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d can --gpu 0
python ./tools/predict.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d fabric --gpu 0
python ./tools/predict.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d fruit_jelly --gpu 0
python ./tools/predict.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d rice --gpu 0
python ./tools/predict.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d sheet_metal --gpu 0
python ./tools/predict.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d vial --gpu 0
python ./tools/predict.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d wallplugs --gpu 0
python ./tools/predict.py /home/wj/ai/mldata1/MVTEC/workdir/aseg /home/wj/ai/mldata1/MVTEC/ -d walnuts --gpu 0
```

The final prediction results will be saved in {work_dir}_tiff/predict, the above example will be saved in /home/wj/ai/mldata1/MVTEC/workdir/aseg_tiff/predict.


