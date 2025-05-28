#!/bin/bash

DATADIR_ROOT="" #datadir_root in README.md

WORK_DIR="" #work_dir in README

#TRAIN
python ./tools/train.py $WORK_DIR $DATADIR_ROOT -d can --gpu 0 &
python ./tools/train.py $WORK_DIR $DATADIR_ROOT -d fabric --gpu 1 &
python ./tools/train.py $WORK_DIR $DATADIR_ROOT -d fruit_jelly --gpu 2 &
python ./tools/train.py $WORK_DIR $DATADIR_ROOT -d rice --gpu 3 &
python ./tools/train.py $WORK_DIR $DATADIR_ROOT -d sheet_metal --gpu 4 &
python ./tools/train.py $WORK_DIR $DATADIR_ROOT -d vial --gpu 5 &
python ./tools/train.py $WORK_DIR $DATADIR_ROOT -d wallplugs --gpu 6 &
python ./tools/train.py $WORK_DIR $DATADIR_ROOT -d walnuts --gpu 7 

#EVAL
python ./tools/eval.py $WORK_DIR $DATADIR_ROOT -d can --gpu 0
python ./tools/eval.py $WORK_DIR $DATADIR_ROOT -d fabric --gpu 0
python ./tools/eval.py $WORK_DIR $DATADIR_ROOT -d fruit_jelly --gpu 0
python ./tools/eval.py $WORK_DIR $DATADIR_ROOT -d rice --gpu 0
python ./tools/eval.py $WORK_DIR $DATADIR_ROOT -d sheet_metal --gpu 0
python ./tools/eval.py $WORK_DIR $DATADIR_ROOT -d vial --gpu 0
python ./tools/eval.py $WORK_DIR $DATADIR_ROOT -d wallplugs --gpu 0
python ./tools/eval.py $WORK_DIR $DATADIR_ROOT -d walnuts --gpu 0


#PREDICT
python ./tools/predict.py $WORK_DIR $DATADIR_ROOT -d can --gpu 0
python ./tools/predict.py $WORK_DIR $DATADIR_ROOT -d fabric --gpu 0
python ./tools/predict.py $WORK_DIR $DATADIR_ROOT -d fruit_jelly --gpu 0
python ./tools/predict.py $WORK_DIR $DATADIR_ROOT -d rice --gpu 0
python ./tools/predict.py $WORK_DIR $DATADIR_ROOT -d sheet_metal --gpu 0
python ./tools/predict.py $WORK_DIR $DATADIR_ROOT -d vial --gpu 0
python ./tools/predict.py $WORK_DIR $DATADIR_ROOT -d wallplugs --gpu 0
python ./tools/predict.py $WORK_DIR $DATADIR_ROOT -d walnuts --gpu 0
