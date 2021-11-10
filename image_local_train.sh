#!/usr/bin/env bash

TRAINER_PACKAGE_PATH='myproject/trainer_image_data'
MAIN_TRAINER_MODULE='trainer_image_data.train'
RUNTIME_VERSION=2.4
PYTHON_VERSION=3.7

# Additional user arguments to be forwarded to user code.
# Any relative paths will be relative to the parent directory of --package-path
TRAIN_DATASET=../data/image_data/train_small
EVAL_DATASET=../data/image_data/test
###################################################################
DATE=$(date +"%Y%m%d_%H%M%S")
JOBID=image_data_$DATE
MODEL_FOLDER_DESTINATION=../mdl/image_data/training/$JOBID/

trainer_pars="--export_path=$MODEL_FOLDER_DESTINATION \
              --train_dataset=$TRAIN_DATASET \
              --eval_dataset=$EVAL_DATASET \
              --epochs 1"

gcloud ai-platform local train \
--package-path $TRAINER_PACKAGE_PATH \
--module-name $MAIN_TRAINER_MODULE \
-- \
$trainer_pars

ret=$?
if [ $ret -eq 0 ]; then
    echo -e "\nSUCCESS\n"
else
    echo "error on local run"
fi
exit $ret