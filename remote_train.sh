#!/usr/bin/env bash

# Choosing between local/remote run
POSITION=${1:-"local"}
TRAINER_PACKAGE_PATH='myproject/trainer_tabular_data'
MAIN_TRAINER_MODULE='trainer_tabular_data.train'
RUNTIME_VERSION=2.4
PYTHON_VERSION=3.7
CMLE_REGION='us-central1'

BUCKET_NAME='bbs-2021-opml4b-explainability'
TRAIN_DATASET=gs://${BUCKET_NAME}/data/tabular_data/train.csv
EVAL_DATASET=gs://${BUCKET_NAME}/data/tabular_data/test.csv
###################################################################
DATE=$(date +"%Y%m%d_%H%M%S")
JOBID=tabular_data_$DATE
MODEL_FOLDER_DESTINATION=gs://${BUCKET_NAME}/mdl/tabular_data/$JOBID/
JOB_DIR=gs://${BUCKET_NAME}/jobdir/tabular_data/$JOBID/

trainer_pars="--export_path=$MODEL_FOLDER_DESTINATION \
              --train_dataset=$TRAIN_DATASET \
              --eval_dataset=$EVAL_DATASET  \
              --epochs 3"

gcloud ai-platform jobs submit training $JOBID \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --job-dir $JOB_DIR \
    --region $CMLE_REGION \
    --runtime-version $RUNTIME_VERSION \
    --python-version $PYTHON_VERSION \
    --scale-tier BASIC \
    -- \
    $trainer_pars

ret=$?
if [ $ret -eq 0 ]; then
    echo -e "\nSUCCESS\n"
else
    echo "error on local run"
fi
exit $ret