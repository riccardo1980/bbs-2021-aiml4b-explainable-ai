#!/usr/bin/env bash
set -e

. scripts/config.sh

gsutil ls -b gs://${BUCKET_NAME}  || gsutil mb -l $REGION gs://${BUCKET_NAME}