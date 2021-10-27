#!/usr/bin/env bash
set -e

. scripts/config.sh

# Delete model version resource
gcloud ai-platform versions delete $VERSION --quiet --model $MODEL

# Delete model resource
gcloud ai-platform models delete $MODEL --quiet