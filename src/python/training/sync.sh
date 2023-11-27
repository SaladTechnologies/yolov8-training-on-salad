#!/bin/bash

# Copy images
azcopy cp "https://${STORAGE_NAME}.file.core.windows.net/training/${TRAINING_DATASET}/images${SASTOKEN}" "/app/training/dataset" --recursive=true

# Copy labels
azcopy cp "https://${STORAGE_NAME}.file.core.windows.net/training/${TRAINING_DATASET}/labels${SASTOKEN}" "/app/training/dataset" --recursive=true

# Copy dataset yaml
azcopy cp "https://${STORAGE_NAME}.file.core.windows.net/training/${TRAINING_DATASET}/yolo8.yaml${SASTOKEN}" "/app/training/yolo8.yaml"

# Copy checkpoints folder
azcopy cp "https://${STORAGE_NAME}.file.core.windows.net/runs${SASTOKEN}" "/app/training" --recursive=true

# Run train.py in the background
python3.10 train.py > /app/training/runs/detect/${OUTPUT_MODEL_NAME}train.log 2>&1 &

while true
do
  if [ -f "/app/training/done.txt" ]; then
    break
  fi
  azcopy sync "/app/training/runs" "https://${STORAGE_NAME}.file.core.windows.net/runs${SASTOKEN}" --recursive=true
  sleep 60
done