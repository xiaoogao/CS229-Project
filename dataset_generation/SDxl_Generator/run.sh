#!/bin/bash

# logging
mkdir -p logs

# Parallel execution of 4 batches
for i in 0 1 2 3
do
  echo "Launching batch $i on GPU $i..."
  CUDA_VISIBLE_DEVICES=$i nohup python generate_images.py $i > logs/batch_$i.log 2>&1 &
done

# Wait for all background jobs to finish
wait

# Move data after all jobs are done
mv SDxl_data ..

echo "All nohup jobs started. Check logs/ for output."
