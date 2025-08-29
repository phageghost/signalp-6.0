export KMP_DUPLICATE_LIB_OK=TRUE 
/Users/dylanskola/micromamba/envs/signalp/bin/python3 scripts/train_model.py \
  --data data/train_set.fasta \
  --test_partition 0 \
  --validation_partition 1 \
  --output_dir testruns \
  --experiment_name test_logging_3 \
  --sp_region_labels \
  --region_regularization_alpha 0.5 \
  --average_per_kingdom \
  --lr 0.001 \
  --optimizer adam \
  --batch_size 20 \
  --clip 1.0 \
