#############################################
# Load environment variables
#############################################
ENV_FILE="${ENV_FILE:-../.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  . "$ENV_FILE"
  set +a
fi

# Test standard 3D generation loader (with optional 2D renderings)
python standard_3dgen_loader.py \
    --obj_list $ALL_OBJ_JSON \
    --gs_path $GS_PATH \
    --caption_path $CAPTIONS_PATH \
    --rendering_path $RENDERINGS_DIR \
    --num_images 1 \
    --mean_file $GS_MEAN_FILE \
    --std_file $GS_STD_FILE \
    --batch_size 2 \
    --num_workers 0

# Test fast 3D generation loader (WebDataset format)
# python fast_3dgen_loader.py \
#     --shard_pattern "$SHARD_PATTERN" \
#     --mean_file $MEAN_FILE \
#     --std_file $STD_FILE \
#     --batch_size 5 \
#     --repeat \
#     --num_workers 0