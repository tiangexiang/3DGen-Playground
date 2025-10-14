#############################################
# Load environment variables
#############################################
ENV_FILE="${ENV_FILE:-../.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  . "$ENV_FILE"
  set +a
fi

python make_webdataset.py \
    --gs_path $GS_PATH \
    --captions $CAPTIONS_PATH \
    --output_dir $WEBDATASET_DIR \
    --obj_list $ALL_OBJ_JSON \
    --log_level DEBUG

#### DEBUG ONLY ####
# python make_webdataset.py \
#     --gs_path $GS_PATH \
#     --captions $CAPTIONS_PATH \
#     --output_dir $WEBDATASET_DIR \
#     --obj_list $ALL_OBJ_JSON \
#     --shard_size 20 \
#     --max_shards 5 \
#     --log_level DEBUG