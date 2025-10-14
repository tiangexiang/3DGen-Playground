#############################################
# Load environment variables
#############################################
ENV_FILE="${ENV_FILE:-../.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  . "$ENV_FILE"
  set +a
fi

# Download GObjaverse renderings, reference: https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse
# THIS TAKES A LONG TIME AND REQUIRES A LOT OF DISK SPACE.
python3 download_renderings.py --json_path $ALL_OBJ_JSON --save_dir $RENDERINGS_DIR
echo -e "\033[0;32m2D Renderings downloaded.\033[0m"
