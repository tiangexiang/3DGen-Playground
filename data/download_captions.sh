#############################################
# Load environment variables
#############################################
ENV_FILE="${ENV_FILE:-../.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  . "$ENV_FILE"
  set +a
fi

# download 3DTopia captions, reference: https://github.com/3DTopia/3DTopia/releases
wget -O $TDTOPIA_CAPTION_PATH https://github.com/3DTopia/3DTopia/releases/download/data/3DTopia-objaverse-caption-361k.json
echo -e "\033[0;32m3DTopia captions downloaded.\033[0m"

# download Cap3D captions, reference: https://huggingface.co/datasets/tiange/Cap3D/tree/main
wget -O $CAP3D_CAPTION_PATH https://huggingface.co/datasets/tiange/Cap3D/resolve/main/Cap3D_automated_Objaverse_full.csv
echo -e "\033[0;32mCap3D captions downloaded.\033[0m"

# preprocess the captions
python3 preprocess_captions.py \
    --caption_path $TDTOPIA_CAPTION_PATH \
    --cap3d_caption_path $CAP3D_CAPTION_PATH \
    --json_path $ALL_OBJ_JSON \
    --output_path $CAPTIONS_PATH
    
echo -e "\033[0;32mCaptions preprocessed and ready to use!\033[0m"