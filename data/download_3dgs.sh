#############################################
# Load environment variables
#############################################
ENV_FILE="${ENV_FILE:-../.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  . "$ENV_FILE"
  set +a
fi

# Download GaussianVerse 3DGS files, reference: http://gaussianverse.stanford.edu
python3 download_3dgs.py \
    --save_dir $GS_PATH \
    --download_meta $DOWNLOAD_META \
    --download_aesthetic $DOWNLOAD_AESTHETIC \
    --download_non_aesthetic $DOWNLOAD_NON_AESTHETIC \

echo -e "\033[0;32mGaussianVerse download completed.\033[0m"


# Unzip downloaded GaussianVerese chunks
echo "start unzipping... this may take a while..."
for i in {0..8}; do
    zip_file="$GS_PATH/GaussianVerse_aesthetic_chunk_$i.zip"
    if [ -f "$zip_file" ]; then
        # Use 'zipinfo -1' for a fast check the file is readable instead of 'unzip -tqq'
        if zipinfo -1 "$zip_file" > /dev/null 2>&1; then
            echo "[OK] Unzipping GaussianVerse_aesthetic_chunk_$i.zip"
            unzip -q "$zip_file" -d "$GS_PATH/GaussianVerse_aesthetic_chunk_$i"
        else
            echo -e "\033[0;31m[ERROR] GaussianVerse_aesthetic_chunk_$i.zip is incomplete. Something is wrong with the download.\033[0m"
        fi
    else
        echo -e "\033[0;33m[WARNING] GaussianVerse_aesthetic_chunk_$i.zip not found.\033[0m"
    fi
done

for i in {0..17}; do
    zip_file="$GS_PATH/GaussianVerse_chunk_$i.zip"
    if [ -f "$zip_file" ]; then
        if unzip -tqq "$zip_file"; then
            echo "[OK] Unzipping GaussianVerse_chunk_$i.zip"
            unzip -q "$zip_file" -d "$GS_PATH/GaussianVerse_chunk_$i"
        else
            echo -e "\033[0;31m[ERROR] GaussianVerse_chunk_$i.zip is incomplete. Something is wrong with the download.\033[0m"
        fi
    else
        echo -e "\033[0;33m[WARNING] GaussianVerse_chunk_$i.zip not found.\033[0m"
    fi
done