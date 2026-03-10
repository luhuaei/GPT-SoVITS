#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST="${JETSON_HOST:-nvidia@192.168.1.230}"
REMOTE_DIR="${JETSON_DIR:-/home/nvidia/GPT-SoVITS}"
MODEL_SOURCE="${MODEL_SOURCE:-ModelScope}"
NLTK_TAGGER_URL="https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger.zip"
NLTK_TAGGER_ENG_URL="https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger_eng.zip"

case "$MODEL_SOURCE" in
  HF)
    PRETRAINED_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
    G2PW_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
    NLTK_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
    OPENJTALK_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
    ;;
  HF-Mirror)
    PRETRAINED_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
    G2PW_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
    NLTK_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
    OPENJTALK_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
    ;;
  ModelScope)
    PRETRAINED_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/pretrained_models.zip"
    G2PW_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip"
    NLTK_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/nltk_data.zip"
    OPENJTALK_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/open_jtalk_dic_utf_8-1.11.tar.gz"
    ;;
  *)
    echo "Unsupported MODEL_SOURCE: $MODEL_SOURCE"
    exit 1
    ;;
esac

"$SCRIPT_DIR/sync_to_jetson.sh"

ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" bash -s -- \
  "$REMOTE_DIR" \
  "$PRETRAINED_URL" \
  "$G2PW_URL" \
  "$NLTK_URL" \
  "$NLTK_TAGGER_URL" \
  "$NLTK_TAGGER_ENG_URL" \
  "$OPENJTALK_URL" <<'REMOTE'
set -euo pipefail

REMOTE_DIR="$1"
PRETRAINED_URL="$2"
G2PW_URL="$3"
NLTK_URL="$4"
NLTK_TAGGER_URL="$5"
NLTK_TAGGER_ENG_URL="$6"
OPENJTALK_URL="$7"

MODEL_ROOT="$REMOTE_DIR/jetson/models"
CACHE_ROOT="$REMOTE_DIR/jetson/cache"
PRETRAINED_ROOT="$MODEL_ROOT/pretrained_models"
PRETRAINED_CACHE="$CACHE_ROOT/pretrained_unpack"
G2PW_CACHE="$CACHE_ROOT/g2pw_unpack"
NLTK_CACHE="$CACHE_ROOT/nltk_unpack"

mkdir -p "$MODEL_ROOT/assets" "$PRETRAINED_ROOT" "$MODEL_ROOT/G2PWModel" "$MODEL_ROOT/nltk_data" "$CACHE_ROOT"
cd "$CACHE_ROOT"

download_if_missing() {
  local target="$1"
  local url="$2"
  if [ -s "$target" ]; then
    echo "Reusing cached artifact: $target"
    return
  fi
  wget -O "$target" "$url"
}

download_if_missing pretrained_models.zip "$PRETRAINED_URL"
download_if_missing G2PWModel.zip "$G2PW_URL"
download_if_missing nltk_data.zip "$NLTK_URL"
download_if_missing averaged_perceptron_tagger.zip "$NLTK_TAGGER_URL"
download_if_missing averaged_perceptron_tagger_eng.zip "$NLTK_TAGGER_ENG_URL"
download_if_missing open_jtalk_dic_utf_8-1.11.tar.gz "$OPENJTALK_URL"

rm -rf "$PRETRAINED_ROOT"/* "$MODEL_ROOT/G2PWModel"/* "$MODEL_ROOT/nltk_data"/*
rm -rf "$PRETRAINED_CACHE" "$G2PW_CACHE" "$NLTK_CACHE"
mkdir -p "$PRETRAINED_CACHE" "$G2PW_CACHE" "$NLTK_CACHE"

unzip -oq pretrained_models.zip -d "$PRETRAINED_CACHE"
PRETRAINED_SRC="$PRETRAINED_CACHE"
if [ -d "$PRETRAINED_CACHE/pretrained_models" ]; then
  PRETRAINED_SRC="$PRETRAINED_CACHE/pretrained_models"
fi

mkdir -p \
  "$PRETRAINED_ROOT/gsv-v2final-pretrained" \
  "$PRETRAINED_ROOT/chinese-hubert-base" \
  "$PRETRAINED_ROOT/chinese-roberta-wwm-ext-large" \
  "$PRETRAINED_ROOT/fast_langdetect"

cp -a "$PRETRAINED_SRC/chinese-hubert-base/." "$PRETRAINED_ROOT/chinese-hubert-base/"
cp -a "$PRETRAINED_SRC/chinese-roberta-wwm-ext-large/." "$PRETRAINED_ROOT/chinese-roberta-wwm-ext-large/"
cp -a "$PRETRAINED_SRC/fast_langdetect/." "$PRETRAINED_ROOT/fast_langdetect/"
cp -f "$PRETRAINED_SRC/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt" \
  "$PRETRAINED_ROOT/gsv-v2final-pretrained/"
cp -f "$PRETRAINED_SRC/gsv-v2final-pretrained/s2G2333k.pth" \
  "$PRETRAINED_ROOT/gsv-v2final-pretrained/"
rm -rf "$PRETRAINED_CACHE"

unzip -oq G2PWModel.zip -d "$G2PW_CACHE"
if [ -d "$G2PW_CACHE/G2PWModel" ]; then
  cp -a "$G2PW_CACHE/G2PWModel/." "$MODEL_ROOT/G2PWModel/"
else
  cp -a "$G2PW_CACHE/." "$MODEL_ROOT/G2PWModel/"
fi
rm -rf "$G2PW_CACHE"

unzip -oq nltk_data.zip -d "$NLTK_CACHE"
NLTK_SRC="$NLTK_CACHE"
if [ -d "$NLTK_CACHE/nltk_data" ]; then
  NLTK_SRC="$NLTK_CACHE/nltk_data"
fi

mkdir -p \
  "$MODEL_ROOT/nltk_data/corpora/cmudict" \
  "$MODEL_ROOT/nltk_data/taggers/averaged_perceptron_tagger" \
  "$MODEL_ROOT/nltk_data/taggers/averaged_perceptron_tagger_eng"

if [ -d "$NLTK_SRC/corpora/cmudict" ]; then
  cp -a "$NLTK_SRC/corpora/cmudict/." "$MODEL_ROOT/nltk_data/corpora/cmudict/"
fi
if [ -f "$NLTK_SRC/corpora/cmudict.zip" ]; then
  cp -f "$NLTK_SRC/corpora/cmudict.zip" "$MODEL_ROOT/nltk_data/corpora/cmudict.zip"
fi
rm -rf "$NLTK_CACHE"

unzip -oq averaged_perceptron_tagger.zip -d "$MODEL_ROOT/nltk_data/taggers"
unzip -oq averaged_perceptron_tagger_eng.zip -d "$MODEL_ROOT/nltk_data/taggers"
cp -f averaged_perceptron_tagger.zip "$MODEL_ROOT/nltk_data/taggers/averaged_perceptron_tagger.zip"

cp -f open_jtalk_dic_utf_8-1.11.tar.gz "$MODEL_ROOT/assets/open_jtalk_dic_utf_8-1.11.tar.gz"

du -sh "$PRETRAINED_ROOT" "$MODEL_ROOT/G2PWModel" "$MODEL_ROOT/nltk_data"
find "$MODEL_ROOT" -maxdepth 3 -type f | sort | sed -n '1,120p'
REMOTE
