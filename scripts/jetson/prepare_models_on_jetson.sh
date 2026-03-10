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

ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" "\
  set -euo pipefail; \
  mkdir -p '$REMOTE_DIR/jetson/models/assets' '$REMOTE_DIR/jetson/models/pretrained_models' '$REMOTE_DIR/jetson/models/G2PWModel' '$REMOTE_DIR/jetson/models/nltk_data' '$REMOTE_DIR/jetson/cache'; \
  cd '$REMOTE_DIR/jetson/cache'; \
  wget -c -O pretrained_models.zip '$PRETRAINED_URL'; \
  wget -c -O G2PWModel.zip '$G2PW_URL'; \
  wget -c -O nltk_data.zip '$NLTK_URL'; \
  wget -c -O averaged_perceptron_tagger.zip '$NLTK_TAGGER_URL'; \
  wget -c -O averaged_perceptron_tagger_eng.zip '$NLTK_TAGGER_ENG_URL'; \
  wget -c -O open_jtalk_dic_utf_8-1.11.tar.gz '$OPENJTALK_URL'; \
  rm -rf '$REMOTE_DIR/jetson/models/pretrained_models'/*; \
  rm -rf '$REMOTE_DIR/jetson/models/G2PWModel'/*; \
  rm -rf '$REMOTE_DIR/jetson/models/nltk_data'/*; \
  unzip -oq pretrained_models.zip -d '$REMOTE_DIR/jetson/cache/pretrained_unpack'; \
  if [ -d '$REMOTE_DIR/jetson/cache/pretrained_unpack/pretrained_models' ]; then cp -a '$REMOTE_DIR/jetson/cache/pretrained_unpack/pretrained_models/.' '$REMOTE_DIR/jetson/models/pretrained_models/'; else cp -a '$REMOTE_DIR/jetson/cache/pretrained_unpack/.' '$REMOTE_DIR/jetson/models/pretrained_models/'; fi; \
  rm -rf '$REMOTE_DIR/jetson/cache/pretrained_unpack'; \
  unzip -oq G2PWModel.zip -d '$REMOTE_DIR/jetson/cache/g2pw_unpack'; \
  if [ -d '$REMOTE_DIR/jetson/cache/g2pw_unpack/G2PWModel' ]; then cp -a '$REMOTE_DIR/jetson/cache/g2pw_unpack/G2PWModel/.' '$REMOTE_DIR/jetson/models/G2PWModel/'; else cp -a '$REMOTE_DIR/jetson/cache/g2pw_unpack/.' '$REMOTE_DIR/jetson/models/G2PWModel/'; fi; \
  rm -rf '$REMOTE_DIR/jetson/cache/g2pw_unpack'; \
  unzip -oq nltk_data.zip -d '$REMOTE_DIR/jetson/cache/nltk_unpack'; \
  if [ -d '$REMOTE_DIR/jetson/cache/nltk_unpack/nltk_data' ]; then cp -a '$REMOTE_DIR/jetson/cache/nltk_unpack/nltk_data/.' '$REMOTE_DIR/jetson/models/nltk_data/'; else cp -a '$REMOTE_DIR/jetson/cache/nltk_unpack/.' '$REMOTE_DIR/jetson/models/nltk_data/'; fi; \
  rm -rf '$REMOTE_DIR/jetson/cache/nltk_unpack'; \
  mkdir -p '$REMOTE_DIR/jetson/models/nltk_data/taggers'; \
  cp -f averaged_perceptron_tagger.zip '$REMOTE_DIR/jetson/models/nltk_data/taggers/averaged_perceptron_tagger.zip'; \
  cp -f averaged_perceptron_tagger_eng.zip '$REMOTE_DIR/jetson/models/nltk_data/taggers/averaged_perceptron_tagger_eng.zip'; \
  unzip -oq averaged_perceptron_tagger.zip -d '$REMOTE_DIR/jetson/models/nltk_data/taggers'; \
  unzip -oq averaged_perceptron_tagger_eng.zip -d '$REMOTE_DIR/jetson/models/nltk_data/taggers'; \
  cp -f open_jtalk_dic_utf_8-1.11.tar.gz '$REMOTE_DIR/jetson/models/assets/open_jtalk_dic_utf_8-1.11.tar.gz'; \
  find '$REMOTE_DIR/jetson/models' -maxdepth 2 -type f | sed -n '1,80p'"
