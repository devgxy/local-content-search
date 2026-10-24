#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE="${HF_ENDPOINT:-https://hf-mirror.com}"
EMBED_DIR=".models/bge-large-zh-v1.5"
RERANK_DIR=".models/bge-reranker-large"

mkdir -p "$EMBED_DIR/1_Pooling" "$RERANK_DIR"

size_of() {
  if stat -f%z "$1" >/dev/null 2>&1; then
    stat -f%z "$1"
  else
    stat -c%s "$1"
  fi
}

fetch() {
  local repo="$1" file="$2" dest="$3" min="${4:-1}"
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ] && [ "$(size_of "$dest")" -ge "$min" ]; then
    echo "skip  $file"
    return 0
  fi
  echo "fetch $repo/$file"
  curl -fL --retry 3 --retry-delay 2 -o "$dest" "$BASE/$repo/resolve/main/$file"
}

echo "镜像：$BASE"

fetch "BAAI/bge-large-zh-v1.5" "config.json" "$EMBED_DIR/config.json"
fetch "BAAI/bge-large-zh-v1.5" "tokenizer_config.json" "$EMBED_DIR/tokenizer_config.json"
fetch "BAAI/bge-large-zh-v1.5" "tokenizer.json" "$EMBED_DIR/tokenizer.json"
fetch "BAAI/bge-large-zh-v1.5" "vocab.txt" "$EMBED_DIR/vocab.txt"
fetch "BAAI/bge-large-zh-v1.5" "special_tokens_map.json" "$EMBED_DIR/special_tokens_map.json"
fetch "BAAI/bge-large-zh-v1.5" "sentence_bert_config.json" "$EMBED_DIR/sentence_bert_config.json"
fetch "BAAI/bge-large-zh-v1.5" "modules.json" "$EMBED_DIR/modules.json"
fetch "BAAI/bge-large-zh-v1.5" "1_Pooling/config.json" "$EMBED_DIR/1_Pooling/config.json"
fetch "BAAI/bge-large-zh-v1.5" "pytorch_model.bin" "$EMBED_DIR/pytorch_model.bin" 1000000000

fetch "BAAI/bge-reranker-large" "config.json" "$RERANK_DIR/config.json"
fetch "BAAI/bge-reranker-large" "tokenizer_config.json" "$RERANK_DIR/tokenizer_config.json"
fetch "BAAI/bge-reranker-large" "tokenizer.json" "$RERANK_DIR/tokenizer.json"
fetch "BAAI/bge-reranker-large" "sentencepiece.bpe.model" "$RERANK_DIR/sentencepiece.bpe.model"
fetch "BAAI/bge-reranker-large" "special_tokens_map.json" "$RERANK_DIR/special_tokens_map.json"
fetch "BAAI/bge-reranker-large" "pytorch_model.bin" "$RERANK_DIR/pytorch_model.bin" 1500000000

EMBED_SIZE="$(size_of "$EMBED_DIR/pytorch_model.bin")"
RERANK_SIZE="$(size_of "$RERANK_DIR/pytorch_model.bin")"
if [ "$EMBED_SIZE" -lt 1000000000 ] || [ "$RERANK_SIZE" -lt 1500000000 ]; then
  echo "模型文件不完整：embed=$EMBED_SIZE rerank=$RERANK_SIZE" >&2
  exit 1
fi

cat > .models/READY.json <<'JSON'
{
  "embed": "bge-large-zh-v1.5",
  "rerank": "bge-reranker-large",
  "dim": 1024,
  "verified_at": "2026-10-21"
}
JSON

echo "模型就绪：embed=${EMBED_SIZE}B rerank=${RERANK_SIZE}B"
