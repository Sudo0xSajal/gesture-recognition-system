#!/usr/bin/env bash
# Convert model to TFLite and deploy to Raspberry Pi.
# Usage: bash deploy_edge.sh -c log/<exp>/best_model.pt -m cnn
#        bash deploy_edge.sh -c log/<exp>/best_model.pt -m cnn \
#             -q float16 --host 192.168.1.100 --password raspberry --run
source "$(dirname "$0")/env.sh"
CKPT=""; MODEL="cnn"; QUANT="float16"; HOST=""; PASS=""; KEY=""; RUN=""; QUANTIZED=""
while [[ $# -gt 0 ]]; do
  case $1 in
    -c) CKPT="$2"; shift 2 ;;
    -m) MODEL="$2"; shift 2 ;;
    -q) QUANT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --password) PASS="$2"; shift 2 ;;
    --key) KEY="$2"; shift 2 ;;
    --run) RUN="--run"; shift ;;
    *) shift ;;
  esac
done
[ -z "$CKPT" ] && { echo "Usage: bash deploy_edge.sh -c <ckpt> -m <model>"; exit 1; }
[ "$QUANT" = "int8" ] && QUANTIZED="--quantized"
TFLITE="integrations/edge/model.tflite"

echo "[deploy_edge] Step 1: Convert to TFLite (${QUANT})"
python integrations/edge/convert_tflite.py \
  --checkpoint "$CKPT" --model "$MODEL" --quantize "$QUANT" --output "$TFLITE"

if [ -n "$HOST" ]; then
  echo ""
  echo "[deploy_edge] Step 2: Deploy to Pi @ $HOST"
  PASS_ARG=""; KEY_ARG=""
  [ -n "$PASS" ] && PASS_ARG="--password $PASS"
  [ -n "$KEY"  ] && KEY_ARG="--key $KEY"
  python integrations/edge/deploy_pi.py \
    --model-path "$TFLITE" --host "$HOST" $PASS_ARG $KEY_ARG $RUN $QUANTIZED
else
  echo "[deploy_edge] Model saved to: $TFLITE"
  echo "  To deploy: add --host <pi_ip> --password <password>"
fi
