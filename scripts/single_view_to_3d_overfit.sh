#! /usr/bin/env bash
#
# Matrix3D single-view overfit runner (no source edits required).
# Usage:
#   bash scripts/single_view_to_3d_overfit.sh configs/single_view_overfit_manhattan.env
#
set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
REPO_DIR=$(dirname "$SCRIPT_DIR")
cd "$REPO_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/single_view_to_3d_overfit.sh <config.env>"
  exit 1
fi

CFG_PATH="$1"
if [[ ! -f "$CFG_PATH" ]]; then
  echo "Config file not found: $CFG_PATH"
  exit 1
fi

# shellcheck disable=SC1090
source "$CFG_PATH"

export NERFSTUDIO_METHOD_CONFIGS="splatfacto_matrix3d=splatfacto_matrix3d.splatfacto_configs:splatfacto_method"
export PYTHONPATH="$PYTHONPATH:$REPO_DIR"

NAME_EXT=$(basename "$INPUT_PATH")
NAME="${NAME_EXT%.*}"
SCENE_DIR="results/$EXP_NAME/$NAME"
LOG_DIR="$SCENE_DIR/logs"
mkdir -p "$LOG_DIR"

RUN_TS=$(date +"%Y%m%d-%H%M%S")
PARAM_LOG="$LOG_DIR/params-$RUN_TS.txt"

{
  echo "run_timestamp=$RUN_TS"
  echo "exp_name=$EXP_NAME"
  echo "input_path=$INPUT_PATH"
  echo "gpu_id=$GPU_ID"
  echo "mixed_precision=$MIXED_PRECISION"
  echo "default_fov=$DEFAULT_FOV"
  echo "num_samples=$NUM_SAMPLES"
  echo "random_seed=$RANDOM_SEED"
  echo "checkpoint_path=$CHECKPOINT_PATH"
  echo "stage3_config=$STAGE3_CONFIG"
  echo "iters=$ITERS"
  echo "num_img=$NUM_IMG"
  echo "sh_degree=$SH_DEGREE"
  echo "lpips_lambda=$LPIPS_LAMBDA"
  echo "depth_l1_lambda=$DEPTH_L1_LAMBDA"
  echo "depth_ranking_lambda=$DEPTH_RANKING_LAMBDA"
  echo "l1_captured=$L1_CAPTURED"
  echo "l1_generated=$L1_GENERATED"
  echo "save_every=$SAVE_EVERY"
} | tee "$PARAM_LOG"

echo "[1/3] Generating multiview observations..."
CUDA_VISIBLE_DEVICES="$GPU_ID" python pipeline_single_to_3d.py \
  --config "$STAGE3_CONFIG" \
  --exp_name "$EXP_NAME" \
  --data_path "$INPUT_PATH" \
  --default_fov "$DEFAULT_FOV" \
  --num_samples "$NUM_SAMPLES" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --mixed_precision "$MIXED_PRECISION" \
  --random_seed "$RANDOM_SEED" |& tee "$LOG_DIR/generation-$RUN_TS.log"

cd "$SCENE_DIR"
echo "[2/3] Overfit optimization with splatfacto_matrix3d..."
ns-train splatfacto_matrix3d \
  --data transforms_train.json \
  --mixed-precision False \
  --output-dir outputs \
  --timestamp "overfit-$RUN_TS" \
  --viewer.quit-on-train-completion True \
  --max-num-iterations "$ITERS" \
  --steps-per-save "$SAVE_EVERY" \
  --pipeline.model.num-downscales -1 \
  --pipeline.model.resolution-schedule 1000 \
  --pipeline.datamanager.max-num-iterations "$ITERS" \
  --pipeline.datamanager.num_image_each_iteration "$NUM_IMG" \
  --pipeline.model.background-color white \
  --pipeline.model.warmup-length 200 \
  --pipeline.model.densify-grad-thresh 0.0008 \
  --pipeline.model.cull-alpha-thresh 0.05 \
  --pipeline.model.cull-scale-thresh 0.5 \
  --pipeline.model.cull-screen-size 0.5 \
  --pipeline.model.reset-alpha-every 20 \
  --pipeline.model.refine-every 50 \
  --pipeline.model.use_scale_regularization True \
  --pipeline.model.max-gauss-ratio 3 \
  --pipeline.model.stop-screen-size-at 4000 \
  --pipeline.model.stop-split-at 1000 \
  --pipeline.model.sh-degree "$SH_DEGREE" \
  --pipeline.model.sh-degree-interval 500 \
  --pipeline.model.full-accumulation-lambda 0.0 \
  --pipeline.model.accumulation-lambda 5.0 \
  --pipeline.model.mask_lambda 5.0 \
  --pipeline.model.ssim-lambda 0.2 \
  --pipeline.model.lpips-lambda "$LPIPS_LAMBDA" \
  --pipeline.model.l1-lambda-on-captured-views "$L1_CAPTURED" \
  --pipeline.model.l1-lambda-on-generation-views "$L1_GENERATED" \
  --pipeline.model.apply-annealing False \
  --pipeline.model.rasterize-mode antialiased \
  --pipeline.model.use-absgrad False \
  --pipeline.model.lpips-downsample 1 \
  --pipeline.model.lpips-min-img-size 128 \
  --pipeline.model.lpips-patch-size 512 \
  --pipeline.model.lpips-no-resize True \
  --pipeline.model.depth-l1-lambda "$DEPTH_L1_LAMBDA" \
  --pipeline.model.depth-ranking-lambda "$DEPTH_RANKING_LAMBDA" \
  --pipeline.model.output-depth-during-training True \
  --pipeline.model.use-bilateral-grid False \
  nerfstudio-data --center-method none --orientation-method none --auto-scale-poses False --train-split-fraction 1.0 --load-3D-points True --depth-unit-scale-factor 1.0 |& tee "$LOG_DIR/train-$RUN_TS.log"

echo "[3/3] Rendering train-split frames and writing summary video..."
ns-render dataset \
  --load-config "outputs/splatfacto_matrix3d/overfit-$RUN_TS/config.yml" \
  --image-format png \
  --split=train \
  --output-path "renders-overfit-$RUN_TS" |& tee "$LOG_DIR/render-$RUN_TS.log"

python "$REPO_DIR/utils/write_videos.py" \
  --render_root "renders-overfit-$RUN_TS" \
  --type object |& tee "$LOG_DIR/video-$RUN_TS.log"

echo "Completed. Scene output: $SCENE_DIR"
