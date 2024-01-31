export CUDA_VISIBLE_DEVICES=1

python exp_elas_interp.py \
  --data-path data/Interp \
  --ntrain 1000 \
  --ntest 200 \
  --ntotal 2000 \
  --in_dim 1 \
  --out_dim 1 \
  --h 41 \
  --w 41 \
  --h-down 1 \
  --w-down 1 \
  --batch-size 40 \
  --learning-rate 0.005 \
  --model complex_FFNO_2D \
  --d-model 64 \
  --num-basis 12 \
  --num-token 4 \
  --patch-size 3,3 \
  --padding 7,7 \
  --model-save-path ./checkpoints/elas_interp \
  --model-save-name cono.pt