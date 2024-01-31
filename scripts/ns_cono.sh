export CUDA_VISIBLE_DEVICES=0

python exp_ns.py \
  --data-path data/NavierStokes_V1e-5_N1200_T20 \
  --ntrain 1000 \
  --ntest 200 \
  --ntotal 1200 \
  --in_dim 10 \
  --out_dim 1 \
  --h 64 \
  --w 64 \
  --h-down 1 \
  --w-down 1 \
  --T-in 10 \
  --T-out 10 \
  --batch-size 20 \
  --learning-rate 0.001 \
  --model complex_FFNO_2D \
  --d-model 64 \
  --num-basis 12 \
  --num-token 4 \
  --patch-size 4,4 \
  --padding 0,0 \
  --model-save-path ./checkpoints/ns \
  --model-save-name ns.pt