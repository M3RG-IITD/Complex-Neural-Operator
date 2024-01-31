export CUDA_VISIBLE_DEVICES=1

python exp_elas.py \
  --data-path data/Meshes \
  --ntrain 1000 \
  --ntest 200 \
  --ntotal 2000 \
  --in_dim 2 \
  --out_dim 1 \
  --batch-size 20 \
  --learning-rate 0.0005 \
  --model complex_FFNO_2D_Irregular_Geo \
  --d-model 64 \
  --num-basis 12 \
  --num-token 4 \
  --patch-size 6,6 \
  --model-save-path ./checkpoints/elas \
  --model-save-name cono.pt 