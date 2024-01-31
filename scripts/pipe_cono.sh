export CUDA_VISIBLE_DEVICES=0

python exp_pipe.py \
  --data-path data/Pipe \
  --ntrain 1000 \
  --ntest 200 \
  --ntotal 1200 \
  --in_dim 2 \
  --out_dim 1 \
  --h 129 \
  --w 129 \
  --h-down 1 \
  --w-down 1 \
  --batch-size 40 \
  --learning-rate 0.0005 \
  --model complex_FFNO_2D \
  --d-model 64 \
  --num-basis 12 \
  --num-token 4 \
  --patch-size 9,9 \
  --padding 15,15 \
  --model-save-path ./checkpoints/pipe \
  --model-save-name cono_pipe.pt 