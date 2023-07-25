# python train.py \
#     --dataset-name sarcasm \
#     --balance-unlabeled \
#     --epochs 2048 \
#     --n-test-per-class 500 \
#     --enable-mixmatch False \
#     --n-labeled 512 

python train.py \
    --dataset-name sarcasm \
    --balance-unlabeled \
    --epochs 512 \
    --n-test-per-class 500 \
    --enable-mixmatch False \
    --n-labeled 30000 

# python train.py \
#     --dataset-name sarcasm \
#     --balance-unlabeled \
#     --epochs 1024 \
#     --n-test-per-class 500 \
#     --n-labeled 512 