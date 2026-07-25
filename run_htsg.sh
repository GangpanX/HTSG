source set_env.sh

echo "=========================================="
echo "Run HTSG Train"
echo "=========================================="

python train.py --dataset weibo --lr 0.0006 --dim 64 --num-layers 3 --save 1

python train.py --dataset twitter --lr 0.0006 --dim 16 --num-layers 2 --save 1

python train.py --dataset pheme --lr 0.0005 --dim 32 --dropout 0.35 --num-layers 3 --weight-decay 1e-4 --save 1
    
