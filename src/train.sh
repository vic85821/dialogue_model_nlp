model_path=$1
cuda=$2
python train.py $model_path --device "cuda:$2"