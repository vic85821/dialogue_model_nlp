test_path=$1
pred_path=$2
python3.7 ./src/predict.py ./models/gru_attention/ --test_path=$1 --pred_path=$2