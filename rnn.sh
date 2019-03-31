test_path=$1
pred_path=$2
python3.7 ./src/predict.py ./models/lstm/ --test_path=$1 --pred_path=$2
