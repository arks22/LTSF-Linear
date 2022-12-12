# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=2880
pred_len=60
model_name=DLinear

python3 -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/bitstamp \
  --data_path bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv\
  --model_id BitStamp_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --pred_len $pred_len\
  --enc_in 6 \
  --des 'Exp' \
  --target Weighted_Price \
  --itr 1 \
  --train_epochs 100 \
  --batch_size 1024 \
  --num_workers 16 \
  --date_type unix \
  --learning_rate 0.0005
