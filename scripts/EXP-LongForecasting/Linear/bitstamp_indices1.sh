# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=1440
pred_len=60
model_name=DLinear

python3 -u run_longExp.py \
  --is_training 0 \
  --root_path ./dataset/bitstamp \
  --data_path bitstamp_indices1.csv\
  --model_id BitStamp_indices1_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --pred_len $pred_len\
  --enc_in 38 \
  --des 'Exp' \
  --target Weighted_Price \
  --itr 1 \
  --train_epochs 20 \
  --batch_size 1024 \
  --num_workers 16 \
  --date_type date \
  --learning_rate 0.0005
