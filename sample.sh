
divide_group='{"User": 2,"Time" : 2,"POI":2}'
dataset="NYC"

save_dir="output/${dataset}/sample"

mkdir -p "$save_dir"

python run_nash_divide.py --dataset "$dataset"  --save_dir "$save_dir"  --divide_group  "$divide_group" --divide_epoch 20 --accu_loss 20 --lrdecay 0.75 --deviceID 0
