# --trained_model_path models/lsun_horse.pt
NAME="ffhq64x64"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True --dropout 0.1"
TRAIN_FLAGS="--lr 1e-4 --batch_size 96 --log_interval 10 --save_interval 10000 --lr_warmup_steps 0"
CHECKPOINT_DIR="./checkpoints/$NAME" OPENAI_LOGDIR="./output/$NAME" python image_train.py --data_dir /data/FFHQ/images1024x1024 $MODEL_FLAGS $TRAIN_FLAGS
