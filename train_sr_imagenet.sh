NAME="imagenet64x64_sr256_finetune"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --large_size 256  --small_size 64 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --dropout 0.1"
TRAIN_FLAGS="--lr 1e-4 --batch_size 4 --log_interval 10 --save_interval 10000 --lr_warmup_steps 5000 --finetune models/64_256_upsampler.pt"
CHECKPOINT_DIR="./checkpoints/$NAME" OPENAI_LOGDIR="./output/$NAME" python super_res_train.py --data_dir /data/common/ILSVRC/Data/CLS-LOC/train $MODEL_FLAGS $TRAIN_FLAGS
