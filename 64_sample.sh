arr=($GPUS)
NGPUS=${#arr[@]}

echo "gpus: $GPUS"
echo "num gpus: $NGPUS"

FILES="
ema_0.9999_140000.pt
ema_0.9999_150000.pt
ema_0.9999_160000.pt
ema_0.9999_170000.pt
ema_0.9999_040000.pt
ema_0.9999_050000.pt
ema_0.9999_060000.pt
ema_0.9999_070000.pt
ema_0.9999_080000.pt
ema_0.9999_090000.pt
"
for f in $FILES
do
	export OPENAI_LOGDIR="./samples-ffhq64-notfixseed-corr/$f"
	idx=0
	for gpuid in $GPUS
	do
		idx=$(($idx+1)) 
		echo $(($idx-1))
		(MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True  --timestep_respacing 250 --num_samples 250 --batch_size 16 --idx 0"
		GPU_IDS=$gpuid python image_sample.py $MODEL_FLAGS --model_path output/ffhq64x64-2/$f $SAMPLE_FLAGS --idx $idx) &
	done
	wait
done
