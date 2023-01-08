arr=($GPUS)
NGPUS=${#arr[@]}

echo "gpus: $GPUS"
echo "num gpus: $NGPUS"

#pathmodel="output/imagenet64x64_sr256_finetune"
#FILES="
#ema_0.9999_110000.pt
#"
pathmodel="models"
FILES="
64_256_upsampler.pt
"
pathdata="samples-imagenet64-1k/64x64_diffusion.pt/output"
for f in $FILES
do
	export OPENAI_LOGDIR="./samples-sr_imagenet-1k_pretrain/$f"
	idx=0
	for gpuid in $GPUS
	do
		idx=$(($idx+1)) 
		echo $(($idx-1))
		(MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --small_size 64 --large_size 256 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True  --timestep_respacing 50 --batch_size 32 --data_dir $pathdata --use_ddim True"
		GPU_IDS=$gpuid python super_res_sample.py $MODEL_FLAGS --model_path $pathmodel/$f $SAMPLE_FLAGS --n_split $NGPUS --i_split $(($idx-1)) ) &
	done
done
