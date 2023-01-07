pathmodel="output/ffhq64x64_sr256_finetune"
FILES="
ema_0.9999_120000.pt
"
#pathmodel="models"
#FILES="
#64_256_upsampler.pt
#"
LRS="
ema_0.9999_090000.pt
ema_0.9999_080000.pt
ema_0.9999_070000.pt
ema_0.9999_020000.pt
ema_0.9999_030000.pt
ema_0.9999_040000.pt
ema_0.9999_050000.pt
ema_0.9999_060000.pt
"
pathdata="samples-ffhq64"
for lr in $LRS
do
	for f in $FILES
	do
		export OPENAI_LOGDIR="./samples-sr_ffhq_sr$f/$lr"
		MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --small_size 64 --large_size 256 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True  --timestep_respacing 50 --batch_size 1 --data_dir $pathdata/$lr/output --use_ddim True --return_prefix False"
		python super_res_sample.py $MODEL_FLAGS --model_path $pathmodel/$f $SAMPLE_FLAGS
	done
done
