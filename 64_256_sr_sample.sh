pathmodel="output/imagenet64x64_sr256_finetune"
FILES="
ema_0.9999_110000.pt
"
#pathmodel="models"
#FILES="
#64_256_upsampler.pt
#"
pathdata="../latent-diffusion/data/autoencoders/data/ILSVRC2012_validation/data"
pathdata="samples-imagenet64/64x64_diffusion.pt/output/n01440764"
for f in $FILES
do
	export OPENAI_LOGDIR="./samples-sr_imagenet/$f"
	MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --small_size 64 --large_size 256 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True  --timestep_respacing 50 --batch_size 8 --data_dir $pathdata --use_ddim True"
	python super_res_sample.py $MODEL_FLAGS --model_path $pathmodel/$f $SAMPLE_FLAGS
done
