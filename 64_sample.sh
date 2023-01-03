FILES="
model020000.pt
ema_0.9999_020000.pt
"
for f in $FILES
do
	export OPENAI_LOGDIR="./samples-ffhq64/$f"
	MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True  --timestep_respacing 250 --num_samples 16 --batch_size 16 --idx 0"
	python image_sample.py $MODEL_FLAGS --model_path output/ffhq64x64/$f $SAMPLE_FLAGS
done
