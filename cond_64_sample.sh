pathmodel="models"
FILES="
64x64_diffusion.pt
"
GPUS="
2
3
4
5
"
arr=($GPUS)

NGPUS=${#arr[@]}

echo "gpus: $GPUS"
echo "num gpus: $NGPUS"

for f in $FILES
do
	export OPENAI_LOGDIR="./samples-imagenet64-1k/$f"
	idx=0
	for gpuid in $GPUS
	do
		idx=$(($idx+1)) 
		echo $(($idx-1))
		(MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True  --timestep_respacing 250 --batch_size 32 --data_dir /data/yzeng22/imagenetval_sample1k --imagenet True"
		GPU_IDS=$gpuid python val_classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt --classifier_depth 4 --model_path $pathmodel/$f --n_split $NGPUS --i_split $(($idx-1)) ) &
	done
done
