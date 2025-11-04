OUTPUT_DIR=generate_sft_qwen3_14b

mkdir $OUTPUT_DIR

CUDA_VISIBLE_DEVICES="0" python generate_sft.py --model_name_or_path Qwen/Qwen3-14B --output_dir $OUTPUT_DIR