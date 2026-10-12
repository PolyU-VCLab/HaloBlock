## Original dllm
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
#     --tasks minerva_math \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path output_reproduce/math \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

## our dParallel
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained="Zigeng/dParallel_Dream_7B_Instruct",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg="entropy_threshold",dParallel=True,dual_cache=True,threshold=0.45 \
    --tasks minerva_math \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 4 \
    --output_path output_reproduce/math \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template
