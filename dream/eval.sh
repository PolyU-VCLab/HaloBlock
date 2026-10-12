############################################### gsm8k evaluations ###############################################


## Original dllm
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy",method="vanilla" \
#     --tasks gsm8k_cot_zeroshot \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path output_reproduce/gsm8k \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

## dParallel
# CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained="Zigeng/dParallel_Dream_7B_Instruct",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg="entropy_threshold",dParallel=True,threshold=0.45,method="dparallel" \
#     --tasks gsm8k_cot_zeroshot \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 5 \
#     --output_path output_reproduce/gsm8k \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

## dParallel + haloblock
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained="Zigeng/dParallel_Dream_7B_Instruct",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg="entropy_threshold",dParallel=True,dual_cache=True,threshold=0.45,halo_factor=2,method="halo_dparallel" \
    --tasks gsm8k_cot_zeroshot \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 5 \
    --output_path output_reproduce/gsm8k \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template

## fastdllm
# CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg=confidence_threshold,dParallel=True,dual_cache=True,threshold=0.9,method="fastdllm" \
#     --tasks gsm8k_cot_zeroshot \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 5 \
#     --output_path output_reproduce/gsm8k \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

## adablock
# CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg=confidence_threshold,dParallel=True,dual_cache=True,threshold=0.9,method="adablock" \
#     --tasks gsm8k_cot_zeroshot \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 5 \
#     --output_path output_reproduce/gsm8k \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

## d3llm
# CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained="d3LLM/d3LLM_Dream",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg="entropy_threshold",dParallel=True,dual_cache=True,threshold=0.4,method="d3llm" \
#     --tasks gsm8k_cot_zeroshot \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 5 \
#     --output_path output_reproduce/gsm8k \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

# d2f
# accelerate launch eval_dream.py --model dream_lora \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,lora_path=SJTU-DENG-Lab/D2F_Dream_Instruct_7B_Lora,max_new_tokens=256,diffusion_steps=256,add_bos_token=true,temperature=0.,block_size=32,block_add_threshold=0.1,skip_threshold=0.9,decoded_token_threshold=0.95,dtype=bfloat16,sampling_strategy="default", \
#     --tasks gsm8k_cot \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 5 \
#     --output_path output_reproduce/gsm8k \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template


############################################### minerva_math evaluations ###############################################

## Original dllm
# CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy",method="vanilla" \
#     --tasks minerva_math \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 4 \
#     --output_path output_reproduce/math \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

## dParallel
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained="Zigeng/dParallel_Dream_7B_Instruct",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg="entropy_threshold",dParallel=True,dual_cache=True,threshold=0.45,method="dparallel" \
#     --tasks minerva_math \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 4 \
#     --output_path output_reproduce/math \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

## dParallel + haloblock
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained="Zigeng/dParallel_Dream_7B_Instruct",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg="entropy_threshold",dParallel=True,dual_cache=True,threshold=0.45,halo_factor=2,method="halo_dparallel" \
#     --tasks minerva_math \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 4 \
#     --output_path output_reproduce/math \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

## fastdllm
# CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg=confidence_threshold,dParallel=True,dual_cache=True,threshold=0.9,method="fastdllm" \
#     --tasks minerva_math \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 4 \
#     --output_path output_reproduce/math \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

## adablock
# CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg=confidence_threshold,dParallel=True,dual_cache=True,threshold=0.9,method="adablock" \
#     --tasks minerva_math \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 4 \
#     --output_path output_reproduce/math \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

## d3llm
# CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=d3LLM/d3LLM_Dream,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg="entropy_threshold",dParallel=True,dual_cache=True,threshold=0.4,method="d3llm" \
#     --tasks minerva_math \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 4 \
#     --output_path output_reproduce/math \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

# d2f
# accelerate launch eval_dream.py --model dream_lora \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,lora_path=SJTU-DENG-Lab/D2F_Dream_Instruct_7B_Lora,max_new_tokens=256,diffusion_steps=256,add_bos_token=true,temperature=0.,block_size=32,block_add_threshold=0.1,skip_threshold=0.9,decoded_token_threshold=0.95,dtype=bfloat16,sampling_strategy="default", \
#     --tasks minerva_math \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 4 \
#     --output_path output_reproduce/math \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

############################################### humaneval evaluations ###############################################

#export HF_ALLOW_CODE_EVAL=1

## Original dllm
# CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy",method="vanilla" \
#     --tasks humaneval_instruct \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path output_reproduce/humaneval \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

## dParallel
# CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained="Zigeng/dParallel_Dream_7B_Instruct",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg="entropy_threshold",dParallel=True,dual_cache=True,threshold=0.5,method="dparallel" \
#     --tasks humaneval_instruct \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path output_reproduce/humaneval \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

## dParallel + haloblock
# CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained="Zigeng/dParallel_Dream_7B_Instruct",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg="entropy_threshold",dParallel=True,dual_cache=True,threshold=0.5,halo_factor=2,method="halo_dparallel" \
#     --tasks humaneval_instruct \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path output_reproduce/humaneval \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

## fastdllm
# CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg=confidence_threshold,dParallel=True,dual_cache=True,threshold=0.9,method="fastdllm" \
#     --tasks humaneval_instruct \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path output_reproduce/humaneval \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

# adablock
# CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained="Dream-org/Dream-v0-Instruct-7B",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg="confidence_threshold",dParallel=True,dual_cache=True,threshold=0.9,method="adablock" \
#     --tasks humaneval_instruct \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path output_reproduce/humaneval \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

## d3llm
# CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=d3LLM/d3LLM_Dream,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.1,alg="entropy_threshold",dParallel=True,dual_cache=True,threshold=0.5,method="d3llm" \
#     --tasks humaneval_instruct \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path output_reproduce/humaneval \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

# d2f
# accelerate launch eval_dream.py --model dream_lora \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,lora_path=SJTU-DENG-Lab/D2F_Dream_Instruct_7B_Lora,max_new_tokens=256,diffusion_steps=256,add_bos_token=true,temperature=0.,block_size=32,block_add_threshold=0.9,skip_threshold=0.95,decoded_token_threshold=0.95,dtype=bfloat16,sampling_strategy="default", \
#     --tasks humaneval_instruct \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path output_reproduce/humaneval \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

############################################### mbpp evaluations ###############################################

#export HF_ALLOW_CODE_EVAL=1
## Original dllm
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy",method="vanilla" \
#     --tasks mbpp_instruct \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path output_reproduce/mbpp \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

## dParallel
# CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained="Zigeng/dParallel_Dream_7B_Instruct",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg="entropy_threshold",dParallel=True,dual_cache=True,threshold=0.5,method="dparallel" \
#     --tasks mbpp_instruct \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 3 \
#     --output_path output_reproduce/mbpp \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

## dParallel + haloblock
# CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained="Zigeng/dParallel_Dream_7B_Instruct",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg="entropy_threshold",dParallel=True,dual_cache=True,threshold=0.5,halo_factor=2,method="halo_dparallel" \
#     --tasks mbpp_instruct \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 3 \
#     --output_path output_reproduce/mbpp \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

## fastdllm
# CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg=confidence_threshold,dParallel=True,dual_cache=True,threshold=0.9,method="fastdllm" \
#     --tasks mbpp_instruct \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 3 \
#     --output_path output_reproduce/mbpp \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

# adablock
# CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained="Dream-org/Dream-v0-Instruct-7B",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg=confidence_threshold,dParallel=True,dual_cache=True,threshold=0.9,method="adablock" \
#     --tasks mbpp_instruct \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 3 \
#     --output_path output_reproduce/mbpp \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

# d3llm
# CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained="d3LLM/d3LLM_Dream",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg="entropy_threshold",dParallel=True,dual_cache=True,threshold=0.4,method="d3llm" \
#     --tasks mbpp_instruct \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 3 \
#     --output_path output_reproduce/mbpp \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

# d2f
# accelerate launch eval_dream.py --model dream_lora \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,lora_path=SJTU-DENG-Lab/D2F_Dream_Instruct_7B_Lora,max_new_tokens=256,diffusion_steps=256,add_bos_token=true,temperature=0.,block_size=32,block_add_threshold=0.1,skip_threshold=0.9,decoded_token_threshold=0.95,dtype=bfloat16,sampling_strategy="default", \
#     --tasks mbpp_instruct \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 3 \
#     --output_path output_reproduce/mbpp \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template







