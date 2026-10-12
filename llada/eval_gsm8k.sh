# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export NCCL_NVLS_ENABLE=0

task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))
factor=1.0
method='haloblock'
model_path='GSAI-ML/LLaDA-8B-Instruct'
#model_path='GSAI-ML/LLaDA-1.5'
#model_path='Zigeng/dParallel-LLaDA-8B-instruct'
#model_path='d3LLM/d3LLM_LLaDA'


# # fast dllm baseline

# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True

# # fast dllm prefix cache
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,threshold=0.9,show_speed=True,method='fastdllm'

# # fast dllm prefix cache + haloblock
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},halo_factor=2.0,use_cache=True,threshold=0.9,show_speed=True,method='haloblock'

# # fast dllm dual cache
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},window_radius=${window_radius},left_radius=0,right_radius=0,use_cache=True,dual_cache=True,threshold=0.9,show_speed=True,method='fastdllm'

# # fast dllm dual cache factor
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},halo_factor=2.0,use_cache=True,dual_cache=True,factor=${factor},show_speed=True,method='fastdllm'

# # fast dllm dual cache + haloblock
accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},halo_factor=2.0,use_cache=True,dual_cache=True,factor=${factor},show_speed=True,method='haloblock'

# # dparallel dual cache
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='Zigeng/dParallel-LLaDA-8B-instruct',gen_length=${length},steps=${length},block_length=${block_length},halo_factor=2.0,use_cache=True,dual_cache=True,threshold=0.5,show_speed=True,method='dparallel'

# # dparallel dual cache + haloblock
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='Zigeng/dParallel-LLaDA-8B-instruct',gen_length=${length},steps=${length},block_length=${block_length},halo_factor=2.0,use_cache=True,dual_cache=True,threshold=0.5,show_speed=True,method='halo_dparallel'

# # adablock dual cache
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},halo_factor=2.0,use_cache=True,dual_cache=True,threshold=0.9,show_speed=True,method='adablock'

# # adablock dual cache + haloblock
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},halo_factor=2.0,use_cache=True,dual_cache=True,threshold=0.9,show_speed=True,method='halo_adablock'

# # d3llm
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='d3LLM/d3LLM_LLaDA',gen_length=${length},steps=${length},block_length=${block_length},halo_factor=2.0,use_cache=True,dual_cache=True,threshold=0.5,block_add_threshold=0.1,decoded_token_threshold=0.95,cache_delay_iter=2,refresh_interval=3,show_speed=True,method='d3llm'

# d2f
# accelerate launch eval_d2f.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model dream_lora \
# --model_args pretrained=GSAI-ML/LLaDA-8B-Instruct,lora_path=SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora,max_new_tokens=256,diffusion_steps=256,add_bos_token=true,temperature=0,block_size=32,block_add_threshold=0.1,skip_threshold=0.9,decoded_token_threshold=0.95,dtype=bfloat16,sampling_strategy=default,save_dir=eval_tmp
