# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=humaneval
length=256
block_length=32
window_radius=16
left_radius=0
right_radius=0
steps=$((length / block_length))
factor=1.0
num_fewshot=0
#model_path='GSAI-ML/LLaDA-8B-Instruct'
#model_path='GSAI-ML/LLaDA-1.5'
model_path='Zigeng/dParallel-LLaDA-8B-instruct'
#model_path='d3LLM/d3LLM_LLaDA'

# baseline
# accelerate launch eval_llada.py --tasks ${task} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True \
# --output_path evals_results/baseline --log_samples


# prefix cache
# accelerate launch eval_llada.py --tasks ${task} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,show_speed=True \
# --output_path evals_results/prefix_cache/humaneval-ns0-${length} --log_samples

# parallel
# accelerate launch eval_llada.py --tasks ${task} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},factor=${factor},left_radius=${left_radius},right_radius=${right_radius},threshold=0.9,show_speed=True \
# --output_path evals_results/parallel/humaneval-ns0-${length} --log_samples


# prefix cache+parallel
# accelerate launch eval_llada.py --tasks ${task} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,window_factor=10,threshold=0.9,show_speed=True \
# --output_path evals_results/cache_parallel/humaneval-ns0-${length} --log_samples

#dual cache+parallel
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},window_factor=2.0,tau=10,use_cache=True,dual_cache=True,threshold=0.9,show_speed=True \
# --output_path evals_results/dual_cache_parallel/humaneval-ns0-${length} --log_samples

# d2f
# accelerate launch eval_d2f.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model dream_lora \
# --model_args pretrained=GSAI-ML/LLaDA-8B-Instruct,lora_path=SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora,max_new_tokens=256,diffusion_steps=256,temperature=0,add_bos_token=true,escape_until=true,block_size=32,block_add_threshold=0.1,skip_threshold=0.9,decoded_token_threshold=0.95,dtype=bfloat16,sampling_strategy=default,save_dir=eval_tmp \
# --output_path evals_results/dual_cache_parallel/humaneval-ns0-${length} --log_samples

#dparallel
accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},threshold=0.5,show_speed=True \
--output_path evals_results/dual_cache_parallel/humaneval-ns0-${length} --log_samples

# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},window_factor=2,tau=10,use_cache=True,dual_cache=True,threshold=0.5,show_speed=True \
# --output_path evals_results/dual_cache_parallel/humaneval-ns0-${length} --log_samples

## NOTICE: use postprocess for humaneval
#python postprocess_code.py ${out}/samples_humaneval_2026-06-05T16-55-54.056089.jsonl
