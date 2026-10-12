# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=mbpp
length=256
block_length=32
steps=$((length / block_length))
factor=0.7
num_fewshot=3
#model_path='Zigeng/dParallel-LLaDA-8B-instruct'
model_path='GSAI-ML/LLaDA-8B-Instruct'
#model_path='GSAI-ML/LLaDA-1.5'
#model_path='d3LLM/d3LLM_LLaDA'

# fast dllm baseline
# accelerate launch eval_llada.py --tasks ${task} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True \
# --output_path evals_results/mbpp/fastdllm/baseline/mbpp-ns0-${length} --log_samples

# fast dllm prefix cache
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},halo_factor=2.0,use_cache=True,threshold=0.9,show_speed=True,method='fastdllm' \
# --output_path evals_results/mbpp/fastdllm/prefixcache/mbpp-ns0-${length} --log_samples

# fast dllm dual cache
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},halo_factor=2.0,use_cache=True,dual_cache=True,threshold=0.9,show_speed=True,method='fastdllm' \
# --output_path evals_results/mbpp/fastdllm/dualcache/mbpp-ns0-${length} --log_samples

# fast dllm dual cache + haloblock
accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},halo_factor=2.0,use_cache=True,dual_cache=True,threshold=0.9,show_speed=True,method='haloblock' \
--output_path evals_results/mbpp/fastdllm/dualcache/mbpp-ns0-${length} --log_samples

# # dparallel dual cache
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='Zigeng/dParallel-LLaDA-8B-instruct',gen_length=${length},steps=${length},block_length=${block_length},halo_factor=2.0,use_cache=True,dual_cache=True,threshold=0.45,show_speed=True,method='dparallel' \
# --output_path evals_results/mbpp/dparallel/mbpp-ns0-${length} --log_samples

# # dparallel dual cache + haloblock
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='Zigeng/dParallel-LLaDA-8B-instruct',gen_length=${length},steps=${length},block_length=${block_length},halo_factor=2.0,use_cache=True,dual_cache=True,threshold=0.45,show_speed=True,method='halo_dparallel' \
# --output_path evals_results/mbpp/dparallel/mbpp-ns0-${length} --log_samples

# # adablock dual cache
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},halo_factor=2.0,use_cache=True,dual_cache=True,threshold=0.9,show_speed=True,method='adablock' \
# --output_path evals_results/mbpp/adablock/mbpp-ns0-${length} --log_samples

# # adablock dual cache + haloblock
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},halo_factor=2.0,use_cache=True,dual_cache=True,threshold=0.9,show_speed=True,method='halo_adablock' \
# --output_path evals_results/mbpp/adablock/mbpp-ns0-${length} --log_samples

# # d3llm
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='d3LLM/d3LLM_LLaDA',gen_length=${length},steps=${length},block_length=${block_length},halo_factor=2.0,use_cache=True,dual_cache=True,threshold=0.4,block_add_threshold=0.1,decoded_token_threshold=0.95,cache_delay_iter=1,refresh_interval=10000,show_speed=True,method='d3llm' \
# --output_path evals_results/mbpp/d3llm/mbpp-ns0-${length} --log_samples

# accelerate launch eval_d2f.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model dream_lora \
# --model_args pretrained=GSAI-ML/LLaDA-8B-Instruct,lora_path=SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora,max_new_tokens=256,diffusion_steps=256,add_bos_token=true,temperature=0,block_size=32,block_add_threshold=0.1,skip_threshold=0.9,decoded_token_threshold=0.95,dtype=bfloat16,sampling_strategy=default,save_dir=eval_tmp \
# --output_path evals_results/mbpp/d2f/mbpp-ns0-${length} --log_samples

## NOTICE: use postprocess for mbpp
#python postprocess_code_mbpp.py ${out}/samples_mbpp_2026-06-05T16-55-54.056089.jsonl
