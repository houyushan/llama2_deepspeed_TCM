在目录下运行(以hpcadmin用户为例)：

```
export 
deepspeed --num_nodes=2 --num_gpus=2 --hostfile hostfile --launcher pdsh --launcher_args="-l hpcadmin 'export CUDA_HOME=~/houys2/hpcadmin_minicoda3/envs/hpcadmin_env_deepspeed && export PATH=\$CUDA_HOME/bin:\$PATH && export LD_LIBRARY_PATH=\$CUDA_HOME/lib:\$LD_LIBRARY_PATH && export PATH=/home/hpcadmin/houys2/pdsh-2.29/bin:\$PATH &&'" main.py --data_path local/jsonfile --data_split 2,4,4 --model_name_or_path /path/to/Llama-2-7b-hf/ --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --max_seq_len 64 --learning_rate 9.65e-6 --weight_decay 0. --num_train_epochs 2 --gradient_accumulation_steps 1 --lr_scheduler_type cosine --num_warmup_steps 0 --seed 1234 --gradient_checkpointing --zero_stage 3 --offload --deepspeed --output_dir /home/hpcadmin/houys2/deepspeed_out_dir/llama2_train_deepspeed/ --data_output_path /home/hpcadmin/houys2/deepspeed_out_dir/llama2_data_outputdir/
```

hostfile 格式如下：
```
shield-c1 slots=2
shield-c2 slots=2
```