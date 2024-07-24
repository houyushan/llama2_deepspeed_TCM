
运行环境：
安装miniconda3 到用户目录：

```Shell
#下载miniconda3
wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
 
#运行sh文件，指定安装目录到用户目录
bash Miniconda3-latest-Linux-x86_64.sh
```


environment.yml 文件：

```Shell
channels:
  - nvidia/label/cuda-11.8.0
  - pytorch # or pytorch-nightly
  - conda-forge
dependencies:
  - pytorch
  - torchvision
  - torchaudio
  - cuda
  - pytorch-cuda=11.8
  - compilers
  - sysroot_linux-64==2.17
  - gcc==11.4
  - ninja
  - py-cpuinfo
  - libaio
  - ca-certificates
  - certifi
  - openssl
  - python=3.10
  - pydantic
```


创建conda虚拟环境：

```Shell
conda env create -f environment.yml -n myenv_deepspeed
```


安装deepspeed

```Shell
pip install deepspeed transformers sentencepiece
```




安装pdsh：

下载pdsh最新版：pdsh-2.29.tar.bz2，下载地址：[https://sourceforge.net/projects/pdsh/
](https://sourceforge.net/projects/pdsh/)下载之后，执行命令：`tar -jxvf pdsh-2.29.tar.bz2 -C /tmp`，解压至/tmp/pdsh-2.26
执行命令：cd /tmp/pdsh-2.26/；进入目录
执行configure命令，如下：

```Shell
./configure \
--prefix=/home/pdsh/2.26/ \
--with-timeout=60 \
--with-ssh \
--with-exec \
--with-nodeupdown \
--with-readline \
--with-rcmd-rank-list=ssh
```


执行`make && make install`，进行编译和安装。完成之后，将命令路径添加至环境变量







导入pdsh 和 conda中的cuda到环境变量：

```Shell
export CUDA_HOME=~/houys2/minicoda3_dir/miniconda33/envs/myenv_deepspeed
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

export PATH=/home/hpcadmin/houys2/pdsh-2.29/bin:$PATH
```


注：可将变量写入.deepspeed_env，来传播用户定义的环境变量。

export DS_ENV_FILE="/path/to/.deepspeed_env"

```Shell
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=eth0

CUDA_HOME=~/houys2/minicoda3_dir/miniconda33/envs/myenv_deepspeed
PATH=$CUDA_HOME/bin:$PATH
LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

PATH=/home/hpcadmin/houys2/pdsh-2.29/bin:$PATH
```




运行deepspeed程序：

编写hostfile：

```Shell
host02 slots 2
host02 slots 2
```




运行：

```Shell
cd /home/hpcadmin/houys2/deepspeed_test_dir/deepspeed-example-mn/deepspeed_src
deepspeed --num_nodes=2 --num_gpus=2 --hostfile hostfile --launcher pdsh cifar10_deepspeed_offload.py --stage 3 --dtype bf16 --deepspeed
```




例子：

在目录下运行(以hpcadmin用户为例)：

```Shell
deepspeed --num_nodes=2 --num_gpus=2 --hostfile hostfile --launcher pdsh --launcher_args="-l hpcadmin 'export CUDA_HOME=~/houys2/hpcadmin_minicoda3/envs/hpcadmin_env_deepspeed && export PATH=\$CUDA_HOME/bin:\$PATH && export LD_LIBRARY_PATH=\$CUDA_HOME/lib:\$LD_LIBRARY_PATH && export PATH=/home/hpcadmin/houys2/pdsh-2.29/bin:\$PATH &&'" main.py --data_path local/jsonfile --data_split 2,4,4 --model_name_or_path /path/to/Llama-2-7b-hf/ --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --max_seq_len 64 --learning_rate 9.65e-6 --weight_decay 0. --num_train_epochs 2 --gradient_accumulation_steps 1 --lr_scheduler_type cosine --num_warmup_steps 0 --seed 1234 --gradient_checkpointing --zero_stage 3 --offload --deepspeed --output_dir /home/hpcadmin/houys2/deepspeed_out_dir/llama2_train_deepspeed/ --data_output_path /home/hpcadmin/houys2/deepspeed_out_dir/llama2_data_outputdir/
```

附：下载Llama-2-7b-hf

minigpt-4:https://github.com/Vision-CAIR/MiniGPT-4
中文llama 预训练模型地址：https://github.com/ymcui/Chinese-LLaMA-Alpaca-2?tab=readme-ov-file
