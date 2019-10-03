documenting my ongoing effort to train a Russian-language GPT-2 on 4x V100/32Gb

CUDA_VISIBLE_DEVICES="0,1,2,3"; export CUDA_VISIBLE_DEVICES;  PYTHONPATH=src; export PYTHONPATH; horovodrun -np 4 -H localhost:4 python3 train-horovod-msg-sp.py --dataset newsru-n-noendoftext.npz

for 117M model it's possible to use batch_size = 8 on a 32Gb GPU with gradient checkpointing

for 345M the batch_size should be 4

