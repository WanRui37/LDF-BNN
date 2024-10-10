model=ldfbnn18

mkdir .models

python3 train_assistant_group_amp.py \
--model $model \
--distillation True \
--teacher_num 0 \
--assistant_teacher_num 1 \
--weak_teacher EfficientNet_B0 \
--mixup 0.0 --cutmix 0.0 --aug-repeats 1 \
--dali_cpu  --multiprocessing-distributed \
--dist-url 'tcp://127.0.0.1:33506' --dist-backend 'nccl' --world-size 1 --rank 0 \
--dataset CIFAR100 \
--data=/home/ic611/workspace/wr/BNext-main_14/dataset/ \
--job_dir $model/ \
--baseline None \
--baseline_model None \
--batch_size 128 \
--learning_rate=1e-3 \
--epochs=256 \
--weight_decay=0
