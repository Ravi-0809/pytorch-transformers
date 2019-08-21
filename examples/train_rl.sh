python -m torch.distributed.launch --nproc_per_node=2 train_rl.py \
    # --model_type bert \
    # --model_name_or_path bert-large-uncased-whole-word-masking \
    --do_train \
    # --do_eval \
    --do_lower_case \
    --train_file /home/ubuntu/question_generation/data/train-v2.0.json \
    # --predict_file $SQUAD_DIR/dev-v1.1.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./outputs/ \
    # --per_gpu_eval_batch_size=3   \
    --per_gpu_train_batch_size=2   \
    --save_steps 10000  \
    --overwrite_output_dir  \
    --local_rank -1 \
    --fp16  \