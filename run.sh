for s in  0
do
    python Main.py \
        --dataset banking \
        --known_cls_ratio 0.7 \
        --cluster_num_factor 2 \
        --seed $s \
        --lambda_cluster 0.99 \
        --labeled_ratio 1\
        --data_augumentation_type 2 \
        --freeze_bert_parameters \
        --save_model \
        --method PLPCL \
        --train_batch_size 128 \
        --pre_train_batch_size 128 \
        --gpu_id 7 \
        --num_pretrain_epochs 100\
        --num_train_epochs 100\
        --pretrain \
        --pretrain_dir pretrain_models_v1_0.7_cescl_seed42_banking
done

