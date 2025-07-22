CUDA_VISIBLE_DEVICES=1 python train_meta_model.py \
    --terms_path /home/jovyan/rahmatullaev/rand_exps/LLMs4OL-Challenge/2025/TaskC-TaxonomyDiscovery/SchemaOrg/train/schemaorg_train_types_spaced.txt \
    --relations_path /home/jovyan/rahmatullaev/rand_exps/LLMs4OL-Challenge/2025/TaskC-TaxonomyDiscovery/SchemaOrg/train/schemaorg_train_pairs_spaced.json \
    --output_dir experiments/schemaorg_try \
    --dataset_name SchemaOrg \
    --freeze_strategy lora \
    --epochs 50 \
    --eval_every 50 \
    --save_every 50 \
    --lr 1e-5
