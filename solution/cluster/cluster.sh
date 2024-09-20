CUDA_VISIBLE_DEVICES=0 python ./step1_embedding_batch.py \
   --model_path /home/disk2/zhangjingjun/data_syn/caches/models/bge \
   --instruction_path data/input_data.jsonl \
   --save_embedding_path data/embedding/

CUDA_VISIBLE_DEVICES=0 python ./step2_kmeans_sample.py \
   --embedding_path data/embedding/1581.npy \
   --instruction_path data/input_data.jsonl \
   --save_path output/output_kmeans.json