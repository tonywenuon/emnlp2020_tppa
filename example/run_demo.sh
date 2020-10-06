

data_set='wizard'

python train_auto_pointer_sent.py --exp_name=tppa --data_set=$data_set --src_seq_length=30 --sample_seq_length=30 --query_retrieval_number=10 --negative_number=30  --auto_pointer_rate=0.7

