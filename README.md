# TPPA

## Introduction
This work is the implementation of the paper [*Approximation of Response Knowledge Retrieval in Knowledge-grounded Dialogue Generation*](https://www.aclweb.org/anthology/2020.findings-emnlp.321.pdf). This work is concerned with improving dialogue generation models through injection of knowledge, e.g., content relevant to the post that can increase the quality of responses. Past research extends the training of the generative models by incorporating statistical properties of posts, responses and related knowledge, without explicitly assessing the knowledge quality. In our work, we demonstrate the importance of knowledge relevance and adopt a two-phase approach. We first apply a novel method, Transformer & Post based Posterior Approximation (TPPA) to select knowledge, and then use the Transformer with Expanded Decoder (TED) model to generate responses from both the post and the knowledge. TPPA method processes posts, post related knowledge, and response related knowledge at both word and sentence level. More details refer to the following paper:

> Approximation of Response Knowledge Retrieval in Knowledge-grounded Dialogue Generation

## Data Format
In the data folder, all of the datasets are put there. The directionary lever should be like:
-data
--wizard
---train
---valid
---test
In each train/valid/test folder, the following files should be in:

* pro_qa.txt. Main post-response pair. The format: `index \t post \t reponse`.
* question_retrieval.txt. This is the PRK illustrated in the paper, using the post as the query to retrieve from the knowledge set. The format: `index \t knowldge1 \t knowledge2 ...`.
* response_retrieval.txt. This is the RRK illustrated in the paper, using the response as the query to retrieve from the knowledge set. The format is the same as question_retrieval.txt.
* negative_facts.txt. This is the negative samples. The knowledge are randomly selected from the data set. The format is the same as question_retrieval.txt.
* global.src.token.dic. This is the vocabulary of the post. Format is: `token \t token_index`. The token_index can be different from the index in pro_qa.txt.
* global.tar.token.dic. This is the vocabulary of the response. The format is the same as global.src.token.dic. In our setting, these two vocabulary files are exactly the same.

Note that the index in files: pro_qa.txt, question_retrieval.txt, response_retrieval.txt and negative_facts.txt should be the same for a certain line, otherwise, the code would throw an assertion error. For the knowledge retrieval, we use pylucene. Please organise your own data set to the required data format to use.


## Usage
In the example folder, there is a script 'run_demo.sh'. Go into the example folder and run the following code. BTW, the configuration can be seen and changed in folder 'configuration/'.

```
python train_auto_pointer_sent.py \
  --exp_name=tppa \
  --data_set=$data_set \
  --src_seq_length=30 \
  --sample_seq_length=30 \
  --query_retrieval_number=10 \
  --negative_number=30  \
  --auto_pointer_rate=0.7
```

## Citation

We appreciate anyone who uses this repo or gets insight from this work to cite the paper. Many thanks!

```
@inproceedings{zheng2020approximation,
  title={Approximation of Response Knowledge Retrieval in Knowledge-grounded Dialogue Generation},
  author={Zheng, Wen and Milic-Frayling, Natasa and Zhou, Ke},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings},
  pages={3581--3591},
  year={2020}
}
```
