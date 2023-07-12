
rm -r unilm
bash setup.sh token_cls
sudo apt-get install unzip

# prep data (unzip)
wget https://guillaumejaume.github.io/FUNSD/dataset.zip
unzip dataset.zip && mv dataset data && rm -rf dataset.zip __MACOSX

# preprocessing
python unilm/layoutlm/examples/seq_labeling/preprocess.py --data_dir data/dataset/training_data/annotations \
                                                      --data_split train \
                                                      --output_dir data \
                                                      --model_name_or_path microsoft/layoutlm-base-uncased \
                                                      --max_len 510

python unilm/layoutlm/examples/seq_labeling/preprocess.py --data_dir data/dataset/testing_data/annotations \
                                                      --data_split test \
                                                      --output_dir data \
                                                      --model_name_or_path microsoft/layoutlm-base-uncased \
                                                      --max_len 510

cat data/train.txt | cut -d$'\t' -f 2 | grep -v "^$"| sort | uniq > data/labels.txt

export MLFLOW_TRACKING_URI="http://127.0.0.1:5001"
python mlflow_peft_lora_token_cls.py 2>&1 | tee mlflow_peft_lora_token_cls.log