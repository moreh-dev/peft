
# bash setup.sh text2image
export MLFLOW_TRACKING_URI="http://127.0.0.1:5001"
export MODEL_NAME="CompVis/stable-diffusion-v1-4" 
export INSTANCE_DIR="./dog"
export OUTPUT_DIR="./save_model"

# prepare data (instance dir)
python prep_dog_data.py

# accelerate config default
python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"
accelerate launch mlflow_train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 27 \
  --lora_text_encoder_r 16 \
  --lora_text_encoder_alpha 17 \
  --learning_rate=1e-4 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=800 2>&1 | tee peft_mlflow_dreambooth.log