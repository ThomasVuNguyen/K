# GTX 1050 Ti compatibility fixes - disable compilation
import unsloth
import os
import csv
import json
from datetime import datetime
from transformers import TrainerCallback
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# CONFIGURATION LOADING FROM JSON
# =============================================================================

# Load configuration from JSON file
with open('config.json', 'r') as f:
    config = json.load(f)

# Extract configuration sections
model_config = config['model_config']
dataset_config = config['dataset_config']
lora_config = config['lora_config']
training_config = config['training_config']
inference_config = config['inference_config']
saving_config = config['saving_config']
logging_config = config['logging_config']
checkpoint_config = config.get('checkpoint_config', {})

# Model Configuration
HUB_MODEL_NAME = model_config['hub_model_name']
MODEL_NAME = model_config['base_model_name']
MAX_SEQ_LENGTH = model_config['max_seq_length']
LOAD_IN_4BIT = model_config['load_in_4bit']
LOAD_IN_8BIT = model_config['load_in_8bit']
FULL_FINETUNING = model_config['full_finetuning']

# Dataset Configuration
CHAT_TEMPLATE = dataset_config['chat_template']
DEFAULT_DATASET_SPLIT = dataset_config.get('dataset_split', 'train')
RAW_DATASET_SPECS = dataset_config.get('datasets')
TEST_OBJECTS = dataset_config.get('test_objects', [])

def normalize_dataset_spec(spec):
    dataset_name = spec.get('name') or spec.get('dataset_name')
    if not dataset_name:
        raise ValueError("Each dataset entry must define a 'name'.")
    dataset_split = spec.get('split') or spec.get('dataset_split') or DEFAULT_DATASET_SPLIT
    return {"name": dataset_name, "split": dataset_split}

if RAW_DATASET_SPECS:
    DATASET_SPECS = [normalize_dataset_spec(spec) for spec in RAW_DATASET_SPECS]
else:
    dataset_name = dataset_config.get('dataset_name')
    if not dataset_name:
        raise ValueError("dataset_config must include either 'datasets' or 'dataset_name'.")
    DATASET_SPECS = [{"name": dataset_name, "split": DEFAULT_DATASET_SPLIT}]

# LoRA Configuration
'''  Rank to Percentage Table:

  | Rank (r) | LoRA Parameters | % of 270M | Memory Est. |
  |----------|-----------------|-----------|-------------|
  | r=8      | 9,533,440       | 3.5%      | ~150MB      |
  | r=16     | 19,066,880      | 7.1%      | ~300MB      |
  | r=32     | 38,133,760      | 14.1%     | ~600MB      |
  | r=64     | 76,267,520      | 28.2%     | ~1.2GB      |
  | r=128    | 152,535,040     | 56.5%     | ~2.4GB      |
'''
LORA_R = lora_config['r']
LORA_ALPHA = LORA_R * lora_config['alpha_multiplier']
LORA_DROPOUT = lora_config['dropout']
LORA_BIAS = lora_config['bias']
USE_GRADIENT_CHECKPOINTING = lora_config['use_gradient_checkpointing']
RANDOM_STATE = lora_config['random_state']
USE_RSLORA = lora_config['use_rslora']
LOFTQ_CONFIG = lora_config['loftq_config']
TARGET_MODULES = lora_config['target_modules']

# Training Configuration
PER_DEVICE_TRAIN_BATCH_SIZE = training_config['per_device_train_batch_size']
GRADIENT_ACCUMULATION_STEPS = training_config['gradient_accumulation_steps']
WARMUP_STEPS = training_config['warmup_steps']
MAX_STEPS = training_config['max_steps']
NUM_TRAIN_EPOCHS = training_config['num_train_epochs']
LEARNING_RATE = training_config['learning_rate']
WEIGHT_DECAY = training_config['weight_decay']
LR_SCHEDULER_TYPE = training_config['lr_scheduler_type']
SEED = training_config['seed']
OUTPUT_DIR = training_config['output_dir']
REPORT_TO = training_config['report_to']
OPTIM = training_config['optim']
LOGGING_STEPS = training_config['logging_steps']
SAVE_STRATEGY = training_config.get('save_strategy', 'steps')
SAVE_STEPS = training_config.get('save_steps', 100)
SAVE_TOTAL_LIMIT = training_config.get('save_total_limit', 3)
EVAL_STRATEGY = training_config.get('eval_strategy', 'no')
EVAL_STEPS = training_config.get('eval_steps', 100)

# Inference Configuration
MAX_NEW_TOKENS = inference_config['max_new_tokens']
TEMPERATURE = inference_config['temperature']
TOP_P = inference_config['top_p']
TOP_K = inference_config['top_k']
DO_SAMPLE = inference_config['do_sample']

# Model Saving Configuration
SAVE_LOCAL = saving_config['save_local']
SAVE_16BIT = saving_config['save_16bit']
SAVE_4BIT = saving_config['save_4bit']
SAVE_LORA = saving_config['save_lora']
PUSH_TO_HUB = saving_config['push_to_hub']

# CSV Logging Configuration
CSV_LOG_ENABLED = logging_config['csv_log_enabled']
CSV_LOG_FILE = f"{HUB_MODEL_NAME}/training_metrics.csv"
CSV_PLOT_FILE = f"{HUB_MODEL_NAME}/training_metrics.png"

# Available Models (for reference)
FOURBIT_MODELS = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
    # Other popular models!
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/Llama-3.3-70B",
    "unsloth/mistral-7b-instruct-v0.3",
    "unsloth/Phi-4",
]  # More models at https://huggingface.co/unsloth

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Load environment variables from .env file (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Environment variables loaded from .env file")
except ImportError:
    print("python-dotenv not installed. Using system environment variables only.")
    print("To install: pip install python-dotenv")

# Set CUDA environment variables for GTX 1050 Ti compatibility
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_INDUCTOR"] = "0"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastModel
import torch

# Disable Triton and dynamic compilation
torch._dynamo.config.suppress_errors = True
torch._dynamo.reset()
torch._dynamo.config.disable = True

# =============================================================================
# CSV LOGGING CALLBACK
# =============================================================================

class CSVMetricsCallback(TrainerCallback):
    """Callback to log training metrics to CSV file"""
    
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.metrics_data = []
        self.fieldnames = ['step', 'epoch', 'loss', 'grad_norm', 'learning_rate', 'timestamp']
        
        # Create CSV file with headers
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Called when trainer logs metrics"""
        if logs is not None and CSV_LOG_ENABLED:
            # Extract metrics from logs
            metrics = {
                'step': state.global_step,
                'epoch': logs.get('epoch', 0),
                'loss': logs.get('loss', None),
                'grad_norm': logs.get('grad_norm', None),
                'learning_rate': logs.get('learning_rate', None),
                'timestamp': datetime.now().isoformat()
            }
            
            # Only log if we have meaningful data
            if any(v is not None for v in [metrics['loss'], metrics['grad_norm'], metrics['learning_rate']]):
                self.metrics_data.append(metrics)
                
                # Write to CSV file
                with open(self.csv_file_path, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                    writer.writerow(metrics)
                
                print(f"Logged metrics: Step {metrics['step']}, Loss: {metrics['loss']}, LR: {metrics['learning_rate']}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called when training ends"""
        if CSV_LOG_ENABLED:
            print(f"\nTraining metrics saved to: {self.csv_file_path}")
            print(f"Total logged entries: {len(self.metrics_data)}")

# =============================================================================
# MODEL LOADING
# =============================================================================

model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
    load_in_8bit=LOAD_IN_8BIT,
    full_finetuning=FULL_FINETUNING,
    # token = "hf_...", # use one if using gated models
)

"""We now add LoRA adapters so we only need to update a small amount of parameters!"""

model = FastModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias=LORA_BIAS,
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    random_state=RANDOM_STATE,
    use_rslora=USE_RSLORA,
    loftq_config=LOFTQ_CONFIG,
)

"""<a name="Data"></a>
### Data Prep
We now use the `Gemma-3` format for conversation style finetunes. We use the reformatted [Tulu-3 SFT Personas Instruction Following](https://huggingface.co/datasets/ThomasTheMaker/tulu-3-sft-personas-instruction-following) dataset. Gemma-3 renders multi turn conversations like below:

```
<bos><start_of_turn>user
Hello!<end_of_turn>
<start_of_turn>model
Hey there!<end_of_turn>
```

We use our `get_chat_template` function to get the correct chat template. We support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3, phi4, qwen2.5, gemma3` and more.
"""

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template=CHAT_TEMPLATE,
)

from datasets import load_dataset, concatenate_datasets

train_splits = []
eval_splits = []
for spec in DATASET_SPECS:
    current_dataset = load_dataset(spec["name"], split=spec["split"])
    split_dataset = current_dataset.train_test_split(test_size=0.05, seed=SEED)
    train_splits.append(split_dataset['train'])
    eval_splits.append(split_dataset['test'])
    print(
        f"Loaded dataset '{spec['name']}' ({spec['split']}) -> "
        f"train: {len(split_dataset['train'])}, val: {len(split_dataset['test'])}"
    )

if not train_splits or not eval_splits:
    raise RuntimeError("No datasets were loaded. Please check dataset_config.")

train_dataset = concatenate_datasets(train_splits) if len(train_splits) > 1 else train_splits[0]
eval_dataset = concatenate_datasets(eval_splits) if len(eval_splits) > 1 else eval_splits[0]

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(eval_dataset)}")

"""We now use `convert_to_chatml` to convert the reformatted dataset (with input/output/system columns) to the correct format for finetuning purposes!"""

def convert_to_chatml(example):
    return {
        "conversations": [
            {"role": "user", "content": "hey cadmonkey, create me a " + example["name"]},
            {"role": "assistant", "content": example["code"]}
        ]
    }

train_dataset = train_dataset.map(convert_to_chatml)
eval_dataset = eval_dataset.map(convert_to_chatml)

"""Let's see how row 100 looks like!"""

train_dataset[100]

"""We now have to apply the chat template for `Gemma3` onto the conversations, and save it to `text`."""

def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

train_dataset = train_dataset.map(formatting_prompts_func, batched = True)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched = True)

"""Let's see how the chat template did!

"""

train_dataset[100]['text']

"""<a name="Train"></a>
### Train the model
Now let's train our model. We do 100 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.
"""

from trl import SFTTrainer, SFTConfig

# Create model directory and copy config
import shutil
os.makedirs(HUB_MODEL_NAME, exist_ok=True)

# Copy the JSON config to the model folder for reproducibility
config_copy_path = f"{HUB_MODEL_NAME}/config.json"
shutil.copy2('config.json', config_copy_path)
print(f"Configuration copied to: {config_copy_path}")

# Initialize CSV logging callback
csv_callback = None
if CSV_LOG_ENABLED:
    csv_callback = CSVMetricsCallback(CSV_LOG_FILE)
    print(f"CSV logging enabled. Metrics will be saved to: {CSV_LOG_FILE}")

# Prepare training arguments - handle max_steps properly
training_args = {
    "dataset_text_field": "text",
    "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "warmup_steps": WARMUP_STEPS,
    "learning_rate": LEARNING_RATE,
    "logging_steps": LOGGING_STEPS,
    "optim": OPTIM,
    "weight_decay": WEIGHT_DECAY,
    "lr_scheduler_type": LR_SCHEDULER_TYPE,
    "seed": SEED,
    "output_dir": OUTPUT_DIR,
    "report_to": REPORT_TO,
    "save_strategy": SAVE_STRATEGY,
    "save_steps": SAVE_STEPS,
    "save_total_limit": SAVE_TOTAL_LIMIT,
    "eval_strategy": EVAL_STRATEGY,
    "eval_steps": EVAL_STEPS,
}

# Add either max_steps OR num_train_epochs, not both
if MAX_STEPS and MAX_STEPS > 0:
    training_args["max_steps"] = MAX_STEPS
else:
    training_args["num_train_epochs"] = NUM_TRAIN_EPOCHS

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(**training_args),
)

# Add CSV callback to trainer
if csv_callback:
    trainer.add_callback(csv_callback)

"""We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs. This helps increase accuracy of finetunes!"""

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

"""Let's verify masking the instruction part is done! Let's print the 100th row again."""

if len(trainer.train_dataset) > 100:
    tokenizer.decode(trainer.train_dataset[100]["input_ids"])

    """Now let's print the masked out example - you should see only the answer is present:"""

    tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

"""Let's train the model! To resume a training run, set `trainer.train(resume_from_checkpoint = True)`"""

# Check for existing checkpoints and resume if available
import glob
checkpoints = glob.glob(f"{OUTPUT_DIR}/checkpoint-*")
resume_from_checkpoint = None
if checkpoints:
    # Get the latest checkpoint by sorting
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
    resume_from_checkpoint = latest_checkpoint
    print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
else:
    print("No checkpoints found. Starting training from scratch.")

trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# @title Show CSV metrics summary and generate plot
# Always generate plot if CSV file exists with data
if os.path.exists(CSV_LOG_FILE):
    try:
        df = pd.read_csv(CSV_LOG_FILE)

        # Only show summary and plot if there's actual data
        if len(df) > 0:
            print("\n" + "="*50)
            print("TRAINING METRICS SUMMARY")
            print("="*50)

            # Calculate summary statistics
            if 'loss' in df.columns:
                losses = df['loss'].dropna()
                if len(losses) > 0:
                    print(f"Final Loss: {losses.iloc[-1]:.4f}")
                    print(f"Initial Loss: {losses.iloc[0]:.4f}")
                    print(f"Loss Reduction: {((losses.iloc[0] - losses.iloc[-1]) / losses.iloc[0] * 100):.2f}%")
                    print(f"Min Loss: {losses.min():.4f}")
                    print(f"Max Loss: {losses.max():.4f}")

            if 'learning_rate' in df.columns:
                learning_rates = df['learning_rate'].dropna()
                if len(learning_rates) > 0:
                    print(f"Final Learning Rate: {learning_rates.iloc[-1]:.2e}")
                    print(f"Initial Learning Rate: {learning_rates.iloc[0]:.2e}")

            if 'grad_norm' in df.columns:
                grad_norms = df['grad_norm'].dropna()
                if len(grad_norms) > 0:
                    print(f"Final Gradient Norm: {grad_norms.iloc[-1]:.4f}")
                    print(f"Average Gradient Norm: {grad_norms.mean():.4f}")

            print(f"Total Logged Steps: {len(df)}")
            print(f"CSV File: {CSV_LOG_FILE}")
            print("="*50)

            # Plot training metrics
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')

            # Plot 1: Loss over steps
            if 'loss' in df.columns and df['loss'].notna().any():
                axes[0, 0].plot(df['step'], df['loss'], linewidth=2, color='#e74c3c')
                axes[0, 0].set_xlabel('Step', fontweight='bold')
                axes[0, 0].set_ylabel('Loss', fontweight='bold')
                axes[0, 0].set_title('Training Loss')
                axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Learning rate over steps
            if 'learning_rate' in df.columns and df['learning_rate'].notna().any():
                axes[0, 1].plot(df['step'], df['learning_rate'], linewidth=2, color='#3498db')
                axes[0, 1].set_xlabel('Step', fontweight='bold')
                axes[0, 1].set_ylabel('Learning Rate', fontweight='bold')
                axes[0, 1].set_title('Learning Rate Schedule')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

            # Plot 3: Gradient norm over steps
            if 'grad_norm' in df.columns and df['grad_norm'].notna().any():
                axes[1, 0].plot(df['step'], df['grad_norm'], linewidth=2, color='#2ecc71')
                axes[1, 0].set_xlabel('Step', fontweight='bold')
                axes[1, 0].set_ylabel('Gradient Norm', fontweight='bold')
                axes[1, 0].set_title('Gradient Norm')
                axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Loss over epochs
            if 'epoch' in df.columns and 'loss' in df.columns and df['loss'].notna().any():
                axes[1, 1].plot(df['epoch'], df['loss'], linewidth=2, color='#9b59b6')
                axes[1, 1].set_xlabel('Epoch', fontweight='bold')
                axes[1, 1].set_ylabel('Loss', fontweight='bold')
                axes[1, 1].set_title('Loss over Epochs')
                axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(CSV_PLOT_FILE, dpi=300, bbox_inches='tight')
            print(f"\n✓ Training metrics plot saved to: {CSV_PLOT_FILE}")
            plt.close()
        else:
            print("\nNo training data found in CSV file - skipping metrics summary and plot")

    except Exception as e:
        print(f"\nWarning: Failed to generate training metrics summary/plot: {e}")

"""<a name="Inference"></a>
### Inference
Let's run the model via Unsloth native inference! According to the `Gemma-3` team, the recommended settings for inference are `temperature = 1.0, top_p = 0.95, top_k = 64`
"""

def build_test_prompt(object_name):
    return f"hey cadmonkey, create me a {object_name}".strip()

def generate_response(prompt_text):
    messages = [{"role": 'user', 'content': prompt_text}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ).removeprefix('<bos>')

    inputs = tokenizer(text, return_tensors = "pt").to("cuda")
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        do_sample=DO_SAMPLE,
        repetition_penalty=1.1,  # Reduce repetition in generated text
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,  # Disable cache to avoid compatibility issues
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if TEST_OBJECTS:
    inference_prompts = [build_test_prompt(obj) for obj in TEST_OBJECTS]
else:
    try:
        fallback_prompt = train_dataset['conversations'][10][0]['content']
    except Exception:
        fallback_prompt = train_dataset['conversations'][0][0]['content']
    inference_prompts = [fallback_prompt]

for prompt in inference_prompts:
    print("\n" + "=" * 50)
    print(f"Prompt: {prompt}")
    print("Generated response:")
    print(generate_response(prompt))

"""<a name="Save"></a>
### Saving, loading finetuned models
To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.

**[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!
"""

# Get Hugging Face token from environment or login
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    try:
        from huggingface_hub import HfFolder
        hf_token = HfFolder.get_token()
    except Exception:
        hf_token = None

# Save LoRA adapters if enabled
if SAVE_LORA:
    print("\n" + "="*50)
    print("SAVING LORA ADAPTERS")
    print("="*50)

    if SAVE_LOCAL:
        model.save_pretrained(HUB_MODEL_NAME)
        tokenizer.save_pretrained(HUB_MODEL_NAME)
        print(f"✓ LoRA adapters saved locally to: {HUB_MODEL_NAME}")

    if PUSH_TO_HUB and hf_token:
        model.push_to_hub(HUB_MODEL_NAME, token=hf_token)
        tokenizer.push_to_hub(HUB_MODEL_NAME, token=hf_token)
        print(f"✓ LoRA adapters uploaded to: https://huggingface.co/{HUB_MODEL_NAME}")
    elif PUSH_TO_HUB and not hf_token:
        print("Warning: HF_TOKEN not found. Skipping upload to Hugging Face.")
else:
    print("LoRA adapter saving disabled in config")

"""Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:"""

if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "gemma-3-270m-tulu-3-sft-personas-instruction-following", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        load_in_4bit = False,
    )

"""### Saving to float16 for VLLM

We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.
"""

# Save and upload 16-bit merged model (without -16bit suffix)
if SAVE_16BIT:
    print("\n" + "="*50)
    print("SAVING 16-BIT MERGED MODEL")
    print("="*50)
    
    # Save locally first
    local_16bit_path = f"{HUB_MODEL_NAME}_16bit"
    model.save_pretrained_merged(local_16bit_path, tokenizer, save_method="merged_16bit")
    print(f"16-bit model saved locally to: {local_16bit_path}")
    
    # Upload to HF Hub with original name (no -16bit suffix)
    if PUSH_TO_HUB and hf_token:
        print(f"Uploading 16-bit merged model to: {HUB_MODEL_NAME}")
        model.push_to_hub_merged(HUB_MODEL_NAME, tokenizer, save_method="merged_16bit", token=hf_token)
        print("✓ 16-bit merged model uploaded to Hugging Face Hub")
        
        # Upload training metrics CSV and plot if they exist (always, not just when CSV_LOG_ENABLED)
        from huggingface_hub import HfApi
        api = HfApi()

        # Upload CSV
        if os.path.exists(CSV_LOG_FILE):
            try:
                print(f"Uploading training metrics: {CSV_LOG_FILE}")
                api.upload_file(
                    path_or_fileobj=CSV_LOG_FILE,
                    path_in_repo="training_metrics.csv",
                    repo_id=HUB_MODEL_NAME,
                    token=hf_token
                )
                print("✓ Training metrics CSV uploaded to Hugging Face Hub")
            except Exception as e:
                print(f"Warning: Failed to upload training metrics CSV: {e}")

        # Upload plot
        if os.path.exists(CSV_PLOT_FILE):
            try:
                print(f"Uploading training plot: {CSV_PLOT_FILE}")
                api.upload_file(
                    path_or_fileobj=CSV_PLOT_FILE,
                    path_in_repo="training_metrics.png",
                    repo_id=HUB_MODEL_NAME,
                    token=hf_token
                )
                print("✓ Training metrics plot uploaded to Hugging Face Hub")
            except Exception as e:
                print(f"Warning: Failed to upload training metrics plot: {e}")
        
        print(f"\n🎉 Model available at: https://huggingface.co/{HUB_MODEL_NAME}")
    elif PUSH_TO_HUB and not hf_token:
        print("Warning: HF_TOKEN not found. Skipping model upload to Hugging Face.")
    
    # Keep the local directory for manual upload if needed
    print(f"16-bit model retained locally at: {local_16bit_path}")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
if PUSH_TO_HUB and hf_token:
    if SAVE_16BIT:
        print(f"✓ 16-bit merged model uploaded to: https://huggingface.co/{HUB_MODEL_NAME}")
    if SAVE_LORA:
        print(f"✓ LoRA adapters uploaded to: https://huggingface.co/{HUB_MODEL_NAME}")
    if os.path.exists(CSV_LOG_FILE):
        print(f"✓ Training metrics (CSV & plot) included in repository")
else:
    print("Model training completed but not uploaded to Hugging Face")
print("="*50)
