#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatic GGUF Conversion and Upload Script
Converts trained model to GGUF format using llama.cpp conversion tools
"""

from unsloth import FastLanguageModel
import os
import json
import subprocess
import shutil
from huggingface_hub import HfApi, create_repo

def create_gguf_directory():
    """Create gguf directory if it doesn't exist"""
    os.makedirs("gguf", exist_ok=True)
    print("Created gguf/ directory")

def load_config():
    """Load configuration from algebra.json"""
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config

def load_model(config):
    """Load the fine-tuned model from config"""
    model_config = config['model_config']
    model_name = model_config['hub_model_name']
    max_seq_length = model_config['max_seq_length']

    # Download model locally first
    from huggingface_hub import snapshot_download
    print(f"Downloading model: {model_name}")
    local_path = snapshot_download(model_name)
    print(f"Model downloaded to: {local_path}")

    # Load model in full precision (not 4bit) for GGUF conversion
    print(f"Loading model from local path: {local_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=local_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # Must be False for GGUF conversion
        dtype=None,  # Auto-detect dtype
        trust_remote_code=True
    )
    print("Model loaded successfully!")
    return model, tokenizer

def convert_to_gguf(model, tokenizer, quantization_type="f16", base_model_name="model"):
    """Convert model to GGUF format using llama.cpp directly"""
    # Variables
    gguf_directory = "gguf"
    
    # Create the main gguf directory if it doesn't exist
    os.makedirs(gguf_directory, exist_ok=True)
    
    # Create a temporary directory for the merged model
    temp_model_dir = f"temp_model_{quantization_type.lower()}"
    final_filename = f"{gguf_directory}/{base_model_name}-{quantization_type.lower()}.gguf"
    
    print(f"Converting to GGUF format with {quantization_type} quantization...")
    print(f"Target file: {final_filename}")
    
    try:
        # Save the model using Unsloth's merged save method
        print("Saving model and tokenizer...")
        model.save_pretrained_merged(temp_model_dir, tokenizer, save_method="merged_16bit")

        # Verify the model directory has the required files
        config_path = os.path.join(temp_model_dir, "config.json")
        if not os.path.exists(config_path):
            raise Exception(f"config.json not found in {temp_model_dir}")
        
        print(f"Model saved to: {temp_model_dir}")
        print("Files in temp directory:")
        for file in os.listdir(temp_model_dir):
            print(f"  - {file}")
        
        # Use llama.cpp conversion script
        print("Converting to GGUF using llama.cpp...")

        # Try to find llama.cpp directory
        llama_cpp_dir = os.getenv("LLAMA_CPP_DIR")
        if not llama_cpp_dir:
            # Try common locations
            possible_paths = [
                "/home/riftuser/bob/llama.cpp",
                os.path.expanduser("~/llama.cpp"),
                os.path.join(os.getcwd(), "llama.cpp"),
                "/usr/local/llama.cpp",
            ]
            for path in possible_paths:
                if os.path.exists(path) and os.path.exists(os.path.join(path, "convert_hf_to_gguf.py")):
                    llama_cpp_dir = path
                    break

        if not llama_cpp_dir or not os.path.exists(llama_cpp_dir):
            raise Exception(
                "llama.cpp directory not found. Please either:\n"
                "  1. Set LLAMA_CPP_DIR environment variable, or\n"
                "  2. Clone llama.cpp to ~/llama.cpp or ./llama.cpp\n"
                "  Example: git clone https://github.com/ggerganov/llama.cpp.git"
            )

        convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
        if not os.path.exists(convert_script):
            raise Exception(f"convert_hf_to_gguf.py not found in {llama_cpp_dir}")

        print(f"Using llama.cpp from: {llama_cpp_dir}")
        
        # Map quantization types
        outtype_map = {
            "F32": "f32",
            "F16": "f16", 
            "BF16": "bf16",
            "Q8_0": "q8_0",
            # K-quants (modern, high-quality quantization methods)
            "Q2_K": "q2_k",
            "Q3_K_S": "q3_k_s",
            "Q3_K_M": "q3_k_m",
            "Q3_K_L": "q3_k_l",
            "Q4_K_S": "q4_k_s",
            "Q4_K_M": "q4_k_m",  # Popular: good quality, small size
            "Q5_K_S": "q5_k_s",
            "Q5_K_M": "q5_k_m",  # Higher quality 5-bit
            "Q6_K": "q6_k",      # Near-lossless 6-bit
        }
        
        outtype = outtype_map.get(quantization_type.upper(), "f16")
        
        # Run the conversion
        cmd = [
            "python", convert_script,
            os.path.abspath(temp_model_dir),  # Use absolute path
            "--outfile", os.path.abspath(final_filename),  # Use absolute path
            "--outtype", outtype,
            "--verbose"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=llama_cpp_dir)
        
        if result.returncode != 0:
            print(f"Conversion failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise Exception(f"llama.cpp conversion failed: {result.stderr}")
        
        print(f"✓ GGUF conversion completed: {final_filename}")
        
        # Verify the file was created
        if os.path.exists(final_filename):
            file_size = os.path.getsize(final_filename)
            print(f"✓ File created successfully: {final_filename} ({file_size:,} bytes)")
        else:
            raise Exception(f"GGUF file was not created: {final_filename}")
        
        # Clean up temporary directory
        if os.path.exists(temp_model_dir):
            shutil.rmtree(temp_model_dir)
            print(f"Cleaned up {temp_model_dir}")
            
    except Exception as e:
        print(f"Error during conversion: {e}")
        # Clean up on error
        if os.path.exists(temp_model_dir):
            shutil.rmtree(temp_model_dir)
        raise
    
    return final_filename

def upload_to_hf(gguf_files, repo_name, hf_token):
    """Upload GGUF files to Hugging Face"""
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment variables. Skipping upload.")
        return
    
    try:
        api = HfApi()
        
        # Create repository if it doesn't exist
        try:
            create_repo(repo_name, token=hf_token, exist_ok=True)
            print(f"Repository {repo_name} ready")
        except Exception as e:
            print(f"Repository creation info: {e}")
        
        # Upload each GGUF file
        for gguf_file in gguf_files:
            if os.path.exists(gguf_file):
                filename = os.path.basename(gguf_file)
                print(f"Uploading {filename} to {repo_name}...")
                api.upload_file(
                    path_or_fileobj=gguf_file,
                    path_in_repo=filename,
                    repo_id=repo_name,
                    token=hf_token
                )
                print(f"✓ Uploaded {filename}")
            else:
                print(f"Warning: {gguf_file} not found")
        
        print(f"\n🎉 All GGUF files uploaded to: https://huggingface.co/{repo_name}")
        
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")

def main():
    """Main conversion function - fully automated"""
    # Load configuration
    config = load_config()
    model_config = config['model_config']
    
    # Get model name and create GGUF repo name
    hub_model_name = model_config['hub_model_name']
    base_model_name = hub_model_name.split('/')[-1]  # Get just the model name part
    gguf_repo_name = f"{hub_model_name}-gguf"
    
    print(f"Source model: {hub_model_name}")
    print(f"GGUF repository: {gguf_repo_name}")
    
    # Quantization types to convert
    # Only exporting Q8_0 (8-bit quantization, high quality)
    quantization_types = [
        "Q8_0",     # 8-bit quantization (high quality)
    ]
    
    # Create gguf directory
    create_gguf_directory()
    
    # Load model
    model, tokenizer = load_model(config)
    
    print(f"\nConverting all quantization types automatically...")
    gguf_files = []
    
    # Convert all quantization types
    for qtype in quantization_types:
        try:
            print(f"\n{'='*50}")
            print(f"Converting {qtype}...")
            gguf_file = convert_to_gguf(model, tokenizer, qtype, base_model_name)
            if os.path.exists(gguf_file):
                gguf_files.append(gguf_file)
                print(f"✓ {qtype} conversion completed: {gguf_file}")
            else:
                print(f"✗ {qtype} conversion failed")
        except Exception as e:
            print(f"✗ Error converting {qtype}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print(f"GGUF conversion completed!")
    print(f"Successfully converted {len(gguf_files)} out of {len(quantization_types)} formats")
    
    # Upload to Hugging Face
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        try:
            from huggingface_hub import HfFolder
            hf_token = HfFolder.get_token()
        except Exception:
            hf_token = None
    
    if gguf_files:
        print(f"\nUploading to Hugging Face: {gguf_repo_name}")
        upload_to_hf(gguf_files, gguf_repo_name, hf_token)
    else:
        print("\nNo GGUF files to upload")
    
    print("\n🎉 Process completed!")
    if gguf_files:
        print(f"📁 Local files in: gguf/")
        if hf_token:
            print(f"🤗 Hugging Face: https://huggingface.co/{gguf_repo_name}")
        for f in gguf_files:
            print(f"   - {os.path.basename(f)}")

if __name__ == "__main__":
    main()
