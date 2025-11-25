#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Evaluation Script
Evaluates the finetuned model by generating OpenSCAD code and checking render success
"""

import os
import json
import subprocess
import re
import tempfile
import requests
import base64
from pathlib import Path
from huggingface_hub import hf_hub_download
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# =============================================================================
# CONFIGURATION
# =============================================================================

# VLM Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llava:7b"
MAX_WORKERS = min(4, multiprocessing.cpu_count())  # Parallel workers for rendering/judging

def load_config():
    """Load configuration from config.json"""
    with open('config.json', 'r') as f:
        return json.load(f)

# Test objects will be loaded from config.json
TEST_OBJECTS = []

# =============================================================================
# MODEL SETUP
# =============================================================================

def download_gguf_model(config):
    """Download Q8 GGUF model from Hugging Face"""
    model_config = config['model_config']
    hub_model_name = model_config['hub_model_name']
    base_model_name = hub_model_name.split('/')[-1]
    gguf_repo_name = f"{hub_model_name}-gguf"
    
    # Model filename
    model_filename = f"{base_model_name}-q8_0.gguf"
    
    print(f"🤖 Downloading GGUF model: {model_filename}")
    print(f"   Repository: {gguf_repo_name}")
    
    try:
        model_path = hf_hub_download(
            repo_id=gguf_repo_name,
            filename=model_filename,
            cache_dir="./models",
        )
        print(f"✓ Model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        raise

def get_llama_cli_path():
    """Find llama-cli executable"""
    possible_paths = [
        "/home/riftuser/bob/llama.cpp/build/bin/llama-cli",
        os.path.expanduser("~/llama.cpp/build/bin/llama-cli"),
        os.path.join(os.getcwd(), "llama.cpp/build/bin/llama-cli"),
        "llama-cli",
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            print(f"✓ Found llama-cli at: {path}")
            return path
    
    raise Exception(
        "llama-cli not found. Please build llama.cpp:\n"
        "  cd ~/bob/llama.cpp && mkdir build && cd build\n"
        "  cmake .. && cmake --build . --config Release"
    )

# =============================================================================
# INFERENCE
# =============================================================================

def run_inference(model_path, object_name, llama_cli_path):
    """Run inference using llama-cli and extract token count"""
    prompt = f"hey cadmonkey, create me a {object_name}"

    print(f"  🎯 Generating: {object_name}")

    try:
        # Build command - simpler approach
        cmd = [
            llama_cli_path,
            "-m", model_path
        ]

        # Run with stdin for the prompt
        result = subprocess.run(
            cmd,
            input=prompt + "\n",
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        error_msg = None
        if result.returncode != 0:
            print(f"    ⚠ Return code: {result.returncode}")
            error_msg = f"Return code: {result.returncode}"
            if result.stderr:
                print(f"    Error: {result.stderr[:300]}")
                error_msg += f" - {result.stderr[:500]}"

        # Extract token count from stderr (llama-cli outputs stats there)
        tokens_generated = 0
        if result.stderr:
            # Look for pattern like "tokens_evaluated = X" or similar
            import re
            match = re.search(r'(\d+)\s+(?:tokens?|t/s)', result.stderr)
            if match:
                try:
                    tokens_generated = int(match.group(1))
                except ValueError:
                    pass

        # If we couldn't extract from stderr, estimate from output length
        if tokens_generated == 0 and result.stdout:
            # Rough estimate: ~4 chars per token
            tokens_generated = len(result.stdout) // 4

        return result.stdout if result.stdout else None, tokens_generated, error_msg

    except subprocess.TimeoutExpired:
        error_msg = "Timeout (>300s) generating response"
        print(f"    ✗ {error_msg}")
        return None, 0, error_msg
    except FileNotFoundError:
        error_msg = f"llama-cli not found at: {llama_cli_path}"
        print(f"    ✗ {error_msg}")
        return None, 0, error_msg
    except Exception as e:
        error_msg = str(e)
        print(f"    ✗ Error: {e}")
        return None, 0, error_msg

# =============================================================================
# OPENSCAD CODE EXTRACTION
# =============================================================================

def extract_openscad_code(text):
    """Extract OpenSCAD code from the generated text"""
    if not text:
        return None

    # Strip llama-cli REPL prompt markers
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        # Skip the EOF marker line
        if line.strip() == '> EOF by user' or line.strip() == '>EOF by user':
            continue

        # Remove '> ' prefix from lines that start with it
        if line.startswith('> '):
            cleaned_lines.append(line[2:])  # Remove '> ' (2 characters)
        elif line.startswith('>') and len(line) > 1 and line[1] != '>':
            cleaned_lines.append(line[1:])  # Remove '>' (1 character)
        else:
            cleaned_lines.append(line)

    # Join and strip whitespace
    cleaned_code = '\n'.join(cleaned_lines).strip()

    if cleaned_code:
        return cleaned_code

    return None

# =============================================================================
# RENDERING
# =============================================================================

def get_openscad_path():
    """Find OpenSCAD executable"""
    possible_paths = [
        "/usr/bin/openscad",
        "/snap/bin/openscad",
        os.path.expanduser("~/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD"),
        "openscad",
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    
    return None

def get_blender_path():
    """Find Blender executable"""
    possible_paths = [
        "/usr/bin/blender",
        "/snap/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
        os.path.expanduser("~/Applications/Blender.app/Contents/MacOS/Blender"),
        "blender",
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    
    return None

def render_openscad(scad_code, object_name, evaluation_dir):
    """Render OpenSCAD code to STL/PNG using Blender for high-quality images"""
    openscad_path = get_openscad_path()
    blender_path = get_blender_path()

    if not openscad_path:
        error_msg = "OpenSCAD not found, skipping rendering"
        print(f"    ⚠ {error_msg}")
        return False, error_msg
    
    if not blender_path:
        error_msg = "Blender not found, falling back to OpenSCAD PNG rendering"
        print(f"    ⚠ {error_msg}")
        return render_openscad_fallback(scad_code, object_name, evaluation_dir, openscad_path)

    try:
        # Create evaluation directory if it doesn't exist
        os.makedirs(evaluation_dir, exist_ok=True)
        
        # Clean object name for filename (remove spaces and special chars)
        safe_name = re.sub(r'[^\w\-_]', '_', object_name)
        
        scad_file = os.path.join(evaluation_dir, f"{safe_name}.scad")
        stl_file = os.path.join(evaluation_dir, f"{safe_name}.stl")
        png_file = os.path.join(evaluation_dir, f"{safe_name}.png")

        # Clean the code: convert \n escape sequences to actual newlines
        cleaned_code = scad_code.replace('\\n', '\n')
        
        # Write OpenSCAD code to file
        with open(scad_file, 'w') as f:
            f.write(cleaned_code)

        print(f"    Running OpenSCAD to STL...")
        
        # First create STL file with OpenSCAD
        stl_result = subprocess.run([
            openscad_path, "-o", stl_file, scad_file
        ], capture_output=True, timeout=30, text=True)
        
        if stl_result.returncode != 0:
            error_msg = f"OpenSCAD STL creation failed: {stl_result.stderr[:200]}"
            print(f"    ✗ {error_msg}")
            return False, error_msg
        
        if not os.path.exists(stl_file) or os.path.getsize(stl_file) == 0:
            error_msg = "STL file not created or empty"
            print(f"    ✗ {error_msg}")
            return False, error_msg
        
        stl_size = os.path.getsize(stl_file)
        print(f"    ✓ STL created: {object_name} ({stl_size} bytes)")
        
        # Now render with Blender for high-quality PNG
        print(f"    Running Blender render...")
        
        blender_script = f'''
import bpy
import sys
import mathutils

# Clear existing mesh
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Import STL
bpy.ops.wm.stl_import(filepath='{stl_file}')

# Get the imported object and calculate its bounds
imported_obj = None
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH' and obj.name != 'Cube':  # Skip default cube
        imported_obj = obj
        break

if imported_obj:
    # Calculate object bounds
    bbox_corners = [imported_obj.matrix_world @ mathutils.Vector(corner) for corner in imported_obj.bound_box]
    min_coords = mathutils.Vector((min(corner.x for corner in bbox_corners),
                                  min(corner.y for corner in bbox_corners),
                                  min(corner.z for corner in bbox_corners)))
    max_coords = mathutils.Vector((max(corner.x for corner in bbox_corners),
                                  max(corner.y for corner in bbox_corners),
                                  max(corner.z for corner in bbox_corners)))
    
    # Calculate center and size
    center = (min_coords + max_coords) / 2
    size = max_coords - min_coords
    max_size = max(size.x, size.y, size.z)
    
    # Position camera at appropriate distance
    distance = max_size * 3  # 3x the object size
    camera_location = center + mathutils.Vector((distance, -distance, distance))
else:
    # Fallback if no object found
    center = mathutils.Vector((0, 0, 0))
    camera_location = mathutils.Vector((10, -10, 10))

# Set up camera and lighting
bpy.ops.object.camera_add(location=camera_location)
camera = bpy.context.object
camera.rotation_euler = (1.1, 0, 0.785)

# Set camera as active
bpy.context.scene.camera = camera

# Add lighting
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))

# Set render settings
bpy.context.scene.render.resolution_x = 800
bpy.context.scene.render.resolution_y = 600
bpy.context.scene.render.filepath = '{png_file}'

# Render
bpy.ops.render.render(write_still=True)
'''
        
        # Write Blender script to temporary file
        script_file = os.path.join(evaluation_dir, f"{safe_name}_render_script.py")
        with open(script_file, 'w') as f:
            f.write(blender_script)
        
        # Run Blender headless
        blender_result = subprocess.run([
            blender_path, "--background", "--python", script_file
        ], capture_output=True, timeout=60, text=True)
        
        # Clean up script file
        try:
            os.remove(script_file)
        except:
            pass
        
        if blender_result.returncode != 0:
            error_msg = f"Blender rendering failed: {blender_result.stderr[:200]}"
            print(f"    ✗ {error_msg}")
            return False, error_msg
        
        # Check if PNG was created successfully
        if os.path.exists(png_file) and os.path.getsize(png_file) > 0:
            png_size = os.path.getsize(png_file)
            print(f"    ✓ Successfully rendered PNG: {object_name} ({png_size} bytes)")
            return True, None
        else:
            error_msg = "Blender did not create PNG file"
            print(f"    ✗ {error_msg}")
            return False, error_msg

    except subprocess.TimeoutExpired:
        error_msg = "Rendering timeout (>60s)"
        print(f"    ✗ {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Render error: {str(e)}"
        print(f"    ✗ {error_msg}")
        return False, error_msg

def render_openscad_fallback(scad_code, object_name, evaluation_dir, openscad_path):
    """Fallback rendering using only OpenSCAD (when Blender is not available)"""
    try:
        # Create evaluation directory if it doesn't exist
        os.makedirs(evaluation_dir, exist_ok=True)
        
        # Clean object name for filename (remove spaces and special chars)
        safe_name = re.sub(r'[^\w\-_]', '_', object_name)
        
        scad_file = os.path.join(evaluation_dir, f"{safe_name}.scad")
        stl_file = os.path.join(evaluation_dir, f"{safe_name}.stl")
        png_file = os.path.join(evaluation_dir, f"{safe_name}.png")

        # Clean the code: convert \n escape sequences to actual newlines
        cleaned_code = scad_code.replace('\\n', '\n')
        
        # Write OpenSCAD code to file
        with open(scad_file, 'w') as f:
            f.write(cleaned_code)

        print(f"    Running OpenSCAD render (fallback)...")
        
        # Render to STL
        stl_result = subprocess.run([
            openscad_path, "-o", stl_file, scad_file
        ], capture_output=True, timeout=30, text=True)
        
        if stl_result.returncode != 0:
            error_msg = f"OpenSCAD STL creation failed: {stl_result.stderr[:200]}"
            print(f"    ✗ {error_msg}")
            return False, error_msg
        
        if not os.path.exists(stl_file) or os.path.getsize(stl_file) == 0:
            error_msg = "STL file not created or empty"
            print(f"    ✗ {error_msg}")
            return False, error_msg
        
        stl_size = os.path.getsize(stl_file)
        print(f"    ✓ STL created: {object_name} ({stl_size} bytes)")
        
        # Render to PNG using OpenSCAD
        png_result = subprocess.run([
            openscad_path, "-o", png_file, "--imgsize", "800,600", scad_file
        ], capture_output=True, timeout=30, text=True)
        
        if png_result.returncode != 0:
            print(f"    ⚠ PNG render failed, but STL succeeded")
            return True, "PNG render failed"
        
        if os.path.exists(png_file) and os.path.getsize(png_file) > 0:
            png_size = os.path.getsize(png_file)
            print(f"    ✓ PNG created: {object_name} ({png_size} bytes)")
            return True, None
        else:
            print(f"    ⚠ PNG render failed, but STL succeeded")
            return True, "PNG render failed"

    except subprocess.TimeoutExpired:
        error_msg = "OpenSCAD timeout (>30s)"
        print(f"    ✗ {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Render error: {str(e)}"
        print(f"    ✗ {error_msg}")
        return False, error_msg

# =============================================================================
# VLM JUDGING
# =============================================================================

def image_to_base64(image_path):
    """Convert image to base64 string for VLM"""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        print(f"    Error converting image to base64: {e}")
        return None

def judge_image_similarity(image_path, object_name):
    """Use VLM to judge if the rendered image resembles the requested object"""
    if not os.path.exists(image_path):
        return False, "Image file not found"
    
    # Convert image to base64
    image_b64 = image_to_base64(image_path)
    if not image_b64:
        return False, "Failed to convert image to base64"
    
    # Create prompt for VLM
    prompt = f"Look at this 3D rendered object. Does this image resemble a {object_name}? Answer only 'yes' or 'no'."
    
    # Prepare payload for vision model
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        if response.status_code == 200:
            answer = response.json()['response'].strip().lower()
            # Extract yes/no from response
            if 'yes' in answer and 'no' not in answer:
                return True, answer
            elif 'no' in answer and 'yes' not in answer:
                return False, answer
            else:
                # Fallback: look for explicit yes/no
                if answer.startswith('yes'):
                    return True, answer
                elif answer.startswith('no'):
                    return False, answer
                else:
                    return False, f"Unclear response: {answer}"
        else:
            return False, f"VLM API error: {response.status_code}"
    except Exception as e:
        return False, f"VLM API error: {str(e)}"

def process_object_parallel(args):
    """Process a single object: generate and render (for parallel execution)"""
    model_path, object_name, llama_cli_path, evaluation_dir, index = args
    
    print(f"  🎯 Processing: {object_name}")
    
    # Generate OpenSCAD code
    output, tokens_generated, inference_error = run_inference(model_path, object_name, llama_cli_path)
    
    if not output:
        return {
            'object': object_name,
            'code_extracted': False,
            'render_success': False,
            'tokens_generated': 0,
            'code': None,
            'inference_error': inference_error,
            'render_error': None,
        }
    
    # Extract code
    scad_code = extract_openscad_code(output)
    code_extracted = scad_code is not None and len(scad_code.strip()) > 0
    
    render_success = False
    render_error = None
    
    if code_extracted:
        print(f"    ✓ OpenSCAD code extracted ({len(scad_code)} chars, ~{tokens_generated} tokens)")
        
        # Try to render
        render_success, render_error = render_openscad(scad_code, object_name, evaluation_dir)
        
        if render_success:
            print(f"    ✓ Render successful")
        else:
            print(f"    ✗ Render failed: {render_error}")
    else:
        print(f"    ✗ Failed to extract OpenSCAD code")
        render_error = "No code extracted from response"
    
    return {
        'object': object_name,
        'code_extracted': code_extracted,
        'render_success': render_success,
        'tokens_generated': tokens_generated,
        'code': scad_code,
        'inference_error': inference_error,
        'render_error': render_error,
    }

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(config):
    """Main evaluation function"""
    print("\n" + "="*60)
    print("🚀 MODEL EVALUATION - OpenSCAD Generation Test")
    print("="*60)

    # Get model path and name
    model_path = download_gguf_model(config)
    llama_cli_path = get_llama_cli_path()
    model_name = config['model_config']['hub_model_name']
    
    # Load test objects from config
    test_objects = config.get('dataset_config', {}).get('test_objects', [])
    
    if not test_objects:
        print("❌ Error: No test_objects found in config.json under dataset_config")
        print("   Please add test_objects to your config.json, e.g.:")
        print('   "dataset_config": {')
        print('     "test_objects": ["cat", "car", "tree"]')
        print('   }')
        exit(1)

    print(f"\n📋 Test Configuration:")
    print(f"   Model: {model_name}")
    print(f"   Objects to test: {len(test_objects)}")
    print(f"   Objects: {', '.join(test_objects)}")
    print()

    # Create evaluation directory and results file
    timestamp = datetime.now()
    base_evaluation_dir = "evaluation"
    model_name_clean = model_name.replace('/', '_').replace(':', '_')  # Clean model name for folder
    run_folder = f"{model_name_clean}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    evaluation_dir = os.path.join(base_evaluation_dir, run_folder)
    os.makedirs(evaluation_dir, exist_ok=True)
    results_file = os.path.join(evaluation_dir, "evaluation_results.json")

    results = []

    def save_results():
        """Helper function to save current results to JSON"""
        total = len(results)
        code_success = sum(1 for r in results if r['code_extracted'])
        render_success = sum(1 for r in results if r['render_success'])
        avg_tokens = sum(r['tokens_generated'] for r in results) / total if total > 0 else 0

        with open(results_file, 'w') as f:
            json.dump({
                'model_name': model_name,
                'timestamp': timestamp.isoformat(),
                'date': timestamp.strftime('%Y-%m-%d'),
                'time': timestamp.strftime('%H:%M:%S'),
                'total_tests': len(test_objects),
                'completed_tests': total,
                'code_extraction_success': code_success,
                'code_extraction_rate': f"{code_success/total*100:.1f}%" if total > 0 else "0%",
                'render_success': render_success,
                'render_success_rate': f"{render_success/total*100:.1f}%" if total > 0 else "0%",
                'average_tokens_generated': f"{avg_tokens:.0f}",
                'results': results
            }, f, indent=2)

    # Initialize empty results file
    save_results()
    print(f"📁 Evaluation run folder: {evaluation_dir}/")
    print(f"📝 Results will be saved to: {results_file}")
    print(f"🖼️  Rendered images will be saved to: {evaluation_dir}/")
    print(f"⚡ Parallel processing with {MAX_WORKERS} workers")

    # Prepare arguments for parallel processing
    process_args = [
        (model_path, obj, llama_cli_path, evaluation_dir, i)
        for i, obj in enumerate(test_objects, 1)
    ]

    print(f"\n🚀 Starting parallel processing of {len(test_objects)} objects...")
    print("-" * 60)

    # Process objects in parallel
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_object_parallel, args): args[4]
            for args in process_args
        }

        # Process results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            
            try:
                result = future.result()
                results.append(result)
                
                # Save results after each completion
                save_results()
                
                # Print progress
                print(f"\n[{len(results)}/{len(test_objects)}] Completed: {result['object']}")
                print(f"  Code: {'✓' if result['code_extracted'] else '✗'}")
                print(f"  Render: {'✓' if result['render_success'] else '✗'}")
                
            except Exception as e:
                print(f"\n[{index}] Error processing: {e}")
                # Add error result
                results.append({
                    'object': f"object_{index}",
                    'code_extracted': False,
                    'render_success': False,
                    'tokens_generated': 0,
                    'code': None,
                    'inference_error': str(e),
                    'render_error': None,
                })
                save_results()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)

    code_success = sum(1 for r in results if r['code_extracted'])
    render_success = sum(1 for r in results if r['render_success'])
    total = len(results)
    avg_tokens = sum(r['tokens_generated'] for r in results) / total if total > 0 else 0

    print(f"\n✨ Code Extraction Success Rate: {code_success}/{total} ({code_success/total*100:.1f}%)")
    print(f"🎨 Render Success Rate: {render_success}/{total} ({render_success/total*100:.1f}%)")
    print(f" Average Tokens Generated: {avg_tokens:.0f}")

    print(f"\n{'Object':<30} {'Code':<6} {'Render':<8} {'Tokens':<8}")
    print("-" * 60)
    for r in results:
        code_status = "✓" if r['code_extracted'] else "✗"
        render_status = "✓" if r['render_success'] else "✗"
        print(f"{r['object']:<30} {code_status:<6} {render_status:<8} {r['tokens_generated']:<8}")

    print(f"\n💾 Final results saved to: {results_file}")
    print(f"🖼️  All rendered images saved to: {evaluation_dir}/")
    print("="*60)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    try:
        config = load_config()
        evaluate_model(config)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)
