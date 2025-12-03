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
from pathlib import Path
from huggingface_hub import hf_hub_download
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# Sequential processing - one object at a time

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
        # Build command with proper flags for one-shot generation
        # Use -p for prompt input, -n to limit tokens, --single-turn to exit after one response
        cmd = [
            llama_cli_path,
            "-m", model_path,
            "-p", prompt,
            "-n", "2048",  # Limit to 2048 tokens max
            "--single-turn",  # Run conversation for a single turn only, then exit
        ]

        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3000  # 50 minute timeout
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
        error_msg = "Timeout (>3000s) generating response"
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
    """Find OpenSCAD executable, preferring openscad-nightly if available"""
    # Check for openscad-nightly first (newer versions with better headless support)
    possible_paths = [
        "openscad-nightly",  # Check PATH first
        "/snap/bin/openscad-nightly",
        "/usr/bin/openscad-nightly",
        "/usr/bin/openscad",
        "/snap/bin/openscad",
        os.path.expanduser("~/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD"),
        "openscad",
    ]
    
    for path in possible_paths:
        # For commands in PATH, check if they exist
        if '/' not in path:
            result = subprocess.run(["which", path], capture_output=True, text=True)
            if result.returncode == 0:
                found_path = result.stdout.strip()
                if os.path.exists(found_path) and os.access(found_path, os.X_OK):
                    return found_path
        elif os.path.exists(path) and os.access(path, os.X_OK):
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

def get_xvfb_path():
    """Find xvfb-run executable for headless rendering"""
    possible_paths = [
        "/usr/bin/xvfb-run",
        "/usr/local/bin/xvfb-run",
        "xvfb-run",
    ]
    
    for path in possible_paths:
        # Check if command exists (works for commands in PATH)
        result = subprocess.run(["which", path.split('/')[-1]], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        elif os.path.exists(path) and os.access(path, os.X_OK):
            return path
    
    return None

def render_openscad(scad_code, object_name, evaluation_dir):
    """Render OpenSCAD code to PNG preserving colors and centering the object"""
    openscad_path = get_openscad_path()

    if not openscad_path:
        error_msg = "OpenSCAD not found, skipping rendering"
        print(f"    ⚠ {error_msg}")
        return False, error_msg
    
    # Use OpenSCAD direct rendering to preserve colors
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
    
    # Position camera at appropriate distance, centered on object
    distance = max_size * 2.5  # 2.5x the object size
    # Position camera at isometric angle looking at center
    camera_location = center + mathutils.Vector((distance, -distance, distance))
    
    # Make camera look at the center
    direction = center - camera_location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera_rotation = rot_quat.to_euler()
else:
    # Fallback if no object found
    center = mathutils.Vector((0, 0, 0))
    camera_location = mathutils.Vector((10, -10, 10))
    camera_rotation = (1.1, 0, 0.785)

# Set up camera and lighting
bpy.ops.object.camera_add(location=camera_location)
camera = bpy.context.object
camera.rotation_euler = camera_rotation

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
        
        # Render to PNG using OpenSCAD with camera settings to center the object
        # OpenSCAD requires an X server for PNG rendering, so we use xvfb-run for headless rendering
        xvfb_path = get_xvfb_path()
        
        if not xvfb_path:
            error_msg = ("xvfb-run not found. OpenSCAD requires an X server for PNG rendering.\n"
                        "   Install it with: sudo apt-get install xvfb")
            print(f"    ✗ {error_msg}")
            return False, error_msg
        
        # Use xvfb-run for headless rendering with proper camera settings
        # Set environment variables to help with headless rendering
        env = os.environ.copy()
        env['LIBGL_ALWAYS_SOFTWARE'] = '1'
        env['GALLIUM_DRIVER'] = 'llvmpipe'
        env['MESA_GL_VERSION_OVERRIDE'] = '3.3'
        
        # Try with --render flag for full geometry evaluation (CGAL) instead of preview mode
        # This might be more stable than preview mode in headless environments
        png_result = subprocess.run([
            xvfb_path, "-a", "-s", "-screen 0 1024x768x24",
            openscad_path,
            "-o", png_file,
            "--render",  # Use full render instead of preview
            "--autocenter",
            "--viewall",
            "--imgsize=800,600",
            "--projection=ortho",
            scad_file
        ], capture_output=True, timeout=30, text=True, env=env)
        
        # Check if file was created successfully (sometimes OpenSCAD returns non-zero but still creates file)
        if os.path.exists(png_file) and os.path.getsize(png_file) > 0:
            png_size = os.path.getsize(png_file)
            print(f"    ✓ PNG created with colors preserved: {object_name} ({png_size} bytes)")
            return True, None
        
        if png_result.returncode != 0:
            # Try without --render (preview mode) but with other settings
            png_result = subprocess.run([
                xvfb_path, "-a", "-s", "-screen 0 1024x768x24",
                openscad_path,
                "-o", png_file,
                "--autocenter",
                "--viewall",
                "--imgsize=800,600",
                scad_file
            ], capture_output=True, timeout=30, text=True, env=env)
            
            # Check if file was created
            if os.path.exists(png_file) and os.path.getsize(png_file) > 0:
                png_size = os.path.getsize(png_file)
                print(f"    ✓ PNG created with colors preserved: {object_name} ({png_size} bytes)")
                return True, None
            
            if png_result.returncode != 0:
                # Try minimal command without viewall
                png_result = subprocess.run([
                    xvfb_path, "-a", "-s", "-screen 0 1024x768x24",
                    openscad_path,
                    "-o", png_file,
                    "--autocenter",
                    "--imgsize=800,600",
                    scad_file
                ], capture_output=True, timeout=30, text=True, env=env)
                
                # Check if file was created
                if os.path.exists(png_file) and os.path.getsize(png_file) > 0:
                    png_size = os.path.getsize(png_file)
                    print(f"    ✓ PNG created with colors preserved: {object_name} ({png_size} bytes)")
                    return True, None
                
                if png_result.returncode != 0:
                    # Show both stdout and stderr for debugging
                    error_output = (png_result.stderr or png_result.stdout or "Unknown error")[:500]
                    error_msg = f"OpenSCAD PNG render failed: {error_output}"
                    print(f"    ⚠ {error_msg}")
                    print(f"    Note: OpenSCAD PNG rendering may fail in headless environments.")
                    print(f"    STL file was created successfully, but PNG rendering requires a display.")
                    return False, error_msg
        
        if os.path.exists(png_file) and os.path.getsize(png_file) > 0:
            png_size = os.path.getsize(png_file)
            print(f"    ✓ PNG created with colors preserved: {object_name} ({png_size} bytes)")
            return True, None
        else:
            error_msg = "PNG file not created or empty (OpenSCAD may require a display for PNG rendering)"
            print(f"    ⚠ {error_msg}")
            print(f"    STL file was created successfully.")
            return False, error_msg

    except subprocess.TimeoutExpired:
        error_msg = "OpenSCAD timeout (>30s)"
        print(f"    ✗ {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Render error: {str(e)}"
        print(f"    ✗ {error_msg}")
        return False, error_msg

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
    print(f"⚡ Sequential processing (1 object at a time)")

    print(f"\n🚀 Starting sequential processing of {len(test_objects)} objects...")
    print("-" * 60)

    # Process objects sequentially (one at a time)
    for i, obj in enumerate(test_objects, 1):
        try:
            print(f"\n[{i}/{len(test_objects)}] Processing: {obj}")
            
            # Process single object
            result = process_object_parallel((model_path, obj, llama_cli_path, evaluation_dir, i))
            results.append(result)
            
            # Save results after each completion
            save_results()
            
            # Print progress
            print(f"\n[{i}/{len(test_objects)}] Completed: {result['object']}")
            print(f"  Code: {'✓' if result['code_extracted'] else '✗'}")
            print(f"  Render: {'✓' if result['render_success'] else '✗'}")
            
        except Exception as e:
            print(f"\n[{i}] Error processing: {e}")
            # Add error result
            results.append({
                'object': obj if 'obj' in locals() else f"object_{i}",
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

    render_success = sum(1 for r in results if r['render_success'])
    total = len(results)

    print(f"\n🎨 Render Success Rate: {render_success}/{total} ({render_success/total*100:.1f}%)")

    print(f"\n💾 Final results saved to: {results_file}")
    print(f"🖼️  All rendered images saved to: {evaluation_dir}/")
    
    # List all rendered images and errors
    print(f"\n📸 Rendered Images:")
    print("-" * 60)
    image_files = []
    errors = []
    
    for r in results:
        safe_name = re.sub(r'[^\w\-_]', '_', r['object'])
        png_file = os.path.join(evaluation_dir, f"{safe_name}.png")
        
        if r['render_success'] and os.path.exists(png_file):
            png_size = os.path.getsize(png_file)
            image_files.append((r['object'], png_file, png_size))
            print(f"  ✓ {r['object']:<25} → {png_file} ({png_size} bytes)")
        else:
            # Collect errors
            error_info = {
                'object': r['object'],
                'inference_error': r.get('inference_error'),
                'render_error': r.get('render_error'),
            }
            if error_info['inference_error'] or error_info['render_error']:
                errors.append(error_info)
    
    if not image_files:
        print("  (No images were successfully rendered)")
    else:
        print(f"\n  Total images rendered: {len(image_files)}")
    
    # Show errors if any
    if errors:
        print(f"\n❌ Errors:")
        print("-" * 60)
        for err in errors:
            print(f"  {err['object']}:")
            if err['inference_error']:
                print(f"    Inference: {err['inference_error']}")
            if err['render_error']:
                print(f"    Render: {err['render_error']}")
    
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
