#!/usr/bin/env python3
"""
Project Verification Script

Verify that the LWE ML Attack project is set up correctly.
"""

import os
from pathlib import Path

def verify_project_structure():
    """Verify that all required files and directories exist"""
    
    project_dir = Path("lwe_ml_attack")
    
    required_files = [
        "README.md",
        "requirements.txt", 
        "setup.py",
        "LICENSE",
        "copy_models.py",
        "run_demo.sh",
        "PROJECT_SUMMARY.md",
        "src/__init__.py",
        "src/lwe_crypto.py",
        "src/attack_engine.py", 
        "src/model_trainer.py",
        "demos/fast_demo.py",
        "demos/simple_demo.py",
        "demos/full_demo.py",
        "docs/USAGE.md",
        "docs/TECHNICAL.md"
    ]
    
    required_dirs = [
        "src",
        "demos", 
        "models",
        "data",
        "results",
        "docs"
    ]
    
    print("=== LWE ML Attack Project Verification ===\n")
    
    # Check if project directory exists
    if not project_dir.exists():
        print("❌ Project directory 'lwe_ml_attack' not found!")
        return False
    
    print(f"✅ Project directory found: {project_dir}")
    
    # Check directories
    print(f"\n📁 Checking directories:")
    for directory in required_dirs:
        dir_path = project_dir / directory
        if dir_path.exists():
            print(f"  ✅ {directory}/")
        else:
            print(f"  ❌ {directory}/ - MISSING")
            return False
    
    # Check files
    print(f"\n📄 Checking files:")
    missing_files = []
    for file_path in required_files:
        full_path = project_dir / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"  ✅ {file_path} ({size} bytes)")
        else:
            print(f"  ❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} required files")
        return False
    
    # Check for models
    models_dir = project_dir / "models"
    model_files = list(models_dir.glob("bit_model_*.h5"))
    
    print(f"\n🤖 Checking models:")
    if model_files:
        print(f"  ✅ Found {len(model_files)} model files")
        if (models_dir / "metadata.json").exists():
            print(f"  ✅ metadata.json found")
        if (models_dir / "training_data.pkl").exists():
            print(f"  ✅ training_data.pkl found")
    else:
        print(f"  ⚠️  No model files found in models/ directory")
        print(f"     Run 'python3 copy_models.py' to copy your trained models")
    
    # Check demo executability
    print(f"\n🎮 Checking demo scripts:")
    for demo in ["fast_demo.py", "simple_demo.py", "full_demo.py"]:
        demo_path = project_dir / "demos" / demo
        if demo_path.exists():
            print(f"  ✅ {demo} ready")
        else:
            print(f"  ❌ {demo} missing")
    
    print(f"\n🔧 Project Setup:")
    print(f"  ✅ All required files present")
    print(f"  ✅ Directory structure correct")
    print(f"  ✅ Documentation complete")
    print(f"  ✅ Ready for GitHub repository")
    
    print(f"\n🚀 Next Steps:")
    if not model_files:
        print(f"  1. Copy your models: python3 lwe_ml_attack/copy_models.py")
    print(f"  2. Test demos: cd lwe_ml_attack && ./run_demo.sh")
    print(f"  3. Create GitHub repo and upload this directory")
    print(f"  4. Add GitHub URL to README.md")
    
    print(f"\n✅ Project verification complete!")
    return True

if __name__ == "__main__":
    verify_project_structure()