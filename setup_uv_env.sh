#!/bin/bash

# MAE-CP One-Click Installation Script (using uv environment)
# Python Version: 3.11

set -e  # Exit immediately if a command exits with a non-zero status

echo "🚀 Starting MAE-CP environment setup..."

# 1. Install uv
echo "📦 Step 1/5: Installing uv..."
if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    pip3 install uv
else
    echo "uv already installed, skipping installation"
fi

# 2. Clone stable-pretraining repository
echo "📥 Step 2/5: Cloning stable-pretraining repository..."
if [ -d "stable-pretraining/.git" ]; then
    echo "stable-pretraining repository already exists, skipping clone"
    cd stable-pretraining
    git pull origin main || git pull origin master || echo "Unable to pull latest code, using existing version"
    cd ..
else
    echo "Cloning stable-pretraining repository..."
    git clone git@github.com:galilai-group/stable-pretraining.git
fi

# 3. Create Python 3.11 virtual environment
echo "🐍 Step 3/5: Creating Python 3.11 virtual environment..."
uv venv .venv --python 3.11

# 4. Activate virtual environment
echo "✅ Step 4/5: Activating virtual environment..."
source .venv/bin/activate

# 5. Install dependencies
echo "📚 Step 5/5: Installing project dependencies..."

# Enter stable-pretraining directory and install
cd stable-pretraining/
echo "Installing stable-pretraining with all extensions..."
uv pip install -e ".[all]"

# Install other dependencies
echo "Installing additional dependencies..."
uv pip install transformers datasets==2.20.0 medmnist

cd ..

echo ""
echo "✨ Installation completed!"
echo ""
echo "🎯 Usage:"
echo "  Activate environment:"
echo "  source .venv/bin/activate"
echo ""

