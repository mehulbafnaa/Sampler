#!/bin/bash

# Log all commands and their outputs
exec 1> >(tee "setup_log.txt") 2>&1
echo "Starting TPU setup at $(date)"

# Function to check command status
check_status() {
    if [ $? -eq 0 ]; then
        echo "✓ $1 completed successfully"
    else
        echo "✗ Error during $1"
        echo "Please check setup_log.txt for details"
        exit 1
    fi
}

# Function to print section headers
print_section() {
    echo "============================================"
    echo "$1"
    echo "============================================"
}

print_section "1. Initial System Updates and GPG Key Fix"
# Fix GPG key issues
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
check_status "GPG key installation"

# Update package lists
sudo apt-get update
sudo apt-get install -y software-properties-common
check_status "Initial system update"

print_section "2. Cleaning Up Existing Python Installations"
# Remove existing Python installations
sudo apt-get remove -y python3.8 python3.9 python3.11 python3-pip
sudo apt-get autoremove -y
sudo apt-get clean

# Remove existing Python directories
sudo rm -rf /usr/lib/python3.8
sudo rm -rf /usr/lib/python3.9
sudo rm -rf /usr/lib/python3.11
sudo rm -rf /usr/local/lib/python3.8
sudo rm -rf /usr/local/lib/python3.9
sudo rm -rf /usr/local/lib/python3.11

# Remove alternatives
sudo update-alternatives --remove-all python3 2>/dev/null || true
sudo update-alternatives --remove-all python 2>/dev/null || true
check_status "Python cleanup"

print_section "3. Installing Python 3.10"
# Add and install Python 3.10
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-distutils python3.10-dev
check_status "Python 3.10 installation"

# Make Python 3.10 the default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
check_status "Python default setup"

print_section "4. Setting up Python Environment"
# Install pip for Python 3.10
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
check_status "Pip installation"

# Update PATH
echo 'export PATH="/usr/bin/python3.10:$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo 'export PYTHONPATH="/usr/lib/python3.10:$PYTHONPATH"' >> ~/.bashrc
sudo ln -sf /usr/bin/python3.10 /usr/local/bin/python3
sudo ln -sf /usr/bin/python3.10 /usr/local/bin/python
source ~/.bashrc
check_status "PATH setup"

print_section "5. Installing System Dependencies"
sudo apt-get install -y \
    git \
    libtpu1 \
    wget \
    curl \
    build-essential
check_status "System dependencies installation"

print_section "6. Installing Python Packages"
# Install required Python packages
python3.10 -m pip install --upgrade pip
python3.10 -m pip install --user \
    jupyter \
    notebook \
    torch \
    "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    torch_xla \
    tensorflow \
    flax \
    optax \
    tensorboard \
    ipykernel \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    tqdm
check_status "Python packages installation"

print_section "7. Setting up Jupyter Environment"
# Create TPU kernel for Jupyter
python3.10 -m ipykernel install --user --name tpu_kernel --display-name "Python 3.10 (TPU)"
check_status "Jupyter kernel setup"

print_section "8. Verifying Installation"
# Verify Python installation
echo "Python version:"
python3 --version
echo "Pip version:"
pip3 --version
echo "Python executable location:"
which python3
which python

# Verify TPU setup
echo "Checking JAX TPU setup..."
python3.10 -c "import jax; print('JAX version:', jax.__version__); print('Available devices:', jax.devices())"
check_status "Installation verification"

print_section "9. Final System Check"
# Print system information
echo "System PATH:"
echo $PATH
echo "Python path:"
python3.10 -c "import sys; print('\n'.join(sys.path))"

# Print completion message
print_section "Setup Complete!"
echo "TPU Environment Setup completed at $(date)"
echo "Important Notes:"
echo "1. Check setup_log.txt for detailed installation log"
echo "2. Use 'python3' or 'python' to run Python 3.10"
echo "3. TPU kernel is available in Jupyter as 'Python 3.10 (TPU)'"
echo "4. Remember to run 'source ~/.bashrc' in new terminals"
echo "5. If using VS Code, restart it to recognize the new Python setup"
