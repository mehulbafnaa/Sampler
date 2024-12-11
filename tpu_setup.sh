#!/bin/bash

# Log all commands and their outputs
LOGFILE="setup_log.txt"
exec 1> >(tee -a "$LOGFILE") 2>&1
echo "Starting TPU setup at $(date)"

# Function to print section headers
print_section() {
    echo "============================================"
    echo "$1"
    echo "============================================"
}

# Function to handle errors
handle_error() {
    echo "Error: $1"
    echo "Setup failed at $(date)"
    exit 1
}

print_section "1. Initial System Update"
sudo apt-get update || handle_error "Failed to update package lists"
sudo apt-get install -y software-properties-common || handle_error "Failed to install software-properties-common"

print_section "2. Adding Python Repository"
sudo add-apt-repository -y ppa:deadsnakes/ppa || handle_error "Failed to add Python repository"
sudo apt-get update || handle_error "Failed to update package lists after adding repository"

print_section "3. Installing Python 3.10"
sudo apt-get install -y python3.10 python3.10-distutils python3.10-dev || handle_error "Failed to install Python 3.10"

print_section "4. Installing pip"
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py || handle_error "Failed to download get-pip.py"
python3.10 get-pip.py --user || handle_error "Failed to install pip"

print_section "5. Setting up PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

print_section "6. Installing TPU Dependencies"
# Fix GPG key first
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add - || handle_error "Failed to add Google Cloud GPG key"
sudo apt-get update || handle_error "Failed to update package lists"
sudo apt-get install -y libtpu1 || handle_error "Failed to install TPU dependencies"

print_section "7. Installing Python Packages"
python3.10 -m pip install --user --upgrade pip || handle_error "Failed to upgrade pip"
python3.10 -m pip install --user \
    jupyter \
    notebook \
    "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    torch \
    torch_xla \
    tensorflow \
    flax \
    optax \
    tensorboard \
    ipykernel || handle_error "Failed to install Python packages"

print_section "8. Setting up Jupyter Kernel"
python3.10 -m ipykernel install --user --name tpu_kernel --display-name "Python 3.10 (TPU)" || handle_error "Failed to setup Jupyter kernel"

print_section "9. Verifying Installation"
echo "Python version:"
python3.10 --version || handle_error "Failed to verify Python installation"
echo "Pip version:"
python3.10 -m pip --version || handle_error "Failed to verify pip installation"

print_section "Setup Complete!"
echo "TPU environment setup completed at $(date)"
echo "Please run: source ~/.bashrc"
echo "Use 'python3.10' to run Python 3.10"
