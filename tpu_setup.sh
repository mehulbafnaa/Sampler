#!/bin/bash

# Log all commands and their outputs
LOGFILE="setup_log.txt"
exec 1> >(tee -a "$LOGFILE") 2>&1

# Color codes for formatted output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print section headers
print_section() {
    echo -e "${YELLOW}============================================"
    echo "$1"
    echo "============================================${NC}"
}

# Function to handle errors
handle_error() {
    echo -e "${RED}Error: $1${NC}"
    echo "Setup failed at $(date)"
    exit 1
}

# Function to print success message
print_success() {
    echo -e "${GREEN}✔ $1${NC}"
}

# Function to check command success
check_command() {
    if ! "$@"; then
        handle_error "Command failed: $*"
    fi
}

# Trap any unexpected errors
set -e
trap 'handle_error "An unexpected error occurred on line $LINENO"' ERR

print_section "1. Initial System Update"
# Add retry mechanism for apt operations
apt_update_with_retry() {
    local retries=3
    local count=0
    until [ $count -ge $retries ]
    do
        if sudo apt-get update; then
            break
        fi
        count=$((count + 1))
        echo "Retry attempt $count of $retries for apt-get update"
        sleep 5
    done
    if [ $count -ge $retries ]; then
        handle_error "Failed to update package lists after $retries attempts"
    fi
}

apt_update_with_retry
check_command sudo apt-get upgrade -y
check_command sudo apt-get install -y software-properties-common curl gnupg
print_success "System update completed"

print_section "2. Adding Python and Cloud Repositories"
# More secure way to add Google Cloud repository
check_command sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/cloud.google.gpg
sudo chmod a+r /etc/apt/keyrings/cloud.google.gpg
echo "deb [signed-by=/etc/apt/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Add Python repository with retry
check_command sudo add-apt-repository -y ppa:deadsnakes/ppa
apt_update_with_retry
print_success "Repositories added successfully"

print_section "3. Installing Python 3.10"
check_command sudo apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils
print_success "Python 3.10 installed"

print_section "4. Installing pip"
if ! command -v pip3.10 &> /dev/null; then
    check_command curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    check_command sudo python3.10 get-pip.py
    rm get-pip.py
fi
print_success "pip installed"

print_section "5. Setting up Virtual Environment"
VENV_PATH="$HOME/tpu_env"
check_command python3.10 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
print_success "Virtual environment created and activated"

print_section "6. Installing TPU Dependencies"
# Update package lists
apt_update_with_retry

# Install libtpu with fallback options
if ! sudo apt-get install -y libtpu1; then
    echo -e "${YELLOW}Warning: Standard libtpu1 package not found. Installing from Google's repository...${NC}"
    # Fallback to direct download if needed
    curl -fsSL https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/libtpu/latest/libtpu.so -o libtpu.so
    sudo mv libtpu.so /usr/lib/
fi

print_section "7. Installing Python Packages"
# Update pip first
check_command pip install --upgrade pip setuptools wheel

# Install packages with retry mechanism
install_package() {
    local package=$1
    local retries=3
    local count=0
    until [ $count -ge $retries ]
    do
        if pip install "$package"; then
            return 0
        fi
        count=$((count + 1))
        echo "Retry attempt $count of $retries for installing $package"
        sleep 5
    done
    return 1
}

# Install core packages
packages=(
    "jupyter"
    "notebook"
    "torch_xla"
    "torch>=2.0.0"
    "tensorflow"
    "flax"
    "optax"
    "tensorboard"
    "ipykernel"
    "transformers"
    "datasets"
)

for package in "${packages[@]}"; do
    check_command install_package "$package"
done

# Install JAX separately due to special URL requirement
check_command pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

print_success "Python packages installed"

print_section "8. Setting up Jupyter Kernel"
check_command python -m ipykernel install --user --name tpu_kernel --display-name "Python 3.10 (TPU)"
print_success "Jupyter kernel installed"

print_section "9. Verifying Installation"
echo "Python version:"
check_command python --version
echo "Pip version:"
check_command pip --version

print_section "10. Setting up Environment Variables"
# Add environment variables to .bashrc if not already present
{
    echo 'export PATH="$HOME/.local/bin:$PATH"'
    echo "export PYTHONPATH=\"$VENV_PATH/lib/python3.10/site-packages:\$PYTHONPATH\""
    echo "export LD_LIBRARY_PATH=/usr/lib:\$LD_LIBRARY_PATH"
} >> ~/.bashrc

print_section "Setup Complete!"
echo -e "${GREEN}TPU environment setup completed successfully at $(date)${NC}"
echo "Please run: source ~/.bashrc"
echo "To activate the virtual environment, run: source $VENV_PATH/bin/activate"

# Create a simple test script to verify TPU setup
cat > test_tpu_setup.py << 'EOL'
import jax
import torch_xla.core.xla_model as xm
import tensorflow as tf

print("\nJAX devices:", jax.devices())
print("\nPyTorch XLA devices:", xm.get_xla_supported_devices())
print("\nTensorFlow devices:", tf.config.list_physical_devices())
EOL

print_success "Created test script: test_tpu_setup.py"
echo "Run 'python test_tpu_setup.py' to verify TPU setup"
