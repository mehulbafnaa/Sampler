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

# Trap any unexpected errors
set -e
trap 'handle_error "An unexpected error occurred on line $LINENO"' ERR

print_section "1. Initial System Update"
sudo apt-get update || handle_error "Failed to update package lists"
sudo apt-get upgrade -y
sudo apt-get install -y software-properties-common curl gnupg || handle_error "Failed to install required system packages"
print_success "System update completed"

print_section "2. Adding Python and Cloud Repositories"
# Modern way to add Google Cloud repository
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/cloud.google.gpg
echo "deb [signed-by=/etc/apt/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list

# Add Python repository
sudo add-apt-repository -y ppa:deadsnakes/ppa || handle_error "Failed to add Python repository"
sudo apt-get update || handle_error "Failed to update package lists after adding repositories"
print_success "Repositories added successfully"

print_section "3. Installing Python 3.10"
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils || handle_error "Failed to install Python 3.10"
print_success "Python 3.10 installed"

print_section "4. Installing pip"
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py || handle_error "Failed to download get-pip.py"
sudo python3.10 get-pip.py || handle_error "Failed to install pip"
rm get-pip.py
print_success "pip installed"

print_section "5. Setting up PATH"
mkdir -p "$HOME/.local/bin"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
print_success "PATH updated"

print_section "6. Installing TPU Dependencies"
# Update package lists
sudo apt-get update

# Attempt TPU package installation with fallback
if ! sudo apt-get install -y libtpu1; then
    echo -e "${YELLOW}Warning: Standard libtpu1 package not found. Proceeding with alternative installation methods.${NC}"
fi

print_section "7. Installing Python Packages"
python3.10 -m pip install --user --upgrade pip setuptools wheel || handle_error "Failed to upgrade pip"
python3.10 -m pip install --user \
    jupyter \
    notebook \
    "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    torch \
    tensorflow \
    flax \
    optax \
    tensorboard \
    ipykernel || handle_error "Failed to install Python packages"
print_success "Python packages installed"

print_section "8. Setting up Jupyter Kernel"
python3.10 -m ipykernel install --user --name tpu_kernel --display-name "Python 3.10 (TPU)" || handle_error "Failed to setup Jupyter kernel"
print_success "Jupyter kernel installed"

print_section "9. Verifying Installation"
echo "Python version:"
python3.10 --version || handle_error "Failed to verify Python installation"
echo "Pip version:"
python3.10 -m pip --version || handle_error "Failed to verify pip installation"

print_section "Setup Complete!"
echo -e "${GREEN}TPU environment setup completed successfully at $(date)${NC}"
echo "Please run: source ~/.bashrc"
echo "Use 'python3.10' to run Python 3.10"
