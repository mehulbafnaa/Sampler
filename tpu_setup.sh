#!/bin/bash

# Log all commands and their outputs
LOGFILE="setup_log.txt"
exec 1> >(tee -a "$LOGFILE") 2>&1
echo "Starting TPU setup at $(date)"

# Function to check command status
check_status() {
    if [ $? -eq 0 ]; then
        echo "✓ $1 completed successfully"
    else
        echo "✗ Error during $1"
        echo "Check $LOGFILE for details"
        return 1
    fi
}

# Function to print section headers
print_section() {
    echo "============================================"
    echo "$1"
    echo "============================================"
}

# Function to backup Python environment
backup_python_env() {
    print_section "Backing up Python Environment"
    # Create backup directory
    BACKUP_DIR="python_backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup Python related configurations
    if [ -f /usr/bin/python3 ]; then
        cp -L /usr/bin/python3 "$BACKUP_DIR/" 2>/dev/null || true
    fi
    python3 -V > "$BACKUP_DIR/python_version.txt" 2>/dev/null || true
    pip3 freeze > "$BACKUP_DIR/requirements.txt" 2>/dev/null || true
    echo "Python environment backed up to $BACKUP_DIR"
}

# Function to verify Python installation
verify_python() {
    local python_cmd=$1
    if command -v $python_cmd >/dev/null 2>&1; then
        echo "$python_cmd found: $($python_cmd -V)"
        return 0
    else
        echo "$python_cmd not found"
        return 1
    fi
}

# Main installation steps
main() {
    # 1. Backup current environment
    backup_python_env
    
    # 2. System Updates and Prerequisites
    print_section "System Updates and Prerequisites"
    # Fix GPG key issues
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    check_status "GPG key installation" || return 1
    
    # Update package lists
    sudo apt-get update
    sudo apt-get install -y software-properties-common
    check_status "System update" || return 1
    
    # 3. Install Python 3.10
    print_section "Installing Python 3.10"
    # Add deadsnakes PPA
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update
    
    # Install Python 3.10 and related packages
    sudo apt-get install -y \
        python3.10 \
        python3.10-venv \
        python3.10-distutils \
        python3.10-dev
    check_status "Python 3.10 installation" || return 1
    
    # 4. Set up Python environment
    print_section "Setting up Python Environment"
    # Install pip for Python 3.10
    curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3.10 get-pip.py --user
    check_status "Pip installation" || return 1
    
    # 5. Install TPU dependencies
    print_section "Installing TPU Dependencies"
    sudo apt-get install -y libtpu1
    check_status "TPU dependencies installation" || return 1
    
    # 6. Install Python packages
    print_section "Installing Python Packages"
    python3.10 -m pip install --user --upgrade pip
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
        ipykernel
    check_status "Python packages installation" || return 1
    
    # 7. Set up Jupyter kernel
    print_section "Setting up Jupyter Kernel"
    python3.10 -m ipykernel install --user --name tpu_kernel --display-name "Python 3.10 (TPU)"
    check_status "Jupyter kernel setup" || return 1
    
    # 8. Verify installation
    print_section "Verifying Installation"
    verify_python "python3.10"
    python3.10 -m pip --version
    
    # 9. Set up convenient aliases
    echo 'alias python="python3.10"' >> ~/.bashrc
    echo 'alias pip="python3.10 -m pip"' >> ~/.bashrc
    source ~/.bashrc
    
    print_section "Installation Complete!"
    echo "Python 3.10 and TPU environment setup completed successfully"
    echo "Please run 'source ~/.bashrc' to use the new aliases"
    echo "Use 'jupyter notebook' to start using notebooks"
}

# Run main installation
main
if [ $? -eq 0 ]; then
    echo "Setup completed successfully!"
else
    echo "Setup failed. Please check $LOGFILE for details"
    exit 1
fi
