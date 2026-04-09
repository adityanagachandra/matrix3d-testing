#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/adityanagachandra/matrix3d-testing}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/matrix3d-testing}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-matrix3d}"

MATRIX3D_CKPT_URL="https://ml-site.cdn-apple.com/models/matrix3d/matrix3d_512.pt"
ISNET_GDRIVE_ID="1XHIzgTzY5BQHw140EDIgwIb53K659ENH"

log() {
  printf "\n[%s] %s\n" "$(date +'%H:%M:%S')" "$*"
}

install_miniconda_if_needed() {
  if command -v conda >/dev/null 2>&1; then
    log "Conda already available."
    return
  fi

  log "Installing Miniconda..."
  wget -O "$HOME/miniconda.sh" "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  bash "$HOME/miniconda.sh" -b -p "$HOME/miniconda3"
  eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
  conda init bash
}

activate_conda() {
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
  fi
}

clone_or_update_repo() {
  if [[ -d "$INSTALL_DIR/.git" ]]; then
    log "Repository already exists at $INSTALL_DIR. Pulling latest..."
    git -C "$INSTALL_DIR" pull --ff-only
  else
    log "Cloning repository..."
    git clone "$REPO_URL" "$INSTALL_DIR"
  fi
}

setup_git_auth() {
  if command -v gh >/dev/null 2>&1; then
    log "GitHub CLI detected."
  else
    log "Installing GitHub CLI (gh)..."
    sudo apt-get update
    sudo apt-get install -y gh
  fi

  if gh auth status >/dev/null 2>&1; then
    log "GitHub auth already configured."
  else
    log "Starting browser login for GitHub auth..."
    gh auth login --git-protocol https --web
  fi
}

setup_env_and_deps() {
  activate_conda

  if conda env list | awk '{print $1}' | grep -Fx "$CONDA_ENV_NAME" >/dev/null 2>&1; then
    log "Conda env '$CONDA_ENV_NAME' already exists."
  else
    log "Creating conda env '$CONDA_ENV_NAME'..."
    conda create -y -n "$CONDA_ENV_NAME" python=3.10
  fi

  conda activate "$CONDA_ENV_NAME"

  log "Installing Python dependencies..."
  cd "$INSTALL_DIR"
  python -m pip install --upgrade pip
  python -m pip install -r requirements_no_tcnn.txt
  python -m pip install gdown
}

download_checkpoints() {
  cd "$INSTALL_DIR"
  mkdir -p checkpoints

  log "Downloading matrix3d_512.pt checkpoint..."
  wget -O checkpoints/matrix3d_512.pt "$MATRIX3D_CKPT_URL"

  log "Downloading isnet-general-use.pth checkpoint via gdown..."
  gdown --id "$ISNET_GDRIVE_ID" -O checkpoints/isnet-general-use.pth
}

main() {
  install_miniconda_if_needed
  activate_conda
  clone_or_update_repo
  setup_git_auth
  setup_env_and_deps
  download_checkpoints

  log "Bootstrap complete."
  log "Next: conda activate $CONDA_ENV_NAME && cd $INSTALL_DIR"
}

main "$@"

