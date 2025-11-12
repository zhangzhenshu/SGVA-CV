# Descargar:
# https://www.python.org/downloads/release/python-3110

# Para ejecutar script, abrir PowerShell y ejecutar:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# .\set_env.ps1

# Crear entorno virtual con Python 3.11
python -m venv env
.\env\Scripts\Activate.ps1

# Actualizar pip
python -m pip install --upgrade pip setuptools wheel

# Instalar PyTorch (CUDA 12.1 o superior)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Instalar Librerías necesarias
pip install numpy scipy pandas