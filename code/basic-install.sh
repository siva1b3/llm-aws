sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv python3-dev build-essential git curl wget


cd llm-aws/code

python3 -m venv llm-env
source llm-env/bin/activate
python -V
which python


pip install --upgrade pip


pip install torch --index-url https://download.pytorch.org/whl/cpu


# python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"

pip install transformers accelerate huggingface_hub jupyter ipywidgets


# python -c "import transformers; print(transformers.__version__)"
# python -c "import jupyter; print('jupyter ok')"



















