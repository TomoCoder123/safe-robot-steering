# safe-robot-steering
```conda create -y -n safesteer python=3.10
conda activate safesteer
conda install ffmpeg -c conda-forge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install -y -c nvidia -c conda-forge cuda-toolkit=11.8
pip install -r third_party/libero/requirements.txt
pip install -e third_party/libero
pip install -e .
pip install -e ".[pi]"
```
