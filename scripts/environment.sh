conda create -n "slotssm" python=3.10 -y 
conda activate slotssm

conda install pip -y

pip install --upgrade pip

conda install -c nvidia cuda-toolkit -y

pip install torch torchvision torchaudio

pip install flash-attn --no-build-isolation

pip install transformer accelerate decord causal-conv1d==1.4.0

pip install git+https://github.com/state-spaces/mamba