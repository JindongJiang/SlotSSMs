conda create -n "slotssm" python=3.10 -y 
conda activate slotssm

conda install pip -y

pip install --upgrade pip

conda install -c nvidia cuda-toolkit -y

pip install torch torchvision torchaudio

pip install flash-attn --no-build-isolation

pip install transformer accelerate decord

pip install git+https://github.com/Dao-AILab/causal-conv1d

pip install git+https://github.com/state-spaces/mamba