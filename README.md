Code implementation for EMNLP 24 findings: Pseudo Label Enhanced Prototypical Contrastive Learning Framework towards Uniformed Intent Discovery.
Prepare the environment:
```
python==3.6.13
pip install -r requirements.txt
```
If you use the NIVIDIA A100 80GB PCIe GPU with PyTorch, the torch version should be updated
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
Run the experiments by: 
```
run.sh
```
You can change the parameters here.

Some code references the following repositories:

- [DKT](https://github.com/myt517/DKT)

- [DeepAligned](https://github.com/thuiar/DeepAligned-Clustering)
