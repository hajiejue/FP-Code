FP Code
=======

This FP is based on two well designed neural netwrok. The first paper called
"Jointly Adversarial Enhancement Training for Robust End-to-End Speech Recognition". https://www.isca-speech.org/archive/pdfs/interspeech_2019/liu19_interspeech.pdf

The second paper called "SEGAN: Speech Enhancement Generative Adversarial Network"https://arxiv.org/pdf/1703.09452.pdf.  
 
# Installation
## the follwing model is just trained in their own repo
## Get code and dependencies for first paper
- `git clone https://github.com/bliunlpr/Robust_e2e_gan.git`
- Install the dependencies listed in 
  - `pip install pytho==3.5`
  - `pip install pyTorch==0.4.0`
or install dependencies by conda env
## Get code and dependencies for second paper
- `git clone https://github.com/santi-pdp/segan_pytorch.git`
- `pip install -r /path/to/requirements.txt`
## Get speech dataset and noise
after downloading frist repo,
- `sh run.sh`
- It will download the speech dataset and spilt it into train, test and val. More details about this dataset can be found here: http://www.aishelltech.com
- Download the Noise dataset from here: https://datashare.ed.ac.uk/handle/10283/1942
- According my FA report, split the noise dataset into match and unmatch data.
## Mix speech dataset and noise
- `python3 data/prepare_feats.py data_dir feat_dir noise_repeat_num`
- It will generate clean_feats.scp noisy_feats.scp and text for match data and unmatch data. In each time, you have to specify the noise data type(match and unmatch)
## Pretrain the code for first paper
### E2E ASR training
- `python3 asr_train.py --dataroot Your data directory(including train, dev and test dataset)` 
### Enhancment training with no GAN
- `python3 enhance_fbank_train.py --dataroot Your data directory`  
### Enhancment training with GAN
- `python3 enhance_gan_train.py --dataroot Your data directory`
### Joint Training
You can jointly train the enhancement network and end-to-end ASR network by the ASR loss.
- `python3 joint_base_train.py --dataroot Your data directory`
### Decoding
- `python3 asr_recog.py`
## Pretrain the code for second paper
- `python train.py --save_path ckpt_segan+ --batch_size 300 \
		--clean_trainset data/clean_trainset \
		--noisy_trainset data/noisy_trainset \
		--cache_dir data/cache`
## Joint training for SEGAN and ASR
Download this repo
- 




