FP Code
=======

This FP is based on two well designed neural netwrok. The first paper is called "Jointly Adversarial Enhancement Training for Robust End-to-End Speech Recognition". https://www.isca-speech.org/archive/pdfs/interspeech_2019/liu19_interspeech.pdf

The second paper is called "SEGAN: Speech Enhancement Generative Adversarial Network". https://arxiv.org/pdf/1703.09452.pdf.  
 
# Installation
## Get code and dependencies for first paper
- `git clone https://github.com/bliunlpr/Robust_e2e_gan.git`
- Install the dependencies listed in 
  - `pip install python==3.5`
  - `pip install pytorch==0.4.0`
or install dependencies by conda env
## Get code and dependencies for second paper
- `git clone https://github.com/santi-pdp/segan_pytorch.git`
- `pip install -r /path/to/requirements.txt`
## Get speech dataset and noise
After downloading the first repository,
- `sh run.sh`
- It will download the speech dataset and spilt it into train, test and validation datasets. More details about this dataset can be found here: http://www.aishelltech.com.
- Download the noise dataset from https://datashare.ed.ac.uk/handle/10283/1942.
- Split the noise dataset into "match" and "unmatch" data. "Match" dataset includes "White Noise", "Factory Floor Noise 1", "Cockpit Noise 1", "Cockpit Noise 3", "Engine Room Noise", "Military Vehicle Noise", "Machine Gun Noise", "Vehicle Interior Noise" and "HF Channel Noise". "Unmatch" dataset includes "Pink Noise", "Factory Floor Noise 2", "Cockpit Noise 2", "Operations Room" and "Military Vehicle Noise".
## Mix speech dataset and noise
Mix the train and validation datasets with "match" noise and mix the test dataset with "unmatch" dataset. The following code will mix the datasets.
- `python3 data/prepare_feats.py data_dir feat_dir noise_repeat_num`
- The output of data/prepare_feats.py will generate "clean_feats.scp", "noisy_feats.scp" and text. "clean_feats.scp" is the file which include clean speech datasets, 
"noisy_feats.scp" is the file in which mix the noise and clean speech datasets. "text" contains corresponding sentence/words about speech. 
## Pretrain the code of first paper
### E2E ASR training
- `python3 asr_train.py --dataroot Your data directory(including train, dev and test dataset)` 
### Enhancment training with without GAN
- `python3 enhance_fbank_train.py --dataroot Your data directory`  
### Enhancment training with GAN
- `python3 enhance_gan_train.py --dataroot Your data directory`
### Joint Training
- `python3 joint_base_train.py --dataroot Your data directory`
### Decoding
- `python3 asr_recog.py`
## Pretrain the code for second paper
- `python train.py --save_path ckpt_segan+ --batch_size 300 \
		--clean_trainset data/clean_trainset \
		--noisy_trainset data/noisy_trainset \
		--cache_dir data/cache`
## Joint training for SEGAN and ASR
Download this repository,
- `git clone https://github.com/hajiejue/FP-Code.git`
- `python train.py --save_path ckpt_segan+ --batch_size 300 \
		--clean_trainset data/clean_trainset \
		--noisy_trainset data/noisy_trainset \
		--cache_dir data/cache`
## Evaluation fo Joint training for SEGAN and ASR
Downloat the subjective evaluation function
- `https://www.crcpress.com/downloads/K14513/ K14513_CD_Files.zip`
- Set up the enhanced speech directory, clean speech directory and also noise speech directory for K14513_CD_Files/MATLAB_code/objective_measures/quality pesq.m, composite.m.
- Run pesq.m, composite.m.
## Configuration
The trained paramerters for joint traning can be found in "ckpt_segan+/config" file. There is a link for download the trained parameters for "weights_EOE_D-Discriminator-26510.ckpt" and "segan+_generator.ckpt". The chickpoint file alse can be found in this link.



