import torch
import numpy as np
def compute_cmvn_epoch(opt, train_loader, enhance_model, feat_model):
    enhance_model.eval()
    feat_model.eval()
    torch.set_grad_enabled(False)
    ##print(enhance_model.state_dict())
    enhance_cmvn_file = os.path.join(opt.exp_path, 'enhance_cmvn.npy')
    for i, (data) in enumerate(train_loader, start=0):
        utt_ids, spk_ids, clean_inputs, clean_log_inputs, mix_inputs, mix_log_inputs, cos_angles, targets, input_sizes, target_sizes = data
        enhance_out = enhance_model(mix_inputs, mix_log_inputs, input_sizes)
        enhance_cmvn = feat_model.compute_cmvn(enhance_out, input_sizes)
        if enhance_cmvn is not None:
            np.save(enhance_cmvn_file, enhance_cmvn)
            print('save enhance_cmvn to {}'.format(enhance_cmvn_file))
            break
    enhance_cmvn = torch.FloatTensor(enhance_cmvn)
    enhance_model.train()
    feat_model.train()
    torch.set_grad_enabled(True)
    return enhance_cmvn