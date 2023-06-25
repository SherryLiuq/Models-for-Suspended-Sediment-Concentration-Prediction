import os
import torch

def SaveModel(epoch, best_performance, model, optimizer, save_root, best=False):
    saveDict = {
        'pre_epoch':epoch,
        'performance':best_performance,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict()
    }
    savePath = os.path.join(save_root, '%s.pth' %('latest' if not best else 'best'))
    torch.save(saveDict, savePath)

def LoadModel(check_path):

    stateDict = torch.load(check_path)

    return stateDict['pre_epoch'], stateDict['performance'], \
           stateDict['model_state_dict'], stateDict['optimizer_state_dict']