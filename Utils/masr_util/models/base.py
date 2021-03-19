import torch
import torch.nn as nn
import json

class MASRModel(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config
        
        
    # 1090126 
    @classmethod
    def load_with_new_vocabulary(cls, path, new_vocabulary):
        print("dgx")
        package = torch.load(path)
        state_dict = package.state_dict()
        config = package.config
        config['vocabulary'] = new_vocabulary
        new_units = len(new_vocabulary)
        state_dict['cnn.10.bias'] = torch.randn(new_units)        
        state_dict['cnn.10.weight_g'] = torch.randn(new_units,1,1)
        state_dict['cnn.10.weight_v'] = torch.randn(new_units,1000,1)        
        m = cls(**config)
#        print(m)
        m.load_state_dict(state_dict)
        return m
    
    
    @classmethod
    def load(cls, path):
        package = torch.load(path)
        state_dict = package.state_dict()# 1090121
        #print("state_dict",state_dict)
        config = package.config# 1090121
        #print("config",config)
#        state_dict = package["state_dict"]
#        config = package["config"]
        m = cls(**config)
        m.load_state_dict(state_dict)
        
        
        
        return m

    def to_train(self):
        from .trainable import TrainableModel

        self.__class__.__bases__ = (TrainableModel,)
        return self

    def predict(self, *args):
        raise NotImplementedError()

    # -> texts: list, len(list) = B
    def _default_decode(self, yp, yp_lens):
        idxs = yp.argmax(1)
        texts = []                   
#        with open("data_aishell/labels.json") as f:
#            Vocabulary = json.load(f)
#            Vocabulary = "".join(Vocabulary)
        for idx, out_len in zip(idxs, yp_lens):
            idx = idx[:out_len]
            text = ""
            last = None
            for i in idx:
                if i.item() not in (last, self.blank):

#                    text += Vocabulary[i.item()]
                    text += self.vocabulary[i.item()]
                last = i
            texts.append(text)
        return texts

    def decode(self, *outputs):  # texts -> list of size B
        return self._default_decode(*outputs)
