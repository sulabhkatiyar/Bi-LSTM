import torch
from torch import nn
import torchvision
import pretrainedmodels
import json
from tqdm import tqdm
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder_choice = 0

class Encoder(nn.Module):
    
    def __init__(self, encoded_image_size=1):
        super(Encoder, self).__init__()        
        self.enc_image_size = encoded_image_size

        if encoder_choice==0:
            vgg16 = torchvision.models.vgg16(pretrained = True)
            self.features_nopool = nn.Sequential(*list(vgg16.features.children())[:-1])
            self.features_pool = list(vgg16.features.children())[-1]
            self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-1]) 

    def forward(self, images):
        if encoder_choice==0:
            x = self.features_nopool(images)
            x_pool = self.features_pool(x)
            x_feat = x_pool.view(x_pool.size(0), -1)
            y = self.classifier(x_feat)
            return y
            
        return out


class DecoderWithAttention_choice(nn.Module):

    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(DecoderWithAttention_choice, self).__init__()

        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, self.embed_dim)  
        self.embedding_reverse = nn.Embedding(vocab_size, self.embed_dim) 
        self.dropout = nn.Dropout(p=self.dropout)        
        self.decode_step1 = nn.LSTMCell(embed_dim, decoder_dim, bias=True)
        self.decode_step2 = nn.LSTMCell(decoder_dim + decoder_dim, decoder_dim, bias=True)
        self.decode_step_reverse1 = nn.LSTMCell(embed_dim, decoder_dim, bias=True)
        self.decode_step_reverse2 = nn.LSTMCell(decoder_dim + decoder_dim, decoder_dim, bias=True)
        self.img_forward = nn.Linear(encoder_dim, decoder_dim)  
        self.img_backward = nn.Linear(encoder_dim, decoder_dim)  
        self.fc = nn.Linear(decoder_dim, vocab_size)  
        self.fc_r = nn.Linear(decoder_dim, vocab_size) 
        self.init_weights() 

    def init_weights(self):   
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.embedding_reverse.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc_r.bias.data.fill_(0)
        self.fc_r.weight.data.uniform_(-0.1, 0.1)

    def get_img_features(self, encoder_out):
        img_f = self.img_forward(encoder_out)  
        img_b = self.img_backward(encoder_out)
        return img_f, img_b
        
    def forward(self, encoder_out, encoded_captions, caption_lengths):

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        encoder_out = encoder_out.view(batch_size, encoder_dim) 
        
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        zero_float = torch.zeros(encoded_captions.shape[0], encoded_captions.shape[1])
        encoded_captions_reverse = zero_float.long()
        for p in range(encoded_captions.shape[0]):
            reversed_list = encoded_captions.tolist()[p][:caption_lengths.tolist()[p]]
            reversed_list.reverse()
            encoded_captions_reverse[p][:caption_lengths.tolist()[p]] = torch.LongTensor(reversed_list)

        encoded_captions_reverse = encoded_captions_reverse.to(device)

        embeddings = self.embedding(encoded_captions)  
        embeddings_reverse = self.embedding_reverse(encoded_captions_reverse)  
        img_f, img_r = self.get_img_features(encoder_out)
        h, c = torch.zeros_like(img_f), torch.zeros_like(img_f)
        h1, c1 = torch.zeros_like(img_f), torch.zeros_like(img_f)
        hr, cr = torch.zeros_like(img_f), torch.zeros_like(img_f)
        hr1, cr1 = torch.zeros_like(img_f), torch.zeros_like(img_f)
            
        decode_lengths = (caption_lengths - 1).tolist()

        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        predictions_reverse = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])            
            h, c = self.decode_step1(embeddings[:batch_size_t, t, :],(h[:batch_size_t], c[:batch_size_t]))    
            h1, c1 = self.decode_step2(torch.cat([h[:batch_size_t], img_f[:batch_size_t]], dim = 1),(h1[:batch_size_t], c1[:batch_size_t]))     
            preds = self.fc(self.dropout(h1))  
            predictions[:batch_size_t, t, :] = preds

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])            
            hr, cr = self.decode_step_reverse1(embeddings_reverse[:batch_size_t, t, :],(hr[:batch_size_t], cr[:batch_size_t]))    
            hr1, cr1 = self.decode_step_reverse2(torch.cat([hr[:batch_size_t], img_r[:batch_size_t]], dim = 1),(hr1[:batch_size_t], cr1[:batch_size_t]))     
            preds_r = self.fc_r(self.dropout(hr1))  
            predictions_reverse[:batch_size_t, t, :] = preds_r

        return [predictions, predictions_reverse], [encoded_captions, encoded_captions_reverse], decode_lengths, sort_ind
