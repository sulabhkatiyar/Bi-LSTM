import torch
from torch import nn
import torchvision
import pretrainedmodels
import json
from tqdm import tqdm
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""HERE WE HAVE CHOICE OF ENCODER. 0 for ResNet101, 1 for InceptionResNetV2, 2 for NasNetLarge; more to be added soon."""
encoder_choice = 31
final_embeddings_dim = 512
embeddings_ensemble_available = False
ensemble_dim = 927

#CHANGE BOTH THESE WHILE CHANGING THE DATASET
# word_map_file = '/home/sulabh00/for_jupyter/papers/hindi/caption_data/WORDMAP_cocohindi_5_cap_per_img_5_min_word_freq.json'
# karpathy_split = '/home/sulabh00/for_jupyter/datasets/karpathy_split/dataset_flickr30khindi.json'

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=1):
        super(Encoder, self).__init__()        
        self.enc_image_size = encoded_image_size

        if encoder_choice==0:
            resnet = torchvision.models.resnet152(pretrained=True)  # pretrained ImageNet ResNet-152
            modules = list(resnet.children())[:-1]   # Resnet101/Resnet152
            self.resnet = nn.Sequential(*modules)

            # modules_roi = list(resnet_roi.children())[:-2]
            # self.resnet_roi = nn.Sequential(*modules_roi)
            # self.activation = torch.nn.Tanh()
            # self.combine = torch.nn.Linear(4096,2048)        

            # resnet_roi = torchvision.models.resnet152(pretrained=True)  # pretrained ImageNet ResNet-101
            
        if encoder_choice==1: # Downloaded this unofficial implementation of InceptionResNetv2.
                              # Included in library 'pretrainedmodels' now
            inceptionresnet=pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000,pretrained='imagenet')
            modules_incepres = list(inceptionresnet.children())[:-2]  # InceptionResNetv2
            self.incepres = nn.Sequential(*modules_incepres)
        if encoder_choice==2:
            naslarge_model=pretrainedmodels.__dict__['nasnetalarge'](num_classes=1000,pretrained='imagenet')
            self.naslarge_model=naslarge_model   # this is needed for fine_tune() function
            self.fine_tune()  
            self.nasnetlarge = naslarge_model.features   # NasNetlarge
        if encoder_choice==3:
            vgg16 = torchvision.models.vgg16_bn(pretrained=True)
            modules_vgg16 = list(vgg16.children())[:-2]
            self.vgg = nn.Sequential(*modules_vgg16)
        if encoder_choice==4:
            alexnet = torchvision.models.alexnet(pretrained=True)
            self.alexnet = torch.nn.Sequential(*list(alexnet.children())[:-1])
        if encoder_choice==5:
            squeezenet = torchvision.models.squeezenet1_0(pretrained = True)
            modules_squeezenet = list(squeezenet.children())[:-1]
            self.squeezenet  = nn.Sequential(*modules_squeezenet)
        if encoder_choice==6:
            densenet = torchvision.models.densenet201(pretrained = True)
            modules_densenet = list(densenet.children())[:-1]
            self.densenet = nn.Sequential(*modules_densenet)
        if encoder_choice==7:
            googlenet = torchvision.models.googlenet(pretrained = True)
            modules_googlenet = list(googlenet.children())[:-1]
            self.googlenet = nn.Sequential(*modules_googlenet)
        if encoder_choice==8:
            shufflenet = torchvision.models.shufflenet_v2_x1_0(pretrained = True)
            modules_shufflenet = list(shufflenet.children())[:-1]
            self.shufflenet = nn.Sequential(*modules_shufflenet)
        if encoder_choice==9:
            mobilenet = torchvision.models.mobilenet_v2(pretrained = True)
            modules_mobilenet = list(mobilenet.children())[:-1]
            self.mobilenet = nn.Sequential(*modules_mobilenet)
        if encoder_choice==10:
            resnext = torchvision.models.resnext101_32x8d(pretrained = True)
            modules_resnext = list(resnext.children())[:-2]
            self.resnext = nn.Sequential(*modules_resnext)
        if encoder_choice==11:
            wideresnet = torchvision.models.wide_resnet101_2(pretrained = True)
            modules_wideresnet = list(wideresnet.children())[:-2]
            self.wideresnet = nn.Sequential(*modules_wideresnet)
        if encoder_choice==12:
            mnasnet = torchvision.models.mnasnet1_0(pretrained = True)
            modules_mnasnet = list(mnasnet.children())[:-1]
            self.mnasnet = nn.Sequential(*modules_mnasnet)
        if encoder_choice==13:
            xception = pretrainedmodels.__dict__['xception'](num_classes=1000, pretrained='imagenet')
            modules_xception = list(xception.children())[:-2]
            self.xception = nn.Sequential(*modules_xception)
        if encoder_choice==14:
            inception = pretrainedmodels.__dict__['inceptionv4'](num_classes=1000, pretrained='imagenet')
            self.inception = inception.features
        if encoder_choice==15:
            nasnetamobile = pretrainedmodels.__dict__['nasnetamobile'](num_classes=1000, pretrained='imagenet')
            self.nasnetamobile = nasnetamobile.features
        if encoder_choice==16:
            dpn131 = pretrainedmodels.__dict__['dpn131'](num_classes=1000, pretrained='imagenet')
            self.dpn131 = dpn131.features
        if encoder_choice==17:
            senet154 = pretrainedmodels.__dict__['senet154'](num_classes=1000, pretrained='imagenet')
            modules_senet154 = list(senet154.children())[:-3]
            self.senet154 = nn.Sequential(*modules_senet154)
        if encoder_choice==18:
            pnasnet5large = pretrainedmodels.__dict__['pnasnet5large'](num_classes=1000, pretrained='imagenet')
            self.pnasnet5large = pnasnet5large.features
        if encoder_choice==19:
            polynet = pretrainedmodels.__dict__['polynet'](num_classes=1000, pretrained='imagenet')
            modules_polynet = list(polynet.children())[:-3]
            self.polynet = nn.Sequential(*modules_polynet)
        if encoder_choice==20:
            resnet18 = torchvision.models.resnet18(pretrained=True)  # pretrained ImageNet ResNet-18
            modules_resnet18 = list(resnet18.children())[:-2]   
            self.resnet18 = nn.Sequential(*modules_resnet18)
        if encoder_choice==21:
            resnet34 = torchvision.models.resnet34(pretrained=True) 
            modules_resnet34 = list(resnet34.children())[:-2]   
            self.resnet34 = nn.Sequential(*modules_resnet34)
        if encoder_choice==22:
            resnet50 = torchvision.models.resnet50(pretrained=True) 
            modules_resnet50 = list(resnet50.children())[:-2]   
            self.resnet50 = nn.Sequential(*modules_resnet50)
        if encoder_choice==23:
            resnet101 = torchvision.models.resnet101(pretrained=True) 
            modules_resnet101 = list(resnet101.children())[:-2]   
            self.resnet101 = nn.Sequential(*modules_resnet101)
        if encoder_choice==24:
            vgg11 = torchvision.models.vgg11_bn(pretrained = True)
            modules_vgg11 = list(vgg11.children())[:-1]
            self.vgg11 = nn.Sequential(*modules_vgg11)
        if encoder_choice==25:
            vgg13 = torchvision.models.vgg13_bn(pretrained = True)
            modules_vgg13 = list(vgg13.children())[:-1]
            self.vgg13 = nn.Sequential(*modules_vgg13)
        if encoder_choice==26:
            vgg19 = torchvision.models.vgg19_bn(pretrained = True)
            modules_vgg19 = list(vgg19.children())[:-1]
            self.vgg19 = nn.Sequential(*modules_vgg19)
        if encoder_choice==27:
            densenet121 = torchvision.models.densenet121(pretrained = True)
            modules_densenet121 = list(densenet121.children())[:-1]
            self.densenet121 = nn.Sequential(*modules_densenet121)
        if encoder_choice==28:
            densenet169 = torchvision.models.densenet169(pretrained = True)
            modules_densenet169 = list(densenet169.children())[:-1]
            self.densenet169 = nn.Sequential(*modules_densenet169)
        if encoder_choice==29:
            densenet161 = torchvision.models.densenet161(pretrained = True)
            modules_densenet161 = list(densenet161.children())[:-1]
            self.densenet161 = nn.Sequential(*modules_densenet161)
        if encoder_choice==30:
            vgg16 = torchvision.models.vgg16(pretrained=True)
            modules_vgg16 = list(vgg16.children())[:-2]
            self.vgg16_nobn = nn.Sequential(*modules_vgg16)
        if encoder_choice==31:
            vgg16 = torchvision.models.vgg16(pretrained = True)
            self.features_nopool = nn.Sequential(*list(vgg16.features.children())[:-1])
            self.features_pool = list(vgg16.features.children())[-1]
            self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-1]) 




        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        
        """IMPORTANT MODIFICATION: Added multiple encoders here on 23rd July,2020"""
        # As metioned in opening class comments, we will use multiple encoders as needed.
        global encoder_choice
        
        if encoder_choice==0:
            out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        elif encoder_choice==1:
            out = self.incepres(images) # (batch_size, 1536, image_size/32, image_size/32)
        elif encoder_choice==2:
            out = self.nasnetlarge(images)  # (batch_size, 4032, image_size/32, image_size/32)
        elif encoder_choice==3:
            out = self.vgg(images)     # (batch_size, 512, image_size/32, image_size/32)
        elif encoder_choice==4:
            out = self.alexnet(images)     # (batch_size, 256, 6, 6)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 6, 6, 256)
            return out
        elif encoder_choice == 5:
            out = self.squeezenet(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 15, 15, 512)
            return out
        elif encoder_choice == 6:
            out = self.densenet(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 1920) 
            return out
        elif encoder_choice == 7:
            out = self.googlenet(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 1024) 
            return out
        elif encoder_choice == 8:
            out = self.shufflenet(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 1024) 
            return out
        elif encoder_choice == 9:
            out = self.mobilenet(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 1280) 
            return out
        elif encoder_choice == 10:
            out = self.resnext(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 2048) 
            return out
        elif encoder_choice == 11:
            out = self.wideresnet(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 2048) 
            return out
        elif encoder_choice == 12:
            out = self.mnasnet(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 1280) 
            return out
        elif encoder_choice == 13:
            out = self.xception(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 2048) 
            return out
        elif encoder_choice == 14:
            out = self.inception(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 1536) 
            return out
        elif encoder_choice == 15:
            out = self.nasnetamobile(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 1056) 
            return out
        elif encoder_choice == 16:
            out = self.dpn131(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 2688) 
            return out
        elif encoder_choice == 17:
            out = self.senet154(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 2048) 
            return out
        elif encoder_choice == 18:
            out = self.pnasnet5large(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 4320) 
            return out
        elif encoder_choice == 19:
            out = self.polynet(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 6, 6, 2048) 
            return out
        elif encoder_choice == 20:
            out = self.resnet18(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 512) 
            return out
        elif encoder_choice == 21:
            out = self.resnet34(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 512) 
            return out
        elif encoder_choice == 22:
            out = self.resnet50(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 2048) 
            return out
        elif encoder_choice == 23:
            out = self.resnet101(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 2048) 
            return out
        elif encoder_choice == 24:
            out = self.vgg11(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 7, 7, 512) 
            return out
        elif encoder_choice == 25:
            out = self.vgg13(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 7, 7, 512) 
            return out
        elif encoder_choice == 26:
            out = self.vgg19(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 7, 7, 512) 
            return out
        elif encoder_choice == 27:
            out = self.densenet121(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 1024) 
            return out
        elif encoder_choice == 28:
            out = self.densenet169(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 1664) 
            return out
        elif encoder_choice == 29:
            out = self.densenet161(images)
            out = out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 2208) 
            return out
        elif encoder_choice==30:
            out = self.vgg16_nobn(images)     # (batch_size, 512, image_size/32, image_size/32)
        elif encoder_choice==31:
            # print(images.shape)
            x = self.features_nopool(images)
            # print(x.shape)
            x_pool = self.features_pool(x)
            # print(x_pool.shape)
            x_feat = x_pool.view(x_pool.size(0), -1)
            # print(x_feat.shape)
            y = self.classifier(x_feat)
            # print(y.shape)
            # return y.unsqueeze(1).unsqueeze(1)
            return y
            
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        # if type(images) == list:
        #     out = self.activation(self.combine(out))
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        if encoder_choice==0:
            for p in self.resnet.parameters():
                p.requires_grad = False
            # If fine-tuning, only fine-tune convolutional blocks 2 through 4
            for c in list(self.resnet.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune
        elif encoder_choice==1:
            for p in self.incepres.parameters():
                p.requires_grad = False
            # If fine-tuning, only fine-tune convolutional blocks 2 through 4
            for c in list(self.incepres.children())[10:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune
        elif encoder_choice==2:
            for p in self.naslarge_model.parameters():
                p.requires_grad = False
            # If fine-tuning, only fine-tune convolutional blocks 2 through 4
            for c in list(self.naslarge_model.children())[15:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune
        elif encoder_choice==3:
            for p in self.vgg.parameters():
                p.requires_grad = False
            # If fine-tuning, only fine-tune convolutional blocks after 20
            for c in list(self.vgg.children())[0][20:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune



class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha

        
    

class DecoderWithAttention_choice(nn.Module):
    """
    Decoder.
    """
    """
    We have a choice now as to what RNN we want to use. 0:LSTM, 1: Bidirectional LSTM, 2: GRU, 3: Bidirectional GRU
    """
    


    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5,choice=0, final_embeddings_dim=512, num_layers=2):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention_choice, self).__init__()

        global embeddings_ensemble_available, ensemble_dim

        self.encoder_dim = encoder_dim
        # self.attention_dim = attention_dim
        if embeddings_ensemble_available == True:
            self.embed_dim = ensemble_dim
        else:
            self.embed_dim = embed_dim
        self.final_embeddings_dim = final_embeddings_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.choice = choice
        self.num_layers = num_layers

        # self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        
        self.embedding = nn.Embedding(vocab_size, self.embed_dim)  # embedding layer
        self.embedding_reverse = nn.Embedding(vocab_size, self.embed_dim)  # reverse embedding layer

        self.final_embeddings = nn.Linear(self.embed_dim, final_embeddings_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        
        #FEATURES TENSOR initialization
        #change cut-off frequency here
        # self.features_tensor = Features_tensor(cut_off_freq, self.encoder_dim, self.embed_dim)
        
        if self.choice==0:
            if embeddings_ensemble_available:
                self.decode_step = nn.LSTMCell(final_embeddings_dim, decoder_dim, bias=True)  # decoding LSTMCell 
            else:
                self.decode_step1 = nn.LSTMCell(embed_dim, decoder_dim, bias=True)
                self.decode_step2 = nn.LSTMCell(decoder_dim + decoder_dim, decoder_dim, bias=True)
                self.decode_step_reverse1 = nn.LSTMCell(embed_dim, decoder_dim, bias=True)
                self.decode_step_reverse2 = nn.LSTMCell(decoder_dim + decoder_dim, decoder_dim, bias=True)
            self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
            self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
            self.init_h_r = nn.Linear(encoder_dim, decoder_dim)  
            self.init_c_r = nn.Linear(encoder_dim, decoder_dim)  
            self.img_forward = nn.Linear(encoder_dim, decoder_dim)  
            self.img_backward = nn.Linear(encoder_dim, decoder_dim)  

        
        elif self.choice==1:
            if embeddings_ensemble_available:
                self.decode_step = nn.LSTMCell(final_embeddings_dim, decoder_dim, bias=True)  # decoding LSTMCell 
            else:
                self.decode_step = nn.LSTMCell(embed_dim, decoder_dim, bias=True)
            self.decode_step_f = nn.LSTMCell(embed_dim, decoder_dim, bias=True)
            self.decode_step_b = nn.LSTMCell(embed_dim, decoder_dim, bias=True)
            self.init_hf = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of Bidirectional LSTM
            self.init_cf = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of Bidirectional LSTM
            self.init_hb = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of Bidirectional LSTM
            self.init_cb = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of Bidirectional LSTM

        elif self.choice == 2:
            if embeddings_ensemble_available:
                self.decode_step1 = nn.LSTMCell(final_embeddings_dim, decoder_dim, bias=True)  # decoding LSTMCell 
            else:
                self.decode_step1 = nn.LSTMCell(embed_dim, decoder_dim, bias=True)
            self.decode_step2 = nn.LSTMCell(decoder_dim, decoder_dim, bias=True)
            self.decode_step3 = nn.LSTMCell(decoder_dim, decoder_dim, bias=True)
            self.init_h1 = nn.Linear(encoder_dim, decoder_dim)
            self.init_c1 = nn.Linear(encoder_dim, decoder_dim)
            self.init_h2 = nn.Linear(encoder_dim, decoder_dim)
            self.init_c2 = nn.Linear(encoder_dim, decoder_dim)
            self.init_h3 = nn.Linear(encoder_dim, decoder_dim)
            self.init_c3 = nn.Linear(encoder_dim, decoder_dim)

        
        #self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        #self.sigmoid = nn.Sigmoid()

        if self.choice==0:
            self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
            self.fc_r = nn.Linear(decoder_dim, vocab_size) 
        elif self.choice==1:
            # self.fc = nn.Linear(2*decoder_dim, vocab_size)
            self.fc = nn.Linear(decoder_dim, vocab_size)
        elif self.choice==2:
            self.fc = nn.Linear(decoder_dim, vocab_size)

        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.embedding_reverse.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc_r.bias.data.fill_(0)
        self.fc_r.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)


    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        # mean_encoder_out = encoder_out.mean(dim=1)
        # h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        # c = self.init_c(mean_encoder_out)
        h = self.init_h(encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(encoder_out)
        return h, c

    def init_hidden_state_reverse(self, encoder_out):
        # mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h_r(encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c_r(encoder_out)
        return h, c

    def get_img_features(self, encoder_out):
        img_f = self.img_forward(encoder_out)  
        img_b = self.img_backward(encoder_out)
        return img_f, img_b


    def init_hidden_state_bidirectional(self, encoder_out):
        """
        This is for bidirectional LSTM. We make two versions of init_h to get two hidden states which we
        concatenate together. init_hf, init_hb and init_ch, init_cb are defined in self.__init__()
        """
        mean_encoder_out=encoder_out.mean(dim=1)  #We mean out one of the encoded_image_size dimensions of encoder_out
        h_f = self.init_hf(mean_encoder_out)  # (batch_size, decoder_dim)
        c_f = self.init_cf(mean_encoder_out)
        h_b = self.init_hb(mean_encoder_out)
        c_b = self.init_cb(mean_encoder_out)
        h = torch.cat((h_f.unsqueeze(0),h_b.unsqueeze(0)),dim=0)
        c = torch.cat((c_f.unsqueeze(0),c_b.unsqueeze(0)),dim=0)
        return h, c, h_f, h_b, c_f, c_b

    def init_hidden_state_deeplstm(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)  #We mean out one of the encoded_image_size dimensions of encoder_out
        h_1 = self.init_h1(mean_encoder_out)  # (batch_size, decoder_dim)
        c_1 = self.init_c1(mean_encoder_out)
        h_2 = self.init_h2(mean_encoder_out)
        c_2 = self.init_c2(mean_encoder_out)
        h_3 = self.init_h3(mean_encoder_out)
        c_3 = self.init_c3(mean_encoder_out)
        return h_1, h_2, h_3, c_1, c_2, c_3
        
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        global final_embeddings_dim, embeddings_ensemble_available, ensemble_dim

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        # encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        # num_pixels = encoder_out.size(1)
        encoder_out = encoder_out.view(batch_size, encoder_dim)  # (batch_size, encoder_dim)

        # Sort input data by decreasing lengths; why? apparent below
        # print(caption_lengths)
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        # print("caption_lengths is {}".format(caption_lengths))
        # print("sort_ind is {}".format(sort_ind))
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        zero_float = torch.zeros(encoded_captions.shape[0], encoded_captions.shape[1])
        encoded_captions_reverse = zero_float.long()
        # print(encoded_captions.tolist()[0])
        for p in range(encoded_captions.shape[0]):
            reversed_list = encoded_captions.tolist()[p][:caption_lengths.tolist()[p]]
            reversed_list.reverse()
            encoded_captions_reverse[p][:caption_lengths.tolist()[p]] = torch.LongTensor(reversed_list)

        encoded_captions_reverse = encoded_captions_reverse.to(device)
        # print(encoded_captions[0:3], encoded_captions_reverse[0:3])        


        #getting the feature map here with the same sequence
        # feature_map = self.features_tensor(encoder_out, encoded_captions)

        # Embedding
        if self.choice==0:
            embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, self.embed_dim)
            embeddings_reverse = self.embedding_reverse(encoded_captions_reverse)  # (batch_size, max_caption_length, self.embed_dim)
            # h, c = self.init_hidden_state(encoder_out)     # Initialize LSTM state
            # hr, cr = self.init_hidden_state_reverse(encoder_out)     # Initialize Reverse LSTM
            img_f, img_r = self.get_img_features(encoder_out)
            h, c = torch.zeros_like(img_f), torch.zeros_like(img_f)
            h1, c1 = torch.zeros_like(img_f), torch.zeros_like(img_f)
            hr, cr = torch.zeros_like(img_f), torch.zeros_like(img_f)
            hr1, cr1 = torch.zeros_like(img_f), torch.zeros_like(img_f)
            
        elif self.choice==1:
            #temp_embeddings = self.embedding(encoded_captions)
            #embeddings= temp_embeddings.unsqueeze(0)
            #h, c, _,_,_,_ = self.init_hidden_state_bidirectional(encoder_out)
            embeddings =self.embedding(encoded_captions)
            _, _, h_f, h_b, c_f, c_b = self.init_hidden_state_bidirectional(encoder_out)
        elif self.choice == 2:
            embeddings = self.embedding(encoded_captions)
            h_1, h_2, h_3, c_1, c_2, c_3 = self.init_hidden_state_deeplstm(encoder_out)
        

        if embeddings_ensemble_available == True:
            embeddings = self.final_embeddings(embeddings)
            # print(embeddings.shape)

        # we MERGE THE EMBEDDINGS WITH FEATURE TENSOR
        # assert(embeddings.shape == feature_map.shape)
        # embeddings = embeddings + feature_map
   
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        predictions_reverse = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        # alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            #attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],h[:batch_size_t])
            #gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            #attention_weighted_encoding = gate * attention_weighted_encoding
            if self.choice==0:
                # print(h.shape)
                h, c = self.decode_step1(embeddings[:batch_size_t, t, :],(h[:batch_size_t], c[:batch_size_t]))     # (batch_size_t, decoder_dim)
                h1, c1 = self.decode_step2(torch.cat([h[:batch_size_t], img_f[:batch_size_t]], dim = 1),(h1[:batch_size_t], c1[:batch_size_t]))     

                preds = self.fc(self.dropout(h1))  # (batch_size_t, vocab_size)

            predictions[:batch_size_t, t, :] = preds
            #alphas[:batch_size_t, t, :] = alpha

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            if self.choice==0:
                hr, cr = self.decode_step_reverse1(embeddings_reverse[:batch_size_t, t, :],(hr[:batch_size_t], cr[:batch_size_t]))     # (batch_size_t, decoder_dim)
                hr1, cr1 = self.decode_step_reverse2(torch.cat([hr[:batch_size_t], img_r[:batch_size_t]], dim = 1),(hr1[:batch_size_t], cr1[:batch_size_t]))     
                preds_r = self.fc_r(self.dropout(hr1))  
            predictions_reverse[:batch_size_t, t, :] = preds_r

        return [predictions, predictions_reverse], [encoded_captions, encoded_captions_reverse], decode_lengths, sort_ind
