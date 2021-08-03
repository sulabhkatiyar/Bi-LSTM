import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention_choice
from datasets import *
from utils import *
from new_utils import *
from nltk.translate.bleu_score import corpus_bleu
from eval_val import evaluate
import json
import argparse

data_folder = 'path_to_data_files'  # folder with data files saved by create_input_files.py
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'  # base name shared by data files
dataset='flickr8k'
emb_dim = 512  
attention_dim = 512  
decoder_dim = 512  
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
cudnn.benchmark = True  

start_epoch = 0
epochs = 20  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
best_bleu1, best_bleu2, best_bleu3, best_bleu4 = 0.,0.,0.,0.  # BLEU scores right now
guiding_bleu= 1 # 1: BLEU 1, 2: BLEU-2, 3: BLEU-3, 4: BLEU4 #THE BLEU METRIC USED TO GUIDE THE PROCESS
print_freq = 1000  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None # path to checkpoint; None if none

encoder_dim=4096  
use_image_transform=False
choice = 0


def main():
    global best_bleu1, best_bleu2, best_bleu3, best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder
    global encoder_dim, data_name, word_map, guiding_bleu, choice, val_loader_single, device

    # Remove previous checkpoints if they exist in same directory 
    if checkpoint == None:
        if 'BEST_checkpoint_'+dataset+'_5_cap_per_img_5_min_word_freq.pth.tar' in os.listdir(os.getcwd()):
            os.remove('BEST_checkpoint_'+dataset+'_5_cap_per_img_5_min_word_freq.pth.tar')
        if 'checkpoint_'+dataset+'_5_cap_per_img_5_min_word_freq.pth.tar' in os.listdir(os.getcwd()):
            os.remove('checkpoint_'+dataset+'_5_cap_per_img_5_min_word_freq.pth.tar')


    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention_choice(embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       encoder_dim=encoder_dim,
                                       dropout=dropout, choice=choice)

        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu_scores'][3]
        best_bleu3 = checkpoint['bleu_scores'][2]
        best_bleu2 = checkpoint['bleu_scores'][1]
        best_bleu1 = checkpoint['bleu_scores'][0]
        decoder = checkpoint['decoder']
        # Note to self: Finish Ph.d quickly and don't waste too much time coding.
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        
        if fine_tune_encoder is True:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)
        else:
            encoder_optimizer = checkpoint['encoder_optimizer']


    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    horizontal=transforms.RandomHorizontalFlip(p=0.5)
    vertical= transforms.RandomVerticalFlip(p=0.5)
    totensor=transforms.ToTensor()
    topil=transforms.ToPILImage()    

    transforms_to_apply=[horizontal]
    transforms_list=[topil] + transforms_to_apply + [totensor] if use_image_transform else []

    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose(transforms_list + [normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)    #other transforms are topil,horizontal,vertical,totensor,

    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    val_loader_single = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=workers, pin_memory=True)
    
    # Epochs
    for epoch in range(start_epoch, epochs):        
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 2 == 0:
            adjust_learning_rate(decoder_optimizer, 0.95)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.9)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu1,recent_bleu2,recent_bleu3,recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)
        
        # Check if there was an improvement
        if guiding_bleu==4:
            is_best = recent_bleu4 > best_bleu4 
        elif guiding_bleu==3:
            is_best = recent_bleu3 > best_bleu3
        elif guiding_bleu==2:
            is_best = recent_bleu2 > best_bleu2
        elif guiding_bleu==1:
            is_best = recent_bleu1 > best_bleu1
        
        best_bleu1 = max(recent_bleu1, best_bleu1)
        best_bleu2 = max(recent_bleu2, best_bleu2)
        best_bleu3 = max(recent_bleu3, best_bleu3)
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
        
        bleu_list=[recent_bleu1,recent_bleu2,recent_bleu3,recent_bleu4]

        # Save validation set loss for each epoch in the file
        with open('validation_logs.txt', 'a') as vl:
            if epoch == 0:
                vl.write('\n\nThe dataset is {}. \nThe BLEU scores for epoch {} are {}.\n'.format(data_name,epoch,bleu_list))
            else:
                vl.write('The BLEU scores for epoch {} are {}.\n'.format(epoch, bleu_list))

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, bleu_list, is_best)

    if not fine_tune_encoder:
        increase_run_number()

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):   
    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  
    data_time = AverageMeter()  
    losses = AverageMeter()  
    top5accs = AverageMeter() 

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens, image_name) in enumerate(train_loader):
        data_time.update(time.time() - start)        
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)        
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)
        # For forward LSTM
        targets = caps_sorted[0][:, 1:]
        scores[0] = pack_padded_sequence(scores[0], decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        # Calculate loss for forward
        loss = criterion(scores[0], targets)
        # For reverse LSTM
        targets_r = caps_sorted[1][:, 1:]
        scores[1] = pack_padded_sequence(scores[1], decode_lengths, batch_first=True).data
        targets_r = pack_padded_sequence(targets_r, decode_lengths, batch_first=True).data
        # Calculate loss for reverse
        loss = loss + criterion(scores[1], targets_r)
       
        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores[0], targets, 5) + accuracy(scores[1], targets_r, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):  
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    with torch.no_grad():       
        metrics_list = evaluate(val_loader_single, encoder, decoder, criterion, word_map, device)

        # metrics_list[0] has overall model validation performance; metrics_list[1] has forward model performance and metrics_list[2] has backward model performance
        bleu1,bleu2,bleu3,bleu4 = metrics_list[0]

        print("\nReturning from Validation with BLEU scores: {}".format([bleu1,bleu2,bleu3,bleu4]))

    return bleu1,bleu2,bleu3,bleu4


if __name__ == '__main__':
    main()
