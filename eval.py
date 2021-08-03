import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from new_utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import json
import argparse


# Parameters
data_folder = 'path_to_data_files'  # folder with data files saved by create_input_files.py
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'   # base name shared by data files
checkpoint = 'BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint
word_map_file = 'path_to_data_folder' + '/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json'  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  

captions_dump=True
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

embeddings_ensemble_available=False

#Arguments to main()
parser = argparse.ArgumentParser(description = 'Evaluation of IC model')
parser.add_argument('beam_size', type=int,  help = 'Beam size for evaluation')
args = parser.parse_args()


def evaluate(beam_size):
    global captions_dump, data_name, embeddings_ensemble_available    
    empty_hypo = 0
    empty_hypo_r = 0    
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    references = list()
    hypotheses = list()
    hypotheses_f = list()
    hypotheses_r = list()
    captions_dict=dict()
    image_names = list()
    for i, (image, caps, caplens, allcaps, image_name) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
        k = beam_size
        image = image.to(device)          
        encoder_out = encoder(image)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(1, encoder_dim)  

        encoder_out = encoder_out.expand(k, encoder_dim)  
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  
        seqs = k_prev_words  # (k, 1)
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        complete_seqs = list()
        complete_seqs_scores = list()        
        step = 1
        img_f, img_r = decoder.get_img_features(encoder_out)
        h, c = torch.zeros_like(img_f), torch.zeros_like(img_f)
        h1, c1 = torch.zeros_like(img_f), torch.zeros_like(img_f)

        while True:
            embeddings = decoder.embedding(k_prev_words).squeeze(1)                
            h, c = decoder.decode_step1(embeddings, (h, c))  
            h1, c1 = decoder.decode_step2(torch.cat([h, img_f], dim = 1),(h1, c1))  
            scores = decoder.fc(h1)  
            scores = F.log_softmax(scores, dim=1)

            scores = top_k_scores.expand_as(scores) + scores  
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  

            prev_word_inds = top_k_words / vocab_size  
            next_word_inds = top_k_words % vocab_size  

            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  

            if k == 0:
                break
                
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            img_f = img_f[prev_word_inds[incomplete_inds]]
            h1, c1 = h1[prev_word_inds[incomplete_inds]], c1[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
           
            if step > 50:
                break
            step += 1


        k = beam_size
        encoder_out = encoder(image)  
        encoder_out = encoder_out.view(1, encoder_dim)  
        encoder_out = encoder_out.expand(k, encoder_dim)  
        k_prev_words_r = torch.LongTensor([[word_map['<end>']]] * k).to(device)  
        seqs_r = k_prev_words_r     
        top_k_scores_r = torch.zeros(k, 1).to(device)  # (k, 1)        
        complete_seqs_r = list()
        complete_seqs_scores_r = list()        
        step = 1                   
        hr, cr = torch.zeros_like(img_r), torch.zeros_like(img_r)
        hr1, cr1 = torch.zeros_like(img_r), torch.zeros_like(img_r)

        while True:
            embeddings_reverse = decoder.embedding_reverse(k_prev_words_r).squeeze(1) 
            hr, cr = decoder.decode_step_reverse1(embeddings_reverse,(hr, cr)) 
            hr1, cr1 = decoder.decode_step_reverse2(torch.cat([hr, img_r], dim = 1),(hr1, cr1))
            scores_r = decoder.fc_r(hr1)  # (s, vocab_size)
            scores_r = F.log_softmax(scores_r, dim=1)

            scores_r = top_k_scores_r.expand_as(scores_r) + scores_r  
            if step == 1:
                top_k_scores_r, top_k_words_r = scores_r[0].topk(k, 0, True, True) 
            else:
                top_k_scores_r, top_k_words_r = scores_r.view(-1).topk(k, 0, True, True) 

            prev_word_inds_r = top_k_words_r / vocab_size  
            next_word_inds_r = top_k_words_r % vocab_size  

            seqs_r = torch.cat([seqs_r[prev_word_inds_r], next_word_inds_r.unsqueeze(1)], dim=1)             
            incomplete_inds_r = [ind for ind, next_word in enumerate(next_word_inds_r) if
                               next_word != word_map['<start>']]
            complete_inds_r = list(set(range(len(next_word_inds_r))) - set(incomplete_inds_r))
            
            if len(complete_inds_r) > 0:
                complete_seqs_r.extend(seqs_r[complete_inds_r].tolist())
                complete_seqs_scores_r.extend(top_k_scores_r[complete_inds_r])
            k -= len(complete_inds_r)  
            
            if k == 0:
                break
            seqs_r = seqs_r[incomplete_inds_r]            
            hr = hr[prev_word_inds_r[incomplete_inds_r]]
            cr = cr[prev_word_inds_r[incomplete_inds_r]]
            img_r = img_r[prev_word_inds_r[incomplete_inds_r]]
            hr1, cr1 = hr1[prev_word_inds_r[incomplete_inds_r]], cr1[prev_word_inds_r[incomplete_inds_r]]
    
            encoder_out = encoder_out[prev_word_inds_r[incomplete_inds_r]]
            top_k_scores_r = top_k_scores_r[incomplete_inds_r].unsqueeze(1)
            k_prev_words_r = next_word_inds_r[incomplete_inds_r].unsqueeze(1)
            
            if step > 50:
                break
            step += 1


        try:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
        except:
            seq = []
            empty_hypo += 1

        try:
            ir = complete_seqs_scores_r.index(max(complete_seqs_scores_r))
            seq_r = complete_seqs_r[ir]
            seq_r.reverse()
        except:
            seq_r = []
            empty_hypo_r += 1

        if seq != [] and seq_r != []:
            if max(complete_seqs_scores) >= max(complete_seqs_scores_r):
                seq_total = seq
            else: 
                seq_total = seq_r
        elif seq_r == []:
            seq_total = seq
        elif seq == []:            
            seq_total = seq_r
        
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        hypotheses.append([w for w in seq_total if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        hypotheses_f.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        hypotheses_r.append([w for w in seq_r if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        
        image_names.append(image_name)
        assert len(references) == len(hypotheses) == len(hypotheses_f) == len(hypotheses_r) == len(image_names)

    captions_dict['references']=references
    captions_dict['hypotheses']=hypotheses  
    captions_dict['hypotheses_r']=hypotheses_r  
    captions_dict['hypotheses_f']=hypotheses_f  
    captions_dict['image_names'] = image_names              
    if captions_dump==True:
        with open('generated_captions_f8k.json', 'w') as gencap:
            json.dump(captions_dict, gencap)
        save_captions_mscoco_format(word_map_file,references,hypotheses,image_names,str(beam_size)+'_f8ktest')
        save_captions_mscoco_format(word_map_file,references,hypotheses_f,image_names,str(beam_size)+'_f8ktest_f')
        save_captions_mscoco_format(word_map_file,references,hypotheses_r,image_names,str(beam_size)+'_f8ktest_r')

    bleu4 = corpus_bleu(references, hypotheses)
    bleu3 = corpus_bleu(references, hypotheses, (1.0/3.0,1.0/3.0,1.0/3.0,))
    bleu2 = corpus_bleu(references, hypotheses, (1.0/2.0,1.0/2.0,))
    bleu1 = corpus_bleu(references, hypotheses, (1.0/1.0,))

    bleu4_f = corpus_bleu(references, hypotheses_f)
    bleu3_f = corpus_bleu(references, hypotheses_f, (1.0/3.0,1.0/3.0,1.0/3.0,))
    bleu2_f = corpus_bleu(references, hypotheses_f, (1.0/2.0,1.0/2.0,))
    bleu1_f = corpus_bleu(references, hypotheses_f, (1.0/1.0,))

    bleu4_r = corpus_bleu(references, hypotheses_r)
    bleu3_r = corpus_bleu(references, hypotheses_r, (1.0/3.0,1.0/3.0,1.0/3.0,))
    bleu2_r = corpus_bleu(references, hypotheses_r, (1.0/2.0,1.0/2.0,))
    bleu1_r = corpus_bleu(references, hypotheses_r, (1.0/1.0,))

    print("The BLEU scores for overall model are {} \n for forward LSTM are {} \n and for backward LSTM are {}".format([bleu1,bleu2,bleu3,bleu4],
                                                                                   [bleu1_f,bleu2_f,bleu3_f,bleu4_f],[bleu1_r,bleu2_r,bleu3_r,bleu4_r]))
    with open('eval_run_logs.txt', 'a') as eval_run:
        eval_run.write("For beam-size {} the BLEU scores for overall model are {},\n for forward LSTM are {} and for backward LSTM are {}".format(beam_size, [bleu1,bleu2,bleu3,bleu4],
                                                                                   [bleu1_f,bleu2_f,bleu3_f,bleu4_f],[bleu1_r,bleu2_r,bleu3_r,bleu4_r]))

    return bleu1,bleu2,bleu3,bleu4


def main():
    beam_size = args.beam_size
    was_fine_tuned=False
    scores=evaluate(args.beam_size)
    print("\nBLEU scores @ beam size of %d is %.4f, %.4f, %.4f, %.4f." % (beam_size, scores[0],scores[1],scores[2],scores[3]))
    with open('eval_run_logs.txt', 'a') as eval_run:
        eval_run.write('The BLEU scores are {bleu_1}, {bleu_2}, {bleu_3}, {bleu_4}.\n'
                       'The beam_size was {beam}.
                       'The model was trained for {epochs} epochs.\n\n\n'.format( bleu_1=scores[0], bleu_2=scores[1], bleu_3=scores[2],
                                                          bleu_4=scores[3], beam=beam_size, epochs=checkpoint['epoch']))

if __name__ == '__main__':
    main()
