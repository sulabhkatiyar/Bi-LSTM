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

cudnn.benchmark = True  

captions_dump=True
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def evaluate(loader, encoder, decoder, criterion, word_map, device):
    global captions_dump
    
    empty_hypo = 0
    empty_hypo_r = 0    
    references = list()
    hypotheses = list()
    hypotheses_f = list()
    hypotheses_r = list()

    beam_size = 1
    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)
    image_names = list()

    for i, (image, caps, caplens, allcaps, image_name) in enumerate(
            tqdm(loader, desc="EVALUATING ON VALIDATION SET")):

        k = beam_size
        image = image.to(device) 

        encoder_out = encoder(image) 
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(1, encoder_dim)  
        encoder_out = encoder_out.expand(k, encoder_dim) 
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device) 
        seqs = k_prev_words  
        top_k_scores = torch.zeros(k, 1).to(device) 

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

            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
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
        top_k_scores_r = torch.zeros(k, 1).to(device)  
        complete_seqs_r = list()
        complete_seqs_scores_r = list()
        step = 1
        hr, cr = torch.zeros_like(img_r), torch.zeros_like(img_r)
        hr1, cr1 = torch.zeros_like(img_r), torch.zeros_like(img_r)

        while True:
            embeddings_reverse = decoder.embedding_reverse(k_prev_words_r).squeeze(1)                
            hr, cr = decoder.decode_step_reverse1(embeddings_reverse,(hr, cr)) 
            hr1, cr1 = decoder.decode_step_reverse2(torch.cat([hr, img_r], dim = 1),(hr1, cr1))
            scores_r = decoder.fc_r(hr1)  
            scores_r = F.log_softmax(scores_r, dim=1)

            scores_r = top_k_scores_r.expand_as(scores_r) + scores_r  

            if step == 1:
                top_k_scores_r, top_k_words_r = scores_r[0].topk(k, 0, True, True) 
            else:
                top_k_scores_r, top_k_words_r = scores_r.view(-1).topk(k, 0, True, True) 
            
            prev_word_inds_r = top_k_words_r / vocab_size 
            next_word_inds_r = top_k_words_r % vocab_size 
            
            seqs_r = torch.cat([seqs_r[prev_word_inds_r], next_word_inds_r.unsqueeze(1)], dim=1) 

            incomplete_inds_r = [ind for ind, next_word in enumerate(next_word_inds_r) if next_word != word_map['<start>']]
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
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}], img_caps)) 
        references.append(img_captions)
        
        hypotheses.append([w for w in seq_total if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        hypotheses_f.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])        
        hypotheses_r.append([w for w in seq_r if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])        
        image_names.append(image_name)
        assert len(references) == len(hypotheses) == len(hypotheses_f) == len(hypotheses_r) == len(image_names)

    # Calculate BLEU scores
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

    print("The Validation set BLEU scores for overall model are {} \nfor forward LSTM are {}\n and for backward LSTM are {}".format([bleu1,bleu2,bleu3,bleu4],
                                                                                   [bleu1_f,bleu2_f,bleu3_f,bleu4_f],[bleu1_r,bleu2_r,bleu3_r,bleu4_r]))
    with open('val_run_logs.txt', 'a') as eval_run:
        eval_run.write("The Validation set BLEU scores, with beam-size {}, for overall model are {}\nfor forward LSTM are {} and for backward LSTM are {}".format(beam_size, [bleu1,bleu2,bleu3,bleu4],
                                                                                   [bleu1_f,bleu2_f,bleu3_f,bleu4_f],[bleu1_r,bleu2_r,bleu3_r,bleu4_r]))

    return [bleu1,bleu2,bleu3,bleu4], [bleu1_f,bleu2_f,bleu3_f,bleu4_f], [bleu1_r,bleu2_r,bleu3_r,bleu4_r]


