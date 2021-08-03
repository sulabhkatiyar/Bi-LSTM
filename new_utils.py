import json
from utils import *


class create_logs():
        pass

#Gives the top level summary of a model; useful for visualizing encoders
def arch_summary(arch):
    model = arch(False)
    tot = 0
    for i, l in enumerate(model.children()):
        n_layers = len(flatten_model(l))
        tot += n_layers
        print(f'({i}) {l.__class__.__name__:<12}: {n_layers:<4}layers (total: {tot})')



#read the first line to get the run number from log file
def get_run_num():
    with open('run_logs.txt', 'r') as rl:
        run_num= int(rl.readline().split('\n')[0])
    return run_num

#Run logs from running train.py 
def save_string_to_run_logs(string_to_save):
    with open('run_logs.txt', 'a') as rl:
        rl.write(string_to_save+'\n\n')

#read the top line and increase the run number 
def increase_run_number():
    with open('run_logs.txt', 'r') as rl:
        all_lines=rl.readlines()
    run_num=int(all_lines[0].split('\n')[0])+1
    all_lines[0]=str(run_num)+'\n'
    with open('run_logs.txt', 'w') as rl:
        rl.write(''.join(all_lines))

def save_captions_mscoco_format(word_map_file,references,hypotheses,image_names, data_name):
    with open(word_map_file, 'r') as word_map:
        wm=json.load(word_map)
    word_map_reverse={value:key for (key,value) in wm.items()}
    
    info= {'contributor': 'unknown','date_created': 'unknown',
           'description': data_name+' images for inference using COCO evaluation','url': 'unknown',
           'version': '1.0','year': 2020}
    type_='captions'
    images=list()
    licenses=list()
    
    captions_list=list()
    annotations=list()
    for i in range(len(references)):
        images.append({'date_captured': 'unknown','file_name': image_names[i],
                       'height': 0,'id': i,'license': 100,'url': 'unknown','width': 0})
        licenses.append({'id': i,'name': 'Attribution-NonCommercial-ShareAlike License','url': 'unknown'})

        candidate, actual= hypotheses[i], references[i]
        image_id = i
        caption = ' '.join([word_map_reverse[num] for num in candidate])
        captions_list.append({'caption': caption, 'image_id': image_id})
        for j in range(len(actual)):
            caption =' '.join(word_map_reverse[num] for num in actual[j])
            id_ = int(str(i) + '00' + str(j))
            annotations.append({'caption':caption, 'id': id_, 'image_id': image_id, 'image_name': image_names[i]})
            
    annotations_dict={'info': info, 'type':type_, 'images':images, 'licenses':licenses,
                      'annotations':annotations}
    
    with open('captions_'+data_name+'_st_results.json', 'w') as cf:
        json.dump(captions_list, cf)
        
    with open('captions_'+data_name+'.json', 'w') as af:
        json.dump(annotations_dict, af)
