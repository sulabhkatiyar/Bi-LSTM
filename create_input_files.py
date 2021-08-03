from utils import create_input_files
import os

# Do you want to delete the previous files which may have different minimum word frequency and other attributes?
delete_previous_files = True

if __name__ == '__main__':
    if delete_previous_files == True:
        for file_name in os.listdir('path_to_data'):
            if 'flickr8k' in file_name.split('_'):
                os.remove('path_to_data' + '/' + file_name)

    # Create input files (along with word map)
    create_input_files(dataset='flickr8k',
                       karpathy_json_path='path_to_karpathy_splits' + '/dataset_flickr8k.json',
                       image_folder= 'path_to_images' + '/f8k/flickr_data/Flickr_Data/Images',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='path_to_output_folder',
                       max_len=34)
