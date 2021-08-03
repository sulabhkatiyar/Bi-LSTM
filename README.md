# Bi-LSTM
## Partial re-implementation of Image Captioning with Deep Bidirectional LSTMs

### Note: this is a work in progress. I will upload results with other datasets and a detailed explanation soon.

### Introduction:
This model is similar to Bi-LSTM model proposed in _Image Captioning with Deep Bidirectional LSTMs_ published in _24th ACM international conference on Multimedia_ [[link]](https://dl.acm.org/doi/abs/10.1145/2964284.2964299) . An updated paper was published in journal ACM TOMM [[link]](https://dl.acm.org/doi/abs/10.1145/3115432).
There are following differences in our implementation:
1. I have not used Data Augmentation in this implementation. However, I have included options for horizontal and vertical data augmentation in the code which can be used by setting use_data_augmentation = True in train.py.
2. I have used batch size of 32 for all experiments and learning rate of 0.0001.
3. I have used VGG-16 CNN for image feature extraction whereas the authors used both AlexNet and VGG-16 for experiments.
4. Since both forward and backward LSTMs are trained for caption generation, I have experimented with both the inference strategy used in the paper (where the most likely sentence generated by forward or backward LSTMs is used as caption) and separate inference with backward and forward LSTMs.

### Method
I have used two-layered Bi-Directional LSTM as described in paper. The Text-LSTM (T-LSTM) takes as input, the word vector representations. In my implementation, the output of Text-LSTM (T-LSTM) and Image Feature representation are concatenated together before being used as input to Multimodal-LSTM (M-LSTM). It is mentioned in the paper that M-LSTM uses both image representation and T-LSTM hidden state but it's not clear to me how both quantities are used. So I have merged them by concatenation and fed them as input to M-LSTM. I have evaluate with 1, 3, 5, 10, 15 and 20 as beam sizes. In the paper, authors have used greedy evaluation with beam size as 1.

In the paper, both forward and backward LSTMs are trained to generate captions and their losses are combined. During evaluation the captions generated by forward and backward LSTM are evaluated and most likely caption is selected at each time-step. In our implementation, we save captions generated by both forward and backward LSTMs separately and also record caption generated by overall model (i.e., the most likely caption, from backward and forward LSTMs, recorded at each time-step.) 

### Results

**For Flickr8k dataset:**
The following table contains results obtained from overall model (best captions selected from forward and backward LSTMs):
|Result |Beam | BLEU-1 | BLEU-2 | BLEU-3| BLEU-4| METEOR | CIDEr | SPICE | ROUGE-L |
|---|---|---|---|---|---|---|---|---|---|
|Paper | 1 |  |  |  |  | |  |  |  |
|Our | 1 | 0.632 | 0.436 | 0.286 | 0.181 | 0.193 | 0.455 | 0.127 | 0.441 |
|Our | 3 | 0.602 | 0.418 | 0.277 | 0.179 | 0.174 | 0.454 | 0.124 | 0.425 |
|Our | 5 | 0.583 | 0.403 | 0.269 | 0.176 | 0.169 | 0.453 | 0.121 | 0.420 |
|Our | 10 | 0.563 | 0.394 | 0.265 | 0.171 | 0.165 | 0.421 | 0.118 | 0.421 |
|Our | 15 | 0.546 | 0.380 | 0.254 | 0.160 | 0.162 | 0.414 | 0.117 | 0.413 |
|Our | 20 | 0.535 | 0.371 | 0.247 | 0.155 | 0.158 | 0.406 | 0.116 | 0.409 |


### Reproducing the results:
1. Download 'Karpathy Splits' for train, validation and testing from [here](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).
2. For evaluation, the model already generates BLEU scores. In addition, it saves results and image annotations as needed in MSCOCO evaluation format. So for generation of METEOR, CIDEr, ROUGE-L and SPICE evaluation metrics, the evaluation code can be downloaded from [here](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI).

#### Prerequisites:
1. This code has been tested on python 3.6.9 but should word on all python versions > 3.6.
2. Pytorch v1.5.0
3. CUDA v10.1
4. Torchvision v0.6.0
5. Numpy v.1.15.0
6. pretrainedmodels v0.7.4 (Install from [source](https://github.com/Cadene/pretrained-models.pytorch.git)). (I think all versions will work but I have listed here for the sake of completeness.)


#### Execution:
1. First set the path to Flickr8k/Flickr30k/MSCOCO data folders in create_input_files_dataname.py file ('dataname' replaced by f8k/f30k/coco).
2. Create processed dataset by running: 
> python create_input_files_dataname.py

3. To train the model:
> python train_dataname.py

4. To evaluate: 
> python eval_dataname.py beamsize 

(eg.: python train_f8k.py 20)


