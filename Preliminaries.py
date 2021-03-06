#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('/opt/cocoapi/PythonAPI')
from pycocotools.coco import COCO
get_ipython().system('pip install nltk')
import nltk
nltk.download('punkt')
from data_loader import get_loader
from torchvision import transforms
import nltk

# Define a transform to pre-process the training images.
transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Set the minimum word count threshold.
vocab_threshold = 5

# Specify the batch size.
batch_size = 10

# Obtain the data loader.
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=False)



sample_caption = 'A person doing a trick on a rail while riding a skateboard.'



sample_tokens = nltk.tokenize.word_tokenize(str(sample_caption).lower())
print(sample_tokens)



sample_caption = []

start_word = data_loader.dataset.vocab.start_word
print('Special start word:', start_word)
sample_caption.append(data_loader.dataset.vocab(start_word))
print(sample_caption)




sample_caption.extend([data_loader.dataset.vocab(token) for token in sample_tokens])
print(sample_caption)



end_word = data_loader.dataset.vocab.end_word
print('Special end word:', end_word)

sample_caption.append(data_loader.dataset.vocab(end_word))
print(sample_caption)




import torch

sample_caption = torch.Tensor(sample_caption).long()
print(sample_caption)




# Preview the word2idx dictionary.
dict(list(data_loader.dataset.vocab.word2idx.items())[:10])


# Print the total number of keys in the word2idx dictionary.
print('Total number of tokens in vocabulary:', len(data_loader.dataset.vocab))




# Modify the minimum word count threshold.
vocab_threshold = 4

# Obtain the data loader.
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=False)




# Print the total number of keys in the word2idx dictionary.
print('Total number of tokens in vocabulary:', len(data_loader.dataset.vocab))


# There are also a few special keys in the `word2idx` dictionary.  You are already familiar with the special start word (`"<start>"`) and special end word (`"<end>"`).  There is one more special token, corresponding to unknown words (`"<unk>"`).  All tokens that don't appear anywhere in the `word2idx` dictionary are considered unknown words.  In the pre-processing step, any unknown tokens are mapped to the integer `2`.


unk_word = data_loader.dataset.vocab.unk_word
print('Special unknown word:', unk_word)

print('All unknown words are mapped to this integer:', data_loader.dataset.vocab(unk_word))


# Check this for yourself below, by pre-processing the provided nonsense words that never appear in the training captions. 



print(data_loader.dataset.vocab('jfkafejw'))
print(data_loader.dataset.vocab('ieowoqjf'))




# Obtain the data loader (from file). Note that it runs much faster than before!
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_from_file=True)




from collections import Counter

# Tally the total number of training captions with each length.
counter = Counter(data_loader.dataset.caption_lengths)
lengths = sorted(counter.items(), key=lambda pair: pair[1], reverse=True)
for value, count in lengths:
    print('value: %2d --- count: %5d' % (value, count))


import numpy as np
import torch.utils.data as data

# Randomly sample a caption length, and sample indices with that length.
indices = data_loader.dataset.get_train_indices()
print('sampled indices:', indices)

# Create and assign a batch sampler to retrieve a batch with the sampled indices.
new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
data_loader.batch_sampler.sampler = new_sampler
    
# Obtain the batch.
images, captions = next(iter(data_loader))
    
print('images.shape:', images.shape)
print('captions.shape:', captions.shape)

# (Optional) Uncomment the lines of code below to print the pre-processed images and captions.
# print('images:', images)
# print('captions:', captions)




# Watch for any changes in model.py, and re-load it automatically.
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Import EncoderCNN and DecoderRNN. 
from model import EncoderCNN, DecoderRNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Specify the dimensionality of the image embedding.
embed_size = 256


# Initialize the encoder. (Optional: Add additional arguments if necessary.)
encoder = EncoderCNN(embed_size)

# Move the encoder to GPU if CUDA is available.
encoder.to(device)
    
# Move last batch of images (from Step 2) to GPU if CUDA is available.   
images = images.to(device)

# Pass the images through the encoder.
features = encoder(images)

print('type(features):', type(features))
print('features.shape:', features.shape)

# Check that your encoder satisfies some requirements of the project! :D
assert type(features)==torch.Tensor, "Encoder output needs to be a PyTorch Tensor." 
assert (features.shape[0]==batch_size) & (features.shape[1]==embed_size), "The shape of the encoder output is incorrect."



# Specify the number of features in the hidden state of the RNN decoder.
hidden_size = 512


# Store the size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Initialize the decoder.
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Move the decoder to GPU if CUDA is available.
decoder.to(device)
    
# Move last batch of captions (from Step 1) to GPU if CUDA is available 
captions = captions.to(device)

# Pass the encoder output and captions through the decoder.
outputs = decoder(features, captions)

print('type(outputs):', type(outputs))
print('outputs.shape:', outputs.shape)

# Check that your decoder satisfies some requirements of the project! :D
assert type(outputs)==torch.Tensor, "Decoder output needs to be a PyTorch Tensor."
assert (outputs.shape[0]==batch_size) & (outputs.shape[1]==captions.shape[1]) & (outputs.shape[2]==vocab_size), "The shape of the decoder output is incorrect."




# # Specify the number of features in the hidden state of the RNN decoder.
# hidden_size = 512
# #-#-#-# Do NOT modify the code below this line. #-#-#-#
# # Store the size of the vocabulary.
# vocab_size = len(data_loader.dataset.vocab)
# # Initialize the decoder.
# decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
# # Move the decoder to GPU if CUDA is available.
# # Issue as at https://knowledge.udacity.com/questions/1448
# #decoder.to(device)
# use_gpu = torch.cuda.is_available() # check if we are using GPU
# if use_gpu:
#     model = model.cuda() # enable GPU on model
    
# # wrap them in Variable
# if use_gpu:
#     inputs = Variable(inputs.cuda())
#     labels = Variable(labels.cuda())
# else:
#     inputs, labels = Variable(inputs), Variable(labels)
    
# # Move last batch of captions (from Step 1) to GPU if CUDA is available 
# captions = captions.to(device)
# # Pass the encoder output and captions through the decoder.
# outputs = decoder(features, captions)
# print('type(outputs):', type(outputs))
# print('outputs.shape:', outputs.shape)
# # Check that your decoder satisfies some requirements of the project! :D
# assert type(outputs)==torch.Tensor, "Decoder output needs to be a PyTorch Tensor."
# assert (outputs.shape[0]==batch_size) & (outputs.shape[1]==captions.shape[1]) & (outputs.shape[2]==vocab_size), "The shape of the decoder output is incorrect."


