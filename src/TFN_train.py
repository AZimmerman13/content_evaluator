from __future__ import print_function
from TFN_model import TFN
#from utils import MultimodalDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import os
import argparse
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd


def preprocess(data_path):
    data = np.load(data_path)
    labels = data[:,0]
    text = data[:, 1:1001]
    img = data[:,1001:]

    img_dim = img.shape[1]
    print("Image feature dimension is: {}".format(img_dim))
    text_dim = text.shape[1]
    print("Text feature dimension is: {}".format(text_dim))

    input_dims = (img_dim, text_dim)
    d = {'labels':labels, 'img':img, 'text':text}
    # output = pd.DataFrame(data=d)

    return labels, img, text, input_dims
    
    """
    # parse the input args
#     dataset = options['dataset']
#     epochs = options['epochs']
#     model_path = options['model_path']
#     max_len = options['max_len']

    # prepare the paths for storing models
    model_path = os.path.join(
        model_path, "tfn.pt")
    print("Temp location for saving model: {}".format(model_path))

    # prepare the datasets
    print("Currently using {} dataset.".format(dataset))
#     mosi = MultimodalDataset(dataset, max_len=max_len)

# We Need to add our own data
    train_set, valid_set, test_set = mosi.train_set, mosi.valid_set, mosi.test_set

    audio_dim = train_set[0][0].shape[1]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[1]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = train_set[0][2].shape[1]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim)

    # normalize the visual features
    visual_max = np.max(np.max(np.abs(train_set.visual), axis=0), axis=0)
    visual_max[visual_max==0] = 1
    train_set.visual = train_set.visual / visual_max
    valid_set.visual = valid_set.visual / visual_max
    test_set.visual = test_set.visual / visual_max

    # for visual and audio modality, we average across time
    # here the original data has shape (max_len, num_examples, feature_dim)
    # after averaging they become (1, num_examples, feature_dim)
    train_set.visual = np.mean(train_set.visual, axis=0, keepdims=True)
    train_set.audio = np.mean(train_set.audio, axis=0, keepdims=True)
    valid_set.visual = np.mean(valid_set.visual, axis=0, keepdims=True)
    valid_set.audio = np.mean(valid_set.audio, axis=0, keepdims=True)
    test_set.visual = np.mean(test_set.visual, axis=0, keepdims=True)
    test_set.audio = np.mean(test_set.audio, axis=0, keepdims=True)

    # remove possible NaN values
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0

    return train_set, valid_set, test_set, input_dims
"""

def display(test_roc_auc, test_loss, test_binacc, test_precision, test_recall, test_f1, test_septacc, test_corr):
    print("ROC AUC on test set is {}".format(test_roc_auc))
    print("MAE on test set is {}".format(test_loss))
    print("Binary accuracy on test set is {}".format(test_binacc))
    print("Precision on test set is {}".format(test_precision))
    print("Recall on test set is {}".format(test_recall))
    print("F1 score on test set is {}".format(test_f1))
    print("Seven-class accuracy on test set is {}".format(test_septacc))
    print("Correlation w.r.t human evaluation on test set is {}".format(test_corr))

def main():
    DTYPE = torch.FloatTensor
    train_labels, train_img, train_text, input_dims = preprocess('data/train.npy')
    valid_labels, valid_img, valid_text, _ = preprocess('data/validate.npy')
    test_labels, test_img, test_text, _ = preprocess('data/test.npy')

    batch_sz = 85

    model = TFN(input_dims, (64, 128), 64, (0.3, 0.3, 0.3), 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cuda':
        model = model.cuda()
        DTYPE = torch.cuda.FloatTensor
    print("Model initialized")
    criterion = nn.L1Loss(size_average=False)
    optimizer = optim.Adam(list(model.parameters())[2:]) # don't optimize the first 2 params, they should be fixed (output_range and shift)
    
    # setup training
    complete = True
    min_valid_loss = float('Inf')
    
    patience = 20
    epochs = 10
    model_path = 'models/best_model.pth'
    
    train_set = list(zip(train_labels, train_img, train_text))
    valid_set = list(zip(valid_labels, valid_img, valid_text))
    test_set = list(zip(test_labels, test_img, test_text))

    
    train_iterator = DataLoader(train_set, batch_size=batch_sz, num_workers=4, shuffle=True)
    valid_iterator = DataLoader(valid_set, batch_size=len(valid_set), num_workers=4, shuffle=True)
    test_iterator = DataLoader(test_set, batch_size=len(test_set), num_workers=4, shuffle=True)
    # curr_patience = patience
    for e in range(epochs):
        model.train()
        model.zero_grad()
        train_loss = 0.0
        for labels, img, text in train_iterator:
            model.zero_grad()

            # the provided data has format [batch_size, seq_len, feature_dim] or [batch_size, 1, feature_dim]
            # x = batch[:-1]
            # x_a = Variable(x[0].float().type(DTYPE), requires_grad=False).squeeze()
            x_i = Variable(img.float().type(DTYPE), requires_grad=False).squeeze()
            x_t = Variable(text.float().type(DTYPE), requires_grad=False)
            y = Variable(labels.view(-1, 1).float().type(DTYPE), requires_grad=False)
            output = model(x_i, x_t)
            loss = criterion(output, y)
            loss.backward()
            train_loss += loss.item() / len(train_set)
            optimizer.step()

        print("Epoch {} complete! Average Training loss: {}".format(e, train_loss))

        # Terminate the training process if run into NaN
        if np.isnan(train_loss):
            print("Training got into NaN values...\n\n")
            complete = False
            break

        # On validation set we don't have to compute metrics other than MAE and accuracy
        model.eval()
        for labels, img, text in valid_iterator:
            # x = batch[:-1]
            # x_a = Variable(x[0].float().type(DTYPE), requires_grad=False).squeeze()
            x_i = Variable(img.float().type(DTYPE), requires_grad=False).squeeze()
            x_t = Variable(text.float().type(DTYPE), requires_grad=False)
            y = Variable(labels.view(-1, 1).float().type(DTYPE), requires_grad=False)
            output = model(x_i, x_t)
            valid_loss = criterion(output, y)
        output_valid = output.cpu().data.numpy().reshape(-1)
        y = y.cpu().data.numpy().reshape(-1)

        if np.isnan(valid_loss.item()):
            print("Training got into NaN values...\n\n")
            complete = False
            break

        # valid_binacc = accuracy_score(output_valid>=0, y>=0)
        valid_binacc = accuracy_score(output_valid>=0.5, y)
        #zbreakpoint()
        roc_auc = roc_auc_score(y, output_valid)

        print("Validation loss is: {}".format(valid_loss.item() / len(valid_set)))
        print("Validation binary accuracy is: {}".format(valid_binacc))
        print("ROC AUC is: {}".format(roc_auc))

        if (valid_loss.item() < min_valid_loss):
            curr_patience = patience
            min_valid_loss = valid_loss.item()
            torch.save(model.state_dict(), model_path)
            print("Found new best model, saving to disk...")
        else:
            curr_patience -= 1
        
        if curr_patience <= 0:
            break
        print("\n\n")
    '''
    if complete:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        # best_model = torch.load(model_path)
        model.eval()
        for img, text, labels in test_iterator:
            # x = batch[:-1]
            # x_a = Variable(x[0].float().type(DTYPE), requires_grad=False).squeeze()
            x_i = Variable(img.float().type(DTYPE), requires_grad=False).squeeze()
            x_t = Variable(text.float().type(DTYPE), requires_grad=False)
            y = Variable(labels.view(-1, 1).float().type(DTYPE), requires_grad=False)
            output_test = model(x_i, x_t)
            loss_test = criterion(output_test, y)
            test_loss = loss_test.item()
        output_test = output_test.cpu().data.numpy().reshape(-1)
        y = y.cpu().data.numpy().reshape(-1)

        test_binacc = accuracy_score(output_test>=0.5, y)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y, output_test>=0.5, average='binary')
        test_septacc = (output_test.round() == y.round()).mean()
        test_roc_auc = roc_auc_score(y, output_test)

        # compute the correlation between true and predicted scores
        test_corr = np.corrcoef([output_test, y])[0][1]  # corrcoef returns a matrix
        test_loss = test_loss / len(test_set)
        
        display(test_roc_auc, test_loss, test_binacc, test_precision, test_recall, test_f1, test_septacc, test_corr)
    '''
    return


if __name__ == "__main__":
    
#     OPTIONS = argparse.ArgumentParser()
#     OPTIONS.add_argument('--dataset', dest='dataset',
#                          type=str, default='MOSI')
#     OPTIONS.add_argument('--epochs', dest='epochs', type=int, default=50)
#     OPTIONS.add_argument('--batch_size', dest='batch_size', type=int, default=32)
#     OPTIONS.add_argument('--patience', dest='patience', type=int, default=20)
#     OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=False)
#     OPTIONS.add_argument('--model_path', dest='model_path',
#                          type=str, default='models')
#     OPTIONS.add_argument('--max_len', dest='max_len', type=int, default=20)
#     PARAMS = vars(OPTIONS.parse_args())
    main()