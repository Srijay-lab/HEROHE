from __future__ import print_function, division

import argparse
import collections
import copy
import glob
import os
import random
from datetime import datetime
import configparser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
import imageio
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from torch.optim import lr_scheduler
from torchvision import models
import utils
from torch.utils.tensorboard import SummaryWriter

class HEROHE:

    def __init__(self, configParser):

        self.data_folder = configParser.get('paths', 'data_folder')
        self.label_file = configParser.get('paths', 'label_file')
        self.batch_size = configParser.getint('model', 'batch_size')
        self.epochs = configParser.getint('model', 'epochs')
        self.lr = configParser.getfloat('model', 'lr')
        self.train_valid_splt = configParser.get('model', 'train_valid_split')
        self.model_folder = configParser.get('paths', 'model_folder')
        self.log_folder = configParser.get('paths', 'log_folder')
        self.trained_model_path = configParser.get('paths', 'trained_model_path')
        self.mode = configParser.get('model', 'mode')
        self.seed = configParser.getint('model', 'seed')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_aggregate_method = configParser.get('model', 'eval_aggregate_method')
        self.augment = configParser.getboolean('model', 'augment')
        self.one_label_smoothing = configParser.getboolean('model', 'one_label_smoothing')

        if (self.mode == "collect_stats"):
            self.collect_annotated_images_stats()
            exit(0)

        current_date_time = str(datetime.now())
        current_date_time=current_date_time.replace(" ","_")
        current_date_time=current_date_time.replace(":","")
        self.train_instance_folder_name = str(current_date_time + "/")
        print("Instance => ",self.train_instance_folder_name)
        self.log_file = os.path.join(self.log_folder,str(current_date_time))

        print("Dataset => ", self.data_folder)
        print("learning rate => ", self.lr)
        print("Batch Size => ", self.batch_size)
        print("Seed => ", self.seed)
        print("Augment => ",self.augment)
        print("One Label Smoothing => ", self.one_label_smoothing)

        self.make_folds()

        if(self.mode == "train"):
            os.makedirs(self.model_folder + self.train_instance_folder_name)

        self.load_data()
        self.create_model()

    def collect_annotated_images_stats(self):
        df = pd.read_csv(self.label_file)
        cases = df['Case']
        case_labels = df['HER2 Status']
        case_labels = [1 if x == 'Positive' else 0 for x in case_labels]
        self.case_label_map = dict(zip(cases, case_labels))

        print("Collecting images statistics")
        slide_images_dict = {}
        for image_path in glob.glob(self.data_folder + "/*.png"):
            image_name = os.path.split(image_path)[1]
            splt = image_name.split("_")
            slide_num = int(splt[0])
            if (slide_num in slide_images_dict):
                slide_images_dict[slide_num] += 1
            else:
                slide_images_dict[slide_num] = 1

        print("Slide to images map ", slide_images_dict)
        num_slides = len(slide_images_dict)
        num_total_images = sum(slide_images_dict.values())
        print("Total Number of slides are ", num_slides)
        print("Total Number of images are ", num_total_images)
        slide_images_dict = collections.OrderedDict(sorted(slide_images_dict.items()))
        plt.bar(range(len(slide_images_dict)), list(slide_images_dict.values()), align='center')
        plt.xticks(range(len(slide_images_dict)), list(slide_images_dict.keys()))
        plt.title("Distribution of number of patches extracted from annotated regions of WSI, Total = " + str(num_total_images))#, White Filter Threshold => "+str(self.WHITE_FILTER_THRESHOLD))
        plt.xlabel("WSI numbers " + str(num_slides))
        plt.ylabel("Number of patches extracted from annotated regions")
        plt.show()

        return 0

    def make_folds(self):

        df = pd.read_csv(self.label_file)
        cases = df['Case']
        case_labels = df['HER2 Status']
        case_labels = [1 if x == 'Positive' else 0 for x in case_labels]
        self.case_label_map = dict(zip(cases, case_labels))
        print("Case label map is ",self.case_label_map)

        slide_nums=[]
        pos_cases = []
        neg_cases = []
        for image_path in glob.glob(self.data_folder + "/*.png"):
            image_name=os.path.split(image_path)[1]
            splt = image_name.split("_")
            slide_num = int(splt[0])
            if(slide_num not in slide_nums):
                slide_nums.append(slide_num)
                if (self.case_label_map[slide_num] >= 0.5):
                    pos_cases.append(slide_num)
                else:
                    neg_cases.append(slide_num)

        num_pos_cases = len(pos_cases)
        num_neg_cases = len(neg_cases)
        num_cases = num_pos_cases + num_neg_cases

        print("Pos cases list ",pos_cases)
        print("Neg cases list ",neg_cases)
        print("Number of Positive Cases are ", num_pos_cases)
        print("Number of Negative Cases are ", num_neg_cases)
        print("Total cases => ",num_cases)

        random.Random(self.seed).shuffle(pos_cases)
        random.Random(self.seed).shuffle(neg_cases)

        train_fold,valid_fold,test_fold = map(int,self.train_valid_splt.split(':'))
        total_folds = train_fold + valid_fold + test_fold
        train_fraction = train_fold/total_folds
        valid_fraction = valid_fold/total_folds

        train_offset = int(train_fraction*num_pos_cases)
        valid_offset = train_offset + int(valid_fraction*num_pos_cases)
        train_pos_ids = pos_cases[0:train_offset]
        valid_pos_ids = pos_cases[train_offset:valid_offset]
        test_pos_ids = pos_cases[valid_offset:]

        train_offset = int(train_fraction*num_neg_cases)
        valid_offset = train_offset + int(valid_fraction*num_neg_cases)
        train_neg_ids = neg_cases[0:train_offset]
        valid_neg_ids = neg_cases[train_offset:valid_offset]
        test_neg_ids = neg_cases[valid_offset:]

        print("train_pos_ids is ",len(train_pos_ids))
        print("train_neg_ids is ", len(train_neg_ids))
        print("valid_pos_ids is ", len(valid_pos_ids))
        print("valid_neg_ids is ", len(valid_neg_ids))
        print("test_pos_ids is ", len(test_pos_ids))
        print("test_neg_ids is ", len(test_neg_ids))

        train_ids = train_pos_ids + train_neg_ids
        valid_ids = valid_pos_ids + valid_neg_ids
        test_ids = test_pos_ids + test_neg_ids
        self.fold_ids = [train_ids,valid_ids,test_ids]

    def load_data(self):

        train_ids = self.fold_ids[0]
        valid_ids = self.fold_ids[1]
        test_ids = self.fold_ids[2]
        num_train_samples = len(train_ids)
        num_valid_samples = len(valid_ids)
        num_test_samples = len(test_ids)
        print("Train WSI Ids are ",train_ids)
        print("Valid WSI Ids are ",valid_ids)
        print("Test WSI Ids are ", test_ids)
        print("Total train samples ",num_train_samples)
        print("Total valid samples ",num_valid_samples)
        print("Total test samples ", num_test_samples)

        self.train_images = []
        self.val_images = []
        self.test_images = []
        self.train_labels = []
        self.val_labels = []
        self.test_labels = []
        self.train_image_names = []
        self.val_image_names = []
        self.test_image_names = []

        image_paths = glob.glob(self.data_folder + "/*.png")

        # Read Images
        for image_path in image_paths:
            image_name = os.path.split(image_path)[1]
            splt = image_name.split("_")
            case_num = int(splt[0])
            image = imageio.imread(image_path)
            images = []
            if(self.augment and (case_num in train_ids)):
                images = utils.augment_image(image,4) #Augment each image with 4 similar images
            images.append(image)
            for image in images:
                avg = np.mean(image)
                image = image / 255
                height, width, depth = image.shape
                image = np.reshape(image, (depth, height, width))
                if (case_num in train_ids):
                    self.train_images.append(image)
                    self.train_labels.append(self.case_label_map[case_num])
                    self.train_image_names.append(image_name)
                elif (case_num in valid_ids):
                    self.val_images.append(image)
                    self.val_labels.append(self.case_label_map[case_num])
                    self.val_image_names.append(image_name)
                else:
                    self.test_images.append(image)
                    self.test_labels.append(self.case_label_map[case_num])
                    self.test_image_names.append(image_name)

        num_train_images = len(self.train_labels)
        num_val_images = len(self.val_labels)
        num_test_images = len(self.test_labels)
        print("Data is loaded with total instances taken for experiment = ",(num_train_images + num_val_images + num_test_images))
        print("num_train_pos_images = ", sum(self.train_labels))
        print("num_train_neg_images = ", num_train_images - sum(self.train_labels))
        print("num_val_pos_images = ", sum(self.val_labels))
        print("num_val_neg_images = ", num_val_images - sum(self.val_labels))
        print("num_test_pos_images = ", sum(self.test_labels))
        print("num_test_neg_images = ", num_test_images - sum(self.test_labels))

        self.train_images,self.train_labels,self.train_image_names = utils.shuffle_list_pair(self.train_images, self.train_labels, self.train_image_names, self.seed)
        self.val_images, self.val_labels,self.val_image_names = utils.shuffle_list_pair(self.val_images, self.val_labels, self.val_image_names, self.seed)
        self.test_images, self.test_labels,self.test_image_names = utils.shuffle_list_pair(self.test_images, self.test_labels, self.test_image_names, self.seed)

    def create_model(self):
        resnet_model = models.resnet18(pretrained=True)
        ''' Uncomment it if you want to freeze the network parameters
        for param in model_conv.parameters():
            param.requires_grad = False
        '''
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = resnet_model.fc.in_features
        resnet_model.fc = nn.Linear(num_ftrs, 2)
        #resnet_model = nn.Sequential(
        #    resnet_model,
         #   nn.Softmax(1)
        #)
        self.model = resnet_model.to(self.device)

    def train_model(self):

        if(os.path.exists(self.trained_model_path)):
            self.model.load_state_dict(torch.load(self.trained_model_path))
            print("Loading previously trained model")
        else:
            print("Start fresh training")

        criterion = nn.CrossEntropyLoss()
        if(self.one_label_smoothing):
            criterion = nn.BCELoss()

        optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr, weight_decay=0.05)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        best_epoch = 1

        logfile = open(self.log_file,"w")
        logfile.write("Learning rate => " + str(self.lr))
        logfile.write("Seed => " + str(self.seed))
        logfile.write("Batch Size => " + str(self.batch_size))
        logfile.write("Dataset used => " + str(self.data_folder))
        logfile.write("Augment => " + str(self.augment))
        logfile.write("One Label Smoothing => " + str(self.one_label_smoothing))

        writer = SummaryWriter()

        for epoch in range(self.epochs):
            print('Epoch {}/{}'.format(epoch+1, self.epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:

                input_images = self.train_images
                input_labels = self.train_labels
                curr_data_size = len(input_labels)

                if phase == 'train':

                    self.model.train()  # Set model to training mode
                    print("train len ",curr_data_size)

                    running_loss = 0.0
                    running_corrects = 0

                    indices = list(range(0, curr_data_size))
                    batch_indices_list = utils.make_batches_from_indices_list(indices, self.batch_size)

                    for batch_indices in batch_indices_list:

                        batch_input_images = torch.from_numpy(np.array(list(map(input_images.__getitem__, batch_indices)))).float().to(self.device)
                        batch_input_labels = list(map(input_labels.__getitem__, batch_indices))
                        if (self.one_label_smoothing):
                            batch_input_labels_smoooth = np.array(list(map(utils.one_label_smoothing, batch_input_labels)))
                            batch_input_labels_smoooth = torch.from_numpy(np.array(batch_input_labels_smoooth)).float().to(self.device)
                        batch_input_labels = torch.from_numpy(np.array(batch_input_labels)).long().to(self.device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # track history if only in train
                        with torch.set_grad_enabled(True):

                            outputs = self.model(batch_input_images)
                            _, preds = torch.max(outputs, 1)
                            if (self.one_label_smoothing):
                                outputs = nn.functional.softmax(outputs, dim=1)
                                outputs = outputs[:, 1]
                                loss = criterion(outputs, batch_input_labels_smoooth)
                            else:
                                loss = criterion(outputs, batch_input_labels)

                            loss.backward()
                            optimizer.step()

                        running_loss += loss.item() * batch_input_labels.size(0)
                        running_corrects += torch.sum(preds == batch_input_labels.data)

                    epoch_loss = running_loss / curr_data_size
                    epoch_acc = running_corrects.double() / curr_data_size

                    scheduler.step()
                    writer.add_scalar('Loss/train', epoch_loss, epoch)
                    writer.add_scalar('Accuracy/train', epoch_acc, epoch)

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))

                    logfile.write(str(epoch + 1) + " " + phase + " " + " accuracy - " + str(epoch_acc.item()) + "\n")

                else:
                    self.model.eval()   # Set model to evaluate mode

                    print("valid len ", len(self.val_labels))
                    patch_level_accuracy, wsi_accuracy, roc_auc, _, _ = self.evaluate_on_wsi(self.val_images,self.val_labels,self.val_image_names)
                    print('valid : Patch Level Accuracy : {:.4f} , WSI Level Accuracy : {:.4f} , AUC ROC : {:.4f}'.format(patch_level_accuracy, wsi_accuracy, roc_auc))

                    print("test len ", len(self.test_labels))
                    patch_level_accuracy, wsi_accuracy, roc_auc, _, _ = self.evaluate_on_wsi(self.test_images,
                                                                                             self.test_labels,
                                                                                             self.test_image_names)

                    writer.add_scalar('Patch Accuracy/valid', patch_level_accuracy, epoch)
                    writer.add_scalar('WSI Accuracy/valid', wsi_accuracy, epoch)
                    writer.add_scalar('AUC ROC/valid', roc_auc, epoch)

                    print('test : Patch Level Accuracy : {:.4f} , WSI Level Accuracy : {:.4f} , AUC ROC : {:.4f}'.format(
                        patch_level_accuracy, wsi_accuracy, roc_auc))

                    if(patch_level_accuracy > best_acc): #Save the best model as per accuracy
                        best_acc = patch_level_accuracy
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        best_epoch = epoch + 1

            if ((epoch+1)%5==0): #Save model after every 5 Epochs
                print("Saving the model after epoch number ",(epoch+1))
                torch.save(self.model.state_dict(), self.model_folder + self.train_instance_folder_name + str(epoch) + '_resnet_herohe.pth')

        print("Training completed")
        print('Best Accuracy: {:4f}'.format(best_acc))
        print("Best model saved after " + str(best_epoch) + " epochs")
        logfile.close()

        # load best model weights and save to file
        self.model.load_state_dict(best_model_wts)
        torch.save(self.model.state_dict(), self.model_folder + self.train_instance_folder_name + "best.pth")

    def eval_model(self):

        print("Evaluating")

        if(self.mode == "eval"):
            print("Loading the trained model")
            self.model.load_state_dict(torch.load(self.trained_model_path))
            print("Model is loaded")

        self.model.eval()

        print("Evaluating model on training data =>")
        _, _, roc_auc, fpr, tpr = self.evaluate_on_wsi(self.train_images,self.train_labels,self.train_image_names)
        self.plot_auc(fpr,tpr,roc_auc)
        print("Evaluating model on validation data =>")
        _, _, roc_auc, fpr, tpr = self.evaluate_on_wsi(self.val_images,self.val_labels,self.val_image_names)
        self.plot_auc(fpr, tpr, roc_auc)
        print("Evaluating model on Test data =>")
        _, _, roc_auc, fpr, tpr = self.evaluate_on_wsi(self.test_images, self.test_labels, self.test_image_names)
        self.plot_auc(fpr, tpr, roc_auc)

    def evaluate_on_wsi(self,input_images,input_labels,inpur_image_names):

        #Patch level computation
        input_len = len(input_labels)
        print("Number of patches => ",input_len)
        input_labels = [1 if x>=0.5 else 0 for x in input_labels]
        indices = list(range(0, input_len))
        batch_indices_list = utils.make_batches_from_indices_list(indices, self.batch_size)
        pred_patch_labels = torch.tensor(np.array([]),device='cuda').long()
        pred_patch_pos_probs = torch.tensor(np.array([]),device='cuda').float()
        pred_patch_neg_probs = torch.tensor(np.array([]),device='cuda').float()
        for batch_indices in batch_indices_list:
            batch_input_images = torch.from_numpy(np.array(list(map(input_images.__getitem__, batch_indices)))).float().to(self.device)
            with torch.set_grad_enabled(False):
                outputs = self.model(batch_input_images)
                pos_probs = outputs[:,1]
                neg_probs = outputs[:,0]
                _,preds = torch.max(outputs, 1)
                pred_patch_labels=torch.cat([pred_patch_labels,preds])
                pred_patch_pos_probs = torch.cat([pred_patch_pos_probs, pos_probs])
                pred_patch_neg_probs = torch.cat([pred_patch_neg_probs, neg_probs])

        assert (len(pred_patch_labels) == input_len)
        assert (len(pred_patch_pos_probs) == input_len)
        assert (len(inpur_image_names) == input_len)
        pred_patch_labels = pred_patch_labels.cpu().numpy()
        #print("Right combinations found => ",np.sum(input_labels==pred_patch_labels))
        patch_level_accuracy = np.sum(input_labels==pred_patch_labels)/input_len
        #print("Patch Level Accuracy => ", patch_level_accuracy)

        #WSI level computation
        pred_patch_pos_probs = pred_patch_pos_probs.cpu().numpy()
        pred_patch_neg_probs = pred_patch_neg_probs.cpu().numpy()

        slide_imagenum_dict = {}
        slide_posprob_dict = {}
        slide_negprob_dict = {}
        slide_winner_dict = {}

        i=0
        for image_name in inpur_image_names:
            splt = image_name.split("_")
            slide_num = int(splt[0])
            posprob = pred_patch_pos_probs[i]
            negprob = pred_patch_neg_probs[i]
            k = -1
            if(posprob >= negprob):
               k = 1
            posprob,negprob = utils.softmax(np.array([posprob,negprob]))
            if(slide_num in slide_imagenum_dict):
                slide_imagenum_dict[slide_num] += 1
                slide_posprob_dict[slide_num] += posprob
                slide_negprob_dict[slide_num] += negprob
                slide_winner_dict[slide_num] += k
            else:
                slide_imagenum_dict[slide_num] = 1
                slide_posprob_dict[slide_num] = posprob
                slide_negprob_dict[slide_num] = negprob
                slide_winner_dict[slide_num] = k
            i+=1

        num_slides = len(slide_imagenum_dict)
        pred_wsi_labels = []
        pred_pos_wsi_probs = []
        pred_neg_wsi_probs = []
        true_wsi_labels = []
        slides = []
        for slide_num in slide_imagenum_dict:
            if(slide_winner_dict[slide_num] >= 0):
                pred_wsi_labels.append(1)
            else:
                pred_wsi_labels.append(0)
            slides.append(slide_num)
            true_wsi_labels.append(self.case_label_map[slide_num])
            pred_pos_wsi_probs.append(slide_posprob_dict[slide_num]/slide_imagenum_dict[slide_num])
            pred_neg_wsi_probs.append(slide_negprob_dict[slide_num]/slide_imagenum_dict[slide_num])

        true_wsi_labels = np.array(true_wsi_labels)
        pred_wsi_labels = np.array(pred_wsi_labels)

        wsi_accuracy = np.sum(true_wsi_labels == pred_wsi_labels) / num_slides
        f1_macro = f1_score(true_wsi_labels, pred_wsi_labels, average='macro')
        f1_micro = f1_score(true_wsi_labels, pred_wsi_labels, average='micro')

        fpr, tpr, threshold = metrics.roc_curve(true_wsi_labels, pred_pos_wsi_probs)
        roc_auc = metrics.auc(fpr, tpr)

        if(self.mode == "eval"):
            print("Confusion matrix =>")
            print(confusion_matrix(true_wsi_labels, pred_wsi_labels))
            print("Slides => ", slides)
            print("True Labels => ", true_wsi_labels)
            print("Predicted Labels => ", pred_wsi_labels)
            print("Predicted Pos Probs => ", pred_pos_wsi_probs)
            print("Predicted Neg Probs => ", pred_neg_wsi_probs)
            print("ACCURACY => ", wsi_accuracy)
            print("AUC ROC => ", roc_auc)
            print("F1 Macro => ", f1_macro)
            print("F1 Micro => ", f1_micro)
            print("AUC ROC => ",roc_auc)

        return patch_level_accuracy,wsi_accuracy,roc_auc,fpr,tpr

    def plot_auc(self,fpr,tpr,roc_auc):

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

if __name__ == "__main__":
    print("Let us solve HEROHE")
    parser = argparse.ArgumentParser(description='HEROHE Baseline ResNet18')
    parser.add_argument('--config', default='config.txt', help='path to config file')
    args = parser.parse_args()

    configParser = configparser.RawConfigParser()
    configParser.read(args.config)
    mode = configParser.get('model', 'mode')

    Herohe = HEROHE(configParser)

    if (mode == "train"):
        Herohe.train_model()
    else:
        Herohe.eval_model()