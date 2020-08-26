# -*- coding: utf-8 -*-

import numpy as np
import torch
from Model import Test
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as Data
import datetime

cita = 1.003
loss_batch = 20

def train(Textfeaturemodel, Imgpredictmodel, Textpredictmodel, Imgmodel, Attentionmodel, Predictmodel, dataset,
          supervise_epochs = 200, text_supervise_epochs = 50, img_supervise_epochs = 50, 
          lr_supervise = 0.01, text_lr_supervise = 0.0001, img_lr_supervise = 0.0001,
          weight_decay = 0, batchsize = 32,lambda1=0.01,lambda2=1, textbatchsize = 32, imgbatchsize = 32, cuda = False, savepath = ''):
    #acc1, acc2, acc3 = Test.test(Textfeaturemodel, Imgpredictmodel, Textpredictmodel, Imgmodel, Attentionmodel, testdataset, batchsize = batchsize, cuda = cuda)
    #print('supervise', acc1, acc2, acc3)
    #exit(-1)
    '''
    pretrain ImgNet
    '''
    if cuda:
        Imgmodel.cuda()  
        Imgpredictmodel.cuda()     
    Imgmodel.train()
    Imgpredictmodel.train()

    par = []
    par.append({'params': Imgmodel.parameters()})
    par.append({'params': Imgpredictmodel.parameters()})
    optimizer = optim.Adam(par, lr = img_lr_supervise, weight_decay = weight_decay)
    scheduler = StepLR(optimizer, step_size = 500, gamma = 0.9)  
    criterion = torch.nn.BCELoss()
    train_img_supervise_loss = []
    batch_count = 0
    loss = 0
    for epoch in range(1, img_supervise_epochs + 1):
        print('train img supervise data:', epoch)
        data_loader = Data.DataLoader(dataset = dataset.supervise_(), batch_size = imgbatchsize, shuffle = True, num_workers = 0)
        for batch_index, (x, y) in enumerate(data_loader, 1):
            batch_count += 1
            scheduler.step()
            img_xx = x[0]
            label = y
            img_xx = img_xx.float()
            label = label.float()
            img_xx = Variable(img_xx).cuda() if cuda else Variable(img_xx)  
            label = Variable(label).cuda() if cuda else Variable(label)  
            imgxx = Imgmodel(img_xx)
            imgyy = Imgpredictmodel(imgxx)
            img_supervise_batch_loss = criterion(imgyy, label)
            loss += img_supervise_batch_loss.data.item()
            if batch_count >= loss_batch:
                loss = loss/loss_batch
                train_img_supervise_loss.append(loss)
                loss = 0
                batch_count = 0
            optimizer.zero_grad()
            img_supervise_batch_loss.backward()
            optimizer.step()
            #if batch_index >= 2:
            #    break
        if epoch % img_supervise_epochs == 0:
            filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
            torch.save(Imgmodel, savepath + filename + 'pretrainimgfeature.pkl')
            torch.save(Imgpredictmodel, savepath + filename + 'pretrainimgpredict.pkl')
            np.save(savepath + filename + "imgsuperviseloss.npy", train_img_supervise_loss)
            acc = Test.Imgtest(Imgmodel, Imgpredictmodel, dataset.test_(), batchsize = imgbatchsize, cuda = cuda)
            print('img supervise', epoch, acc)
            np.save(savepath + filename + "imgsuperviseacc.npy", [acc])
    

    '''
    pretrain TextNet.
    ''' 
    
    if cuda:
        Textfeaturemodel.cuda()  
        Textpredictmodel.cuda()     
    Textfeaturemodel.train()
    Textpredictmodel.train()

    par = []
    par.append({'params': Textfeaturemodel.parameters()})
    par.append({'params': Textpredictmodel.parameters()})
    optimizer = optim.Adam(par, lr = text_lr_supervise, weight_decay = weight_decay)
    scheduler = StepLR(optimizer, step_size = 500, gamma = 0.9)  
    criterion = torch.nn.BCELoss()
    train_text_supervise_loss = []
    batch_count = 0
    loss = 0
    for epoch in range(1, text_supervise_epochs + 1):
        print('train text supervise data:', epoch)
        data_loader = Data.DataLoader(dataset = dataset.supervise_(), batch_size = textbatchsize, shuffle = True, num_workers = 0)
        for batch_index, (x, y) in enumerate(data_loader, 1):
            batch_count += 1
            scheduler.step()
            text_xx = x[1]
            label = y
            text_xx = text_xx.float()
            label = label.float()
            text_xx = Variable(text_xx).cuda() if cuda else Variable(text_xx)  
            label = Variable(label).cuda() if cuda else Variable(label)  
            textxx = Textfeaturemodel(text_xx)
            textyy = Textpredictmodel(textxx)
            text_supervise_batch_loss = criterion(textyy, label)
            loss += text_supervise_batch_loss.data.item()
            if batch_count >= loss_batch:
                loss = loss/loss_batch
                train_text_supervise_loss.append(loss)
                loss = 0
                batch_count = 0
            optimizer.zero_grad()
            text_supervise_batch_loss.backward()
            optimizer.step()
            #if batch_index >= 2:
            #    break
        if epoch % text_supervise_epochs == 0:
            filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
            torch.save(Textfeaturemodel, savepath + filename + 'pretraintextfeature.pkl')
            torch.save(Textpredictmodel, savepath + filename + 'pretraintextpredict.pkl')
            np.save(savepath + filename + "textsuperviseloss.npy", train_text_supervise_loss)
            acc = Test.texttest(Textfeaturemodel, Textpredictmodel, dataset.test_(), batchsize = textbatchsize, cuda = cuda)
            print('text supervise', epoch, acc)
            np.save(savepath + filename + "textsuperviseacc.npy", [acc])
    
    '''
    train data mode.
    '''   
    
    if cuda:
        Textfeaturemodel.cuda()
        Imgpredictmodel.cuda()
        Textpredictmodel.cuda()
        Imgmodel.cuda()
        Attentionmodel.cuda()
        Predictmodel.cuda()
    Textfeaturemodel.train()
    Imgpredictmodel.train()
    Textpredictmodel.train()        
    Attentionmodel.train()
    Imgmodel.train()
    Predictmodel.train()

    par = []
    par.append({'params': Imgmodel.parameters()})
    par.append({'params': Attentionmodel.parameters()})
    par.append({'params': Textfeaturemodel.parameters()})
    par.append({'params': Imgpredictmodel.parameters()})
    par.append({'params': Textpredictmodel.parameters()})

    optimizer = optim.Adam(par, lr = lr_supervise, weight_decay = weight_decay)
    scheduler = StepLR(optimizer, step_size = 500, gamma = 0.9)  
    criterion = torch.nn.BCELoss()

    train_supervise_loss = []
    batch_count = 0
    loss = 0
    for epoch in range(1, supervise_epochs + 1):
        print('train supervise data:', epoch)
        data_loader = Data.DataLoader(dataset = dataset.unsupervise_(), batch_size = batchsize, shuffle = True, num_workers = 0)
        for batch_index, (x, y) in enumerate(data_loader, 1):
            batch_count += 1
            scheduler.step()
            x[0] = torch.cat(x[0], 0)
            x[1] = torch.cat(x[1], 0)
            y = torch.cat(y, 0)
            #print(x[0])
            #print(x[0].size())
            #print(y)
            #print(y.size())
            #exit(-1)
            '''
            Attention architecture and use bceloss.
            '''
            supervise_img_xx = x[0]
            supervise_text_xx = x[1]
            label = y
            supervise_img_xx = supervise_img_xx.float()
            supervise_text_xx = supervise_text_xx.float()
            label = label.float()
            supervise_img_xx = Variable(supervise_img_xx).cuda() if cuda else Variable(supervise_img_xx)  
            supervise_text_xx = Variable(supervise_text_xx).cuda() if cuda else Variable(supervise_text_xx)  
            label = Variable(label).cuda() if cuda else Variable(label)  
            supervise_imghidden = Imgmodel(supervise_img_xx)
            supervise_texthidden = Textfeaturemodel(supervise_text_xx)
            #supervise_imgk = Attentionmodel(supervise_imghidden)
            #supervise_textk = Attentionmodel(supervise_texthidden)
            #img_attention = nn.functional.softmax(supervise_imgk, dim = 0)
            #text_attention = nn.functional.softmax(supervise_textk, dim = 0)
            supervise_imgpredict = Imgpredictmodel(supervise_imghidden)
            supervise_textpredict = Textpredictmodel(supervise_texthidden)
            
            supervise_imgk = Attentionmodel(supervise_imghidden)
            supervise_textk = Attentionmodel(supervise_texthidden)
            modality_attention = []
            modality_attention.append(supervise_imgk)
            modality_attention.append(supervise_textk)
            #print(modality_attention)
            modality_attention = torch.cat(modality_attention, 1)
            #print(modality_attention)
            #exit(-1)
            modality_attention = nn.functional.softmax(modality_attention, dim = 1)
            img_attention = torch.zeros(1, len(y))
            img_attention[0] = modality_attention[:,0]
            img_attention = img_attention.t()
            text_attention = torch.zeros(1, len(y))
            text_attention[0] = modality_attention[:,1]
            text_attention = text_attention.t()
            if cuda:
                img_attention = img_attention.cuda()
                text_attention = text_attention.cuda()
            supervise_feature_hidden = img_attention * supervise_imghidden + text_attention * supervise_texthidden
            supervise_predict = Predictmodel(supervise_feature_hidden)
            #print(img_attention.size())
            #print(supervise_imgpredict.size())
            #print((img_attention*supervise_imgpredict).size())
            #exit(-1)
            #print(img_attention)
            #exit(-1)
            #imgpredict = Imgpredictmodel(imghidden)
            #textpredict = Textpredictmodel(texthidden)
            #feature_hidden = img_attention * imghidden + text_attention * texthidden            
            
            
            totalloss = criterion(supervise_predict, label)
            imgloss = criterion(supervise_imgpredict, label)
            textloss = criterion(supervise_textpredict, label)
            #print(supervise_imgpredict)
            #print(supervise_textpredict)
            #print(torch.max(supervise_imgpredict, supervise_textpredict))
            #exit(-1)
            #totalloss = criterion(img_attention*supervise_imgpredict + text_attention*supervise_textpredict, label)
            #totalloss = criterion(torch.max(supervise_imgpredict, supervise_textpredict), label)
            #imgloss1 = torch.sum(img_attention.t()[0] * torch.mean(imgloss, dim = 1))
            #textloss1 = torch.sum(text_attention.t()[0] * torch.mean(textloss, dim = 1))   
            '''
            Diversity Measure code.
            '''         
            similar = torch.bmm(supervise_imgpredict.unsqueeze(1), supervise_textpredict.unsqueeze(2)).view(supervise_imgpredict.size()[0])
            norm_matrix_img = torch.norm(supervise_imgpredict, 2, dim = 1)
            norm_matrix_text = torch.norm(supervise_textpredict, 2, dim = 1)
            div = torch.mean(similar/(norm_matrix_img * norm_matrix_text))
            #print(div)
            #print((similar/(norm_matrix_img * norm_matrix_text)).size())
            #exit(-1)
            #supervise_loss = imgloss1*10 + textloss1*10 + div
            supervise_loss = imgloss + textloss + totalloss*2
            '''
            Robust Consistency Measure code.
            '''
            x[2] = torch.cat(x[2], 0)
            x[3] = torch.cat(x[3], 0)
            unsupervise_img_xx = x[2]
            unsupervise_text_xx = x[3]
            unsupervise_img_xx = unsupervise_img_xx.float()
            unsupervise_text_xx = unsupervise_text_xx.float()
            unsupervise_img_xx = Variable(unsupervise_img_xx).cuda() if cuda else Variable(unsupervise_img_xx)  
            unsupervise_text_xx = Variable(unsupervise_text_xx).cuda() if cuda else Variable(unsupervise_text_xx)    
            unsupervise_imghidden = Imgmodel(unsupervise_img_xx)
            unsupervise_texthidden = Textfeaturemodel(unsupervise_text_xx)
            unsupervise_imgpredict = Imgpredictmodel(unsupervise_imghidden)
            unsupervise_textpredict = Textpredictmodel(unsupervise_texthidden)
            #dis = torch.sum(unsupervise_imgpredict - unsupervise_textpredict, dim = 1)
            unsimilar = torch.bmm(unsupervise_imgpredict.unsqueeze(1), unsupervise_textpredict.unsqueeze(2)).view(unsupervise_imgpredict.size()[0])
            unnorm_matrix_img = torch.norm(unsupervise_imgpredict, 2, dim = 1)
            unnorm_matrix_text = torch.norm(unsupervise_textpredict, 2, dim = 1)
            #print(unnorm_matrix_img)
            dis = 2 - unsimilar/(unnorm_matrix_img * unnorm_matrix_text)
            #print(dis)
            #print(dis*100)
            #exit(-1)
            tensor1 = dis[torch.abs(dis) < cita]
            tensor2 = dis[torch.abs(dis) >= cita]
            tensor1loss = torch.sum(tensor1 * tensor1/2)
            tensor2loss = torch.sum(cita * (torch.abs(tensor2) - 1/2 * cita))
            unsupervise_loss = (tensor1loss + tensor2loss)/unsupervise_img_xx.size()[0]        
            total_loss = supervise_loss + 0.01* div +  unsupervise_loss
            
            loss += total_loss.data.item()
            if batch_count >= loss_batch:
                loss = loss/loss_batch
                train_supervise_loss.append(loss)
                loss = 0
                batch_count = 0
            #print((imgloss.data.item(), textloss.data.item(), totalloss.data.item(), div.data.item(), unsupervise_loss.data.item()))
            #exit(-1)
            #print(total_loss.data.item())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            #if batch_index >= 2:
            #    break
        if epoch % 1 == 0:
            #filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
            #torch.save(Imgmodel, savepath + 'supervise' + filename +'img.pkl')
            #torch.save(Textfeaturemodel, savepath + 'supervise' + filename + 'Textfeaturemodel.pkl')
            #torch.save(Imgpredictmodel, savepath + 'supervise' + filename + 'Imgpredictmodel.pkl')
            #torch.save(Textpredictmodel, savepath + 'supervise' + filename + 'Textpredictmodel.pkl')
            #torch.save(Attentionmodel, savepath + 'supervise' + filename +'attention.pkl')
            #np.save(savepath + filename + "superviseloss.npy", train_supervise_loss)
            acc1, acc2, acc3, coverage1, coverage2, coverage3, example_auc1, example_auc2, example_auc3, macro_auc1, macro_auc2, macro_auc3, micro_auc1, micro_auc2, micro_auc3, ranking_loss1, ranking_loss2, ranking_loss3 = Test.test(Textfeaturemodel, Imgpredictmodel, Textpredictmodel, Imgmodel, Predictmodel, Attentionmodel, dataset.test_(), batchsize = batchsize, cuda = cuda)
            print('acc', epoch, acc1, acc2, acc3)
            print('coverage', epoch, coverage1, coverage2, coverage3)
            print('example_auc', epoch, example_auc1, example_auc2, example_auc3)
            print('macro_auc', epoch, macro_auc1, macro_auc2, macro_auc3)
            print('micro_auc', epoch, micro_auc1, micro_auc2, micro_auc3)
            print('ranking_loss', epoch, ranking_loss1, ranking_loss2, ranking_loss3)
            #np.save(savepath + filename + "superviseacc.npy", [acc1, acc2, acc3])

    return train_supervise_loss
