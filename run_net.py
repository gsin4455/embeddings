import torch
import torch.optim as optim
import time
import h5py
#from vgg import vgg 
from torch.autograd import Variable
import numpy as np
import argparse
import csv
from resnet import * 
import geoopt
from torch.utils.data.sampler import SubsetRandomSampler
from net import net

'''
Classes:
0) OOK,
1) 4ASK,
2) 8ASK,
3) BPSK,
4) QPSK,
5) 8PSK,
6) 16PSK,
7) 32PSK,
8) 16APSK,
9) 32APSK,
10) 64APSK,
11) 128APSK,
============
Data of focus:
12) 16QAM,
13) 32QAM,
14) 64QAM,
============
15) 128QAM,
16) 256QAM,
============

17) AM-SSB-WC,
18) AM-SSB-SC,
19) AM-DSB-WC,
20) AM-DSB-SC,
21) FM,
22) GMSK,
23) OQPSK
'''





def get_loss_optimizer(net,learning_rate=0.01): 
    #Loss
    loss = torch.nn.CrossEntropyLoss()
    
    #Optimizer
    
    #optimizer = optim.Adam(net.parameters(), lr= learning_rate)
    optimizer = geoopt.optim.RiemannianAdam(net.parameters(), lr= learning_rate)
    
    return(loss, optimizer)

def test_net(test_loader=None,path=None, batch_size= 128):
    #run test loop here
    
    n_batches = len(test_loader)
    model = torch.load(path)
    net = model['model']
    net.load_state_dict(model['state_dict'])
    for par in net.parameters():
        par.requires_grad = False
    net.eval()
    
    #writing results to spreadsheet
    
    fname = "test_pred.csv"
    f_out = open(fname,"w")
    wrt = csv.writer(f_out)
    net = net.cpu()
    #testing metrics
    corr_cnt = 0
    total_iter = 0
    for data in test_loader:
        [inputs,labels,snr] = data
        snr = snr.numpy()
        #inputs, labels,snr = Variable(inputs), Variable(labels), Variable(snr)
        pred = net(inputs.float()).numpy()
        pred = np.argmax(pred,axis=1)
        labels = np.argmax(labels.numpy(),axis=1)
        for s,p,l in zip(snr,pred,labels):
            if(p == l):
                corr_cnt += 1
                wrt.writerow([s,p,l])
            total_iter +=1 
    print("Test done, accr = :" + str(corr_cnt/total_iter))
    f_out.close()

def train_net(train_loader=None,net=None, batch_size=128, n_epochs=500 ,learning_rate = 0.01,opt = 0,saved_model=None):
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    n_batches = len(train_loader)
    
    
    #Create our loss and optimizer functions
    loss, optimizer = get_loss_optimizer(net,learning_rate)
    
    training_start_time = time.time()
    
    #Training results printed to csv-different for each optimizer
    
    f_out = open("tr_" + str(opt) + ".csv","w")
    wrt = csv.writer(f_out)

    total_train_loss = 0
    
    #net = net.to('cuda')
    #Loop for n_epochs
    for epoch in range(n_epochs):
        
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        
        wrt.writerow([epoch,total_train_loss])  

        if (((epoch+1) % 250) == 0):
            checkpoint = {'model':net,
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict()}
            file_name = 'checkpoint.pt'
            torch.save(checkpoint, file_name)
        i = 0
        for data in train_loader:
            
            #Get inputs
            #Wrap them in a Variable object
            [inputs,labels,snr] = data 
            #inputs, labels,snr = Variable(inputs).to('cuda'), Variable(labels).to('cuda'), Variable(snr)
            inputs, labels,snr = Variable(inputs), Variable(labels), Variable(snr)
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            outputs = net(inputs)
            loss_size = loss(outputs,np.argmax(labels,axis=1))
            loss_size.backward()
            optimizer.step()
            
            #Print statistics
            running_loss += loss_size.data
            total_train_loss += loss_size.data

            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), total_train_loss/print_every, time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
            i += 1
    
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    final = {'model':net,
                'state_dict': net.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(final, saved_model)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",help="Test the model on this dataset")
    parser.add_argument("--optim", type=int, help= "Type of optimizer to use")
    parser.add_argument("--data_dir",type=str, help="Path to files")
    parser.add_argument("--steps", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch Size")
    parser.add_argument("--learning_rate", type=float, help="Learning Rate")
    parser.add_argument("--model_path", type=str, help="Path to Save Model")
    parser.add_argument("--classes", type=int, help="Number of classes")
    
    args = parser.parse_args()
        
    data = h5py.File('%s/qam_train3.hdf5' % args.data_dir, 'r')
    
    np.random.seed(2019)
                    
    n_ex = data['X'].shape[0]
    n_train = int(n_ex*(7/8))

    idx = np.random.permutation(n_ex)

    train_idx = idx[:n_train].tolist()
    test_idx = idx[n_train:n_ex].tolist()

    train_samp = SubsetRandomSampler(train_idx)
    test_samp = SubsetRandomSampler(test_idx)
                                                            
    data = torch.utils.data.TensorDataset(torch.from_numpy(np.transpose(data['X'].value, (0,2,1))),torch.from_numpy(data['Y'].value), torch.from_numpy(data['Z'].value))
    #nn = net(args.classes)
    #nn = ResNet18(args.classes)
    
    
    if(args.train):
        #Training data
        '''
        model = torch.load(args.model_path)
        nn = model['model']
        nn.load_state_dict(model['state_dict'])
        
        '''
        if(args.optim == 0):
            nn = net(args.classes,manifold=geoopt.manifolds.Euclidean())
             
        elif(args.optim==1):
            nn = net(args.classes,manifold=geoopt.manifolds.PoincareBall())

        #nn = net(args.classes)

        train_loader = torch.utils.data.DataLoader(data,batch_size =args.batch_size, sampler=train_samp)    
        train_net(train_loader, nn,args.batch_size, args.steps,args.learning_rate,args.optim,args.model_path)
    else:

        test_loader = torch.utils.data.DataLoader(data,batch_size=args.batch_size,sampler=test_samp)
        test_net(test_loader, args.model_path,args.batch_size)
     
    

        
    
