import sys
import copy
import torch
import numpy as np
import pandas as pd
from einops import rearrange
from Utilities import *



# Parameters

NPERIOD    = int(sys.argv[1])    # Predict 1 or 5 or 10 time steps after
EPOC       = int(sys.argv[2])    # Epochs

MN = ['Linear', 'CNN', 'LSTM', 'Transformer']
NM   = len(MN)    # Number of Models
df = pd.read_csv('dataset.csv')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
TRAIN_LOSS = np.zeros((NM, 4), dtype=float)   # Training Loss for Route A, D, B, C
TEST_LOSS  = np.zeros((NM, 4), dtype=float)   # Testing  Loss for Route A, D, B, C
FEA_TYPE   = 0                   # 0->All, 1->Motor, 2->Position
FEA_APPEND = 1                   # 0->+0, 1->+3, 2->3


class Linear_Net(torch.nn.Module):
    def __init__(self, n_time, n_output):
        super(Linear_Net, self).__init__()
        self.FL1o1 = torch.nn.Linear(n_time*7,  32)
        self.FL1o2 = torch.nn.Linear(n_time*13, 32)
        self.FL1o3 = torch.nn.Linear(n_time*3,  32)
        self.FL2o1, self.FL2o2, self.FL2o3 = torch.nn.Linear(32,16), torch.nn.Linear(32,16), torch.nn.Linear(32,16)
        self.FL3o1, self.FL4o1 = torch.nn.Linear(16,  7), torch.nn.Linear(7,  8)
        self.FL3o2, self.FL4o2 = torch.nn.Linear(16, 13), torch.nn.Linear(13, 8)
        self.FL3o3, self.FL4o3 = torch.nn.Linear(16,  3), torch.nn.Linear(3,  8)
        self.FL5o1, self.FL5o2, self.FL5o3 = torch.nn.Linear(8,n_output), torch.nn.Linear(8,n_output), torch.nn.Linear(8,n_output)
        
        self.SL1 = torch.nn.Linear(n_output*3, 16)
        self.SL2 = torch.nn.Linear(16, 4)
        self.SL3 = torch.nn.Linear(4,  n_output)
        self.LRelu = torch.nn.LeakyReLU()
    def forward(self, x):
        x1 = rearrange(x[:,:,:7],   'b t f -> b (t f)')
        x2 = rearrange(x[:,:,7:20], 'b t f -> b (t f)')
        x3 = rearrange(x[:,:,20:],  'b t f -> b (t f)')
        x1 = self.FL5o1( self.LRelu(self.FL4o1( self.LRelu(self.FL3o1( self.LRelu(self.FL2o1( torch.sigmoid(self.FL1o1(x1)))))))))
        x2 = self.FL5o2( self.LRelu(self.FL4o2( self.LRelu(self.FL3o2( self.LRelu(self.FL2o2( torch.sigmoid(self.FL1o2(x2)))))))))
        x3 = self.FL5o3( self.LRelu(self.FL4o3( self.LRelu(self.FL3o3( self.LRelu(self.FL2o3( torch.sigmoid(self.FL1o3(x3)))))))))

        x = torch.cat((x1, x2, x3), 1)
        x = self.LRelu(self.SL1(x))
        x = self.LRelu(self.SL2(x))
        x = self.SL3(x)
        return x, x1, x2, x3
    
class CNN_Net(torch.nn.Module):
    def __init__(self, n_time, n_output):
        super(CNN_Net, self).__init__() 
        self.CV1 = torch.nn.Conv1d(7,   7, 3, stride=1, padding='same')
        self.CV2 = torch.nn.Conv1d(13, 13, 3, stride=1, padding='same')
        self.CV3 = torch.nn.Conv1d(3,   3, 3, stride=1, padding='same')
        self.FL1o1 = torch.nn.Linear(n_time*7,  32)
        self.FL1o2 = torch.nn.Linear(n_time*13, 32)
        self.FL1o3 = torch.nn.Linear(n_time*3,  32)
        self.FL2o1, self.FL2o2, self.FL2o3 = torch.nn.Linear(32,16), torch.nn.Linear(32,16), torch.nn.Linear(32,16)
        self.FL3o1, self.FL4o1 = torch.nn.Linear(16,  7), torch.nn.Linear(7,  8)
        self.FL3o2, self.FL4o2 = torch.nn.Linear(16, 13), torch.nn.Linear(13, 8)
        self.FL3o3, self.FL4o3 = torch.nn.Linear(16,  3), torch.nn.Linear(3,  8)
        self.FL5o1, self.FL5o2, self.FL5o3 = torch.nn.Linear(8,n_output), torch.nn.Linear(8,n_output), torch.nn.Linear(8,n_output)

        self.SL1 = torch.nn.Linear(n_output*3, 16)
        self.SL2 = torch.nn.Linear(16, 4)
        self.SL3 = torch.nn.Linear(4,  n_output)
        self.LRelu = torch.nn.LeakyReLU()
    def forward(self, x):
        x1 = rearrange(x[:,:,:7],   'b t f -> b f t')
        x2 = rearrange(x[:,:,7:20], 'b t f -> b f t')
        x3 = rearrange(x[:,:,20:],  'b t f -> b f t')
        x1, x2, x3 = torch.sigmoid(self.CV1(x1)), torch.sigmoid(self.CV2(x2)), torch.sigmoid(self.CV3(x3))
        x1 = rearrange(x1, 'b f t -> b (t f)')
        x2 = rearrange(x2, 'b f t -> b (t f)')
        x3 = rearrange(x3, 'b f t -> b (t f)')
        x1 = self.FL5o1( self.LRelu(self.FL4o1( self.LRelu(self.FL3o1( self.LRelu(self.FL2o1( torch.sigmoid(self.FL1o1(x1)))))))))
        x2 = self.FL5o2( self.LRelu(self.FL4o2( self.LRelu(self.FL3o2( self.LRelu(self.FL2o2( torch.sigmoid(self.FL1o2(x2)))))))))
        x3 = self.FL5o3( self.LRelu(self.FL4o3( self.LRelu(self.FL3o3( self.LRelu(self.FL2o3( torch.sigmoid(self.FL1o3(x3)))))))))

        x = torch.cat((x1, x2, x3), 1)
        x = self.LRelu(self.SL1(x))
        x = self.LRelu(self.SL2(x))
        x = self.SL3(x)
        return x, x1, x2, x3
    
class LSTM_Net(torch.nn.Module):
    def __init__(self, n_time, n_output):
        super(LSTM_Net, self).__init__()
        self.LS1 = torch.nn.LSTM(7,   7, 2, bidirectional=False, batch_first=True)
        self.LS2 = torch.nn.LSTM(13, 13, 2, bidirectional=False, batch_first=True)
        self.LS3 = torch.nn.LSTM(3,   3, 2, bidirectional=False, batch_first=True)
        self.FL1o1 = torch.nn.Linear(n_time*7,  32)
        self.FL1o2 = torch.nn.Linear(n_time*13, 32)
        self.FL1o3 = torch.nn.Linear(n_time*3,  32)
        self.FL2o1, self.FL2o2, self.FL2o3 = torch.nn.Linear(32,16), torch.nn.Linear(32,16), torch.nn.Linear(32,16)
        self.FL3o1, self.FL4o1 = torch.nn.Linear(16,  7), torch.nn.Linear(7,  8)
        self.FL3o2, self.FL4o2 = torch.nn.Linear(16, 13), torch.nn.Linear(13, 8)
        self.FL3o3, self.FL4o3 = torch.nn.Linear(16,  3), torch.nn.Linear(3,  8)
        self.FL5o1, self.FL5o2, self.FL5o3 = torch.nn.Linear(8,n_output), torch.nn.Linear(8,n_output), torch.nn.Linear(8,n_output)
       
        self.SL1 = torch.nn.Linear(n_output*3, 16)
        self.SL2 = torch.nn.Linear(16, 4)
        self.SL3 = torch.nn.Linear(4,  n_output)
        self.LRelu = torch.nn.LeakyReLU()
    def forward(self, x):
        x1 = torch.sigmoid(self.LS1(x[:,:,:  7])[0])
        x2 = torch.sigmoid(self.LS2(x[:,:,7:20])[0])
        x3 = torch.sigmoid(self.LS3(x[:,:, 20:])[0])
        x1 = rearrange(x1, 'b t f -> b (t f)')
        x2 = rearrange(x2, 'b t f -> b (t f)')
        x3 = rearrange(x3, 'b t f -> b (t f)')
        x1 = self.FL5o1( self.LRelu(self.FL4o1( self.LRelu(self.FL3o1( self.LRelu(self.FL2o1( torch.sigmoid(self.FL1o1(x1)))))))))
        x2 = self.FL5o2( self.LRelu(self.FL4o2( self.LRelu(self.FL3o2( self.LRelu(self.FL2o2( torch.sigmoid(self.FL1o2(x2)))))))))
        x3 = self.FL5o3( self.LRelu(self.FL4o3( self.LRelu(self.FL3o3( self.LRelu(self.FL2o3( torch.sigmoid(self.FL1o3(x3)))))))))

        x = torch.cat((x1, x2, x3), 1)
        x = self.LRelu(self.SL1(x))
        x = self.LRelu(self.SL2(x))
        x = self.SL3(x)
        return x, x1, x2, x3
    
class Transformer(torch.nn.Module):
    def __init__(self, n_time, n_output):
        super(Transformer, self).__init__()
        self.TF1 = torch.nn.Transformer(d_model=7,  nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=16, batch_first=True)
        self.TF2 = torch.nn.Transformer(d_model=13, nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=16, batch_first=True)
        self.TF3 = torch.nn.Transformer(d_model=3,  nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=16, batch_first=True)
        self.FL1o1 = torch.nn.Linear(n_time*7,  32)
        self.FL1o2 = torch.nn.Linear(n_time*13, 32)
        self.FL1o3 = torch.nn.Linear(n_time*3,  32)
        self.FL2o1, self.FL2o2, self.FL2o3 = torch.nn.Linear(32,16), torch.nn.Linear(32,16), torch.nn.Linear(32,16)
        self.FL3o1, self.FL4o1 = torch.nn.Linear(16,  7), torch.nn.Linear(7,  8)
        self.FL3o2, self.FL4o2 = torch.nn.Linear(16, 13), torch.nn.Linear(13, 8)
        self.FL3o3, self.FL4o3 = torch.nn.Linear(16,  3), torch.nn.Linear(3,  8)
        self.FL5o1, self.FL5o2, self.FL5o3 = torch.nn.Linear(8,n_output), torch.nn.Linear(8,n_output), torch.nn.Linear(8,n_output)

        self.SL1 = torch.nn.Linear(n_output*3, 16)
        self.SL2 = torch.nn.Linear(16, 4)
        self.SL3 = torch.nn.Linear(4,  n_output)
        self.LRelu = torch.nn.LeakyReLU()
    def forward(self, x):
        x1 = torch.sigmoid(self.TF1(x[:,:,:7],  x[:,:,:7]  ))
        x2 = torch.sigmoid(self.TF2(x[:,:,7:20],x[:,:,7:20]))
        x3 = torch.sigmoid(self.TF3(x[:,:,20:], x[:,:,20:] ))
        x1 = rearrange(x1, 'b f t -> b (t f)')
        x2 = rearrange(x2, 'b f t -> b (t f)')
        x3 = rearrange(x3, 'b f t -> b (t f)')
        x1 = self.FL5o1( self.LRelu(self.FL4o1( self.LRelu(self.FL3o1( self.LRelu(self.FL2o1( torch.sigmoid(self.FL1o1(x1)))))))))
        x2 = self.FL5o2( self.LRelu(self.FL4o2( self.LRelu(self.FL3o2( self.LRelu(self.FL2o2( torch.sigmoid(self.FL1o2(x2)))))))))
        x3 = self.FL5o3( self.LRelu(self.FL4o3( self.LRelu(self.FL3o3( self.LRelu(self.FL2o3( torch.sigmoid(self.FL1o3(x3)))))))))

        x = torch.cat((x1, x2, x3), 1)
        x = self.LRelu(self.SL1(x))
        x = self.LRelu(self.SL2(x))
        x = self.SL3(x)
        return x, x1, x2, x3

    
    
for im, model_name in enumerate(MN):
    print('========== '+model_name+' ==========')
    print('----- Route A', ' -----')
    trainx, trainy, trainf, MAV_trainy, testx, testy, NF, BI, BN, MEAN, STD = Load_Data(df[df['route']=='A'], 0, 37, 10, NPERIOD, FEA_APPEND, FEA_TYPE, device,0,0,0)
    if   im==0:   net = Linear_Net( n_time=NPERIOD, n_output=1)
    elif im==1:   net = CNN_Net(    n_time=NPERIOD, n_output=1)
    elif im==2:   net = LSTM_Net(   n_time=NPERIOD, n_output=1)
    elif im==3:   net = Transformer(n_time=NPERIOD, n_output=1)
    net.to(device)
    
    BL, PATH  = np.inf, model_name+'-A-'+str(FEA_APPEND)+'_'+str(FEA_TYPE)+'_'+str(NPERIOD)+'_'+str(EPOC)
    loss_fn   = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    for epoc in range(EPOC):
        LP0, LP1, LP2, LP3 = 0, 0, 0, 0
        for b in range(BN):
            t, f = BI[b], BI[b+1]
            pred, pred1, pred2, pred3 = net(trainx[t:f])
            optimizer.zero_grad()
            loss = loss_fn(torch.reshape(pred ,(-1,)), trainy[t:f])
            LP0 += float(loss)
            loss.backward(retain_graph=True)
            loss = loss_fn(torch.reshape(pred1,(-1,)), trainy[t:f])/3.
            LP1 += float(loss)
            loss.backward(retain_graph=True)
            loss = loss_fn(torch.reshape(pred2,(-1,)), trainy[t:f])/3.
            LP2 += float(loss)
            loss.backward(retain_graph=True)
            loss = loss_fn(torch.reshape(pred3,(-1,)), trainy[t:f])/3.
            LP3 += float(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        pred, pred1, pred2, pred3 = net(trainx)
        loss = float(loss_fn(torch.reshape(pred,(-1,)), trainy))
        if epoc%10==9: 
            SUM = LP0+LP1+LP2+LP3
            print(epoc+1, '/', EPOC, '  ', round(float(loss), 5), round(LP0,1), round(LP1,1), round(LP2,1), round(LP3,1), ' - ', round(LP0/SUM,3), round(LP1/SUM,3), round(LP2/SUM,3), round(LP3/SUM,3))
        if loss<BL:
            BL = loss
            torch.save(net.state_dict(), PATH+'.pt') 
            print('Save Best at', epoc, 'with loss of', loss)
    net.load_state_dict(torch.load(PATH+'.pt'))
#     Net  = copy.deepcopy(net)
    pred, pred1, pred2, pred3 = net(trainx)
    TRAIN_LOSS[im][0], TRAIN_LOSS[im][3] = float(loss_fn(torch.reshape(pred,(-1,)), trainy)), np.nan
    print('Training Loss', round(TRAIN_LOSS[im][0],3) )
    pred, pred1, pred2, pred3 = net(testx)
    np.save(PATH+'.npy', pred.cpu().detach().numpy().reshape((-1,)))
    TEST_LOSS[im][0] = float(loss_fn(torch.reshape(pred,(-1,)), testy))
    print('Testing Loss',  round(TEST_LOSS[ im][0],3) )
    
    
    print('----- Route D', ' -----')
    PATH  = model_name+'-D-'+str(FEA_APPEND)+'_'+str(FEA_TYPE)+'_'+str(NPERIOD)+'_'+str(EPOC)
    targetx, targety, trainf, MAV_trainy, testx, testy, NF, BI, BN, mean, std = Load_Data(df[df['route']=='D'], 109, 114, 0, NPERIOD, FEA_APPEND, FEA_TYPE, device, 1, MEAN, STD)
    pred, pred1, pred2, pred3 = net(targetx)
    np.save(PATH+'.npy', pred.cpu().detach().numpy().reshape((-1,)))
    TEST_LOSS[im][3] = float(loss_fn(torch.reshape(pred,(-1,)), targety))
    print('Target Loss', round(TEST_LOSS[im][3],3) )
    
    
    for ir, route in enumerate(['B', 'C']):
        print('----- Route', route, ' -----')
        if ir==0:   F,T=37,73
        if ir==1:   F,T=73,109
        trainx, trainy, trainf, MAV_trainy, testx, testy, NF, BI, BN, mean, std = Load_Data(df[df['route']==route], F, T, 10, NPERIOD, FEA_APPEND, FEA_TYPE, device,0,0,0)
        if   im==0:   net = Linear_Net( n_time=NPERIOD,  n_output=1)
        elif im==1:   net = CNN_Net(    n_time=NPERIOD,  n_output=1)
        elif im==2:   net = LSTM_Net(   n_time=NPERIOD,  n_output=1)
        elif im==3:   net = Transformer(n_time=NPERIOD,  n_output=1)
#         net = copy.deepcopy(Net)
        net.to(device)
        
        BL, PATH = np.inf, model_name+'-'+route+'-'+str(FEA_APPEND)+'_'+str(FEA_TYPE)+'_'+str(NPERIOD)+'_'+str(EPOC)
        loss_fn   = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
        for epoc in range(EPOC):
            LP0, LP1, LP2, LP3 = 0, 0, 0, 0
            for b in range(BN):
                t, f = BI[b], BI[b+1]
                pred, pred1, pred2, pred3 = net(trainx[t:f])
                optimizer.zero_grad()
                loss = loss_fn(torch.reshape(pred ,(-1,)), trainy[t:f])
                LP0 += float(loss)
                loss.backward(retain_graph=True)
                loss = loss_fn(torch.reshape(pred1,(-1,)), trainy[t:f])/3.
                LP1 += float(loss)
                loss.backward(retain_graph=True)
                loss = loss_fn(torch.reshape(pred2,(-1,)), trainy[t:f])/3.
                LP2 += float(loss)
                loss.backward(retain_graph=True)
                loss = loss_fn(torch.reshape(pred3,(-1,)), trainy[t:f])/3.
                LP3 += float(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            pred, pred1, pred2, pred3 = net(trainx)
            loss = float(loss_fn(torch.reshape(pred,(-1,)), trainy))
            if epoc%10==9: 
                SUM = LP0+LP1+LP2+LP3
                print(epoc+1, '/', EPOC, '  ', round(float(loss), 5), round(LP0,1), round(LP1,1), round(LP2,1), round(LP3,1), ' - ', round(LP0/SUM,3), round(LP1/SUM,3), round(LP2/SUM,3), round(LP3/SUM,3))
            if loss<BL:
                BL = loss
                torch.save(net.state_dict(), PATH+'.pt') 
                print('Save Best at', epoc, 'with loss of', loss)
        net.load_state_dict(torch.load(PATH+'.pt'))
        pred, pred1, pred2, pred3 = net(trainx)
        TRAIN_LOSS[im][ir+1] = float(loss_fn(torch.reshape(pred,(-1,)), trainy))
        print('Training Loss', round(TRAIN_LOSS[im][ir+1],3) )
        pred, pred1, pred2, pred3 = net(testx)
        np.save(PATH+'.npy', pred.cpu().detach().numpy().reshape((-1,)))
        TEST_LOSS[im][ir+1] = float(loss_fn(torch.reshape(pred,(-1,)), testy))
        print('Testing Loss',  round(TEST_LOSS[ im][ir+1],3) )
        
    torch.cuda.empty_cache()

print('A (Source)', 'B(Transfer)', 'C (Few)', 'D (Zero)')
for im, model_name in enumerate(MN):
    print(model_name+':')
    print('Train:', end=' ')
    for i in range(4):
        print(round(TRAIN_LOSS[im][i],5), end=', ')
    print()
    print('Test :', end=' ')
    for i in range(4):
        print(round(TEST_LOSS[im][i],5), end=', ')
    print()

np.save('1-2StagePP_'+str(NPERIOD)+'_'+str(EPOC)+'.npy', [TRAIN_LOSS, TEST_LOSS])