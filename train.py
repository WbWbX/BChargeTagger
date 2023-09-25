import argparse
import tensorflow as tf
import numpy as np
import scipy
import h5py
import os
import sys
import glob
import math
import re
import csv
import sklearn.metrics

from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--lr', dest='learningRate',type=float, default=5e-3)
parser.add_argument('--epochs', dest='epochs',type=int, default=80)
parser.add_argument('--output', dest='output',type=str,required=True)
parser.add_argument('--optimize', dest='optimize',action='store_true',default=False)

args = parser.parse_args()

SMALL_SIZE = 15
MEDIUM_SIZE = 15
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

from featureDict import featureDict
from Network import Network

try:
    os.makedirs(args.output)
except Exception as e:
    print (e)

files = glob.glob("/nfs/dust/cms/user/mkomm/WbWbX/BChargeTagger/samples/unpacked*.tfrecord")

print ("files: ",len(files))
files = files[:500]
#baggingFraction = 0.5
splitFraction = 0.8
nSplit = int(splitFraction*(len(files)-2))+1
trainFiles = files[:nSplit]
testFiles = files[nSplit:]

features = {
    "xcharge": tf.io.FixedLenFeature([1], tf.float32),
    "bPartonCharge": tf.io.FixedLenFeature([1], tf.float32),
    "bHadronCharge": tf.io.FixedLenFeature([1], tf.float32),
    "cHadronCharge": tf.io.FixedLenFeature([1], tf.float32),
    "bDecay": tf.io.FixedLenFeature([7], tf.float32),
}
for name,featureGroup in featureDict.items():
    if 'max' in featureGroup.keys():
        features[name] = tf.io.FixedLenFeature([featureGroup['max']*len(featureGroup['branches'])], tf.float32)
    else:
        features[name] = tf.io.FixedLenFeature([len(featureGroup['branches'])], tf.float32)
                
def decode_data(raw_data):
    decoded_data = tf.io.parse_example(raw_data,features)
    for name,featureGroup in featureDict.items():
        if 'max' in featureGroup.keys():
            decoded_data[name] = tf.reshape(decoded_data[name],[-1,featureGroup['max'],len(featureGroup['branches'])])
    return decoded_data

def setup_pipeline(fileList):
    ds = tf.data.Dataset.from_tensor_slices(fileList)
    ds.shuffle(len(fileList),reshuffle_each_iteration=True)
    ds = ds.interleave(
        lambda x: tf.data.TFRecordDataset(
            x, compression_type='GZIP', buffer_size=100000000
        ),
        cycle_length=6, 
        block_length=250, 
        num_parallel_calls=6
    )
    ds = ds.batch(250) #decode in batches (match block_length?)
    ds = ds.map(decode_data, num_parallel_calls=6)
    ds = ds.unbatch()
    ds = ds.shuffle(50000,reshuffle_each_iteration=True)
    ds = ds.batch(10000)
    ds = ds.prefetch(5)
    
    return ds
    
def learningRateWithDecay(lr,epoch):
    return lr/(1+0.15*max(0,epoch-10)**1.5)

dsTrain = setup_pipeline(trainFiles)
dsTest = setup_pipeline(testFiles)


initLearningRate = args.learningRate
network = Network()
model = network.makeModel()
opt = tf.keras.optimizers.Adam(learning_rate=learningRateWithDecay(initLearningRate,0),epsilon=1e-4)
model.compile(opt,loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True,label_smoothing=1e-2))

print ("="*80)
model.summary()

if args.optimize:
    print ("Optimizing learning rate")
    lrScanValues = np.logspace(-7,-1,200)
    logLossValues = np.zeros(len(lrScanValues))
    iScan = 0
    
    initWeights = model.get_weights()
    while (iScan<len(lrScanValues)):
        for _,batch in enumerate(dsTrain):
            if iScan>=len(lrScanValues):
                break
            tf.keras.backend.set_value(
                model.optimizer.learning_rate, 
                lrScanValues[iScan]
            )
            
            inputsList = [
                batch['cpf'],batch['cpf_charge'],
                batch['muon'],batch['muon_charge'],
                batch['electron'],batch['electron_charge'],
                batch['npf'],
                batch['sv'],
                batch['global']
            ]
            truth = tf.keras.utils.to_categorical(
                batch['xcharge'], num_classes=4, dtype='float32'
            ) 
            sample_weight = np.sum(truth*truth.shape[0]/truth.shape[1]/(1.+np.sum(truth,axis=0)),axis=1)
            
            loss = model.train_on_batch(inputsList,[truth],sample_weight=sample_weight)
            logLossValues[iScan] = np.log(loss)
            if iScan%10==0:
                print ("Scan step %04i/%04i: lr=%10.2e, loss=%10.4e"%(
                    iScan,len(lrScanValues),lrScanValues[iScan],loss
                ))
            iScan+=1
            
            
    logLossValuesSmooth = logLossValues
    smoothIteration = 5
    smoothWindow = 4
    for _ in range(smoothIteration):
        logLossValuesSmooth = np.convolve(logLossValuesSmooth,np.ones(smoothWindow),'same')/smoothWindow
    offset = (smoothIteration+1)*smoothWindow
    bestLearningRateIdx = offset+np.argmax(logLossValuesSmooth[offset:len(lrScanValues)-offset]-logLossValuesSmooth[offset+1:len(lrScanValues)-offset+1])

    print ("-"*80)
    print ("Best inital learning rate: %10.2e"%lrScanValues[bestLearningRateIdx])
    fig = plt.figure(figsize=[6.4, 5.8],dpi=300)
    plt.plot(lrScanValues,logLossValues,color='blue',alpha=0.5)
    plt.plot(lrScanValues,logLossValuesSmooth,color='blue',alpha=1.,linewidth=2)
    plt.plot(lrScanValues[bestLearningRateIdx],logLossValues[bestLearningRateIdx],marker='o',color='orange',markersize=3)
    plt.xscale('log')
    plt.grid(which='minor',linestyle='--',color='gray')
    plt.grid(which='major',linestyle='--',color='black')
    plt.xlabel('Learning rate')
    plt.ylabel('log(loss)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output,"lroptimization.png"))
    plt.close()
    
    initLearningRate = lrScanValues[bestLearningRateIdx]
    
    #reload initial random weights
    model.set_weights(initWeights)

for epoch in range(args.epochs+1):
    print ("="*80)
    tstart = datetime.now()
    lr = learningRateWithDecay(initLearningRate,0)
    tf.keras.backend.set_value(
        model.optimizer.learning_rate, 
        lr
    )
    
    if epoch>0:
        model.load_weights("weights_%i.hdf5"%(epoch-1))

    lossTrain = 0.
    accCatTrain = tf.keras.metrics.CategoricalAccuracy()
    accBinTrain = tf.keras.metrics.BinaryAccuracy()
    
    for stepTrain,batch in enumerate(dsTrain):
    
        inputsList = [
            batch['cpf'],batch['cpf_charge'],
            batch['muon'],batch['muon_charge'],
            batch['electron'],batch['electron_charge'],
            batch['npf'],
            batch['sv'],
            batch['global']
        ]
        truth = tf.keras.utils.to_categorical(
            batch['xcharge'], num_classes=4, dtype='float32'
        ) 
        sample_weight = np.sum(truth*truth.shape[0]/truth.shape[1]/(1.+np.sum(truth,axis=0)),axis=1)
        
        loss = model.train_on_batch(inputsList,[truth],sample_weight=sample_weight)
        pred = tf.nn.softmax(model.predict_on_batch(inputsList),axis=1).numpy()
        
        accCatTrain.update_state(truth,pred,sample_weight=sample_weight)
        accBinTrain.update_state(
            truth[:,2:3]+truth[:,3:4],
            pred[:,2:3]+pred[:,3:4],
            sample_weight=sample_weight
        )
        lossTrain+=loss
        if stepTrain%10==0:
            print ("Train step %03i-%04i: loss=%10.4e, acc=%5.2f%% (%5.2f%%)"%(
                epoch,stepTrain,loss,100.*accCatTrain.result().numpy(),100.*accBinTrain.result().numpy()
            ))

    
    model.save_weights("weights_%i.hdf5"%(epoch))
    
    lossTest = 0.
    accCatTest = tf.keras.metrics.CategoricalAccuracy()
    accBinTest = tf.keras.metrics.BinaryAccuracy()
    
    testLabels = []
    testScores = []
    refScores = []
    
    print ("-"*80)
    for stepTest,batch in enumerate(dsTest):
        inputsList = [
            batch['cpf'],batch['cpf_charge'],
            batch['muon'],batch['muon_charge'],
            batch['electron'],batch['electron_charge'],
            batch['npf'],
            batch['sv'],
            batch['global']
        ]
        truth = tf.keras.utils.to_categorical(
            batch['xcharge'], num_classes=4, dtype='float32'
        ) 
        sample_weight = np.sum(truth*truth.shape[0]/truth.shape[1]/(1.+np.sum(truth,axis=0)),axis=1)
        
        loss = model.test_on_batch(inputsList,[truth],sample_weight=sample_weight)
        pred = tf.nn.softmax(model.predict_on_batch(inputsList),axis=1).numpy()
        
        testLabels.append(truth)
        testScores.append(pred)
        
        weight = np.power(batch['cpf'][:,:,0],0.6)
        weightedChargeSum = np.sum(weight*batch['cpf_charge'][:,:,0],axis=1)/(1e-6+np.sum(weight,axis=1))
        refScores.append(weightedChargeSum)
        
        accCatTest.update_state(truth,pred,sample_weight=sample_weight)
        accBinTest.update_state(
            truth[:,2:3]+truth[:,3:4],
            pred[:,2:3]+pred[:,3:4],
            #sample_weight=sample_weight
        )
        lossTest+=loss
        if stepTest%10==0:
            print ("Test step %03i-%04i: loss=%10.4e, acc=%5.2f%% (%5.2f%%)"%(
                epoch,stepTest,loss,100.*accCatTest.result().numpy(),100.*accBinTest.result().numpy()
            ))
            
    lossTrain /= stepTrain
    lossTest /= stepTest
    
    tend = datetime.now()
    print ("-"*80)
    print ("Epoch duration %03i: "%(epoch),tend-tstart)
    
    with open(os.path.join(args.output,"summary.dat"),'w' if epoch==0 else 'a',newline='') as f:
        writer = csv.DictWriter(f,["epoch","lr","lossTrain","lossTest","accCatTrain","accCatTest","accBinTrain","accBinTest"])
        if epoch==0:
            writer.writeheader()
        writer.writerow({
            "epoch":epoch,
            "lr":lr,
            "lossTrain":lossTrain,
            "lossTest":lossTest,
            "accCatTrain":accCatTrain.result().numpy(),
            "accCatTest":accCatTest.result().numpy(),
            "accBinTrain":accBinTrain.result().numpy(),
            "accBinTest":accBinTest.result().numpy()   
        })

    if epoch%2==0:
        testLabels = np.concatenate(testLabels,axis=0)
        testScores = np.concatenate(testScores,axis=0)
        refScores = np.concatenate(refScores,axis=0)
        
        
        fig = plt.figure(figsize=[6.4, 5.8],dpi=300)
        testScores = testScores[:,2]+testScores[:,3]
        colors = ['#17b2eb','#0971e8','#d90202','#e69a17']
        labels = ['$B^{-}$','$\\overline{B}^{0}$','$B^{0}$','$B^{+}$']
        for ihist in range(4):
            plt.hist(
                testScores[testLabels[:,ihist]>0.5], 
                bins=50, range=(0,1), density=True,
                alpha=0.25,color=colors[ihist],label=labels[ihist]
            )
        for ihist in range(4):
            plt.hist(
                testScores[testLabels[:,ihist]>0.5], 
                histtype='step',
                bins=50, range=(0,1), density=True,
                alpha=0.5,color=colors[ihist],
                linewidth=1
            ) 
        plt.xlabel('Score')
        plt.ylabel('$\\langle$#jets$\\rangle$')
        plt.gca().yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        plt.gca().xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        plt.grid(True,which='both',axis='both', linestyle='--',color='#b5b5b5')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output,"score_%i.png"%(epoch)))
        plt.close()
        
        '''
        fig = plt.figure(figsize=[6.4, 5.8],dpi=300)
        
        selectBpm = (testLabels[:,0]+testLabels[:,3])>0
        selectB0 = (testLabels[:,1]+testLabels[:,2])>0
        for name,label,score in [
            ['$\\overline{b}$ vs. $b$ (charge sum)',1.*((testLabels[:,2]+testLabels[:,3])>0),refScores],
            ['$\\overline{b}$ vs. $b$ (tagger)',1.*((testLabels[:,2]+testLabels[:,3])>0),testScores[:,2]+testScores[:,3]],
            ['$B^{-}$ vs. $B^{+}$',1.*((testLabels[selectBpm][:,3])>0),testScores[selectBpm][:,2]+testScores[selectBpm][:,3]],
            ['$\\overline{B}^{0}$ vs. $B^{0}$',1.*((testLabels[selectB0][:,2])>0),testScores[selectB0][:,2]+testScores[selectB0][:,3]],
        ]:
            fpr,tpr,thres = sklearn.metrics.roc_curve(
                label, 
                score,
                pos_label = 1
            )
            auc = sklearn.metrics.auc(fpr,tpr)
            plt.plot(tpr,1.-fpr,label=name)
        
        plt.xlabel('Signal efficiency')
        plt.ylabel('1 - Background efficiency')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.gca().yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        plt.gca().xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        plt.plot([0, 1], [1, 0], linewidth=1, linestyle='--',color='black')
        plt.grid(True,which='both',axis='both', linestyle='--',color='#b5b5b5')
        plt.legend()
        plt.tight_layout()
        plt.savefig("roc_%i.png"%(epoch))
        plt.close()
        '''
    
    

