import argparse
import os
import sys
import numpy as np
import math


import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from featureDict import featureDict
import uproot3 as uproot




originFeatures = [
    "jetorigin_bHadronCharge",
    "jetorigin_bPartonCharge",
    "jetorigin_cHadronCharge",
    
    "jetorigin_bHadronDecay_DiHadronic",
    "jetorigin_bHadronDecay_Electron",
    "jetorigin_bHadronDecay_Muon",
    "jetorigin_bHadronDecay_OtherHadronic",
    "jetorigin_bHadronDecay_SingleHadronic",
    "jetorigin_bHadronDecay_Tau",
    "jetorigin_bHadronDecay_Undefined",
    "jetorigin_bHadronId",
    
    "jetorigin_cHadronDecay_DiHadronic",
    "jetorigin_cHadronDecay_Electron",
    "jetorigin_cHadronDecay_Muon",
    "jetorigin_cHadronDecay_OtherHadronic",
    "jetorigin_cHadronDecay_SingleHadronic",
    "jetorigin_cHadronDecay_Tau",
    "jetorigin_cHadronDecay_Undefined",
    "jetorigin_cHadronId",
    
    "jetorigin_hadronFlavor",
    "jetorigin_partonFlavor",
    
    "jetorigin_nBHadrons",
    "jetorigin_nCHadrons",
    
    "jetorigin_matchedBHadronDeltaR",
    "jetorigin_matchedBHadronPt",
    
    "jetorigin_matchedCHadronDeltaR",
    "jetorigin_matchedCHadronPt",
    
    "jetorigin_matchedGenJetDeltaR",
    "jetorigin_matchedGenJetPt",
]

allBranches = originFeatures
for k in featureDict.keys():
    allBranches+=featureDict[k]['branches']
    if 'max' in featureDict[k].keys():
        allBranches+=[featureDict[k]['length'],featureDict[k]['offset']]
allBranches = list(set(allBranches)) #remove duplicates

parser = argparse.ArgumentParser()
parser.add_argument('--input', dest='input',required=True)
parser.add_argument('--split', dest='split',type=int,default=1)
parser.add_argument('output', nargs=1)
args = parser.parse_args()

#f = uproot.open("root://gfe02.grid.hep.ph.ic.ac.uk/pnfs/hep.ph.ic.ac.uk/data/cms/store/user/mkomm/ST/NANOX_210113/TTToHadronic_TuneCP5_PSweights_13TeV-powheg-pythia8-2016/TTToHadronic_TuneCP5_PSweights_13TeV-powheg-pythia8/TTToHadronic_TuneCP5_PSweights_13TeV-powheg-pythia8-2016/210113_210415/0000/nano_102.root")
f = uproot.open(args.input)

def makeTFWriter(outputName,currentSplit=0,totalSplit=1):
    if totalSplit>1:
        outputName = outputName.rsplit(".",1)[0]+"_"+str(currentSplit)+"."+outputName.rsplit(".",1)[1]
    tfwriter = tf.io.TFRecordWriter(
        outputName,
        options=tf.io.TFRecordOptions(
            compression_type='GZIP',
            compression_level = 4,
            input_buffer_size=100,
            output_buffer_size=100,
            mem_level = 8,
        )
    )
    return tfwriter

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))


def make1DArray(data,ievent,ijet,features):
    arr = np.zeros((len(features),),dtype=np.float32)
    for i,feature in enumerate(features):
        arr[i] = data[feature][ievent][ijet]
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr
       
def make2DArray(data,ievent,indices,nmax,features):
    arr = np.zeros((nmax,len(features)),dtype=np.float32)
    for i,feature in enumerate(features):
        for j,idx in enumerate(indices[:nmax]):
            arr[j,i] = data[feature][ievent][idx]
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr
    
chunkread = 1000
njets= 0
currentSplit = 0

tfwriter = makeTFWriter(args.output[0],currentSplit,args.split)

for ibatch,data in enumerate(f["Events"].iterate(allBranches,entrysteps=chunkread)):
    if ibatch*chunkread>int(1.*len(f["Events"])*(currentSplit+1)/args.split):
        currentSplit+=1
        tfwriter.close()
        tfwriter = makeTFWriter(args.output[0],currentSplit,args.split)
            
    print (ibatch*chunkread,'/',len(f["Events"]),njets)
    
    data = {k.decode('utf-8'):v for k,v in data.items()}
    for ievent in range(len(data['global_pt'])):
        for ijet in range(len(data['global_pt'][ievent])):
            if data['global_pt'][ievent][ijet]<25.:
                continue
            if math.fabs(data['global_eta'][ievent][ijet])>2.4:
                continue
            if abs(data['jetorigin_hadronFlavor'][ievent][ijet])<5:
                continue
            bPtRatio = data['global_pt'][ievent][ijet]/data['jetorigin_matchedBHadronPt'][ievent][ijet]
            if (bPtRatio<0.5):
                continue
                
            #originArr = fillScalar(data,ievent,ijet,originFeatures)

            bDecayFeatures = [
                "jetorigin_bHadronDecay_Electron",
                "jetorigin_bHadronDecay_Muon",
                "jetorigin_bHadronDecay_Tau",
                "jetorigin_bHadronDecay_SingleHadronic",
                "jetorigin_bHadronDecay_DiHadronic",
                "jetorigin_bHadronDecay_OtherHadronic",
                "jetorigin_bHadronDecay_Undefined"
            ]
            
            xcharge = 0*(data['jetorigin_bHadronCharge'][ievent][ijet]==-1)
            xcharge += 1*(data['jetorigin_bHadronCharge'][ievent][ijet]==0)*(data['jetorigin_bPartonCharge'][ievent][ijet]==-1)
            xcharge += 2*(data['jetorigin_bHadronCharge'][ievent][ijet]==0)*(data['jetorigin_bPartonCharge'][ievent][ijet]==1)
            xcharge += 3*(data['jetorigin_bHadronCharge'][ievent][ijet]==1)
            
            tfData = {
                'xcharge': _float_feature(np.array([xcharge],np.float32)), 
                "bPartonCharge": _float_feature(np.array([data['jetorigin_bPartonCharge'][ievent][ijet]],np.float32)), 
                "bHadronCharge": _float_feature(np.array([data['jetorigin_bHadronCharge'][ievent][ijet]],np.float32)), 
                "cHadronCharge": _float_feature(np.array([data['jetorigin_cHadronCharge'][ievent][ijet]],np.float32)), 
                "bDecay": _float_feature(make1DArray(data,ievent,ijet,bDecayFeatures))
            }
            
            for name,featureGroup in featureDict.items():
                if 'max' in featureGroup.keys():
                    length = data[featureGroup['length']][ievent][ijet]
                    offset = data[featureGroup['offset']][ievent][ijet]
                    tfData[name] = _float_feature(make2DArray(data,ievent,range(offset,offset+length),featureGroup['max'],featureGroup['branches']))
                else:
                    tfData[name] = _float_feature(make1DArray(data,ievent,ijet,featureGroup['branches']))

            #print (tfData.keys())
            example = tf.train.Example(features = tf.train.Features(feature = tfData))

            tfwriter.write(example.SerializeToString())
            njets+=1
    

tfwriter.close()
        
        
        
