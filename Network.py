import tensorflow as tf
import sys

from featureDict import featureDict

class ResDenseLayer():
    def __init__(self,prefix,nodeSize,depth=3):
        self.inputDropout = tf.keras.layers.Dropout(0.5,noise_shape=[1,nodeSize],name=prefix+"_input_dropout")
        self.layerList = []
        for i in range(depth):
            self.layerList.extend([
                tf.keras.layers.Dense(
                    nodeSize,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=tf.keras.regularizers.l2(1e-8),
                    activation=None,
                    name=prefix+"_dense_"+str(i+1)
                ),
                tf.keras.layers.Leakyselu(alpha=0.1,name=prefix+"_activation_"+str(i+1)),
                tf.keras.layers.Dropout(0.1,name=prefix+"_dropout_"+str(i+1))
            ])
            
    def __call__(self,inputs):
        result = inputs
        for layer in self.layerList:
            result = layer(result)
        
        drop = self.inputDropout(inputs)
        result = tf.keras.layers.Lambda(lambda x: x[0]+x[1])([drop,result])
        return result
        
class ResConv1DLayer():
    def __init__(self,prefix,nodeSize,depth=3):
        self.inputDropout = tf.keras.layers.Dropout(0.5,noise_shape=[1,1,nodeSize],name=prefix+"_input_dropout")
        self.layerList = []
        for i in range(depth):
            self.layerList.extend([
                tf.keras.layers.Conv1D(
                    filters=nodeSize,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    kernel_initializer="lecun_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(1e-8),
                    bias_initializer="zeros",
                    activation = None,
                    name=prefix+"_conv_"+str(i+1)
                ),
                tf.keras.layers.Leakyselu(alpha=0.1,name=prefix+"_activation_"+str(i+1)),
                tf.keras.layers.Dropout(0.1,name=prefix+"_dropout_"+str(i+1))
            ])
            
    def __call__(self,inputs):
        result = inputs
        for layer in self.layerList:
            result = layer(result)
        
        drop = self.inputDropout(inputs)
        result = tf.keras.layers.Lambda(lambda x: x[0]+x[1])([drop,result])
        return result
        
class Network():
    def __init__(self):
        self.globalInput = tf.keras.layers.Input(shape=(
            len(featureDict['global']['branches']
        ),),name='global_input')
    
        self.cpfInput = tf.keras.layers.Input(shape=(
            featureDict['cpf']['max'],
            len(featureDict['cpf']['branches']
        )),name='cpf_input')
        self.npfInput = tf.keras.layers.Input(shape=(
            featureDict['npf']['max'],
            len(featureDict['npf']['branches']
        )),name='npf_input')
        self.svInput = tf.keras.layers.Input(shape=(
            featureDict['sv']['max'],
            len(featureDict['sv']['branches']
        )),name='sv_input')
        self.muonInput = tf.keras.layers.Input(shape=(
            featureDict['muon']['max'],
            len(featureDict['muon']['branches']
        )),name='muon_input')
        self.electronInput = tf.keras.layers.Input(shape=(
            featureDict['electron']['max'],
            len(featureDict['electron']['branches']
        )),name='electron_input')
        
        self.cpfChargeInput = tf.keras.layers.Input(shape=(
            featureDict['cpf_charge']['max'],
            len(featureDict['cpf_charge']['branches']
        )),name='cpf_charge_input')
        self.muonChargeInput = tf.keras.layers.Input(shape=(
            featureDict['muon_charge']['max'],
            len(featureDict['muon_charge']['branches']
        )),name='muon_charge_input')
        self.electronChargeInput = tf.keras.layers.Input(shape=(
            featureDict['electron_charge']['max'],
            len(featureDict['electron_charge']['branches']
        )),name='electron_charge_input')
        
        
        self.globalPreprocLayer = \
            tf.keras.layers.Lambda(self.preprocessingFct(
                featureDict["global"]["branches"],
                featureDict["global"]["preprocessing"]
            ),name='global_preproc')
            
        self.cpfPreprocLayer = \
            tf.keras.layers.Lambda(self.preprocessingFct(
                featureDict["cpf"]["branches"],
                featureDict["cpf"]["preprocessing"]
            ),name='cpf_preproc')
            
        self.npfPreprocLayer = \
            tf.keras.layers.Lambda(self.preprocessingFct(
                featureDict["npf"]["branches"],
                featureDict["npf"]["preprocessing"]
            ),name='npf_preproc')
            
        self.svPreprocLayer = \
            tf.keras.layers.Lambda(self.preprocessingFct(
                featureDict["sv"]["branches"],
                featureDict["sv"]["preprocessing"]
            ),name='sv_preproc')
            
        self.muonPreprocLayer = \
            tf.keras.layers.Lambda(self.preprocessingFct(
                featureDict["muon"]["branches"],
                featureDict["muon"]["preprocessing"]
            ),name='muon_preproc')
            
        self.electronPreprocLayer = \
            tf.keras.layers.Lambda(self.preprocessingFct(
                featureDict["electron"]["branches"],
                featureDict["electron"]["preprocessing"]
            ),name='electron_preproc')
        
        self.cpfLayers = []
        for i,nodes in enumerate([64,64,64]):
            self.cpfLayers.extend([
                tf.keras.layers.Dense(
                    nodes,
                    kernel_initializer="lecun_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(1e-8),
                    bias_initializer="zeros",
                    activation = 'selu',
                    name='cpf_conv_%i'%(i+1)
                ),
                tf.keras.layers.Dropout(0.1,name='cpf_dropout_%i'%(i+1)),
                #ResConv1DLayer('cpf_res_%i'%(i+1),filters,depth=2)
            ])
        self.cpfLayers.extend([
            tf.keras.layers.Dense(
                16,
                kernel_initializer="lecun_normal",
                use_bias=False,
                activation = 'sigmoid',
                name='cpf_conv_final'
            )
        ])
            
        self.muonLayers = []
        for i,nodes in enumerate([64,64,64]):
            self.muonLayers.extend([
                tf.keras.layers.Dense(
                    nodes,
                    kernel_initializer="lecun_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(1e-8),
                    bias_initializer="zeros",
                    activation = 'selu',
                    name='muon_conv_%i'%(i+1)
                ),
                tf.keras.layers.Dropout(0.1,name='muon_dropout_%i'%(i+1)),
                #ResConv1DLayer('muon_res_%i'%(i+1),filters,depth=2)
            ])
        self.muonLayers.extend([
            tf.keras.layers.Dense(
                16,
                kernel_initializer="lecun_normal",
                use_bias=False,
                activation = 'sigmoid',
                name='muon_conv_final'
            )
        ])
            
        self.electronLayers = []
        for i,nodes in enumerate([64,64,64]):
            self.electronLayers.extend([
                tf.keras.layers.Dense(
                    nodes,
                    kernel_initializer="lecun_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(1e-8),
                    bias_initializer="zeros",
                    activation = 'selu',
                    name='electron_conv_%i'%(i+1)
                ),
                tf.keras.layers.Dropout(0.1,name='electron_dropout_%i'%(i+1)),
                #ResConv1DLayer('electron_res_%i'%(i+1),filters,depth=2)
            ])
        self.electronLayers.extend([
            tf.keras.layers.Dense(
                16,
                kernel_initializer="lecun_normal",
                use_bias=False,
                activation = 'sigmoid',
                name='electron_conv_final'
            )
        ])
            
        self.npfLayers = []
        for i,nodes in enumerate([32,32,32,8]):
            self.npfLayers.extend([
                tf.keras.layers.Dense(
                    nodes,
                    kernel_initializer="lecun_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(1e-8),
                    bias_initializer="zeros",
                    activation = 'selu',
                    name='npf_conv_%i'%(i+1)
                ),
                tf.keras.layers.Dropout(0.1,name='npf_dropout_%i'%(i+1))
                
            ])
            
        self.svLayers = []
        for i,nodes in enumerate([64,64,64,16]):
            self.svLayers.extend([
                tf.keras.layers.Dense(
                    nodes,
                    kernel_initializer="lecun_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(1e-8),
                    bias_initializer="zeros",
                    activation = 'selu',
                    name='sv_conv_%i'%(i+1)
                ),
                tf.keras.layers.Dropout(0.1,name='sv_dropout_%i'%(i+1))
            ])
            
        
            
        self.featureLayers = []
        for i,nodes in enumerate([256,256,256,128]):
            self.featureLayers.extend([
                tf.keras.layers.Dense(
                    nodes,
                    kernel_initializer="lecun_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(1e-8),
                    bias_initializer="zeros",
                    activation = 'selu',
                    name='feature_dense_%i'%(i+1)
                ),
                tf.keras.layers.Dropout(0.1,name='feature_dropout_%i'%(i+1)),
            ])
        #self.featureLayers.extend([
        #    ResDenseLayer('feature_res',128,depth=2)
        #])
        self.featureLayers.extend([
            tf.keras.layers.Dense(
                4,
                kernel_initializer="lecun_normal",
                kernel_regularizer=tf.keras.regularizers.l2(1e-8),
                bias_initializer="zeros",
                activation = None,
                name='final_dense'
            ),
            #note: no activation function to return logits
        ])
        
        self.sumLayer = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x,axis=1),name='sum')
        self.weightedSumLayer = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x[0]*x[1],axis=1),name='weighted_sum')
        
        self.sliceFirstLayer = tf.keras.layers.Lambda(lambda x: x[:,0,:],name='slice_first')
        
        self.concatFeatureLayer = tf.keras.layers.Concatenate(axis=1,name='concat_features')
        self.flattenLayer = tf.keras.layers.Flatten(name='flatten_features')
        
    def preprocessingFct(self,featureNames,preprocDict):
        def applyPreproc(inputFeatures):
            #BYPASS
            #return inputFeatures

            unstackFeatures = tf.unstack(inputFeatures,axis=-1)
            if len(unstackFeatures)!=len(featureNames):
                print ("Number of features ("+str(len(unstackFeatures))+") does not match given list of names ("+str(len(featureNames))+"): "+str(featureNames))
                sys.exit(1)
            unusedPreproc = list(preprocDict.keys())
            if len(unusedPreproc)==0:
                return inputFeatures
            for i,featureName in enumerate(featureNames):
                if featureName in unusedPreproc:
                    unusedPreproc.remove(featureName)
                if featureName in preprocDict.keys():
                    unstackFeatures[i] = preprocDict[featureName](unstackFeatures[i])

            if len(unusedPreproc)>0:
                print ("Following preprocessing not applied: "+str(unusedPreproc))
            return tf.stack(unstackFeatures,axis=-1)
        return applyPreproc
        
        
    def getPrediction(
        self,
        cpfInput,cpfChargeInput,
        muonInput,muonChargeInput,
        electronInput,electronChargeInput,
        npfInput,
        svInput,
        globalInput
    ):
        cpfPreproc = self.cpfPreprocLayer(cpfInput)
        cpf = cpfPreproc
        for layer in self.cpfLayers:
            cpf = layer(cpf)
        cpf = self.weightedSumLayer([cpf,cpfChargeInput])
        
        muonPreproc = self.muonPreprocLayer(muonInput)
        muon = muonPreproc
        for layer in self.muonLayers:
            muon = layer(muon)
        muon = self.weightedSumLayer([muon,muonChargeInput])
            
        electronPreproc = self.electronPreprocLayer(electronInput)
        electron = electronPreproc
        for layer in self.electronLayers:
            electron = layer(electron)
        electron = self.weightedSumLayer([electron,electronChargeInput])
        
        npfPreproc = self.npfPreprocLayer(npfInput)
        npf = npfPreproc
        for layer in self.npfLayers:
            npf = layer(npf)
        npf = self.sumLayer(npf)
        
        svPreproc = self.svPreprocLayer(svInput)
        sv = svPreproc
        for layer in self.svLayers:
            sv = layer(sv)
        sv = self.sumLayer(sv)
            
            
        globalPreproc = self.globalPreprocLayer(globalInput)
        
        full_features = self.concatFeatureLayer([
            self.sliceFirstLayer(cpfPreproc),
            cpf,npf,sv,muon,electron,
            globalPreproc
        ])
        
        prediction = full_features
        for layer in self.featureLayers:
            prediction = layer(prediction)
        return prediction
        
    def makeModel(self):
        prediction = self.getPrediction(
            self.cpfInput,self.cpfChargeInput,
            self.muonInput,self.muonChargeInput,
            self.electronInput,self.electronChargeInput,
            self.npfInput,
            self.svInput,
            self.globalInput
        )
        predictionInv = self.getPrediction(
            self.cpfInput,-1.*self.cpfChargeInput,
            self.muonInput,-1.*self.muonChargeInput,
            self.electronInput,-1.*self.electronChargeInput,
            self.npfInput,
            self.svInput,
            self.globalInput
        )
        model = tf.keras.Model(
            inputs=[
                self.cpfInput,self.cpfChargeInput,
                self.muonInput,self.muonChargeInput,
                self.electronInput,self.electronChargeInput,
                self.npfInput,
                self.svInput,
                self.globalInput
            ],
            outputs = [prediction-predictionInv]
        )
        return model
        
        
