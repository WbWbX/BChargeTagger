import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np


#tf.compat.v1.disable_eager_execution()

from Network import Network

network = Network()
model = network.makeModel()
model.load_weights("weights_60.hdf5")

modelFct = tf.function(
    func=lambda x: tf.identity(model(x),name='prediction'),
    input_signature=[[
        tf.TensorSpec(x.shape, dtype=x.dtype, name=x.name.split(':')[0].replace('_input','')) for x in model.inputs
    ]]
)


modelFct = modelFct.get_concrete_function()

frozenFct = convert_variables_to_constants_v2(modelFct)
#frozenFct.graph.as_graph_def().remove_attribute('batch_dims')
#frozenFct.graph.as_graph_def()

     
print("Frozen model inputs: ")
print(frozenFct.inputs)
print("Frozen model outputs: ")
print(frozenFct.outputs)

tf.io.write_graph(
    graph_or_graph_def=frozenFct.graph,
    logdir='',
    name="frozenModel.pb",
    as_text=False
)

