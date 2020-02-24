from __future__ import absolute_import, division, print_function, unicode_literals
import librosa

from tqdm import tqdm
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import numpy as np
import os
from tf_enc_dec import LSTMAutoencoder
import aux_fn
import logging
import time

from tensorflow.contrib import cudnn_rnn
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

#parameters
batchsize = 2
epoch = 100
sample_rate=6
window_width=sample_rate #this is size of window to put into autoencoder.
LAYER_SIZE = 4 # amount of neurons in each hidden layer
HIDDEN_LAYER_COUNT = 4 #amount of hidden layers

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(filename="results/training.log",filemode='w', level=logging.INFO)

path = 'test_data/audio'

class LogHistory(Callback):
    def on_epoch_end(self, batch, logs={}):
        logging.info(logs)

model = LSTMAutoencoder(batchsize, window_width, LAYER_SIZE, HIDDEN_LAYER_COUNT)

files = aux_fn.get_all_files(path)
print(['Number of files',len(files)])

# data = aux_fn.get_data_from_files(files, sr=sample_rate, ww=window_width)
data = aux_fn.get_data_from_files(files, sr=16000, ww=16000)
# data = np.expand_dims(data, axis=2)
data = data[0:batchsize, 0:sample_rate]


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    sess.run(model.initial_state_enc)
    start_time = time.time()
    for i in range(epoch):
        (loss_val, _) = sess.run([model.loss, model.train], {model.x_placeholder: data})
        print('iter %d:' % (i + 1), loss_val)

    (input_, output_) = sess.run([model.input_, model.output_], {model.x_placeholder: data})
    print('train result :')
    # print('input :', input_[0, :, :].flatten())
    # print('output :', output_[0, :, :].flatten())
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    print('input :', input_[0,:, :].flatten())
    print('output :', output_[0, :, :].flatten())
    tf.io.write_graph(sess.graph_def, './results', 'network_train.pbtxt')
    output_node_names = "decoder/dense_output"
    output_graph = "./result/outputnf.pb"
    output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names.split(",")
                                                                 )
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    graph_io.write_graph(output_graph_def, './results', 'network_frezze.pbtxt', as_text=True)


