from __future__ import absolute_import, division, print_function, unicode_literals
import librosa

from tqdm import tqdm
from tensorflow.keras.callbacks import Callback
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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
batchsize = 300
epochs = 100
sample_rate = 16000
window_width = 16000 #this is size of window to put into autoencoder.
LAYER_SIZE = 15 # amount of neurons in each hidden layer
HIDDEN_LAYER_COUNT = 8 #amount of hiddnot en layers
logs_path = "results/log"
model_path = "results/model"
if not os.path.exists(logs_path):
    os.mkdir(logs_path)
if not os.path.exists(model_path):
    os.mkdir(model_path)
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
data = aux_fn.get_data_from_files(files, sr=sample_rate, ww=window_width)
# data = np.expand_dims(data, axis=2)
# data = data[0:batchsize, 0:sample_rate]


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    train_writer = tf.summary.FileWriter(logs_path, sess.graph)

    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        minibatch = 0
        for i in range( int((data.shape[0]) / batchsize)):
            start = minibatch
            end = minibatch + batchsize
            batch_data = np.array(data[start:end, 0:window_width])
            (loss_val, optimizer) = sess.run([model.loss, model.train], {model.x_placeholder: batch_data})
            print('iter %d:' % (minibatch + 1), loss_val)
            epoch_loss += loss_val
            minibatch += batchsize
            tf.summary.scalar("mini_batch_loss", loss_val)



        batch_data = np.array(data[start:end, 0:window_width])
        print('Epoch ', epoch, 'Completed out of', epochs, "loss: ", epoch_loss)
        (input_, output_) = sess.run([model.input_, model.output_], {model.x_placeholder: batch_data})
        print('train result :')
        print('input :', input_[0, :, :].flatten())
        print('output :', output_[0, :, :].flatten())
        #tf.summary.scalar("epoch_loss", epoch_loss)
        #tf.summary.scalar("avg_epoch_loss", epoch_loss / int((data.shape[0]) / batchsize) )
        #merged = tf.summary.merge_all()
        #train_writer.add_summary(merged, epoch)

    # print('input :', input_[0, :, :].flatten())
    # print('output :', output_[0, :, :].flatten())
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    tf.io.write_graph(sess.graph_def, './results', 'network_train.pbtxt')
    output_node_names = "decoder/dense_output"
    output_graph = os.path.join(model_path, "output_tfgraph.pb")
    output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names.split(",")
                                                                 )
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    graph_io.write_graph(output_graph_def, model_path, 'output_tfgraph.pbtxt', as_text=True)


