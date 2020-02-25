# TF_LSTM_Autoencoder

#Convert Keras model to tensorflow frozen graph which is OpenVINO convertable 
 tested for tensorflow 1.14.0
cmd : python keras_to_tf.py --input_model ./trained_model/saved_model.h5 --num_outputs 1
output : /trained_model/keras_output_graph.pb
output : /trained_model/keras_output_graph.pbtxt

# Tensorflow frozen graph to OpenVINO optimized graph 
python3  /opt/intel/openvino/deployment_tools/model_optimizer/mo.py  --input_model ./trained_model/keras_output_graph.pb --input_shape [1,16000,1] --output dense_5/BiasAdd


# Train Tensorflow API Based Autoencoder which can be converted into OpenVINO
  Training works for both 1.14.0 and 2.1.0 but OpenVINO conversion only works for  tensorflow version 1.14.
 use train_tf_v2.py for training.  

