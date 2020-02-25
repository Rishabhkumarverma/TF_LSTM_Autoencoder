from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import yaml
import logging
from tqdm import tqdm
import numpy as np
import aux_fn
import time
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IECore

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--testparam", help="Required. Path to an test parameter  .yaml file .",
                      required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                           "kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)


    return parser
def load_data(window_width, file, sample_rate, rolling_step=None):
    if rolling_step is None:
        rolling_step = window_width // 2
    data = aux_fn.get_data_from_files(
        [file], sample_rate, window_width, rolling_step)
    print(data.shape)
    return data

def get_response(data,yhat,file_meta, sample_rate, rolling_step=None):
    yhat = np.squeeze(yhat)
    data = np.squeeze(data)
    response = aux_fn.unroll(yhat, rolling_step)
    raw_data = aux_fn.unroll(data, rolling_step)

    anomaly_bounds = None
    if file_meta['event_present']:
        ev_start = round(
            file_meta['event_start_in_mixture_seconds'] * sample_rate)
        ev_end = round(
            ev_start + file_meta['event_length_seconds'] * sample_rate)
        anomaly_bounds = (min(ev_start, len(raw_data)),
                          min(ev_end, len(raw_data)))

    return {'data': raw_data, 'resp': response, 'actual': anomaly_bounds}

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    with open(args.testparam, 'r') as stream:
        test_param = yaml.load(stream, Loader=yaml.FullLoader)
    sample_rate = test_param['sample_rate']
    rolling_step = test_param['rolling_step']  # this is rolling step

    test_folder = test_param['test_path'] + "/audio"
    yaml_file = test_param['test_path'] + \
                "/meta/mixture_recipes_devtest_gunshot.yaml"

    print("Testing folder to proceed:")
    print(test_folder)
    print(model_bin)
    log.info("Creating Inference Engine...")
    ie = IECore()

    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
        # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)


    info_input_blob = None
    input_shape_dims = None
    feed_dict = {}
    for blob_name in net.inputs:
        print(len(net.inputs[blob_name].shape))
        if len(net.inputs[blob_name].shape) == 3:
            input_blob = blob_name
            input_shape_dims = 3
        elif len(net.inputs[blob_name].shape) == 2:
            info_input_blob = blob_name
            input_shape_dims = 2

        else:
            raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                               .format(len(net.inputs[blob_name].shape), blob_name))
    print(len(net.outputs))

    assert len(net.outputs) == 1, "Demo supports only single output topologies"
    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
    exec_net = ie.load_network(network=net, num_requests=1, device_name=args.device)

    # Read and pre-process input image
    if input_shape_dims == 3:
        batch_size, window_width, num_class = net.inputs[input_blob].shape
    elif input_shape_dims == 2 :
        batch_size, window_width = net.inputs[input_blob].shape
    else:
        print("check input shape dimension: ")
        return
    print(net.inputs[input_blob].shape)
    files = aux_fn.get_all_files(test_folder)
    sig_process = {}
    file_meta = None
    with open(yaml_file, 'r') as infile:
        metadata = yaml.load(infile, Loader=yaml.SafeLoader)
    for file in tqdm(files):
        file_time = time()
        print(file)
        logging.info(f' inference on {file}: ')

        filename_wav = os.path.basename(file)
        for i in range(len(metadata)):
            if filename_wav in metadata[i]['mixture_audio_filename']:
                file_meta = metadata[i]
                break

        data = load_data(window_width, file, sample_rate, rolling_step)
        if input_shape_dims == 3:
            data = np.expand_dims(data, axis=2)

        iteration = data.shape[0]

        pred = []

        for iterate in range(iteration):
            print(input_blob)
            if input_shape_dims == 3:
                feed_dict[input_blob] = data[iterate, :, :]
            else:
                feed_dict[input_blob] = data[iterate, :]
            # inf_start = time.time()
            exec_net.start_async(request_id=0, inputs=feed_dict)
            if exec_net.requests[0].wait(-1) == 0:
                # inf_end = time.time()
                # det_time = inf_end - inf_start
                res = exec_net.requests[0].outputs[out_blob]
                print("result shape", res.shape)
                pred.append(list(res))

        sig_process[filename_wav] = get_response(data, pred, file_meta, sample_rate, rolling_step)
        tmp_time = time() - file_time
        logging.info(
            f" \tfile inference time = {tmp_time:.2f} s, {(tmp_time / 117) * 1000:.4f} ms per sample")

main()
