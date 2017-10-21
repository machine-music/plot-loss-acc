#!/usr/bin/python

import numpy as np
import os
import sys
import getopt
import matplotlib.pyplot as plt

MODEL_DIRECTORY = ''
HISTORY_FILE = ''
OUTPUT_FILE = ''

def main(argv):
    hist_file, model_name, out_file = parse_options(argv)
    history = np.load(hist_file).item()

    f, axarr = plot_model(history, model_name)
    f.savefig(out_file)
    print("Created figure at ", out_file)

def plot_model(history, model_name):
    # summarize history for accuracy
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(history['acc'])
    axarr[0].plot(history['val_acc'])
    axarr[0].set_title(model_name + ' model accuracy')
    axarr[0].set_ylabel('accuracy')
    axarr[0].set_xlabel('epoch')
    axarr[0].legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    axarr[1].plot(history['loss'])
    axarr[1].plot(history['val_loss'])
    axarr[1].set_title(model_name + ' model loss')
    axarr[1].set_ylabel('loss')
    axarr[1].set_xlabel('epoch')
    axarr[1].legend(['train', 'test'], loc='upper left')

    f.subplots_adjust(hspace=.5)

    return f, axarr

def parse_options(argv):
    global MODEL_DIRECTORY
    global HISTORY_FILE
    global OUTPUT_FILE
    try:
        opts, args = getopt.getopt(argv, 'm:h:o:', ['modelFile=','historyFile=',
            'outputFile='])
        for opt, arg in opts:
            if opt in ('-m', '--modelFile'):
                MODEL_DIRECTORY = arg
            elif opt in ('-h', '--historyFile'):
                HISTORY_FILE = arg
            elif opt in ('-o', '--outputFile'):
                OUTPUT_FILE = arg
    except getopt.GetoptError as e:
        print(e)

    if not (HISTORY_FILE or MODEL_DIRECTORY):
        print("History file or model directory needed.")
        print("EX python plot.py -h <HISTORY_FILE>.npy")
        print("OR python plot.py -m <MODEL_DIRECTORY>")
        sys.exit(0)

    if(MODEL_DIRECTORY):
        hist_file = os.path.join(MODEL_DIRECTORY, 'model_history.npy')
        model_name = os.path.basename(MODEL_DIRECTORY)
    elif(HISTORY_FILE):
        hist_file = HISTORY_FILE
        model_name, _ = os.path.splitext(os.path.basename(HISTORY_FILE))

    if(OUTPUT_FILE):
        out_file = OUTPUT_FILE
    else:
        out_file = os.path.splitext(os.path.basename(hist_file))[0] + ".png"
        if(MODEL_DIRECTORY):
            out_file = os.path.join(MODEL_DIRECTORY, out_file)

    return hist_file, model_name, out_file

if __name__ == "__main__":
    main(sys.argv[1:])
