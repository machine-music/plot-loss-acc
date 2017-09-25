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
    global MODEL_DIRECTORY
    global HISTORY_FILE
    global OUTPUT_FILE
    try:
        opts, args = getopt.getopt(argv, 'm:h:o:', ['modelFile=','historyFile=',
            'outputFile='])
        for opt, arg in opts:
            if opt in ('-m', '--modelFile'):
                MODEL_FILE = arg
            elif opt in ('-h', '--historyFile'):
                HISTORY_FILE = arg
            elif opt in ('-o', '--outputFile'):
                OUTPUT_FILE = arg
    except getopt.GetoptError as e:
        print(e)

    if(not HISTORY_FILE):
        print("History file needed.")
        print("EX python plot.py -h <HISTORY_FILE>.npy")
        sys.exit(0)

    if(not OUTPUT_FILE):
        OUTPUT_FILE = os.path.splitext(os.path.basename(HISTORY_FILE)) + ".png"

    if(MODEL_DIRECTORY):
        model_name = os.path.basename(MODEL_DIRECTORY)
    elif(OUTPUT_FILE):
        model_name, _ = os.path.splitext(os.path.basename(OUTPUT_FILE))
    else:
        model_name, _ = os.path.splitext(os.path.basename(HISTORY_FILE))

    history = np.load(HISTORY_FILE).item()

    plt.figure(1)
    plt.subplot(211)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title(model_name + ' model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(model_name + ' model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplots_adjust(hspace=.5)

    plt.savefig(OUTPUT_FILE)


if __name__ == "__main__":
    main(sys.argv[1:])
