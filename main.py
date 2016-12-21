#!/usr/bin/python -u
# -*- coding: utf-8 -*-

import tensorflow as tf
import time
import logging
import Architecture
from LanideNN import Parameters


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

params = Parameters.Parameters("PARAMS")

testingModel = False
continueModel = False


# UNCOMMENT MODEL - first must be downloaded from:
#testingModel = "models/WikiMulti.model"
# testingModel = "models/ALTW.model"
testingModel = "models/LanideNN.model"


if continueModel:
    params.continue_model(continueModel)
    logging.info("CONTINUING model {0}. Trained for {1} steps".format(continueModel, params.params["step"]))
    testingModel = params.params["identification"]
elif testingModel:
    params.load_params(testingModel)
    logging.info("Loading model " + testingModel)
else:
    params.add_integer("version", 3)  # version of a code
    params.add_string("corpus_name", "small")  # Folder within data/.
    params.add_bool("unicode_normalization", True)  # normalize unicode
    params.add_integer("size", 500)  # Size of each model layer.
    params.add_integer("embedding_size", 200)  # Size of each model layer.
    params.add_integer("num_layers", 1)  # Number of layers in the model.
    params.add_bool("pad_lines", False) # Never used in experiments for EACL2017
    params.add_float("dropout", 0.5)
    params.add_float("learning_rate", 1e-4)  # earlier 1e-3
    params.add_float("unknown_prob", 0)  # probability that some inputs will be changed for UNK during training to simulate testing
    params.add_integer("min_count", 50)  # minimal number of occurences of element

    # updatable arguments
    params.add_integer("trained_lines", 0)  # how many lines was trained through
    params.add_integer("epochs", 0)  # how many times it went through data
    params.add_integer("step", 0)  # how many steps it was already learning

params.add_integer("max_iters", 1000000)  # How many training steps to do in total.
params.add_integer("steps_per_checkpoint", 5000)  # How many training steps to do per checkpoint.
params.add_integer("batch_size", 64)  # Batch size to use during training.
params.add_string("time_stop", "")  # hour in format HH when the training should stop, anything out of this format won't be considered, if I tried putting minutes there the training sometimes takes more time and skip that minute
params.add_integer("max_length", 200)  # Max length of input.

params.print()

with tf.Session() as sess:
    start = time.time()  # for counting the time

    # TRAINING
    # arch = Architecture.Arch(sess, params, testingModel)
    # arch.training()


    # TESTING
    langs = None
    # langs = ['ces', 'eng', 'fra', 'deu', 'spa']
    arch = Architecture.Arch(sess, params, testingModel,
                             prepare_train_set=False)
    arch.evaluate_dataset("test/LanideNN_testset", langs)




    arch.evaluate_string('Text reading assistance: 昨日すき焼きを食べました.', True, ['jpn','eng'])
    arch.evaluate_string('El chico no tiene en la cabeza nada más que el negocio. Der Junge hat ja nichts im Kopf als das Geschäft.', True, ['deu', 'spa'])
    arch.evaluate_string('La signora lesse il messaggio e volse a Daisy uno sguardo di intesa. The lady read the message and looked up at Daisy in a knowing way.', True, ['ita','eng'])
    arch.evaluate_string("At that time of night there were few people around. A cette heure de la nuit il n'y avait pas grand monde aux alentours.", True, ['fra','eng'])
    arch.evaluate_string('Eravamo tre contro uno, però; e la mozione fu approvata. Men omröstningen utföll tre mot en dock, sa förslaget bifölls.', True, ['ita','swe'])
    arch.evaluate_string('Ich sag Gute Nacht. And I say good night. Schon leuchtet ein Stern. Yes, I see the light.', True, ['deu', 'eng'])


    print("Script finished in " + str(int(time.time() - start)) + " s")
