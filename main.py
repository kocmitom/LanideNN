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

# WIKIMULTI
#testingModel = "models/WikiMulti.model"
# ALTW
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
    params.add_string("corpus_name", "data")  # Folder with data.
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

    arch = Architecture.Arch(sess, params, testingModel)

    #arch.training()


    langs = None
    # langs = ['afr', 'sqi', 'ara', 'hye', 'aze', 'eus', 'bel',
    #                        'ben', 'bul', 'cat', 'hrv', 'ces', 'zho', 'dan',
    #                        'nld', 'eng', 'est', 'fin', 'fra', 'glg', 'kat',
    #                        'deu', 'ell', 'guj', 'hat', 'heb', 'hin', 'hun',
    #                        'isl', 'ind', 'gle', 'ita', 'jav', 'jpn', 'kan',
    #                        'kor', 'lav', 'lit', 'mkd', 'msa', 'mal', 'mlt',
    #                        'mar', 'nep', 'nor', 'ori', 'fas', 'pol', 'por',
    #                        'pan', 'ron', 'rus', 'srp', 'sin', 'slk', 'slv',
    #                        'spa', 'swa', 'swe', 'tgl', 'tam', 'tel', 'tha',
    #                        'tur', 'ukr', 'urd', 'vie', 'cym']

    arch.evaluate_short_dataset(langs)


    #arch.evaluate_string('Ich sag Gute Nacht. And I say good night. Schon leuchtet ein Stern. Yes, I see the light.', True, ['deu', 'eng'])
    #arch.evaluate_string('Text reading assistance: 昨日すき焼きを食べました.', True, ['jpn','eng'])
    #arch.evaluate_string('The Quatrième Étage is a short story written by Jean Hougron', True, ['fra', 'eng'])
    #arch.evaluate_string("Je n'ai pas dormi depuis trois jours. I haven’t slept for three days. Est-ce que vous en avez parlé à la police", True, ['fra','eng'])
    #arch.evaluate_string('La signora lesse il messaggio e volse a Daisy uno sguardo di intesa. The lady read the message and looked up at Daisy in a knowing way.', True, ['ita','eng'])
    #arch.evaluate_string('Nearby was a little note written in pencil. Vedle byla malá cedulka se vzkazem napsaným tužkou.', True, ['ces','eng'])
    #arch.evaluate_string("At that time of night there were few people around. A cette heure de la nuit il n'y avait pas grand monde aux alentours.", True, ['fra','eng'])
    #arch.evaluate_string("Cette église s'élevait sur une place étroite et sombre, près de la grille du Palais. Tämä kirkko oli ahtaalla ja pimeällä paikalla lähellä Oikeuspalatsin aitausta. The church stood in a narrow, gloomy square, not far from the gates of the Palais de Justice.", True, ['eng', 'fra', 'fin'])
    #arch.evaluate_string('She was thirty years old and had been a detective for the past two years. Ей было тридцать лет, и последние два года она была сыщиком.', True,['rus','eng'])
    #arch.evaluate_string('Eravamo tre contro uno, però; e la mozione fu approvata. Men omröstningen utföll tre mot en dock, sa förslaget bifölls.', True, ['ita','swe'])
    #arch.evaluate_string('El chico no tiene en la cabeza nada más que el negocio. Der Junge hat ja nichts im Kopf als das Geschäft.', True, ['deu', 'spa'])


    print("Script finished in " + str(int(time.time() - start)) + " s")
