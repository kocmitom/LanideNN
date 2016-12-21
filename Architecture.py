#!/usr/bin/python3
# -*- coding: utf-8 -*-

import logging
import os
import time

import numpy as np

from LanideNN import Vocabulary, LanguageID, BiRNN_Embed
from LanideNN.CharRepre import Data


class Arch(object):
    def __init__(self, sess, params, trained_model=False,
                 prepare_train_set=True):
        start = time.time()  # for counting the time
        self.sess = sess
        self.params = params
        self.train_set = Data(self.params, "data/" + self.params.get(
            "corpus_name") + "/train", None, only_eval=False)

        if prepare_train_set:
            self.train_set.prepare_data(self.params.get("min_count"))

        self.model = BiRNN_Embed.Model(sess, self.params, self.train_set.vocab_size())

        if trained_model:
            # restore model
            self.model.saver.restore(sess, trained_model)

        print("Model prepared in " + str(int(time.time() - start)) + " s.")

    def get_confusion_matrix(self):
        # confusion matrix
        self.params.params["pad_lines"] = True
        bigdev = Data(self.params, "data/" + self.params.get("corpus_name") + "/dev", "data/" + self.params.get("corpus_name") + "/train")

        confusionMatrix = None
        while not bigdev.is_finished():
            dev_batch_xs, dev_batch_ys, length = bigdev.get_batch()
            outs = self.model.eval(self.sess, dev_batch_xs, length)
            confusionMatrix = self.model.get_confusion_matrix(outs, dev_batch_ys, confusionMatrix)

        confusionMatrix.print_worse_pairs(names=lambda x: self.train_set.get_target_name(x), maximum=200)
        confusionMatrix.print_worse_classes(names=lambda x: self.train_set.get_target_name(x), maximum=200)
        confusionMatrix.print_error_rate()

    def evaluate(self, files, max_langs_per_file, allowed_langs, output_file, threashold, eval_lines=False, eval_blocks=False, smoothing=0, unknown=None, separator=",", code_swaps=None):
        langs_mask = np.zeros(self.model.vocab_sizes[1], dtype=np.int)
        for allowed in self.train_set.get_tagging_classes():
            langs_mask[allowed] = 1  # this allows _UNK a napr HTML

        for l in allowed_langs:
            # try find originally
            id = self.train_set.trg_vocab.get_id(l)
            if id == Vocabulary.Vocab.UNK_ID:
                # try find by ISO3
                iso = LanguageID.LanguageID(l)
                id = self.train_set.trg_vocab.get_id(iso.get_iso3())
            if id == Vocabulary.Vocab.UNK_ID:
                print("UNSUPPORTED LANGUAGE IN MODEL: " + l)
            else:
                langs_mask[id] = 1

        datafile = Data(self.params, None, "data/" + self.params.get("corpus_name") + "/train", only_eval=True, use_eol=eval_lines)

        if smoothing > 0:
            print("USING SMOOTHING OF {0}".format(smoothing))

        with open(output_file, encoding='utf-8', mode='w', buffering=1) as bal:
            for filename in files:
                # files has structure: [folder, outputing_name, possible_encoding]
                if len(filename) > 2:
                    datafile.restart(filename[0] + filename[1], filename[2])
                else:
                    datafile.restart(filename[0] + filename[1])

                guesses = np.zeros(self.train_set.vocab_size()[1], np.int)
                row = np.zeros(self.train_set.vocab_size()[1], np.int)
                row_length = 0
                total = 0
                smooth = []
                while not datafile.is_finished():
                    dev_batch_xs, dev_batch_ys, lengths = datafile.get_batch()
                    outs = self.model.eval(self.sess, dev_batch_xs, lengths, langs_mask=langs_mask)
                    for j in range(len(outs[0])):
                        block_guesses = np.zeros(self.train_set.vocab_size()[1], np.int)
                        for i in range(len(outs)):
                            if dev_batch_xs[i][j] == datafile.trg_vocab.PAD_ID:
                                break
                            # print(datafile.get_source_name(dev_batch_xs[i][j]), datafile.get_target_name(outs[i][j], "orig"))
                            total += 1
                            if eval_lines:
                                if dev_batch_xs[i][j] == datafile.trg_vocab.EOL_ID:  # dev_batch_ys[i][j] == -2: # or
                                    guesses[np.argmax(row)] += row_length
                                    # print("filename {0}, guessed {1}, sum {2}, line length {3}".format(filename[1], datafile.get_target_name(np.argmax(row), "iso2"), row[np.argmax(row)], row_length))
                                    row = np.zeros(self.train_set.vocab_size()[1], np.int)
                                    row_length = 0
                                else:
                                    row[outs[i][j]] += 1
                                    row_length += 1
                            elif eval_blocks:
                                block_guesses[outs[i][j]] += 1
                            elif smoothing > 0:
                                smooth.append(outs[i][j])
                            else:
                                guesses[outs[i][j]] += 1

                        if eval_blocks:
                            guesses[np.argmax(block_guesses)] += i

                if smoothing > 0:
                    for i in range(len(smooth)):
                        if i + smoothing < len(smooth) and smooth[i] == smooth[i + smoothing]:
                            # if first and the last are the same, the inbetween should be too
                            guesses[smooth[i]] += smoothing
                            i += smoothing - 1
                        else:
                            guesses[smooth[i]] += 1

                langs = 0
                last_count = 1
                seznam = ""
                for max in np.argsort(-guesses):
                    if guesses[max] == 0 or langs == max_langs_per_file:
                        break
                    guess_name = datafile.get_target_name(max, "iso2")
                    percent = 100 * guesses[max] / total
                    if guess_name in allowed_langs:
                        if code_swaps is not None and guess_name in code_swaps:
                            guess_name = code_swaps[guess_name]
                        # print at least on language
                        # if langs > 0 and 100 * guesses[max] / last_count < threashold:
                        #     break
                        if langs > 0 and percent < threashold:
                            break
                        seznam += "{0} {1:.0f}; ".format(guess_name, percent)
                        bal.write(filename[1] + separator + guess_name + "\n")
                        # print(filename[1] + "," + guess_name)
                        langs += 1
                        last_count = guesses[max]
                    else:
                        lab = LanguageID.LanguageID(guess_name)
                        print(filename[1] + ", not allowed lang: " + lab.get_all())
                # print("List: "+seznam)
                if langs == 0 and unknown is not None:
                    # no language was outputted
                    bal.write(filename[1] + separator + unknown + "\n")

    def evaluate_dataset(self, source,
                         allowed_languages=None):

        correct_all = 0
        total_all = 0
        with open(source, mode='r') as src:
            for l in src:
                if total_all % 1000 == 0:
                    print("processed lines ", total_all)
                entry = l.strip().split(' ', 1)
                if allowed_languages is not None:
                    guess = self.evaluate_string(entry[1], languages=allowed_languages)
                else:
                    guess = self.evaluate_string(entry[1])
                a = LanguageID.LanguageID(guess[0])
                total_all += 1
                if entry[0] == a.get_iso3():
                    correct_all += 1

        print("Accuracy all: {0} ({1}/{2})".format(correct_all / total_all, correct_all, total_all))

    def evaluate_string(self, text, print_per_character=False, languages = None):
        if languages is not None:
            langs_mask = np.zeros(self.model.vocab_sizes[1], dtype=np.int)
            for allowed in self.train_set.get_tagging_classes():
                langs_mask[allowed] = 1  # this allows _UNK a napr HTML

            for l in languages:
                # try find originally
                id = self.train_set.trg_vocab.get_id(l)
                if id == Vocabulary.Vocab.UNK_ID:
                    # try find by ISO3
                    iso = LanguageID.LanguageID(l)
                    id = self.train_set.trg_vocab.get_id(iso.get_iso3())
                if id == Vocabulary.Vocab.UNK_ID:
                    print("UNSUPPORTED LANGUAGE IN MODEL: " + l)
                else:
                    langs_mask[id] = 1
        datafile = Data(self.params, None, "data/" + self.params.get("corpus_name") + "/train", text_to_eval=text)

        guesses = np.zeros(self.train_set.vocab_size()[1], np.int)
        total = 0
        orig = ""
        classif = ""
        while not datafile.is_finished():
            dev_batch_xs, _, lengths = datafile.get_batch()
            
            if languages is not None:
                outs = self.model.eval(self.sess, dev_batch_xs, lengths, langs_mask = langs_mask)
            else:
                outs = self.model.eval(self.sess, dev_batch_xs, lengths)
            for j in range(len(outs[0])):
                for i in range(len(outs)):
                    if languages is not None:
                        new_out = {}
                        for a in range(len(langs_mask)):
                            if langs_mask[a]==1:
                                new_out[a] = outs[i][j][a]
                    
                    maxim = outs[i][j]
                    
                    if dev_batch_xs[i][j] == datafile.trg_vocab.PAD_ID:
                        break
                    guesses[maxim] += 1

                    total += 1
        max = np.argmax(guesses)
        if print_per_character:
            print(orig)
            print(classif)
        accur = 0
        if total > 0:
            accur = float(guesses[max]) / float(total)
            
        print([datafile.get_target_name(max), accur])

    def training(self, eval=None):
        self.train_set.skip_n_lines(self.params.params["trained_lines"])

        dev = Data(self.params, "data/" + self.params.get("corpus_name") + "/dev", "data/" + self.params.get("corpus_name") + "/train")
        dev.prepare_data(self.params.get("min_count"))
        start = time.time()  # for counting the time
        cycle_time = time.time()
        logging.info("Training process begun.")
        stop = False

        # Keep training until reach max iterations
        while not stop:
            self.params.params["step"] += 1
            batch_xs, batch_ys, lengths = self.train_set.get_batch()
            self.model.run(False, self.sess, batch_xs, batch_ys, lengths, self.params.get("dropout"))

            stop = self.chech_stopfile("STOP_IMMEDIATELY")

            if time.strftime("%H") == self.params.get("time_stop"):
                stop = True

            if self.params.params["step"] % self.params.get("steps_per_checkpoint") == 0 or stop:
                c_time = time.time()
                corr = [0, 0]

                while not dev.is_finished() and eval is None:
                    dev_batch_xs, dev_batch_ys, lengths = dev.get_batch()
                    # if testingModel:
                    #    corectness = model.run(True, sess, test_batch_xs, test_batch_ys, data.rev_vocab)

                    dropout = 1
                    corr = np.sum([corr, self.model.run(True, self.sess, dev_batch_xs, dev_batch_ys, lengths, dropout, lambda x: dev.get_target_name(x))], axis=0)

                if eval is not None:
                    logging.info("Not testing on dev but on special function.")
                    result = eval()
                else:
                    # restart development data
                    dev.restart()
                    result = (corr[0] / corr[1]) * 100

                self.params.params["trained_lines"] = self.train_set.get_trained_lines();

                self.model.save(self.sess, self.params.params["step"], result)

                print("Iter {0}, Total correctness: {1} % {2}, time per step: {3} s, total time: {4} min, {5}".format(self.params.params["step"] * self.params.get("batch_size"), result, corr,
                                                                                                                      (c_time - cycle_time) / self.params.get("steps_per_checkpoint"),
                                                                                                                      int((time.time() - start) / 60), time.strftime("%H:%M:%S")))
                # print((c_time - cycle_time) / self.params.get("steps_per_checkpoint"))
                cycle_time = time.time()

                stop = stop or self.chech_stopfile("STOP_MODEL")  # if it already is True do not change it

                if self.params.params["step"] >= self.params.get("max_iters"):
                    stop = True

            # check if the file was not finished and if it was, start over
            if self.train_set.is_finished():
                self.params.params["epochs"] += 1
                logging.info("Generator read training file completely and starts over")
                self.train_set.restart()

        print("Training finished in " + str(int(time.time() - start)) + " s")

    def chech_stopfile(self, filename):
        stop = False
        with open(filename, mode="r") as stp:
            for line in stp:
                if line.strip() == self.params.params["corpus_name"]:
                    logging.info("Stopping training on command from stopfile.")

                    stop = True
                    break

        if stop:
            # remove command from file
            f = open(filename, "r")
            lines = f.readlines()
            f.close()

            f = open(filename, "w")
            for line in lines:
                if line.strip() != self.params.params["corpus_name"]:
                    f.write(line)
            f.close()

        return stop
