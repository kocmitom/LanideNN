#!/usr/bin/python3
# -*- coding: utf-8 -*-

import random
import re
import os
import logging
import numpy as np
import tensorflow as tf
from LanideNN import Vocabulary
import abc


class Repre(object):
    def __init__(self, original_filename, vocabulary_path, params=None, use_eol = False, text_to_eval = None):
        self.prepared_data = False  # consider data not prepared
        self.original_filename = original_filename
        self.encoding = "utf-8"  # for baldwin dataset use
        if vocabulary_path == None:
            self.vocabulary_path = original_filename
        else:
            self.vocabulary_path = vocabulary_path

        self.batch_number = 0
        self.skip_lines = 0
        self.line_num = 0
        self.use_eol = use_eol
        self.file_finished = False
        self.pad_lines = params.get("pad_lines")
        self.unknown_prob = params.get("unknown_prob")
        self.max_length = params.params.get("max_length")
        self.unicode_normalization = params.params.get("unicode_normalization")
        self.batch_size = params.params.get("batch_size")
        self.version = params.params.get("version")

        self.src_vocab = Vocabulary.Vocab(self.vocabulary_path + ".src.vocab")
        self.trg_vocab = Vocabulary.Vocab(self.vocabulary_path + ".trg.vocab")
        self.gen = None
        self.text_to_eval = text_to_eval

    def prepare_data(self, min_count=0):
        """ Prepares data if they do not exists

        It sequentially loads original data and based on the encoding function prints IDs
        from vocabulary into the datafile.

        Args:
          encode_line: function which encodes line into string of tokens
        """
        filename = self.original_filename + ".ids"
        if not os.path.isfile(filename):
            if min_count > 0 and self.src_vocab.build_vocab:
                logging.info("Data are not prepared yet. Preparing vocabulary.")
                iters = self.file_iterator()
                counter = 0
                for elements in iters:
                    counter += 1
                    if counter % 100000 == 0:
                        logging.info("to vocab line {0}".format(counter))
                self.src_vocab.finish_vocab(min_count)
                self.trg_vocab.finish_vocab(0) # target vocabulary should not be prunned
            logging.info("Data are not prepared yet. Preparing token IDs into {0} file.".format(filename))

            with open(filename, encoding='utf-8', mode='w') as tokens_file:
                counter = 0
                iters = self.file_iterator()
                for elements in iters:
                    tokens_file.write(" ".join(str(x) for x in elements) + "\n")
                    counter += 1
                    if counter % 100000 == 0:
                        logging.info("tokenizing line {0}".format(counter))
            self.src_vocab.save()
            self.trg_vocab.save()
            logging.info("Data prepared.")

        self.prepared_data = True

    def restart(self, new_filename=None, encoding="utf-8"):
        # load new file to process ... needed in baldwin
        if new_filename is not None:
            self.original_filename = new_filename
            self.encoding = encoding

        self.file_finished = False
        self.line_num = 0
        self.gen = self.get_generator()

    def file_iterator(self):
        if self.prepared_data:
            filename = self.original_filename + ".ids"
            with open(filename, encoding='utf-8', mode="r") as file:
                for line in file:
                    self.line_num += 1
                    if self.skip_lines > 0:
                        self.skip_lines -= 1
                        continue

                    yield line.split()
        else:
            if self.text_to_eval is None:
                with open(self.original_filename, encoding=self.encoding, mode='r') as data_file:
                    for line in data_file:
                        self.line_num += 1
                        if self.skip_lines > 0:
                            self.skip_lines -= 1
                            continue

                        line = line.strip()
                        if len(line) == 0:
                            continue
                        yield self.encode_line(line)
            else:
                yield self.encode_line(self.text_to_eval)


    @abc.abstractmethod
    def encode_line(self, line):
        # this method should return list of IDS, where targets are negative IDS
        # target IDS must be moved by -1 to avoid collision over ID 0
        return

    def increase_batch(self):
        self.batch_number += 1

    def skip_n_lines(self, skip_lines=0):
        logging.info("Skipping {0} lines.".format(skip_lines))
        self.skip_lines = skip_lines

    def get_trained_lines(self):
        return self.line_num

    def is_finished(self):
        return self.file_finished

    def get_generator(self):
        """Initialize generator for reading data

        Whenever it gets to the end of file it fills the last batch with PADS.

        Returns:
          an ID of next token
        """
        iters = self.file_iterator()
        trg = -1
        warn = False
        for elements in iters:

            for x in elements:
                x = int(x)
                if x < 0:
                    # targets are moved by -1
                    trg = -x - 1
                    continue
                # simulate unknown
                if self.unknown_prob > 0 and random.random() < self.unknown_prob:
                    x = self.src_vocab.UNK_ID

                yield x, trg

            batch = self.batch_number
            if self.use_eol:
                if self.version<3:
                    if not warn:
                        print("DO NOT USE END OF LINES, IT IS NOT IMPLEMENTED IN THIS MODEL")
                        warn = True
                        yield self.src_vocab.EOL_ID, -2
                    else:
                        yield self.src_vocab.EOL_ID, self.trg_vocab.EOL_ID

            while self.pad_lines and batch == self.batch_number:
                yield self.src_vocab.PAD_ID, self.trg_vocab.PAD_ID

        self.file_finished = True

        while True:
            yield self.src_vocab.PAD_ID, self.trg_vocab.PAD_ID

    def vocab_size(self):
        return [self.src_vocab.size(), self.trg_vocab.size()]
