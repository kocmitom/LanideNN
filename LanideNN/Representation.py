#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import logging


class Representation(object):
    # TODO PAD not implemented yet
    PAD_ID = 1
    _PAD = "_PAD"

    UNK_ID = 0
    _UNK = "_UNK"

    def __init__(self, original_filename, is_source, encode_line, automatic_recycle=False, pad_lines = False):
        self.vocab = None
        self.rev_vocab = None
        self.build_vocab = False
        self.cycle = 0  # number of cycles over the data ... between restarts
        self.batch_number = 0
        self.automatic_recycle = automatic_recycle  # automatically restart generator at the end
        self.prepared_data = False  # consider data not prepared
        self.encode_line = encode_line  # function for line encoding
        self.pad_lines = pad_lines

        self.original_filename = original_filename
        if is_source:
            self.data_filename = original_filename + ".src"
        else:
            self.data_filename = original_filename + ".trg"

        self.src = self.get_generator()

    def vocab_id(self, token):
        """Get ID of token from vocabulary.

        If we are building the vocabulary, we are adding new tokens into the vocab,
        otherwise for unknown tokens return UNK_ID

        Args:
          token: name of a token.

        Returns:
          a n ID
        """
        if self.build_vocab:
            if token not in self.vocab:
                self.vocab[token] = len(self.rev_vocab)
                self.rev_vocab.append(token)
            return self.vocab[token]
        else:
            return self.vocab.get(token, Representation.UNK_ID)

    def load_vocabulary(self, vocabulary_path):
        """Initialize vocabulary from file.

        We assume the vocabulary is stored one-item-per-line, so a file:
          dog
          cat
        will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
        also return the reversed-vocabulary ["dog", "cat"].

        Args:
          vocabulary_path: path to the file containing the vocabulary.

        Returns:
          a pair: the vocabulary (a dictionary mapping string to integers), and
          the reversed vocabulary (a list, which reverses the vocabulary mapping).
        """
        if vocabulary_path is not None and os.path.isfile(vocabulary_path):
            self.rev_vocab = []
            with open(vocabulary_path, encoding='utf-8', mode='r') as f:
                self.rev_vocab.extend(f.readlines())
            self.rev_vocab = [line.strip('\n') for line in self.rev_vocab]
            self.vocab = dict([(x, y) for (y, x) in enumerate(self.rev_vocab)])
        else:
            self.build_vocab = True
            self.vocab = {Representation._UNK: Representation.UNK_ID, Representation._PAD: Representation.PAD_ID}
            self.rev_vocab = [Representation._UNK, Representation._PAD]

    def prepare_data(self):
        """ Prepares data if they do not exists

        It sequentially loads original data and based on the encoding function prints IDs
        from vocabulary into the datafile.

        Args:
          encode_line: function which encodes line into string of tokens
        """
        if not os.path.isfile(self.data_filename):
            logging.info("Data are not prepared yet. Preparing token IDs into {0} file.".format(self.data_filename))

            with open(self.data_filename, encoding='utf-8', mode='w') as tokens_file:
                counter = 0
                iters = self.file_iterator()
                for elements in iters:
                    tokens_file.write(" ".join(str(x) for x in elements) + "\n")
                    counter += 1
                    if counter % 100000 == 0:
                        logging.info("tokenizing line {0}".format(counter))

            logging.info("Data prepared.")

        self.prepared_data = True

    def file_iterator(self):
        if self.prepared_data:
            with open(self.data_filename, encoding='utf-8', mode="r") as file:
                for line in file:
                    yield line.split()
        else:
            with open(self.original_filename, encoding='utf-8', mode='r') as data_file:
                for line in data_file:
                    line = line.strip()
                    encoded = self.encode_line(line)

                    yield [self.vocab_id(token) for token in encoded]

    def increase_batch(self):
        self.batch_number += 1

    def get_generator(self):
        """Initialize generator for reading data

        Whenever it gets to the end of file it fills the last batch with PADS.

        Returns:
          an ID of next token
        """
        while True:
            iters = self.file_iterator()
            for elements in iters:
                batch = self.batch_number
                for x in elements:
                    yield int(x)
                while self.pad_lines and batch == self.batch_number:
                    yield Representation.PAD_ID

            if self.automatic_recycle:
                logging.info("Generator read the complete file {0} and starts over".format(self.data_filename))
                continue

            self.cycle += 1
            while True:
                yield Representation.PAD_ID

