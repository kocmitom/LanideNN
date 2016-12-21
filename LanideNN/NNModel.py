import abc

from LanideNN import ConfusionMatrix, Representation

import numpy as np
import logging


class Model(object):
    def __init__(self, vocab_sizes):
        """abstract class for Models

        Args:
          vocab_sizes: list of two elements, input and output vocabulary size
        """
        self.vocab_sizes = vocab_sizes

    @abc.abstractmethod
    def eval(self, session, inputs):
        """Evaluate inputs and return computed outputs. It must have the same structure as the outputs."""
        return

    # def get_confusion_matrix(self, session, inputs, correct_outputs, confusionMatrix=None):
    #     if confusionMatrix is None:
    #         confusionMatrix = ConfusionMatrix.ConfusionMatrix()
    #
    #     outs = self.eval(session, inputs)
    #     gen = MathUtils.flatten(outs)
    #
    #     for correct in MathUtils.flatten(correct_outputs):
    #         computed = next(gen)
    #
    #         # TODO probably most methods will not have access to this Representation PAD_ID
    #         if correct != Representation.Representation.PAD_ID:
    #             confusionMatrix.increase(correct, computed)
    #
    #     return confusionMatrix


    # TODO DELME only for testing purposes of lanidenn
    def get_confusion_matrix(self, computed_outs, correct_outputs, confusionMatrix=None):
        if confusionMatrix is None:
            logging.info("WARNING this method for generation of confusion matrix should be used only in the project langspot")
            confusionMatrix = ConfusionMatrix.ConfusionMatrix()

        votes = np.zeros(200)

        # TODO only lanidenn has two dimensional matrix
        for b in range(len(computed_outs[0])):
            for l in range(len(computed_outs)):
                correct = correct_outputs[l][b]
                computed = computed_outs[l][b]
                if correct != Representation.Representation.PAD_ID:
                    votes[computed] += 1
            confusionMatrix.increase(correct_outputs[0][b], np.argmax(votes))
            votes = np.zeros(200)

        return confusionMatrix
