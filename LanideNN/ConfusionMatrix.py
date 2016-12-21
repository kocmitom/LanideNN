from collections import defaultdict


class ConfusionMatrix(object):
    def __init__(self):
        # initialize confusion matrix
        self.confusionMatrix = defaultdict(int)

    def increase(self, correct, computed):
        self.confusionMatrix[(correct, computed)] += 1

    def print_worse_pairs(self, names=lambda x: x, maximum=100):
        for value in sorted(self.confusionMatrix, key=self.confusionMatrix.get, reverse=True):
            if value[0] == value[1]:
                #skip correct assignments
                continue
            print("CORRECT: {0}\tWRONG: {1}:\t{2}".format(names(value[0]), names(value[1]), self.confusionMatrix[value]))
            maximum -= 1
            if maximum == 0:
                break

    def print_worse_classes(self, names=lambda x: x, maximum=100):
        incorrectly = defaultdict(int)
        total = defaultdict(int)
        for pair in self.confusionMatrix:
            total[pair[0]] += self.confusionMatrix[pair]
            if pair[0] != pair[1]:
                incorrectly[pair[0]] += self.confusionMatrix[pair]

        for cl in sorted(total, key=lambda k: incorrectly[k]/total[k], reverse=True):
            print("{0}:\terror {1}%".format(names(cl), 100*incorrectly[cl]/total[cl]))
            maximum -= 1
            if maximum == 0:
                break

    def print_error_rate(self):
        errors = 0
        total = 0
        for pair in self.confusionMatrix:
            total += self.confusionMatrix[pair]
            if pair[0] != pair[1]:
                errors += self.confusionMatrix[pair]

        print("Error rate is {0} %".format(100*errors/total))
