import os
import codecs
import collections
import logging


class Parameters(object):
    def __init__(self, filename):
        self.params = collections.OrderedDict()
        self.filename = filename
        self.columns = []

        if os.path.isfile(filename):
            with codecs.open(filename, encoding='utf-8', mode='r') as f:
                pars = f.readline().strip().split('\t')
                for p in pars:
                    data = p.split(' ')
                    self.columns.append([data[0], data[2]])
                    self.params[data[0]] = data[1]
        else:
            file = codecs.open(filename, mode="w", encoding="utf-8")
            self.params["identification"] = "?"
            self.columns.append(["identification", "str"])
            self.params["result"] = "Initialized"
            self.columns.append(["result", "str"])
            file.write('identification ? str\tresult Initialized str\n')
            file.close()

    def add_integer(self, name, default_value):
        if name not in self.params:
            self.columns.append([name, "int"])
            self.create_parameter("{0} {1} int".format(name, default_value))

        self.params[name] = default_value

    def add_bool(self, name, default_value):
        if name not in self.params:
            self.columns.append([name, "bool"])
            self.create_parameter("{0} {1} bool".format(name, default_value))

        self.params[name] = default_value

    def add_float(self, name, default_value):
        if name not in self.params:
            self.columns.append([name, "float"])
            self.create_parameter("{0} {1} float".format(name, default_value))

        self.params[name] = default_value

    def add_string(self, name, default_value):
        if " " in default_value:
            raise NotImplementedError("Space must not be present in default value.")
        if name not in self.params:
            self.columns.append([name, "str"])
            self.create_parameter("{0} {1} str".format(name, default_value))

        self.params[name] = default_value

    def get(self, name):
        if name not in self.params:
            logging.error("Parameter {0} is not set. Exiting.".format(name))
            exit(7)
        return self.params[name]

    def create_parameter(self, header):
        content = ""
        first = True

        with codecs.open(self.filename, encoding='utf-8', mode='r') as f:
            for x in f.readlines():
                if first:
                    content += ''.join([x.strip(), '\t', header, '\n'])
                    first = False
                else:
                    content += x

        with codecs.open(self.filename, encoding='utf-8', mode='w') as f:
            f.writelines(content)

    def save_result(self, result, identification):
        self.params["identification"] = identification
        self.params["result"] = result
        line = ""
        for p in self.params:
            if len(line) > 0:
                line += "\t"
            line += str(self.params[p])

        with codecs.open(self.filename, encoding='utf-8', mode='a') as f:
            f.write(line+"\n")

    def get_column_id(self,name):
        for i in range(len(self.columns)):
            if self.columns[i][0]==name:
                return i
        logging.error("Parameter column {0} not found.".format(name))
        exit(7)


    def load_params(self, identification):
        with codecs.open(self.filename, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().split("\t")
                if params[0] == identification:
                    for i in range(len(params)):
                        if self.columns[i][1] == "int":
                            self.params[self.columns[i][0]] = int(params[i])
                        elif self.columns[i][1] == "bool":
                            self.params[self.columns[i][0]] = params[i] == "True"
                        elif self.columns[i][1] == "float":
                            self.params[self.columns[i][0]] = float(params[i])
                        else:
                            self.params[self.columns[i][0]] = params[i]
                    return

            logging.error("Parameters not found for identification {0}".format(identification))
            exit(7)

    def continue_model(self, corpus_name):
        identification = ""
        with codecs.open(self.filename, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().split("\t")
                if params[self.get_column_id("corpus_name")] == corpus_name:
                    identification = params[0]

            if len(identification) >0:
                self.load_params(identification)
            else:
                logging.error("No model of corpus {0} found.".format(corpus_name))
                exit(7)

    def print(self):
        for par in self.params:
            print(par+": "+str(self.params[par]))

