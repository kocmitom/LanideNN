#!/usr/bin/python3
# -*- coding: utf-8 -*-

from CharRepre import Data
import logging
import time
import ComputePRF
import numpy as np
import CharModel
import LanguageID

import os
import inspect
import numpy as np
from operator import itemgetter

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
from KocmiTF import Vocabulary


class Arch(object):
    def __init__(self, sess, params, trained_model=False):
        start = time.time()  # for counting the time
        self.sess = sess
        self.params = params
        self.train_set = Data(self.params, "data/" + self.params.get("corpus_name") + "/train", None, only_eval=False)
        self.train_set.prepare_data(self.params.get("min_count"))

        self.model = CharModel.Model(sess, self.params, self.train_set.vocab_size())

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

    def evaluate_baldwin(self, improved=False):
        # all model languages TODO CHANGE FOR NEW MODELS
        # prepared
        allowed_langs = ["ru", "new", "gl", "nl", "da", "cmn", "hu", "cs", "en", "ja", "it", "eo", "ber", "bg", "de", "pt", "he", "fr", "ar", "uk", "eu", "sk", "pms", "et", "es", "is", "vi", "lt",
                         "fi", "az", "af", "sv", "lv", "pl", "id", "zh", "tr", "kn", "el", "an", "sr", "gu", "fa", "ko", "mk", "ro", "kk", "ta", "ml", "be", "cy", "ms", "hr", "ur", "te", "nn", "bs",
                         "als", "lb", "fy", "br", "hi", "no", "la", "sl", "ca", "th", "bn", "ka", "tl", "sq", "pa", "si", "mr", "oc", "sw", "bpy", "ht", "ast", "jv", "ku", "ia", "ga", "yi", "lmo",
                         "as", "ks", "hy", "mn", "li", "or", "sco", "ne", "wa", "nds", "vec", "pes", "tlh", "hsb", "jbo", "sah", "gd", "qu", "nb", "tt", "fo", "ug", "pnb", "ceb", "am", "glk", "bcl",
                         "co"]
        # prepared2
        allowed_langs = ["af", "am", "ar", "an", "as", "ast", "az", "ba", "bcl", "be", "bn", "ber", "bpy", "br", "bg", "ca", "ceb", "cs", "ce", "cv", "co", "cy", "da", "de", "dv", "ekk", "el", "en",
                         "et", "eu", "fa", "fi", "fr", "fy", "gd", "ga", "gl", "gom", "gsw", "gu", "ht", "he", "hif", "hi", "hr", "hsb", "hu", "hy", "io", "ilo", "ia", "id", "it", "jv", "ja",
                         "kl", "kn", "ks", "ka", "kk", "ky", "ko", "ku", "la", "lv", "li", "lt", "lb", "lg", "lus", "ml", "mr", "min", "mk", "mg", "mt", "mn", "mi", "ms", "nds", "ne", "new", "nl",
                         "nn", "no", "nso", "oc", "or", "os", "pam", "pa", "pms", "pnb", "pl", "pt", "ps", "rm", "ro", "ru", "sah", "scn", "si", "sk", "sl", "sn", "so", "es", "sq", "sr", "su", "sw",
                         "sv", "ta", "tt", "te", "tg", "tl", "th", "tr", "ug", "uk", "ur", "uz", "vec", "vi", "vo", "wa", "yi", "zh", "zu"] #"is",
        
        #prepared3
        if not improved:
            allowed_langs = ["af","am","ar","an","as","ast","az","ba","bcl","be","bn","ber","bpy","br","bg","ca","ceb","cs","ce","cv","co","cy","da","de","dv","ekk","el","en","et","eu","fa","fi","fr","fy","gd","ga","gl","gom","gsw","gu","ht","he","hif","hi","hr","hsb","hu","hy","io","ilo","ia","id","is","it","jv","ja","kl","kn","ks","ka","kk","ky","ko","ku","la","lv","li","lt","lb","lg","lo","lus","ml","mr","min","mk","mg","mt","mn","mi","ms","nds","ne","new","nl","nn","no","nso","oc","or","os","pam","pa","pms","pnb","pl","pt","ps","rm","ro","ru","sah","scn","si","sk","sl","sn","so","es","sq","sr","su","sw","sv","ta","tt","te","tg","tl","th","tr","ug","uk","ur","uz","vec","vi","vo","wa","yi","zh","zu"]
        else:
            # lanfid
            allowed_langs = ["af", "am", "an", "ar", "as", "az", "be", "bg", "bn", "br", "bs", "ca", "cs", "cy", "da", "de", "dz", "el", "en", "eo", "es", "et", "eu", "fa", "fi", "fo", "fr", "ga",
                                "gl",
                                "gu", "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "ku", "ky", "la", "lb", "lo", "lt", "lv", "mg", "mk", "ml", "mn",
                                "mr",
                                "ms", "mt", "nb", "ne", "nl", "nn", "no", "oc", "or", "pa", "pl", "ps", "pt", "qu", "ro", "ru", "rw", "se", "si", "sk", "sl", "sq", "sr", "sv", "sw", "ta", "te", "th",
                                "tl",
                                "tr", "ug", "uk", "ur", "vi", "vo", "wa", "xh", "zh", "zu"]
        # # actual languages of dataset
            allowed_langs = ["af", "an", "ar", "az", "be", "bg", "bn", "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fr", "ga", "gl", "ha", "he", "hi", "hr", "ht", "hu",
               "id", "io", "is", "it", "ja", "jv", "ka", "kk", "ko", "ku", "ky", "la", "lb", "lo", "lt", "lv", "mk", "ml", "mr", "ms", "ne", "nl", "nn", "no", "oc", "pl", "ps", "pt", "ro", "ru",
                  "sh", "hbs", "sk", "sl", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "uz", "vi", "wa", "zh", "zu"] #+ UNKNOWN

        # # languages per corpus + UNKNOWN
        # allowed_langs = {
        #     "TCL": ["af", "ar", "az", "bg", "bn", "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "fa", "fi", "fr", "ga", "ha", "he", "hi", "hr", "hu", "id", "is", "it", "ja", "kk", "ko", "ky",
        #             "lo", "lv", "mk", "ml", "ne", "nl", "no", "pl", "ps", "pt", "ro", "ru", "sk", "sl", "so", "sq", "sr", "sv", "sw", "ta", "th", "tl", "tr", "uk", "ur", "uz", "vi", "zh", "zu"],
        #     "EuroGOV": ["de", "en", "es", "fi", "fr", "hu", "it", "nl", "pt", "ro"],
        #     "Wikipedia": ["af", "an", "ar", "az", "be", "bg", "bn", "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "et", "eu", "fa", "fi", "fr", "gl", "he", "hi", "hr", "ht", "hu", "id", "io",
        #                   "is", "it", "ja", "jv", "ka", "ko", "ku", "la", "lb", "lt", "lv", "mk", "mr", "ms", "nl", "nn", "no", "oc", "pl", "pt", "ro", "ru", "sh", "hbs", "sk", "sl", "sq", "su", "sv", "ta",
        #                   "te", "th", "tl", "tr", "uk", "vi", "wa", "zh"]}
        # print("DO NOT FORGET TO ADD WHICH LANGUAGE SHOULD BE USED IN ALLOWED LANGS BY: allowed_langs[cor]")

        # evaluate baldwin testing data
        datasets = ["EuroGOV", "TCL", "Wikipedia"]
        # datasets = ["EuroGOV"] # smallest
        folder = "test/baldwin/"
        baldwin = Data(self.params, None, "data/" + self.params.get("corpus_name") + "/train", only_eval=True)

        #code_swaps = {"hbs": "sh"}
        code_swaps = {}

        for cor in datasets:
            output_file = "results/baldwin_" + cor + "_" + str(time.time())

            def __gen():
                with open(folder + cor + ".meta", encoding='utf-8', mode='r') as c:
                    for line in c:
                        attr = line.strip().split('\t')
                        encoding = attr[1]

                        if encoding == "big5" or encoding == "gb2312":
                            encoding = "utf-8"

                        yield [folder + cor + "/", attr[0], encoding]

            files = __gen()

            start = time.time()
            self.evaluate(files, 1, allowed_langs, output_file, 100, unknown="en", separator="\t", code_swaps=code_swaps)
            print("Script finished on corpus " + cor + " in " + str(int(time.time() - start)) + " s")

            logging.info("Results on " + cor + ".meta saved in " + output_file)
            ComputePRF.evaluate(output_file, "test/baldwin/" + cor + ".meta", separator="\t", columnsEG=[1, 2])

    def evaluate_ALTW(self, threashold, eval_lines=False, max_langs=2, improved=False):
        if improved:
            allowed_langs = ["af", "an", "ar", "ast", "az", "be", "bg", "bn", "bpy", "br", "bs", "ca", "ceb", "cs", "cy", "da", "de", "el", "en", "et", "eu", "fa", "fi", "fr", "gl", "he", "hi", "hr",
                         "ht",
                         "hu", "id", "io", "is", "it", "ja", "jv", "ka", "ko", "ku", "la", "lb", "lt", "lv", "mk", "mr", "ms", "nap", "nds", "new", "nl", "nn", "no", "oc", "pl", "pms", "pt", "ro",
                         "ru",
                         "scn", "sh", "sk", "sl", "sq", "su", "sv", "ta", "te", "th", "tl", "tr", "uk", "vi", "wa", "zh"]
        else:
            allowed_langs = ["af","am","ar","an","as","ast","az","ba","bcl","be","bn","ber","bpy","br","bg","ca","ceb","cs","ce","cv","co","cy","da","de","dv","ekk","el","en","et","eu","fa","fi","fr","fy","gd","ga","gl","gom","gsw","gu","ht","he","hif","hi","hr","hsb","hu","hy","io","ilo","ia","id","is","it","jv","ja","kl","kn","ks","ka","kk","ky","ko","ku","la","lv","li","lt","lb","lg","lo","lus","ml","mr","min","mk","mg","mt","mn","mi","ms","nds","ne","new","nl","nn","no","nso","oc","or","os","pam","pa","pms","pnb","pl","pt","ps","rm","ro","ru","sah","scn","si","sk","sl","sn","so","es","sq","sr","su","sw","sv","ta","tt","te","tg","tl","th","tr","ug","uk","ur","uz","vec","vi","vo","wa","yi","zh","zu"]
        output_file = "results/ALTW_" + str(time.time())
        folder = "test/altw2010-langid/tst/"

        def __gen():
            for filename in os.listdir(folder):
                if filename == '.history-kocmanek':
                    continue

                yield [folder, filename]

        files = __gen()

        #code_swaps = {"hbs": "sh"}
        code_swaps = {}

        self.evaluate(files, max_langs, allowed_langs, output_file, threashold, eval_lines, code_swaps=code_swaps)

        logging.info("Evaluation of ALTW with threashold " + str(threashold) + " is in " + output_file)
        return ComputePRF.evaluate(output_file, "test/altw2010-langid/tst-lang", ",")

    def evaluateWikiMulti(self, threashold, max_langs=5, eval_lines=False, eval_blocks=False, smoothing=0, improved=False):
        if improved:
            allowed_langs = ["ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "et", "fa", "fi", "fr", "gl", "he", "hi", "hr", "hu", "id", "it", "ja", "ka", "ko", "lt", "mk", "ms", "nl",
                         "no", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "ta", "te", "th", "tr", "uk", "vi", "zh"]
        else:
            allowed_langs = ["af","am","ar","an","as","ast","az","ba","bcl","be","bn","ber","bpy","br","bg","ca","ceb","cs","ce","cv","co","cy","da","de","dv","ekk","el","en","et","eu","fa","fi","fr","fy","gd","ga","gl","gom","gsw","gu","ht","he","hif","hi","hr","hsb","hu","hy","io","ilo","ia","id","is","it","jv","ja","kl","kn","ks","ka","kk","ky","ko","ku","la","lv","li","lt","lb","lg","lo","lus","ml","mr","min","mk","mg","mt","mn","mi","ms","nds","ne","new","nl","nn","no","nso","oc","or","os","pam","pa","pms","pnb","pl","pt","ps","rm","ro","ru","sah","scn","si","sk","sl","sn","so","es","sq","sr","su","sw","sv","ta","tt","te","tg","tl","th","tr","ug","uk","ur","uz","vec","vi","vo","wa","yi","zh","zu"]
        folder = "test/wikipedia-multi-v6/wikipedia-multi/"
        output_file = "results/WikiMulti_" + str(time.time())

        def __gen():
            read = []
            with open(folder + "all-meta3", mode='r') as all_meta:
                for line in all_meta:
                    line = line.strip().split(',')
                    file = line[0]

                    if file in read:
                        continue

                    read.append(file)

                    yield [folder, file]

        files = __gen()

        self.evaluate(files, max_langs, allowed_langs, output_file, threashold, eval_lines, eval_blocks, smoothing)

        logging.info("Evaluation of MultiWiki with threashold " + str(threashold) + " is in " + output_file)
        return ComputePRF.evaluate(output_file, folder + "all-meta3", ",")

    def evaluate_tweetlid(self):
        output_file = "results/tweetlid_" + str(time.time())
        print("Outputing file is " + output_file)

        test = "test/TweetLID_corpusV2/official_test_tweets.tsv"
        labels = "test/TweetLID_corpusV2/official_test_labels.tsv"

        datafile = Data(self.params, test, "data/" + self.params.get("corpus_name") + "/train")

        with open(output_file, encoding='utf-8', mode='w', buffering=1) as bal:
            with open(labels, encoding='utf-8', mode='r', buffering=1) as lab:
                while not datafile.is_finished():
                    dev_batch_xs, _, lengths = datafile.get_batch()
                    outs = self.model.eval(self.sess, dev_batch_xs, lengths)
                    for j in range(len(outs[0])):
                        label = next(lab, None)
                        guesses = np.zeros(self.train_set.vocab_size()[1], np.int)
                        for i in range(len(outs)):
                            if dev_batch_xs[i][j] == datafile.trg_vocab.PAD_ID:
                                break
                            guesses[outs[i][j]] += 1
                        if label is not None:
                            max = np.argmax(guesses)
                            language = datafile.get_target_name(max, "orig")
                            if language.startswith("_"):
                                language = "und"
                            bal.write(label.strip() + "\t" + language + "\n")
                        else:
                            print("out of LABELS :/ ... something is wrong or the final batch padding happened")

    def evaluate_short_dataset(self, improved=False):
        short_dataset_langs = ['afr', 'sqi', 'ara', 'hye', 'aze', 'eus', 'bel', 'ben', 'bul', 'cat', 'hrv', 'ces', 'zho', 'dan', 'nld', 'eng', 'est', 'fin', 'fra', 'glg', 'kat', 'deu', 'ell', 'guj', 'hat',
                       'heb', 'hin', 'hun', 'isl', 'ind', 'gle', 'ita', 'jav', 'jpn', 'kan', 'kor', 'lav', 'lit', 'mkd', 'msa', 'mal', 'mlt', 'mar', 'nep', 'nor', 'ori', 'fas', 'pol', 'por', 'pan',
                       'ron', 'rus', 'srp', 'sin', 'slk', 'slv', 'spa', 'swa', 'swe', 'tgl', 'tam', 'tel', 'tha', 'tur', 'ukr', 'urd', 'vie', 'cym']
        source = "test/short_dataset"
        correct_all = 0
        correct_inter = 0
        total_all = 0
        total_inter = 0
        with open(source, mode='r') as src:
            for l in src:
                if total_all % 1000 == 0:
                    print("processed out of 13100: ", total_all)
                entry = l.strip().split(' ', 1)
                if improved:
                    guess = self.evaluate_string(entry[1], languages=short_dataset_langs)
                else:
                    guess = self.evaluate_string(entry[1])
                a = LanguageID.LanguageID(guess[0])
                total_all += 1
                if entry[0] in short_dataset_langs:
                    total_inter += 1
                if entry[0] == a.get_iso3():
                    correct_all += 1
                    if entry[0] in short_dataset_langs:
                        correct_inter += 1


        print("Accuracy all: {0} ({1}/{2}), Accuracy inter: {3} ({4}/{5})".format(correct_all / total_all, correct_all, total_all, correct_inter / total_inter, correct_inter, total_inter))

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
        last_class = -1
        ##print(languages[0])
        while not datafile.is_finished():
            dev_batch_xs, _, lengths = datafile.get_batch()
            
            if languages is not None:
                outs = self.model.eval(self.sess, dev_batch_xs, lengths, langs_mask = langs_mask)
            else:
                outs = self.model.eval(self.sess, dev_batch_xs, lengths)
            ##outs = self.model.eval_probs(self.sess, dev_batch_xs, lengths)
            for j in range(len(outs[0])):
                for i in range(len(outs)):
                    if languages is not None:
                        new_out = {}
                        for a in range(len(langs_mask)):
                            if langs_mask[a]==1:
                                new_out[a] = outs[i][j][a]
                        maxsort = sorted(new_out, key=new_out.get) 
                    else:
                        maxsort = outs[i][j].argsort()
                    
                    maxim = outs[i][j]
                    #maxim = maxsort[-1]
                    #maxim2 = maxsort[-2]
                    #maxim3 = maxsort[-3]
                    
                    if dev_batch_xs[i][j] == datafile.trg_vocab.PAD_ID:
                        break
                    guesses[maxim] += 1
                    if print_per_character:
                        sums = outs[i][j][maxim] + outs[i][j][maxim2]
                        #print(datafile.get_source_name(dev_batch_xs[i][j]),datafile.get_target_name(maxim), outs[i][j][maxim]/sums, datafile.get_target_name(maxim2), outs[i][j][maxim2]/sums)
                        print(outs[i][j][self.train_set.trg_vocab.get_id(languages[0])]/sums)
                        if last_class != maxim:
                            while len(orig) < len(classif):
                                orig += " "
                            while len(classif) < len(orig):
                                classif += " "
                            classif += str(datafile.get_target_name(maxim)) + " "
                            last_class = maxim
                        orig += str(datafile.get_source_name(dev_batch_xs[i][j]))

                    total += 1
        max = np.argmax(guesses)
        #print(max)
        if print_per_character:
            print(orig)
            print(classif)
        accur = 0
        if total > 0:
            accur = float(guesses[max]) / float(total)
        return [datafile.get_target_name(max), accur]

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
