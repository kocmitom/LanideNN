from collections import defaultdict
import operator


def compute_prf(tp, fp, fn):
    if tp > 0:
        p = tp / float(tp + fp)
        r = tp / float(tp + fn)
        f = 2 * r * p / float(r + p)
    else:
        p = 0
        r = 0
        f = 0

    return p, r, f


def evaluate(evalfile, goldfile, separator=",", columnsEG=[1, 1], get_confusion=True):
    if isinstance(separator, str):
        separator = [separator, separator]

    gm = defaultdict(list)
    langs = []
    gold_data = defaultdict(list)

    with open(goldfile, mode="r") as gold:
        for line in gold:
            x = line.strip().split(separator[1])
            gold_data[x[0]].append(x[columnsEG[1]])
            gm[x[columnsEG[1]]].append(x[0])
            if x[columnsEG[1]] not in langs:
                langs.append(x[columnsEG[1]])

    em = defaultdict(list)
    eval_data = defaultdict(list)

    with open(evalfile, mode="r") as eval:
        for line in eval:
            x = line.strip().split(separator[0])
            em[x[columnsEG[0]]].append(x[0])
            if x[columnsEG[0]] not in langs:
                langs.append(x[columnsEG[0]])

            eval_data[x[0]].append(x[columnsEG[0]])

    if get_confusion:
        print("CONFUSION WORKS PROPERLY ONLY IF EACH QUERY CAN HAVE ONLY ONE ANSWER")
        confusions = defaultdict(list)
        total_confusions = defaultdict(int)
        for g in gold_data:
            gold_set = []
            for l in gold_data[g]:
                same = False
                if g in eval_data:
                    for l2 in eval_data[g]:
                        if l == l2:
                            same = True
                            break
                if not same:
                    gold_set.append(l)

            eval_set = []
            for l in eval_data[g]:
                same = False
                if g in eval_data:
                    for l2 in gold_data[g]:
                        if l == l2:
                            same = True
                            break
                if not same:
                    eval_set.append(l)
            total_confusions[str(sorted(gold_data[g]))] += 1

            if len(gold_set) == 0 or len(eval_set) == 0:
                continue
            confusions[str(sorted(gold_set))].append(str(sorted(eval_set)))

        for l in sorted(confusions):
            conf = defaultdict(int)
            for x in confusions[l]:
                conf[x] += 1
            print("Correct {0}, Incorrectly: {1} out of {2}".format(l, len(confusions[l]), total_confusions[l]))
            for x in sorted(conf.items(), key=operator.itemgetter(1), reverse=True):
                if x[1] <= 0:
                    break
                print("... confused with {0} {1}x".format(x[0], x[1]))

    scores = defaultdict(lambda: defaultdict(int))
    for lang in langs:
        for doc in em[lang]:
            if doc in gm[lang]:
                scores[lang]["tp"] += 1
            else:
                scores[lang]["fp"] += 1

        for doc in gm[lang]:
            if doc not in em[lang]:
                scores[lang]["fn"] += 1

    tp = 0
    fp = 0
    fn = 0

    for lang in langs:
        tp += scores[lang]["tp"]
        fp += scores[lang]["fp"]
        fn += scores[lang]["fn"]

    p, r, f = compute_prf(tp, fp, fn)

    tpm = defaultdict(int)
    fpm = defaultdict(int)
    fnm = defaultdict(int)
    for lang in langs:
        for doc in em[lang]:
            if doc in gm[lang]:
                tpm[lang] += 1
            else:
                fpm[lang] += 1

    for lang in langs:
        for doc in gm[lang]:
            if doc not in em[lang]:
                fnm[lang] += 1

    pm = 0
    rm = 0
    fm = 0

    for lang in langs:
        ptmp, rtmp, ftmp = compute_prf(tpm[lang], fpm[lang], fnm[lang])
        pm += ptmp
        rm += rtmp
        fm += ftmp

    pm /= float(len(langs))
    rm /= float(len(langs))
    fm /= float(len(langs))

    if rm + pm > 0:
        fms = 2 * rm * pm / float(rm + pm)
    else:
        fms = 0

    print("PM {0:.3f}, RM {1:.3f}, FM {2:.3f}, (f(PM,RM)={3:.3f}), Pm {4:.3f}, Rm {5:.3f}, Fm {6:.3f} ".format(pm, rm, fm, fms, p, r, f))

    return f
