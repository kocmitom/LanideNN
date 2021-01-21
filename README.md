# LanideNN

LanideNN is language identification method based on Bidirectional Recurrent Neural Networks [1] and it performs well in monolingual and multilingual language identification tasks on six testsets. The method keeps its accuracy also for short documents and across domains, so it is ideal for the off-the-shelf use without preparation of training data.

We have also released the LanideNN dataset. More is at https://ufal.mff.cuni.cz/tom-kocmi/lanidenn


# Installation

You will need TensorFlow 0.8 (we are planning to migrate whole framework into
 NeuralMonkey and solve a problem with long running time)

Also you will need following python packages:

  pip install iso-639 abc unicodedata


# Running

Simply edit main.py and run it. You can train your own models or load
our pretrained models at
http://ufallab.ms.mff.cuni.cz/~kocmanek/lanidenn_models.tar.gz

Evaluated on following testset:
https://1drv.ms/t/s!Aq0goPMF_Lnlg-Jf47U0VGYAJL46qA?e=feDcS7


# Contact

In case of any problems please feel free to send an email to
kocmi@ufal.mff.cuni.cz


[1] Tom Kocmi and Ondřej Bojar. LanideNN: Multilingual Language Identification on Character Window. In EACL 2017
