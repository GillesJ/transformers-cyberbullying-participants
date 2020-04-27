# AMiCA Cyberbullying Participants Transformer-based Classification
Post-level cyberbullying participant text classification experiments for the AMiCA Cyberbullying dataset.
Part of NLE submission.

## Docker container
on weoh containername = cbpart

TODO POST CONTAINER ENV to DOCKERHUB WITH CODE BUT LEAVE OUT DATA

## Install (manual)
This depends on the package [SimpleTransformers](https://github.com/ThilinaRajapakse/simpletransformers).

`pipenv install --python 3.7.5 simpletransformers torch pandas torchvision`

or with Pipfile: `pipenv install`

! SET ENV_VAR `CUDA_VISIBLE_DEVICES`: `export CUDA_VISIBLE_DEVICES=0`

##Tensorboard for checking loss and accuracy
You need to install Tensorflow to use Tensorboard on your client (simpletransformers actually uses the PyTorch-fork tensorboardx for its tensorboard output and does not depend on TF.):
First install a python version compatible with TF (latest=3.7.5 as of writing):
`pyenv install 3.7.5`
Now install TensorFlow
`pipx install --python /home/gilles/.pyenv/versions/3.7.5/bin/python tensorflow`
Now run the Tensorboard command on the run dir which was created during training:
`tensorboard`

## Utility commands
- Remove large output files: checkpoints and epoch binaries (DELETES BACKUP)
1. Change to experiment dir: `cd RUNDIR`
2. Check what you are removing `find . \( -name "epoch*" -or -name "checkpoint*" \) -exec echo "{}" \;`
3. Remove it `find . \( -name "epoch*" -or -name "checkpoint*" \) -exec rm -r "{}" \; -prune`

## Model and experiment notes

- Choice of Dutch language model: 2 available pre-trained options: RobBERT and BERTje

## Results

**English**:

Best model in current manuscript:
- CASCADE: f1_macro xval: 55.19 f1_macro holdout: 55.67

Transformer experiments:
- BEST en_roberta-base_epochs-4_seq-256 f1_macro xval: 0.602471 f1_macro holdout 0.5583992223036653
- en_roberta-base_epochs-16_seq-256: f1_macro xval: 0.578473 f1_macro holdout 0.6237261658903018
- en_roberta-base_epochs-8_seq-256 f1_macro xval: 0.578601 f1_macro holdout 0.6085872969255104
- *en_roberta-base_epochs-2_seq-256 f1_macro xval: 0.562697 f1_macro holdout 0.5705378932281459
- *batch_size 64: en_roberta-base_epochs-8_seq-256 f1_macro xval: 0.598845 f1_macro holdout 0.6072391246267734
- *batch_size 64: en_roberta-base_epochs-4_seq-256 f1_macro xval: 0.601382 f1_macro holdout 0.59696208327875

- en_bert-base-uncased_epochs-16_seq-256 f1_macro xval: 0.581963 f1_macro holdout 0.5985772069532231
- en_bert-base-uncased_epochs-8_seq-256 f1_macro xval: 0.584632 f1_macro holdout 0.6008079707237194
- en_bert-base-uncased_epochs-4_seq-256 f1_macro xval: 0.594775 f1_macro holdout 0.6004311799287892

- en_xlnet-base-cased_epochs-16_seq-256: f1_macro xval: 0.549157 f1_macro holdout 0.5384509454843583
- en_xlnet-base-cased_epochs-8_seq-256 f1_macro xval: 0.568525 f1_macro holdout 0.5987210196341299
- en_xlnet-base-cased_epochs-4_seq-256 f1_macro xval: 0.553785 f1_macro holdout 0.5202046860611904

- EN_XLM-ROBERTA-BASE_EPOCHS-16_SEQ-256: f1_macro xval: 0.504973 f1_macro holdout 0.5489888923408954
- BAD Roberta-large 4 epochs: Predicts only majority class in xval and holdout
- BAD en_bert-large-uncased_epochs-16_seq-256: INCOMPLETE RUN but HOLDOUT BAD predicts only majority: f1_macro xval: nan f1_macro holdout 0.2442530028341356

**Dutch**:

Best model in current manuscript:
- Cascading + FS: f1_macro xval: 0.5153 f1_macro holdout 0.5437

Transformer experiments:
- BEST nl_pdelobelle/robBERT-base_epochs-4_seq-256 f1_macro xval: 0.549166 f1_macro holdout 0.5673304389312812
- NL_ROBBERT-BASE_EPOCHS-16_SEQ-256 f1_macro xval: 0.530336 f1_macro holdout 0.5640263470933217
- NL_ROBBERT-BASE_EPOCHS-8_SEQ-256 f1_macro xval: 0.531273 f1_macro holdout 0.5347052061666936
- NL_ROBBERT-BASE_EPOCHS-16_SEQ-128 f1_macro xval: 0.522126 f1_macro holdout 0.5275768783033994

- NL_BERT-BASE-DUTCH-CASED_EPOCHS-16_SEQ-256: f1_macro xval: 0.508923 f1_macro holdout 0.5223214958407013
- NL_BERT-BASE-DUTCH-CASED_EPOCHS-8_SEQ-256: f1_macro xval: 0.514197 f1_macro holdout 0.5258056806369584
- nl_bert-base-dutch-cased_epochs-4_seq-256 f1_macro xval: 0.520648 f1_macro holdout 0.5295773755652862
