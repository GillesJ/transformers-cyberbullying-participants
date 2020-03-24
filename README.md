# AMiCA Cyberbullying Participants Transformer-based Classification
Post-level cyberbullying participant text classification experiments for the AMiCA Cyberbullying dataset.
Part of NLE submission.

## Install
This depends on the package [SimpleTransformers](https://github.com/ThilinaRajapakse/simpletransformers).

`pipenv install --python 3.7.5 simpletransformers torch pandas`

##Tensorboard for checking loss and accuracy
You need to install Tensorflow to use Tensorboard on your client (simpletransformers actually uses the PyTorch-fork tensorboardx for its tensorboard output and does not depend on TF.):
First install a python version compatible with TF (latest=3.7.5 as of writing):
`pyenv install 3.7.5`
Now install TensorFlow
`pipx install --python /home/gilles/.pyenv/versions/3.7.5/bin/python tensorflow`
Now run the Tensorboard command on the run dir which was created during training:
`tensorboard`

## Utility
- Remove large output files: checkpoints and epoch binaries (DELETES BACKUP)
1. Change to experiment dir: `cd RUNDIR`
2. Check what you are removing `find . \( -name "epoch*" -or -name "checkpoint*" \) -exec echo "{}" \;`
3. Remove it `find . \( -name "epoch*" -or -name "checkpoint*" \) -exec rm -r "{}" \; -prune`
