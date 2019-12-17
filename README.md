## Disambiguated skip-gram model

This is a TensorFlow 1.x implementation of the disambiguated skip-gram model introduced in [1].

### Instalation

The dependencies are listed in `requirements.txt`.
We recommend creating a Python 2.x virtual environment and then installing the required packages using:

```sh
pip install -r requirements.txt
```

### Training

To reproduce the results from [1] you first need to download the
[Wesbury Lab Wikipedia corpus](http://www.psych.ualberta.ca/~westburylab/downloads/westburylab.wikicorp.download.html)
and unpack it to the `datasets` directory. You can then run the script that will prepare
the training data, train the model and evaluate it against WSI benchmarks:

```sh
./prepare_train_and_test.sh
```

On a 24-core machine with 128 GB of RAM this script takes around 2.5 days to train the model.

### Disclaimer

This code is intended for replication of the published results and should not be used for commercial
purposes.

### References

[1] Karol Grzegorczyk, Marcin Kurdziel, [Disambiguated skip-gram model](http://aclweb.org/anthology/D18-1174), EMNLP 2018, [appendix](http://anthology.aclweb.org/attachments/D/D18/D18-1174.Attachment.zip)
