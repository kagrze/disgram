#!/bin/sh

DATASETS_DIR=./datasets
BASE_DIR=/tmp/disgram_exp

TRAINSET_FILE=${DATASETS_DIR}/WestburyLab.Wikipedia.Corpus_training.txt

if [ ! -f ${TRAINSET_FILE} ]; then
	python src/prepare.py --original_corpus=${DATASETS_DIR}/WestburyLab.Wikipedia.Corpus.txt --training_set=${TRAINSET_FILE}
fi

DICT_FILE=${DATASETS_DIR}/dict_mc_100_training.pkl.gz

if [ ! -f ${DICT_FILE} ]; then
	python src/build_dictionary.py --corpus_dir=${DATASETS_DIR}
fi

ADA_GRAM_REPO_DATA=https://raw.githubusercontent.com/sbos/AdaGram.jl/master/data

function get_data {
	mkdir -p ${DATASETS_DIR}/$1
	wget -nc ${ADA_GRAM_REPO_DATA}/$1/dataset.txt -P ${DATASETS_DIR}/$1
	wget -nc ${ADA_GRAM_REPO_DATA}/$1/key.txt -P ${DATASETS_DIR}/$1
}

get_data semeval-2007
get_data semeval-2010
get_data wwsi

mkdir -p ${BASE_DIR}

echo "Begin training. Logs in ${BASE_DIR}"

SENSE_NUMBER=5
WORD_DIM_NUM=300
TRAIN_EPOCH_NUM=3

MAIN_SCRIPT=src/disgram.py

# pretrain

WORK_DIR=${BASE_DIR}/pretrain
mkdir -p ${WORK_DIR}

python -u ${MAIN_SCRIPT} --mode=0 --prob_sense_assign=2 --sense_number=${SENSE_NUMBER} --word_dim_num=${WORD_DIM_NUM} --lr_train=0.1 \
--uncertain_penalty=-0.1 --work_dir=${WORK_DIR} --trainset_file=${TRAINSET_FILE} --dict_file=${DICT_FILE} &> ${WORK_DIR}/train.log

# dump parameters

python -u ${MAIN_SCRIPT} --mode=4 --prob_sense_assign=2 --sense_number=${SENSE_NUMBER} --word_dim_num=${WORD_DIM_NUM} \
 --dict_file=${DICT_FILE} --work_dir=${WORK_DIR} &> ${WORK_DIR}/dump.log

# train

INIT_MODEL_DIR=${WORK_DIR}/parameters_dump
WORK_DIR=${BASE_DIR}/train
mkdir -p ${WORK_DIR}

function train_model {
	python -u ${MAIN_SCRIPT} --mode=0 --prob_sense_assign=3 --sense_number=${SENSE_NUMBER} --ctx_softmax_temp=1.0,0.5_0.5_0.5_0.5_0.5 \
	--relaxed_one_hot=1 --work_dir=${WORK_DIR} --init_model_dir=${INIT_MODEL_DIR} --trainset_file=${TRAINSET_FILE} --dict_file=${DICT_FILE} \
	--train_epoch_num=${EPOCH_NUM} --lr_train=0.1,0.05,0.01,0.005,0.001 &> ${WORK_DIR}/train_${EPOCH_NUM}_epoch.log
}

for EPOCH_NUM in $(eval echo {1..${TRAIN_EPOCH_NUM}})
do
	if [ ${EPOCH_NUM} -ne 1 ]
	then
		unset INIT_MODEL_DIR
	fi
	train_model
done

# estimate marginal probabilites of the last epoch

python -u ${MAIN_SCRIPT} --mode=1 --prob_sense_assign=3 --sense_number=${SENSE_NUMBER} --work_dir=${WORK_DIR} \
--trainset_file=${TRAINSET_FILE} --train_epoch_num=${TRAIN_EPOCH_NUM} --dict_file=${DICT_FILE} --worker_number=12 &> ${WORK_DIR}/est_prob.log

# WSI experiments

TEST_WORDS=apple,fox,net,rock,plant,bank,mouse,palm,crane,paris,light,core,table

python -u ${MAIN_SCRIPT} --mode=3 --prob_sense_assign=3 --sense_number=${SENSE_NUMBER} --work_dir=${WORK_DIR} \
--trainset_file=${TRAINSET_FILE} --train_epoch_num=${TRAIN_EPOCH_NUM} --dict_file=${DICT_FILE} \
--wsi_dir=${DATASETS_DIR} --test_words=${TEST_WORDS} &> ${WORK_DIR}/wsi.log
