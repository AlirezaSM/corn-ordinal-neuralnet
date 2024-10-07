# Sebastian Raschka 2020
# coral_pytorch
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

MORPH2_INFO = {
    'TRAIN_CSV_PATH': '/home/raschka/code/github/ordinal-conditional/data/processed/morph2/morph2_train.csv',
    'TEST_CSV_PATH': '/home/raschka/code/github/ordinal-conditional/data/processed/morph2/morph2_valid.csv',
    'VALID_CSV_PATH': '/home/raschka/code/github/ordinal-conditional/data/processed/morph2/morph2_test.csv',
    'IMAGE_PATH': '/home/raschka/code/github/ordinal-conditional/data/processed/morph2/morph2-aligned-nose/jpg',
    'CLASS_COLUMN': 'age'}

MORPH2_BALANCED_INFO = {
    'TRAIN_CSV_PATH': '/home/raschka/code/github/ordinal-conditional/data/processed/morph2/morph2_train_balanced.csv',
    'TEST_CSV_PATH': '/home/raschka/code/github/ordinal-conditional/data/processed/morph2/morph2_valid_balanced.csv',
    'VALID_CSV_PATH': '/home/raschka/code/github/ordinal-conditional/data/processed/morph2/morph2_test_balanced.csv',
    'IMAGE_PATH': '/home/raschka/code/github/ordinal-conditional/data/processed/morph2/morph2-aligned-nose/jpg',
    'CLASS_COLUMN': 'age'}

AFAD_BALANCED_INFO = {
    'TRAIN_CSV_PATH': '/content/corn-ordinal-neuralnet/datasets/afad/afad_train_balanced.csv',
    'TEST_CSV_PATH': '/content/corn-ordinal-neuralnet/datasets/afad/afad_test_balanced.csv',
    'VALID_CSV_PATH': '/content/corn-ordinal-neuralnet/datasets/afad/afad_valid_balanced.csv',
    'IMAGE_PATH': '/content/AFAD-Full',
    'CLASS_COLUMN': 'age'}

AES_INFO = {
    'TRAIN_CSV_PATH': '/home/raschka/code/github/ordinal-conditional/data/processed/aes/aes_train_balanced.csv',
    'TEST_CSV_PATH': '/home/raschka/code/github/ordinal-conditional/data/processed/aes/aes_valid_balanced.csv',
    'VALID_CSV_PATH': '/home/raschka/code/github/ordinal-conditional/data/processed/aes/aes_test_balanced.csv',
    'IMAGE_PATH': '/home/raschka/code/github/ordinal-conditional/data/processed/aes/aes/jpg',
    'CLASS_COLUMN': 'beauty_scores'}

MNIST_INFO = {}