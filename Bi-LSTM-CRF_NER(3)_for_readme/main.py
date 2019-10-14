import ner_data_utils
from model import NERTagger

import config


#dataset
print("\n[ train data set ]")
train = ner_data_utils.data_set("train_data")
print("\n[ valid data set ]")
valid = ner_data_utils.data_set("valid_data")
print("\n[ test data set ]")
test = ner_data_utils.data_set("test_data")
print("\n[ real test data set ]")
test_real = ner_data_utils.data_set("test_data_real")


# NERTagger
model = NERTagger(mdl_path=config.mdl_path,
                  final_mdl_path=config.final_mdl_path,
                  num_units=config.num_units,
                  optimizer=config.optimizer,
                  learning_rate=config.learning_rate,
                  epochs=config.epochs,
                  batch_size=config.batch_size,
                  activation=config.activation,
                  keep_probability=config.keep_probability,
                  keep_probability_d=config.keep_probability_d)

# print("======================== Train ===========================")
# model.train(train, valid)
#
# print("=================== Test (for eval) ======================")
# model.test(test)
# model.metrics("test")

print("================ Test (without NE tags) ===================")
model.test_real(test_real)
model.result()