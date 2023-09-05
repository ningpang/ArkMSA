from data_utils.text_dataset import MVSA, Twitter, MVSAKnow, Twitter17Know
from data_utils.data_loader import get_data_loader, get_plus_data_loader
from models.BERT_cls import BERT_cls
from models.BERT_mlm import BERT_mlm
import json
from collections import Counter
from argparse import ArgumentParser
from config import Config, _MODEL_CLASSES
from framework import CLS_framework, MLM_framework, MLM_plus_framework

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = ArgumentParser(description="Config for MMSD")
    parser.add_argument('--config', default='config_dir/MVSAsKnow.ini')
    args = parser.parse_args()
    config = Config(args.config)
    paths = {'bert': config.bert_path,
             'roberta': config.roberta_path}
    config.bert_path = paths[config.encoder_type]
    config.few_shot = None

    encoder = _MODEL_CLASSES[config.encoder_type]

    processors = {'MVSA': MVSA(config, config.root_dir, encoder['tokenizer']),
                  'MVSAK': MVSAKnow(config, config.root_dir, encoder['tokenizer']),
                  'Twitter': Twitter(config, config.root_dir, encoder['tokenizer']),
                  'TW15K': Twitter17Know(config, config.root_dir, encoder['tokenizer']),
                  'TW17K': Twitter17Know(config, config.root_dir, encoder['tokenizer']),}
    data_processor = processors[config.task]
    config.save_name = '_'.join([config.dataset_name, config.encoder_type, config.model_type])



    train, dev, test = data_processor.load_dataset()
    if config.model_type == 'mlmp':
        train_loader = get_plus_data_loader(config, train, shuffle=True)
        dev_loader = get_plus_data_loader(config, dev)
        test_loader = get_plus_data_loader(config, test)
    else:
        train_loader = get_data_loader(config, train, shuffle=True)
        dev_loader = get_data_loader(config, dev)
        test_loader = get_data_loader(config, test)

    if config.model_type == 'cls':
        model = BERT_cls(config, num_label=3, model=encoder['model']).to(config.device)
        framework = CLS_framework(config)
    else:
        id2name = ["Negative Adverse Unfavorable Pessimistic Hostile Critical Dismal Gloomy Detrimental Defeatist Damaging",
                   "Neutral Impartial Unbiased Objective Uninvolved Indifferent Balanced Nonpartisan Disinterested Equitable Fair-minded",
                   "Positive Optimistic Favorable Encouraging Upbeat Good Constructive Affirmative Bright Promising Supportive"]
        tokenizer = data_processor.tokenizer
        model = BERT_mlm(config, rels_num= 3,
                         device=config.device,
                         id2name=id2name,
                         encoder = encoder['encoder'],
                         tokenizer= tokenizer,
                         init_by_cls=None)
        if config.model_type == 'mlmp':
            framework = MLM_plus_framework(config)
        else:
            framework = MLM_framework(config)

    framework.train(config, model, train_loader, dev_loader, test_loader)



    # with open('./save_results/{}_train_ids.json'.format(dataset_name), 'w') as f:
    #     json.dump(train[0], f)
    # with open('./save_results/{}_dev_ids.json'.format(dataset_name), 'w') as f:
    #     json.dump(dev[0], f)
    # with open('./save_results/{}_test_ids.json'.format(dataset_name), 'w') as f:
    #     json.dump(test[0], f)
    # with open('./save_results/{}_train_labels.json'.format(dataset_name), 'w') as f:
    #     json.dump(train[3], f)
    # with open('./save_results/{}_dev_labels.json'.format(dataset_name), 'w') as f:
    #     json.dump(dev[3], f)
    # with open('./save_results/{}_test_labels.json'.format(dataset_name), 'w') as f:
    #     json.dump(test[3], f)


