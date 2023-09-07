from collections import Counter
import pandas as pd
import csv
import torch
import random
random.seed(2023)
import os
from .dataset_utils import normalize_word

class Twitter(object):
    def __init__(self, config, root_dir, tokenizer):
        self.config = config
        self.root_dir = root_dir
        self.preprocessed = True
        self._dev_ratio = 0.1
        self._test_ratio = 0.1
        self.max_length = config.max_length
        self.splits = ["train", "dev", "test"]
        self.model_type = config.model_type
        self.bert_path = config.bert_path
        self.tokenizer = tokenizer.from_pretrained(self.bert_path, additional_special_tokens=["[P00]"])
        self.templates = ['The following piece of text expresses a ' + self.tokenizer.mask_token + ' sentiment for the target [P00] [Entity] [P00].']
        # self.templates = ['The following piece of text expresses a ' +self.tokenizer.mask_token +' sentiment.']

    def bert_cls_process(self, data):
        ids = data[0]
        texts = data[1]
        describs = data[2]
        reasons = data[3]
        imgs = data[4]
        labels = data[5]
        targets = data[6]
        processed_data = []
        for i in range(len(ids)):
            instance = {}
            text = texts[i].replace('[Entity]', targets[i])

            sent_feature = self.tokenizer(
                self.tokenizer.sep_token.join([targets[i], text]),
                return_token_type_ids=True,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors='pt'
            )

            instance['features'] = sent_feature
            # instance['tokens'] = self.tokenizer.encode(text, padding='max_length', truncation=True, max_length=self.max_length)
            instance['label'] = labels[i]
            processed_data.append(instance)
        return processed_data

    def bert_mlm_process(self, data):
        SEP = '[SEP]'
        ids = data[0]
        texts = data[1]
        describs = data[2]
        reasons = data[3]
        imgs = data[4]
        labels = data[5]
        targets = data[6]
        processed_data = []
        for i in range(len(ids)):
            instance = {}
            text = texts[i]


            label = labels[i]
            all_candidates = []
            template = self.templates[0].replace('[Entity]', targets[i])

            all_candidates.append((template, text))
            sent_feature = self.tokenizer.batch_encode_plus(all_candidates, return_token_type_ids=True,
                                                                          max_length=self.max_length,
                                                                          truncation=True, padding="max_length",
                                                                          return_tensors='pt')


            instance['features'] = sent_feature
            # instance['tokens'] = self.tokenizer.encode(text, padding='max_length', truncation=True, max_length=self.max_length)
            instance['label'] = label
            processed_data.append(instance)
        return processed_data


    def fewshot_dataset(self, train_data):
        ids, texts, descrips, reasons, images, labels, targets = train_data
        sample_ids = random.sample(range(len(ids)), int(len(ids)*self.config.few_shot))
        ids = [ids[i] for i in sample_ids]
        texts = [texts[i] for i in sample_ids]
        descrips = [descrips[i] for i in sample_ids]
        reasons = [reasons[i] for i in sample_ids]
        images = [images[i] for i in sample_ids]
        labels = [labels[i] for i in sample_ids]
        targets = [targets[i] for i in sample_ids]

        return ids, texts, descrips, reasons, images, labels, targets

    def load_dataset(self):
        if self.config.dataset_name == "Twitter15":
            train = json.load(open(os.path.join(self.root_dir, 'tw15_train.json')))
            dev = json.load(open(os.path.join(self.root_dir, 'tw15_dev.json')))
            test = json.load(open(os.path.join(self.root_dir, 'tw15_test.json')))
        else:
            train = json.load(open(os.path.join(self.root_dir, 'tw17_train.json')))
            dev = json.load(open(os.path.join(self.root_dir, 'tw17_dev.json')))
            test = json.load(open(os.path.join(self.root_dir, 'tw17_test.json')))

        def get_data(data):
            ids = []
            labels = []
            texts = []
            images = []
            descrips = []
            reasons = []
            targets = []
            for item in data:
                target = item['target']
                targets.append(target)
                ids.append(str(item['ImageID']))
                images.append(str(item['ImageID']))
                labels.append(int(item['gt_label']) + 1)
                text = item['text'].replace('$T$', 'Jake Paul')

                texts.append(normalize_word(text))
                descrips.append(normalize_word(item['description']))
                reasons.append(normalize_word(item['reason']))
            return ids, texts, descrips, reasons, images, labels, targets

        train_data = get_data(train)
        if self.config.few_shot is not None:
            train_data = self.fewshot_dataset(train_data)
        print('The number of training is : {}'.format(len(train_data[0])))
        dev_data = get_data(dev)
        test_data = get_data(test)

        if self.model_type == 'cls':
            train_data = self.bert_cls_process(train_data)
            dev_data = self.bert_cls_process(dev_data)
            test_data = self.bert_cls_process(test_data)
        elif self.model_type == 'mlm':
            train_data = self.bert_mlm_process(train_data)
            dev_data = self.bert_mlm_process(dev_data)
            test_data = self.bert_mlm_process(test_data)

        else:
            raise NotImplementedError

        return train_data, dev_data, test_data

class MVSA(object):
    def __init__(self, config, root_dir, tokenizer):
        self.config = config
        self.root_dir = root_dir
        self.preprocessed = True
        self._dev_ratio = 0.1
        self._test_ratio = 0.1
        self.max_length = config.max_length
        self.splits = ["train", "dev", "test"]
        self.model_type = config.model_type
        self.bert_path = config.bert_path
        self.tokenizer = tokenizer.from_pretrained(self.bert_path)

        self.templates = ['The texts express a ' +self.tokenizer.mask_token +' sentiment.']

    def bert_cls_process(self, data):
        ids = data[0]
        texts = data[1]
        describs = data[2]
        reasons = data[3]
        imgs = data[4]
        labels = data[5]
        processed_data = []
        for i in range(len(ids)):
            instance = {}
            text = texts[i]
            label = labels[i]
            sent_feature = self.tokenizer(
                text,
                return_token_type_ids=True,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors='pt'
            )
            instance['features'] = sent_feature
            # instance['tokens'] = self.tokenizer.encode(text, padding='max_length', truncation=True, max_length=self.max_length)
            instance['label'] = label
            processed_data.append(instance)
        return processed_data

    def bert_mlm_process(self, data):
        SEP = '[SEP]'
        ids = data[0]
        texts = data[1]
        describs = data[2]
        reasons = data[3]
        imgs = data[4]
        labels = data[5]
        processed_data = []
        for i in range(len(ids)):
            instance = {}
            text = texts[i]
            describ = describs[i].strip().split()
            if len(describ) > self.config.reserve_length:
                describ = describ[:self.config.reserve_length]
            describ = ' '.join(describ)
            reason = reasons[i].strip().split()
            if len(reason) > self.config.reserve_length:
                reason = reason[:self.config.reserve_length]
            reason = ' '.join(reason)

            label = labels[i]
            all_candidates = []
            all_candidates.append((self.templates[0], describ+ SEP+ text))
            sent_feature = self.tokenizer.batch_encode_plus(all_candidates, return_token_type_ids=True,
                                                                          max_length=self.max_length,
                                                                          truncation=True, padding="max_length",
                                                                          return_tensors='pt')

            instance['features'] = sent_feature
            # instance['tokens'] = self.tokenizer.encode(text, padding='max_length', truncation=True, max_length=self.max_length)
            instance['label'] = label
            processed_data.append(instance)
        return processed_data


    def load_dataset(self):
        if self.config.dataset_name == 'single':
            train = json.load(open(os.path.join(self.root_dir, 'single_train.json')))
            dev = json.load(open(os.path.join(self.root_dir, 'single_dev.json')))
            test = json.load(open(os.path.join(self.root_dir, 'single_test.json')))
        elif self.config.dataset_name == 'multiple':
            train = json.load(open(os.path.join(self.root_dir, 'multiple_train.json')))
            dev = json.load(open(os.path.join(self.root_dir, 'multiple_dev.json')))
            test = json.load(open(os.path.join(self.root_dir, 'multiple_test.json')))
        else:
            raise NotImplementedError

        def get_data(data):
            ids = []
            labels = []
            texts = []
            images = []
            descrips = []
            reasons = []
            for item in data:
                ids.append(int(item['ImageID']))
                images.append(int(item['ImageID']))
                labels.append(item['gt_label'])
                text = item['text']

                texts.append(normalize_word(text))
                descrips.append(normalize_word(item['description']))
                reasons.append(normalize_word(item['reason']))
            return ids, texts, descrips, reasons, images, labels

        train_data = get_data(train)

        dev_data = get_data(dev)
        test_data = get_data(test)

        if self.model_type == 'cls':
            train_data = self.bert_cls_process(train_data)
            dev_data = self.bert_cls_process(dev_data)
            test_data = self.bert_cls_process(test_data)
        elif self.model_type == 'mlm':
            train_data = self.bert_mlm_process(train_data)
            dev_data = self.bert_mlm_process(dev_data)
            test_data = self.bert_mlm_process(test_data)
        else:
            raise NotImplementedError

        return train_data, dev_data, test_data

import json
class MVSAKnow(object):
    def __init__(self, config, root_dir, tokenizer):
        self.config = config
        self.root_dir = root_dir
        self.preprocessed = True
        self._dev_ratio = 0.1
        self._test_ratio = 0.1
        self.max_length = config.max_length
        self.splits = ["train", "dev", "test"]
        self.model_type = config.model_type
        self.bert_path = config.bert_path
        self.tokenizer = tokenizer.from_pretrained(self.bert_path)

        self.templates = ['The texts express a ' +self.tokenizer.mask_token +' sentiment.']

    def bert_cls_process(self, data):
        ids = data[0]
        texts = data[1]
        describs = data[2]
        reasons = data[3]
        imgs = data[4]
        labels = data[5]
        processed_data = []
        for i in range(len(ids)):
            instance = {}
            text = texts[i]
            label = labels[i]
            sent_feature = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors='pt'
            )
            instance['features'] = sent_feature
            # instance['tokens'] = self.tokenizer.encode(text, padding='max_length', truncation=True, max_length=self.max_length)
            instance['label'] = label
            processed_data.append(instance)
        return processed_data

    def bert_mlm_process(self, data):
        print('Using prompt-tuning ....')
        SEP = '[SEP]'
        ids = data[0]
        texts = data[1]
        describs = data[2]
        reasons = data[3]
        imgs = data[4]
        labels = data[5]
        processed_data = []
        for i in range(len(ids)):
            instance = {}
            text = texts[i]
            describ = describs[i].strip().split()
            if len(describ) > self.config.reserve_length:
                describ = describ[:self.config.reserve_length]
            describ = ' '.join(describ)
            reason = reasons[i].strip().split()
            if len(reason) > self.config.reserve_length:
                reason = reason[:self.config.reserve_length]
            reason = ' '.join(reason)

            label = labels[i]
            all_candidates = []
            all_candidates.append((text+SEP+describ, self.templates[0]))
            sent_feature = self.tokenizer.batch_encode_plus(all_candidates, return_token_type_ids=True,
                                                                          max_length=self.max_length,
                                                                          truncation=True, padding="max_length",
                                                                          return_tensors='pt')

            instance['features'] = sent_feature
            instance['label'] = label
            processed_data.append(instance)
        return processed_data

    def bert_mlmp_process(self, data):
        SEP = '[SEP]'
        ids = data[0]
        texts = data[1]
        describs = data[2]
        reasons = data[3]
        imgs = data[4]
        labels = data[5]
        # targets = data[6]
        processed_data = []
        sentiments = ['positive', 'neutral', 'negative']
        for i in range(len(ids)):
            instance = {}
            text = texts[i]
            describ = describs[i].strip().split()
            if len(describ) > self.config.reserve_length:
                describ = describ[:self.config.reserve_length]
            describ = ' '.join(describ)
            reason = reasons[i].strip().split()
            if len(reason) > self.config.reserve_length:
                reason = reason[:self.config.reserve_length]
            reason = ' '.join(reason)

            label = labels[i]
            all_candidates = []
            template = self.templates[0]
            answer_template = template.replace(self.tokenizer.mask_token, sentiments[label])

            all_candidates.append((text+ describ,template))

            sent_feature = self.tokenizer.batch_encode_plus(all_candidates, return_token_type_ids=True,
                                                                          max_length=self.max_length,
                                                                          truncation=True, padding="max_length",
                                                                          return_tensors='pt')

            all_candidates = []
            all_candidates.append((text + describ+answer_template, reason))
            reason_feature = self.tokenizer.batch_encode_plus(all_candidates, return_token_type_ids=True,
                                                                          max_length=self.max_length,
                                                                          truncation=True, padding="max_length",
                                                                          return_tensors='pt')

            if self.config.encoder_type == 'roberta':
                x_ids, y_ids = torch.where(reason_feature['input_ids']==self.tokenizer.eos_token_id)
                reason_feature['token_type_ids'][0][y_ids[-2]: y_ids[-1]] = 1

            instance['features'] = sent_feature
            instance['reason_features'] = reason_feature
            # instance['tokens'] = self.tokenizer.encode(text, padding='max_length', truncation=True, max_length=self.max_length)
            instance['label'] = label

            processed_data.append(instance)
        return processed_data


    def load_dataset(self):
        if self.config.dataset_name == 'single':
            train = json.load(open(os.path.join(self.root_dir, 'single_train.json')))
            dev = json.load(open(os.path.join(self.root_dir, 'single_dev.json')))
            test = json.load(open(os.path.join(self.root_dir, 'single_test.json')))
        elif self.config.dataset_name == 'multiple':
            train = json.load(open(os.path.join(self.root_dir, 'multiple_train.json')))
            dev = json.load(open(os.path.join(self.root_dir, 'multiple_dev.json')))
            test = json.load(open(os.path.join(self.root_dir, 'multiple_test.json')))
        else:
            raise NotImplementedError

        def get_data(data):
            ids = []
            labels = []
            texts = []
            images = []
            descrips = []
            reasons = []
            for item in data:
                ids.append(int(item['ImageID']))
                images.append(int(item['ImageID']))
                labels.append(item['gt_label'])
                text = item['text']

                texts.append(normalize_word(text))
                descrips.append(normalize_word(item['description']))
                reasons.append(normalize_word(item['reason']))
            return ids, texts, descrips, reasons, images, labels

        train_data = get_data(train)
        dev_data = get_data(dev)
        test_data = get_data(test)

        if self.model_type == 'cls':
            train_data = self.bert_cls_process(train_data)
            dev_data = self.bert_cls_process(dev_data)
            test_data = self.bert_cls_process(test_data)
        elif self.model_type == 'mlm':
            train_data = self.bert_mlm_process(train_data)
            dev_data = self.bert_mlm_process(dev_data)
            test_data = self.bert_mlm_process(test_data)
        elif self.model_type == 'mlmp':
            train_data = self.bert_mlmp_process(train_data)
            dev_data = self.bert_mlmp_process(dev_data)
            test_data = self.bert_mlmp_process(test_data)
        else:
            raise NotImplementedError

        return train_data, dev_data, test_data

class Twitter17Know(object):
    def __init__(self, config, root_dir, tokenizer):
        self.config = config
        self.root_dir = root_dir
        self.preprocessed = True
        self._dev_ratio = 0.1
        self._test_ratio = 0.1
        self.max_length = config.max_length
        self.splits = ["train", "dev", "test"]
        self.model_type = config.model_type
        self.bert_path = config.bert_path
        self.tokenizer = tokenizer.from_pretrained(self.bert_path, additional_special_tokens=["[P00]"])
        self.templates = ['The following piece of text expresses a ' + self.tokenizer.mask_token + ' sentiment for the target [P00] [Entity] [P00].']
        # self.templates = ['The following piece of text expresses a ' +self.tokenizer.mask_token +' sentiment.']

    def bert_cls_process(self, data):
        ids = data[0]
        texts = data[1]
        describs = data[2]
        reasons = data[3]
        imgs = data[4]
        labels = data[5]
        targets = data[6]
        processed_data = []
        for i in range(len(ids)):
            instance = {}
            text = texts[i].replace('[Entity]', targets[i])
            describ = describs[i].strip().split()
            if len(describ) > self.config.reserve_length:
                describ = describ[:self.config.reserve_length]
            describ = ' '.join(describ)
            reason = reasons[i].strip().split()
            if len(reason) > self.config.reserve_length:
                reason = reason[:self.config.reserve_length]
            reason = ' '.join(reason)

            sent_feature = self.tokenizer(
                self.tokenizer.sep_token.join([targets[i], text,describ]),
                return_token_type_ids=True,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors='pt'
            )

            instance['features'] = sent_feature
            # instance['tokens'] = self.tokenizer.encode(text, padding='max_length', truncation=True, max_length=self.max_length)
            instance['label'] = labels[i]
            processed_data.append(instance)
        return processed_data

    def bert_mlm_process(self, data):
        SEP = '[SEP]'
        ids = data[0]
        texts = data[1]
        describs = data[2]
        reasons = data[3]
        imgs = data[4]
        labels = data[5]
        targets = data[6]
        processed_data = []
        for i in range(len(ids)):
            instance = {}
            text = texts[i]
            describ = describs[i].strip().split()
            if len(describ) > self.config.reserve_length:
                describ = describ[:self.config.reserve_length]
            describ = ' '.join(describ)
            reason = reasons[i].strip().split()
            if len(reason) > self.config.reserve_length:
                reason = reason[:self.config.reserve_length]
            reason = ' '.join(reason)

            label = labels[i]
            all_candidates = []
            template = self.templates[0].replace('[Entity]', targets[i])

            all_candidates.append((template, text+ self.tokenizer.sep_token + describ))
            sent_feature = self.tokenizer.batch_encode_plus(all_candidates, return_token_type_ids=True,
                                                                          max_length=self.max_length,
                                                                          truncation=True, padding="max_length",
                                                                          return_tensors='pt')


            instance['features'] = sent_feature
            # instance['tokens'] = self.tokenizer.encode(text, padding='max_length', truncation=True, max_length=self.max_length)
            instance['label'] = label
            processed_data.append(instance)
        return processed_data

    def bert_mlmp_process(self, data):
        SEP = '[SEP]'
        ids = data[0]
        texts = data[1]
        describs = data[2]
        reasons = data[3]
        imgs = data[4]
        labels = data[5]
        targets = data[6]
        processed_data = []
        sentiments = ['positive', 'neutral', 'negative']
        for i in range(len(ids)):
            instance = {}
            text = texts[i]
            describ = describs[i].strip().split()
            if len(describ) > self.config.reserve_length:
                describ = describ[:self.config.reserve_length]
            describ = ' '.join(describ)
            reason = reasons[i].strip().split()
            if len(reason) > self.config.reserve_length:
                reason = reason[:self.config.reserve_length]
            reason = ' '.join(reason)

            label = labels[i]
            all_candidates = []
            template = self.templates[0].replace('[Entity]', targets[i])
            answer_template = template.replace(self.tokenizer.mask_token, sentiments[label])

            all_candidates.append((template, text+ describ))

            sent_feature = self.tokenizer.batch_encode_plus(all_candidates, return_token_type_ids=True,
                                                                          max_length=self.max_length,
                                                                          truncation=True, padding="max_length",
                                                                          return_tensors='pt')

            all_candidates = []
            all_candidates.append((answer_template + text + describ, reason))
            reason_feature = self.tokenizer.batch_encode_plus(all_candidates, return_token_type_ids=True,
                                                                          max_length=self.max_length,
                                                                          truncation=True, padding="max_length",
                                                                          return_tensors='pt')

            if self.config.encoder_type == 'roberta':
                x_ids, y_ids = torch.where(reason_feature['input_ids']==self.tokenizer.eos_token_id)
                reason_feature['token_type_ids'][0][y_ids[-2]: y_ids[-1]] = 1

            instance['features'] = sent_feature
            instance['reason_features'] = reason_feature
            # instance['tokens'] = self.tokenizer.encode(text, padding='max_length', truncation=True, max_length=self.max_length)
            instance['label'] = label

            processed_data.append(instance)
        return processed_data

    def fewshot_dataset(self, train_data):
        ids, texts, descrips, reasons, images, labels, targets = train_data
        sample_ids = random.sample(range(len(ids)), int(len(ids)*self.config.few_shot))
        ids = [ids[i] for i in sample_ids]
        texts = [texts[i] for i in sample_ids]
        descrips = [descrips[i] for i in sample_ids]
        reasons = [reasons[i] for i in sample_ids]
        images = [images[i] for i in sample_ids]
        labels = [labels[i] for i in sample_ids]
        targets = [targets[i] for i in sample_ids]

        return ids, texts, descrips, reasons, images, labels, targets

    def load_dataset(self):
        if self.config.task == "TW15K":
            train = json.load(open(os.path.join(self.root_dir, 'tw15_train.json')))
            dev = json.load(open(os.path.join(self.root_dir, 'tw15_dev.json')))
            test = json.load(open(os.path.join(self.root_dir, 'tw15_test.json')))
        else:
            train = json.load(open(os.path.join(self.root_dir, 'tw17_train.json')))
            dev = json.load(open(os.path.join(self.root_dir, 'tw17_dev.json')))
            test = json.load(open(os.path.join(self.root_dir, 'tw17_test.json')))

        def get_data(data):
            ids = []
            labels = []
            texts = []
            images = []
            descrips = []
            reasons = []
            targets = []
            for item in data:
                target = item['target']
                targets.append(target)
                ids.append(str(item['ImageID']))
                images.append(str(item['ImageID']))
                labels.append(int(item['gt_label']) + 1)
                text = item['text'].replace('$T$', 'Jake Paul')

                texts.append(normalize_word(text))
                descrips.append(normalize_word(item['description']))
                reasons.append(normalize_word(item['reason']))
            return ids, texts, descrips, reasons, images, labels, targets

        train_data = get_data(train)
        if self.config.few_shot is not None:
            train_data = self.fewshot_dataset(train_data)
        print('The number of training is : {}'.format(len(train_data[0])))
        dev_data = get_data(dev)
        test_data = get_data(test)

        if self.model_type == 'cls':
            train_data = self.bert_cls_process(train_data)
            dev_data = self.bert_cls_process(dev_data)
            test_data = self.bert_cls_process(test_data)
        elif self.model_type == 'mlm':
            train_data = self.bert_mlm_process(train_data)
            dev_data = self.bert_mlm_process(dev_data)
            test_data = self.bert_mlm_process(test_data)
        elif self.model_type == 'mlmp':
            train_data = self.bert_mlmp_process(train_data)
            dev_data = self.bert_mlmp_process(dev_data)
            test_data = self.bert_mlmp_process(test_data)
        else:
            raise NotImplementedError

        return train_data, dev_data, test_data