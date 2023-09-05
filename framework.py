import torch
import torch.nn as nn
import numpy as np
import logging
from sklearn import metrics

class CLS_framework(object):
    def __init__(self, config):
        self.config = config
        self.average = config.average
        # 这里本来想控制不同类别样本的损失权重，发现没什么用，如果需要可以在损失函数声明时加入
        self.weights = torch.tensor([0.3, 0.4, 0.3]).to(config.device)

    def train(self, config, model, train_loader, val_loader, test_loader):

        loss_function = nn.CrossEntropyLoss(weight=self.weights)  # 设置损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=5e-4)

        max_valid_f1 = 0
        best_epoch = 0
        for epoch in range(config.epochs):

            logging.info(f"epoch: {epoch} starts")
            sum_total = 0
            sum_loss = 0.0
            train_loss = 0.0
            model.train()

            for labels, input_ids, attention_mask, token_type_ids in train_loader:
                # 选取对应批次数据的输入和标签
                labels, input_ids, token_type_ids, attention_mask = labels.to(config.device), input_ids.to(config.device), attention_mask.to(config.device), token_type_ids.to(config.device)

                # 模型预测
                logits = model(input_ids, attention_mask, token_type_ids)
                # print(logits)
                loss = loss_function(logits, labels)
                # print(loss)

                optimizer.zero_grad()  # 梯度清零
                loss.backward()  # 计算梯度
                optimizer.step()  # 更新参数

                sum_total += logits.size(0)
                sum_loss += loss.item()
                train_loss = sum_loss / (sum_total/ config.batch_size)

            # valid_acc, valid_p, valid_r, valid_f1 = self.valid(config, model, val_loader)
            test_acc, macf1, wf1 = self.valid(config, model, test_loader)

            if macf1 > max_valid_f1:
                max_valid_f1 = macf1
                best_epoch = epoch
                # torch.save(model.state_dict(), './save_results/' + config.save_name + "_checkpoint.pt")

            # print(
            #     f"epoch: {epoch}, train loss: {train_loss:.4f}, valid accuracy: {valid_acc * 100:.2f}%,\
            #     valid precision: {valid_p * 100:.2f}%, valid recall: {valid_r * 100:.2f}%, valid F1: {valid_f1 * 100:.2f}%")
            print(
                f"epoch: {epoch}, test accuracy: {test_acc * 100:.2f}%,\
                            test mac-F1: {macf1 * 100:.2f}%, w-F1: {wf1 * 100:.2f}")
            print(f"Best epoch: {best_epoch}")

    def valid(self, config, model, val_loader):
        y_pred = []
        y_label = []
        model.eval()
        with torch.no_grad():
            for labels, input_ids, attention_mask, token_type_ids in val_loader:
                # 选取对应批次数据的输入和标签
                labels, input_ids, token_type_ids, attention_mask = labels.to(config.device), input_ids.to(
                    config.device), attention_mask.to(config.device), token_type_ids.to(config.device)

                # 模型预测
                logits = model(input_ids, attention_mask, token_type_ids)

                # 取分类概率最大的类别作为预测的类别
                y_hat = torch.tensor([torch.argmax(_) for _ in logits]).to(config.device)

                y_pred.append(y_hat.cpu().numpy())
                y_label.append(labels.cpu().numpy())

            y_pred = np.concatenate(y_pred)
            y_label = np.concatenate(y_label)

            acc = metrics.accuracy_score(y_true=y_label, y_pred=y_pred)
            macf1 = metrics.f1_score(y_true=y_label, y_pred=y_pred, average='macro')
            wf1 = metrics.f1_score(y_true=y_label, y_pred=y_pred, average='weighted')
            confusion_matrix = metrics.confusion_matrix(y_true=y_label, y_pred=y_pred)
            print("Confusion matrix:")
            print(confusion_matrix)

            return acc, macf1, wf1

    def test(self, config, model, test_loader, ckpt=None):
        if ckpt is not None:
            model.load_state_dict(torch.load(ckpt))
        results = self.valid(config, model, test_loader)
        return results

class MLM_framework(object):
    def __init__(self, config):
        self.config = config
        self.average = config.average
        # 这里本来想控制不同类别样本的损失权重，发现没什么用，如果需要可以在损失函数声明时加入
        self.weights = torch.tensor([0.3, 0.4, 0.3]).to(config.device)

    def train(self, config, model, train_loader, val_loader, test_loader):

        loss_function = nn.CrossEntropyLoss(weight=self.weights)  # 设置损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=5e-4)

        max_valid_f1 = 0
        best_epoch = 0
        for epoch in range(config.epochs):

            logging.info(f"epoch: {epoch} starts")
            sum_total = 0
            sum_loss = 0.0
            train_loss = 0.0
            model.train()

            for labels, input_ids, attention_mask, token_type_ids in train_loader:
                # 选取对应批次数据的输入和标签
                labels, input_ids, token_type_ids, attention_mask = labels.to(config.device), input_ids.to(config.device), attention_mask.to(config.device), token_type_ids.to(config.device)

                # 模型预测
                logits, mask_hidden = model(input_ids=input_ids,
                                            token_type_ids=token_type_ids,
                                            attention_mask=attention_mask,
                                            return_mask_hidden=True)
                loss = loss_function(logits, labels)

                optimizer.zero_grad()  # 梯度清零
                loss.backward()  # 计算梯度
                optimizer.step()  # 更新参数

                sum_total += logits.size(0)
                sum_loss += loss.item()
                train_loss = sum_loss / (sum_total/ config.batch_size)

            # valid_acc, valid_p, valid_r, valid_f1 = self.valid(config, model, val_loader)
            test_acc, macf1, wf1 = self.valid(config, model, test_loader)

            if macf1 > max_valid_f1:
                max_valid_f1 = macf1
                best_epoch = epoch
                # torch.save(model.state_dict(), './save_results/' + config.save_name + "_checkpoint.pt")


            # print(
            #     f"epoch: {epoch}, train loss: {train_loss:.4f}, valid accuracy: {valid_acc * 100:.2f}%,\
            #     valid precision: {valid_p * 100:.2f}%, valid recall: {valid_r * 100:.2f}%, valid F1: {valid_f1 * 100:.2f}%")
            print(
                f"epoch: {epoch}, test accuracy: {test_acc * 100:.2f}%,\
                            test mac-F1: {macf1 * 100:.2f}%, w-F1: {wf1 * 100:.2f}")
            print(f"Best epoch: {best_epoch}")

    def valid(self, config, model, val_loader):
        y_pred = []
        y_label = []
        model.eval()
        with torch.no_grad():
            for labels, input_ids, attention_mask, token_type_ids in val_loader:
                # 选取对应批次数据的输入和标签
                labels, input_ids, token_type_ids, attention_mask = labels.to(config.device), input_ids.to(
                    config.device), attention_mask.to(config.device), token_type_ids.to(config.device)

                # 模型预测
                logits, mask_hidden = model(input_ids=input_ids,
                                                  token_type_ids=token_type_ids,
                                                  attention_mask=attention_mask,
                                                  return_mask_hidden=True)

                # 取分类概率最大的类别作为预测的类别
                y_hat = torch.tensor([torch.argmax(_) for _ in logits]).to(config.device)

                y_pred.append(y_hat.cpu().numpy())
                y_label.append(labels.cpu().numpy())

            y_pred = np.concatenate(y_pred)
            y_label = np.concatenate(y_label)

            acc = metrics.accuracy_score(y_true=y_label, y_pred=y_pred)
            # p = metrics.precision_score(y_true=y_label, y_pred=y_pred, average=self.average)
            # r = metrics.recall_score(y_true=y_label, y_pred=y_pred, average=self.average)
            macf1 = metrics.f1_score(y_true=y_label, y_pred=y_pred, average='macro')
            wf1 = metrics.f1_score(y_true=y_label, y_pred=y_pred, average='weighted')
            confusion_matrix = metrics.confusion_matrix(y_true=y_label, y_pred=y_pred)
            print("Confusion matrix:")
            print(confusion_matrix)

            return acc, macf1, wf1

    def test(self, config, model, test_loader, ckpt=None):
        if ckpt is not None:
            model.load_state_dict(torch.load(ckpt))
        results = self.valid(config, model, test_loader)
        return results

class MLM_plus_framework(object):
    def __init__(self, config):
        self.config = config
        self.average = config.average
        # 这里本来想控制不同类别样本的损失权重，发现没什么用，如果需要可以在损失函数声明时加入
        self.weights = torch.tensor([0.3, 0.4, 0.3]).to(config.device)

    def train(self, config, model, train_loader, val_loader, test_loader):

        loss_function = nn.CrossEntropyLoss(weight=self.weights)  # 设置损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=5e-4)

        max_valid_f1 = 0
        best_epoch = 0
        for epoch in range(config.epochs):

            logging.info(f"epoch: {epoch} starts")
            sum_total = 0
            sum_loss = 0.0
            train_loss = 0.0
            model.train()

            for labels, input_ids, attention_mask, token_type_ids, reason_input_ids, reason_attention_mask, reason_token_type_ids in train_loader:
                # 选取对应批次数据的输入和标签
                labels, input_ids, token_type_ids, attention_mask = labels.to(config.device), input_ids.to(config.device), attention_mask.to(config.device), token_type_ids.to(config.device)
                reason_input_ids, reason_token_type_ids, reason_attention_mask = reason_input_ids.to(config.device), reason_attention_mask.to(config.device), reason_token_type_ids.to(config.device)


                # 模型预测
                logits, mask_hidden, mlm_loss = model.mask_replay_forward(input_ids=input_ids,
                                            token_type_ids=token_type_ids,
                                            attention_mask=attention_mask,
                                            reason_input_ids=reason_input_ids,
                                            reason_token_type_ids=reason_token_type_ids,
                                            reason_attention_mask=reason_attention_mask,
                                            return_mask_hidden=True)
                loss = loss_function(logits, labels)+config.alpha*mlm_loss

                optimizer.zero_grad()  # 梯度清零
                loss.backward()  # 计算梯度
                optimizer.step()  # 更新参数

                sum_total += logits.size(0)
                sum_loss += loss.item()
                train_loss = sum_loss / (sum_total/ config.batch_size)

            # valid_acc, valid_p, valid_r, valid_f1 = self.valid(config, model, val_loader)
            test_acc,macf1, wf1 = self.valid(config, model, test_loader)

            if macf1 > max_valid_f1:
                max_valid_f1 = macf1
                best_epoch = epoch
                torch.save(model.state_dict(), './save_results/' + config.save_name + "_checkpoint.pt")
                # torch.save(model.state_dict(), model.name + "_checkpoint.pt")  # save the ckpt with model name
            # print(
            #     f"epoch: {epoch}, train loss: {train_loss:.4f}, valid accuracy: {valid_acc * 100:.2f}%,\
            #     valid precision: {valid_p * 100:.2f}%, valid recall: {valid_r * 100:.2f}%, valid F1: {valid_f1 * 100:.2f}%")
            print(
                f"epoch: {epoch}, test accuracy: {test_acc * 100:.2f}%,\
                            test macF1: {macf1 * 100:.2f}%, wF1: {wf1 * 100:.2f}%")
            print(f"Best epoch: {best_epoch}")

    def valid(self, config, model, val_loader):
        y_pred = []
        y_label = []
        model.eval()
        with torch.no_grad():
            for labels, input_ids, attention_mask, token_type_ids, _, _, _ in val_loader:
                # 选取对应批次数据的输入和标签
                labels, input_ids, token_type_ids, attention_mask = labels.to(config.device), input_ids.to(
                    config.device), attention_mask.to(config.device), token_type_ids.to(config.device)

                # 模型预测
                logits, mask_hidden = model(input_ids=input_ids,
                                                  token_type_ids=token_type_ids,
                                                  attention_mask=attention_mask,
                                                  return_mask_hidden=True)

                # 取分类概率最大的类别作为预测的类别
                y_hat = torch.tensor([torch.argmax(_) for _ in logits]).to(config.device)

                y_pred.append(y_hat.cpu().numpy())
                y_label.append(labels.cpu().numpy())

            y_pred = np.concatenate(y_pred)
            y_label = np.concatenate(y_label)

            acc = metrics.accuracy_score(y_true=y_label, y_pred=y_pred)

            macf1 = metrics.f1_score(y_true=y_label, y_pred=y_pred, average='macro')
            wf1 = metrics.f1_score(y_true=y_label, y_pred=y_pred, average='weighted')
            confusion_matrix = metrics.confusion_matrix(y_true=y_label, y_pred=y_pred)
            print("Confusion matrix:")
            print(confusion_matrix)

            return acc, macf1, wf1

    def test(self, config, model, test_loader, ckpt=None):
        if ckpt is not None:
            model.load_state_dict(torch.load(ckpt))
        results = self.valid(config, model, test_loader)
        return results
