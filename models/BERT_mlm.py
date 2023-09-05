from transformers import BertForMaskedLM, RobertaForMaskedLM
import torch.nn as nn
import torch
import torch.nn.functional as F

class BERT_mlm(nn.Module):

    def __init__(self, config, rels_num=0, device='cpu', id2name=None, encoder=None, tokenizer=None, init_by_cls=False):
        super(BERT_mlm, self).__init__()

        self.rels_num = rels_num
        self.device = device

        self.encoder = encoder.from_pretrained(config.bert_path)
        # self.encoder = BertForMaskedLM.from_pretrained(config.bert_path)
        self.config = config
        ## 1 is remain for replay-token, 4 is remain for entity marker
        self.encoder.resize_token_embeddings(self.encoder.config.vocab_size +1+ rels_num)
        self.output_size = config.encoder_output_size
        self.tokenizer = tokenizer
        self.replay_token_id = torch.LongTensor([self.encoder.config.vocab_size - rels_num]).to(device)

        if config.p_mask >= 0:
            self.p_mask = config.p_mask
            self.mlm_loss_fn = nn.CrossEntropyLoss()

        if config.encoder_type == 'roberta':
            if id2name:  ## init the new token embedding
                candidate_labels = id2name
                if init_by_cls:
                    ## attention avg of embedding
                    candidate_ids = tokenizer.batch_encode_plus(candidate_labels, add_special_tokens=False)['input_ids']
                    std1 = torch.std(self.encoder.roberta.embeddings.word_embeddings.weight.data[-rels_num:]).cpu().item()
                    for i in range(rels_num):
                        label_token_ids = candidate_ids[i]
                        ## [1,12,L,L]
                        attentions = self.encoder.roberta(input_ids=candidate_ids, output_attentions=True).attentions
                        label_emb_init = self.encoder.roberta.embeddings.word_embeddings.weight.data[label_token_ids].mean(
                            dim=0)
                        self.encoder.roberta.embeddings.word_embeddings.weight.data[-rels_num:][i] = label_emb_init
                    std2 = torch.std(self.encoder.roberta.embeddings.word_embeddings.weight.data[-rels_num:]).cpu().item()
                    print(
                        f"Init the new tokens embedding by attention-weighted-avarage of labels token-emb. The variance from {std1:.4} to {std2:.4} ")

                else:
                    candidate_ids = tokenizer.batch_encode_plus(candidate_labels, add_special_tokens=False)['input_ids']
                    std1 = torch.std(
                        self.encoder.roberta.embeddings.word_embeddings.weight.data[-rels_num:]).cpu().item()
                    for i in range(rels_num):
                        label_token_ids = candidate_ids[i]
                        label_emb_init = self.encoder.roberta.embeddings.word_embeddings.weight.data[
                            label_token_ids].mean(
                            dim=0)

                        self.encoder.roberta.embeddings.word_embeddings.weight.data[-rels_num:][i] = label_emb_init
                    std2 = torch.std(
                        self.encoder.roberta.embeddings.word_embeddings.weight.data[-rels_num:]).cpu().item()
                    self.label_emb_init = self.encoder.roberta.embeddings.word_embeddings.weight.data[-rels_num:]
                    print(
                        f"Init the new tokens embedding by avarage of labels token-emb. The variance from {std1:.4} to {std2:.4} ")
            else:
                print(f"Random init the new tokens embedding.")
        elif config.encoder_type =='bert':
            if id2name:  ## init the new token embedding
                candidate_labels = id2name
                if init_by_cls:
                    ## attention avg of embedding
                    candidate_ids = tokenizer.batch_encode_plus(candidate_labels, add_special_tokens=False)['input_ids']
                    std1 = torch.std(self.encoder.bert.embeddings.word_embeddings.weight.data[-rels_num:]).cpu().item()
                    for i in range(rels_num):
                        label_token_ids = candidate_ids[i]
                        ## [1,12,L,L]
                        attentions = self.encoder.bert(input_ids=candidate_ids, output_attentions=True).attentions
                        label_emb_init = self.encoder.bert.embeddings.word_embeddings.weight.data[label_token_ids].mean(
                            dim=0)
                        self.encoder.bert.embeddings.word_embeddings.weight.data[-rels_num:][i] = label_emb_init
                    std2 = torch.std(self.encoder.bert.embeddings.word_embeddings.weight.data[-rels_num:]).cpu().item()
                    print(
                        f"Init the new tokens embedding by attention-weighted-avarage of labels token-emb. The variance from {std1:.4} to {std2:.4} ")

                else:
                    candidate_ids = tokenizer.batch_encode_plus(candidate_labels, add_special_tokens=False)['input_ids']

                    std1 = torch.std(self.encoder.bert.embeddings.word_embeddings.weight.data[-rels_num:]).cpu().item()
                    for i in range(rels_num):
                        label_token_ids = candidate_ids[i]
                        label_emb_init = self.encoder.bert.embeddings.word_embeddings.weight.data[label_token_ids].mean(dim=0)
                        self.encoder.bert.embeddings.word_embeddings.weight.data[-rels_num:][i] = label_emb_init
                    std2 = torch.std(self.encoder.bert.embeddings.word_embeddings.weight.data[-rels_num:]).cpu().item()
                    self.label_emb_init = self.encoder.bert.embeddings.word_embeddings.weight.data[-rels_num:]
                    print(
                        f"Init the new tokens embedding by avarage of labels token-emb. The variance from {std1:.4} to {std2:.4} ")
            else:
                print(f"Random init the new tokens embedding.")
        else:
            if id2name:  ## init the new token embedding
                candidate_labels = id2name
                if init_by_cls:
                    ## attention avg of embedding
                    candidate_ids = tokenizer.batch_encode_plus(candidate_labels, add_special_tokens=False)['input_ids']
                    std1 = torch.std(self.encoder.deberta.embeddings.word_embeddings.weight.data[-rels_num:]).cpu().item()
                    for i in range(rels_num):
                        label_token_ids = candidate_ids[i]
                        ## [1,12,L,L]
                        attentions = self.encoder.deberta(input_ids=candidate_ids, output_attentions=True).attentions
                        label_emb_init = self.encoder.deberta.embeddings.word_embeddings.weight.data[label_token_ids].mean(
                            dim=0)
                        self.encoder.deberta.embeddings.word_embeddings.weight.data[-rels_num:][i] = label_emb_init
                    std2 = torch.std(self.encoder.deberta.embeddings.word_embeddings.weight.data[-rels_num:]).cpu().item()
                    print(
                        f"Init the new tokens embedding by attention-weighted-avarage of labels token-emb. The variance from {std1:.4} to {std2:.4} ")

                else:
                    candidate_ids = tokenizer.batch_encode_plus(candidate_labels, add_special_tokens=False)['input_ids']

                    std1 = torch.std(self.encoder.deberta.embeddings.word_embeddings.weight.data[-rels_num:]).cpu().item()
                    for i in range(rels_num):
                        label_token_ids = candidate_ids[i]
                        label_emb_init = self.encoder.deberta.embeddings.word_embeddings.weight.data[label_token_ids].mean(dim=0)
                        self.encoder.deberta.embeddings.word_embeddings.weight.data[-rels_num:][i] = label_emb_init
                    std2 = torch.std(self.encoder.deberta.embeddings.word_embeddings.weight.data[-rels_num:]).cpu().item()
                    self.label_emb_init = self.encoder.deberta.embeddings.word_embeddings.weight.data[-rels_num:]
                    print(
                        f"Init the new tokens embedding by avarage of labels token-emb. The variance from {std1:.4} to {std2:.4} ")
            else:
                print(f"Random init the new tokens embedding.")
        self.mask_id = self.tokenizer.mask_token_id
        self.to(self.device)

    def forward(self, input_ids, attention_mask, token_type_ids, return_mask_hidden=False, return_cls_hidden=False):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids, output_hidden_states=return_mask_hidden)

        ## [MASK] position

        x_idxs, y_idxs = torch.where(input_ids == self.mask_id)
        # x_idxs, y_idxs = torch.where(input_ids == 103)

        ## [B, 30552+rels_num]
        logits = out.logits[x_idxs, y_idxs]

        assert logits.shape[0] == out.logits.shape[0]

        if not return_mask_hidden:
            # [B,rels_num]
            return logits[:, -self.rels_num:]
        else:
            # [B,L,768]
            last_hidden_states = out.hidden_states[-1]
            ## [B,768]
            mask_hidden = last_hidden_states[x_idxs, y_idxs]
            assert mask_hidden.shape[0] == last_hidden_states.shape[0]
            ## [B,rels_num], [B,768]
            if not return_cls_hidden:
                return logits[:, -self.rels_num:], mask_hidden
            else:
                cls_hidden = last_hidden_states[:, 0, :]
                return logits[:, -self.rels_num:], mask_hidden, cls_hidden

    def mask_replay_forward(self, input_ids, attention_mask, token_type_ids, reason_input_ids, reason_attention_mask, reason_token_type_ids, return_mask_hidden=False,
                            return_cls_hidden=False):

        B = reason_input_ids.shape[0]
        L = reason_input_ids.shape[1]
        p_randn = torch.rand([B, L]).to(self.device)
        x_mask, y_mask = torch.where((p_randn < self.p_mask) * (reason_token_type_ids ==1))
        mask_reason_token_ids = reason_input_ids[x_mask, y_mask]
        reason_input_ids[x_mask, y_mask] = self.tokenizer.mask_token_id

        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids, output_hidden_states=return_mask_hidden)
        reason_out = self.encoder(input_ids=reason_input_ids, attention_mask=reason_attention_mask,
                            output_hidden_states=return_mask_hidden)

        ## [MASK] position
        x_idxs, y_idxs = torch.where((input_ids == self.tokenizer.mask_token_id))

        ## [B, 30552+rels_num]
        logits = out.logits[x_idxs, y_idxs]
        logits_mask = reason_out.logits[x_mask, y_mask]
        loss_mlm = self.mlm_loss_fn(logits_mask, mask_reason_token_ids)

        if loss_mlm.cpu().isnan() or self.p_mask <= 0:
            loss_mlm = torch.zeros([1])[0].to(self.device)

        assert logits.shape[0] == out.logits.shape[0]

        if not return_mask_hidden:
            # [B,rels_num]
            return logits[:, -self.rels_num:], loss_mlm
        else:
            # [B,L,768]
            last_hidden_states = out.hidden_states[-1]
            ## [B,768]
            mask_hidden = last_hidden_states[x_idxs, y_idxs]
            assert mask_hidden.shape[0] == last_hidden_states.shape[0]
            ## [B,rels_num], [B,768]
            if not return_cls_hidden:
                return logits[:, -self.rels_num:], mask_hidden, loss_mlm
            else:
                cls_hidden = last_hidden_states[:, 0, :]
                return logits[:, -self.rels_num:], mask_hidden, cls_hidden, loss_mlm

    def mlm_forward(self, mask_hidden):
        # [B,30552+rels_num]
        prediction_scores = self.encoder.cls(mask_hidden)
        # [B,rels_num]
        return prediction_scores[:, -self.rels_num:]

