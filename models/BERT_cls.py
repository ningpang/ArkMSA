import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoModel, AutoConfig

class BERT_cls(nn.Module):
    def __init__(self, config, num_label=3, model= None, dropout=0.5, special_tokenizer=None):
        super(BERT_cls, self).__init__()
        # self.config = AutoConfig.from_pretrained(bert_model)
        # self.model = BertModel.from_pretrained(config.bert_path)

        self.model = model.from_pretrained(config.bert_path)
        if special_tokenizer is not None:
            self.model.resize_token_embeddings(len(special_tokenizer))

        self.activation = nn.Tanh()
        if config.encoder_type == 'bert':
            hidden_size = 768
        else:
            hidden_size = 1024
         # classification
        self.classifier = nn.Linear(hidden_size, num_label)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        bert_emb = self.dropout(outputs[1])
        logits = self.classifier(bert_emb)

        return logits