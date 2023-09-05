from configparser import ConfigParser
from transformers import BertTokenizer, BertForMaskedLM, BertModel, BertConfig, \
    RobertaTokenizer, RobertaForMaskedLM, RobertaModel, RobertaConfig, \
    DebertaTokenizer, DebertaForMaskedLM, DebertaModel, DebertaConfig, \
    AutoTokenizer, AutoModelForMaskedLM

_MODEL_CLASSES = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'encoder':BertForMaskedLM,
        'model': BertModel
    },
    'roberta': {
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'encoder':RobertaForMaskedLM,
        'model': RobertaModel
    },
    'deberta': {
        'config': DebertaConfig,
        'tokenizer': AutoTokenizer,
        'encoder': AutoModelForMaskedLM,
        'model': DebertaModel
    },
}

class Config(ConfigParser):
    def __init__(self, config_file):
        raw_config = ConfigParser()
        raw_config.read(config_file, encoding='utf-8')
        self.para_show=f"<Configs Parameters>\n[Path]:\t{config_file}\n"
        self.cast_values(raw_config)

    def cast_values(self, raw_config):
        for section in raw_config.sections():
            self.para_show += f"\n[{str(section)}]\n" 
            for key, value in raw_config.items(section):
                self.para_show += f"{str(key)}{'-'*(80-len(str(value))-len(str(key)))}{str(value)}\n"
                val = None
                if type(value) is str and value.startswith("[") and value.endswith("]"):
                    val = eval(value)
                    setattr(self, key, val)
                    continue
                for attr in ["getint", "getfloat", "getboolean"]:
                    try:
                        val = getattr(raw_config[section], attr)(key)
                        break
                    except:
                        val = value
                setattr(self, key, val)
        print(self.para_show)