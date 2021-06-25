from transformers import AutoModel


def fetch_model_str(backbone: str) -> str:
    return '{}-base-cased'.format(backbone)


def fetch_tokenizer(backbone: str = 'distilbert'):
    from transformers import DistilBertTokenizerFast as TokenizerFast
    return TokenizerFast.from_pretrained(fetch_model_str(backbone))


def fetch_pretrained_bert_model(backbone: str = 'distilbert') -> AutoModel:
    from transformers import DistilBertModel as Model
    return Model.from_pretrained(fetch_model_str(backbone))
