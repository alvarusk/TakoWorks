import re
from pathlib import Path
from functools import lru_cache
from typing import Any, Optional

MODEL_NAME = "tohoku-nlp/bert-base-japanese-char-v3"
WEIGHT_PATH = Path(__file__).resolve().parent / "weight" / "punctuation_position_model.pth"
_TORCH: Optional[Any] = None


@lru_cache(maxsize=1)
def _get_components():
    try:
        import torch  # type: ignore
        from transformers import BertTokenizer, BertModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Falta torch/transformers para restaurar puntuación (pip install torch transformers)."
        ) from exc
    global _TORCH
    _TORCH = torch

    class PunctuationPredictor(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.dropout = torch.nn.Dropout(0.2)
            self.linear = torch.nn.Linear(768, 2)

        def forward(self, input_ids, attention_mask):
            last_hidden_state = self.base_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            return self.linear(self.dropout(last_hidden_state))

    # Si no quieres descargas en runtime, aquí podrías pasar local_files_only=True
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    base_model = BertModel.from_pretrained(MODEL_NAME)

    model = PunctuationPredictor(base_model)
    model.load_state_dict(torch.load(str(WEIGHT_PATH), map_location=torch.device("cpu")))
    model.eval()
    return tokenizer, model


def _insert_punctuation(input_ids_1d, comma_pos, period_pos, tokenizer):
    out = []
    for i, (c, p) in enumerate(zip(comma_pos, period_pos)):
        token_id = input_ids_1d[i].item()
        if token_id > 5:
            if i < len(input_ids_1d) - 1:
                tok = tokenizer.ids_to_tokens[token_id]
                if p:
                    out.append(tok + "。")
                elif c:
                    out.append(tok + "、")
                else:
                    out.append(tok)
            else:
                break
    return "".join(out)


def _capitalize_sentences(text: str) -> str:
    # Mayúscula al inicio y tras . ! ? (también tras salto de línea).
    def _cap(m: re.Match) -> str:
        return m.group(1) + m.group(2).upper()

    # incluye letras latinas con acentos
    return re.sub(r'(^|[\.\!\?]\s+|\n\s*)([a-záéíóúüñà-öø-ÿ])', _cap, text, flags=re.IGNORECASE)


def _postprocess(text: str) -> str:
    if not text:
        return text

    # 1) Quitar espacios antes de puntuación (ASCII + fullwidth)
    text = re.sub(r"\s+([\.,])", r"\1", text)
    text = re.sub(r"\s+([\?!])", r"\1", text)
    text = re.sub(r"\s+([，。、。？！；])", r"\1", text)

    # 2) ? .  -> ?   y   ! . -> !
    text = re.sub(r"\?\s*\.", "?", text)
    text = re.sub(r"!\s*\.", "!", text)

    # 3) ha (palabra completa) -> wa
    #    Nota: si estaba al inicio de frase, luego se capitaliza (Wa)
    text = re.sub(r"\bha\b", "wa", text, flags=re.IGNORECASE)

    # 4) Capitalización de frases (si hay latín; en japonés no afecta)
    text = _capitalize_sentences(text)

    return text


def process_long_text(text, max_length=256, comma_thresh=0.1, period_thresh=0.1):
    tokenizer, model = _get_components()

    # Este modelo inserta '、' y '。'. Quitamos los existentes para evitar dobles.
    text = text.replace("、", "").replace("。", "")
    result = ""

    for i in range(0, len(text), max_length):
        chunk = text[i : i + max_length]
        inputs = tokenizer(
            " ".join(list(chunk)),
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with _TORCH.no_grad():
            output = model(inputs.input_ids, inputs.attention_mask)
            output = _TORCH.sigmoid(output)

        comma_pos = output[0].detach().cpu().numpy().T[0] > comma_thresh
        period_pos = output[0].detach().cpu().numpy().T[1] > period_thresh
        result += _insert_punctuation(inputs.input_ids[0], comma_pos, period_pos, tokenizer)

    # Post-proceso global (fuera del bucle)
    return _postprocess(result)
