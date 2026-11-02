"""Sentence splitting for streamed speech: the first sentence is synthesized and sent while the
rest is still being generated, so time to first audio does not grow with the length of the input."""

from __future__ import annotations

import re

# A sentence ends at . ! ? … (optionally followed by closing quotes/brackets) and whitespace.
_END = re.compile(r"(?<=[.!?…])[\"'”»)\]]*\s+")
# Common abbreviations that end with a period but do not end a sentence (EN/FR/DE).
_ABBREV = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "vs.", "etc.", "e.g.", "i.e.", "approx.",
           "m.", "mme.", "mlle.", "env.", "p.ex.", "z.b.", "bzw.", "ca.", "nr.", "no."}


# Function words per language. Shared words (de, la, que...) count for every language that has
# them; the distinctive ones decide.
_STOPWORDS = {
    "en": set("the and is are you your to of in that it for with this was have be on not at we i".split()),
    "fr": set("le la les et est sont tu ton ta vous de des du un une que qui pour avec ce cette "
              "pas je nous il elle au aux en".split()),
    "de": set("der die das und ist sind du dein ich nicht mit ein eine zu den dem für auf wir "
              "es sie im auch noch".split()),
    "es": set("el los las y es son para con por una del al lo se más pero como está muy también "
              "que de la en un yo tu".split()),
    "it": set("il lo gli le e è sono per con non che di della del una un ma come anche molto io".split()),
    "pt": set("o os as e é são para com não que de do da um uma mas como também muito em eu você".split()),
    "nl": set("de het een en is zijn niet met voor op van dat je ik we maar ook".split()),
}
_ACCENTS = {"fr": "àâçéèêëîïôùûœ", "de": "äöüß", "es": "ñ¿¡áíóú", "pt": "ãõáíóú", "it": "ìòè"}


def detect_language(text: str, allowed: list[str] | None = None) -> str:
    """Cheap guess from function words and accents (EN, FR, DE, ES, IT, PT, NL), good enough to
    pick a TTS voice and pronunciation for an assistant reply. Only languages in `allowed` (the
    model's languages, in order) are considered; when the text gives no clue, or the model's
    languages are ones this cannot recognize, the model's first language wins (English if none)."""
    low = text.lower()
    words = [w.split("'")[-1] for w in re.findall(r"[a-zàáâãäçéèêëíìîïñóòôõöúùûüÿœæß']+", low)]
    scores = {lang: sum(w in sw for w in words) for lang, sw in _STOPWORDS.items()}
    for lang, chars in _ACCENTS.items():
        scores[lang] += 2 * sum(low.count(c) for c in chars)
    order = list(allowed) if allowed else list(_STOPWORDS)
    candidates = [lang for lang in order if lang in scores]
    if not candidates:
        return order[0]
    best = max(candidates, key=lambda lang: (scores[lang], -order.index(lang)))
    return best if scores[best] > 0 else order[0]


def split_sentences(text: str, min_chars: int = 20, max_chars: int = 400) -> list[str]:
    """Split into sentences; merge fragments shorter than min_chars into the next one, and cut
    sentences longer than max_chars at a comma or space so no single chunk runs away."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    raw: list[str] = []
    start = 0
    for m in _END.finditer(text):
        piece = text[start:m.start()].strip()
        last_word = piece.rsplit(" ", 1)[-1].lower()
        if last_word in _ABBREV or re.fullmatch(r"\d+\.", last_word):
            continue
        raw.append(piece)
        start = m.end()
    if text[start:].strip():
        raw.append(text[start:].strip())

    merged: list[str] = []
    buf = ""
    for s in raw:
        buf = f"{buf} {s}".strip() if buf else s
        if len(buf) >= min_chars:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] = f"{merged[-1]} {buf}"
        else:
            merged.append(buf)

    out: list[str] = []
    for s in merged:
        while len(s) > max_chars:
            cut = max(s.rfind(", ", 0, max_chars), s.rfind("; ", 0, max_chars))
            if cut < max_chars // 3:
                cut = s.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            out.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
        if s:
            out.append(s)
    return out
