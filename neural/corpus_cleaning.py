"""Deterministic source-aware text cleaning before Mandarin annotation."""

from html import unescape
from html.entities import html5
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tempfile
import unicodedata


RENDERED_SOURCES = frozenset(("forum", "zhihu", "fineweb"))
HAN_RANGE = "\u3400-\u4dbf\u4e00-\u9fff"
URL = re.compile(r"(?:https?://|www\.)[A-Za-z0-9][A-Za-z0-9._~:/?#\[\]@!$&()*+,;=%+\-]*", re.I)
ASCII_CONTROLS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
ENTITY = re.compile(r"&(?:#[0-9]+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);")
ANCHOR = re.compile(r"</?a(?=[\s/>])[^<>]*>", re.I)
BREAK = re.compile(r"<br\s*/?\s*>", re.I)
HAN_REPEAT = re.compile(r"([" + HAN_RANGE + r"])\1{31,}")
PHRASE_REPEAT = re.compile(r"([^\s]{2,16}?)\1{7,}")
CHINESE_PHRASE = re.compile(r"[" + HAN_RANGE + r"]+[，。！？、；：,.!?;:]*")


def make_simplifier():
    """Load reviewed phrases ahead of installed OpenCC data, returning assets.

    Only configuration paths are resolved; upstream dictionaries stay in their
    installed location. The returned asset paths pin every conversion input.
    """
    import opencc

    profile = Path(__file__).with_name("opencc-t2s-reviewed.json")
    config = json.loads(profile.read_text(encoding="utf-8"))
    installed = Path(opencc.OpenCC("t2s").config).resolve().parent
    assets = {profile.name: profile}

    dictionaries = [config["normalization"][0]["dict"],
                    *config["conversion_chain"][0]["dict"]["dicts"]]
    for dictionary in dictionaries:
        filename = dictionary["file"]
        directory = profile.parent if dictionary["type"] == "text" else installed
        path = (directory / filename).resolve(strict=True)
        dictionary["file"] = str(path)
        assets[filename] = path
    # OpenCC reads standard JSON files. Its wheel's compiled search directory
    # can differ from the runtime install path, hence these absolute paths.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as temporary:
        json.dump(config, temporary, ensure_ascii=False)
        temporary.flush()
        converter = opencc.OpenCC(temporary.name)
    return converter, assets


def decode_entities(text):
    # html.unescape alone accepts partial names without a semicolon, turning
    # URL parameters such as &copyfr= into visible text such as ©fr=.
    def replace(match):
        entity = match.group()
        return unescape(entity) if entity.startswith("&#") or entity[1:] in html5 else entity

    return ENTITY.sub(replace, text)


class _AnchorAttributes(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.platform_link = False

    def handle_starttag(self, tag, attrs):
        self.platform_link = tag == "a" and any(name in ("namecard", "usercard") for name, _ in attrs)


def _platform_links(text):
    """Remove only identified platform wrappers, including unclosed ones."""
    stack = []

    def replace(match):
        tag = match.group()
        if tag.startswith("</"):
            return "" if stack and stack.pop() else tag
        parser = _AnchorAttributes()
        parser.feed(tag)
        if not tag.rstrip().endswith("/>"):
            stack.append(parser.platform_link)
        return "" if parser.platform_link else tag

    return ANCHOR.sub(replace, text)


def _invalid_unicode(text):
    found = set()
    for char in text:
        if char == "\ufffd":
            found.add("replacement_character")
        elif unicodedata.category(char) == "Co":
            found.add("private_use_character")
        elif "\ud800" <= char <= "\udfff":
            found.add("surrogate_character")
    return found


def _repetition_reason(text):
    length = sum(not char.isspace() for char in text)
    if not length:
        return None
    # These signatures occurred as nearly entire forum posts. A discussion
    # merely mentioning a damaged encoding or a short laugh is retained.
    for pattern, name in ((r"烫{16,}", "tang"), (r"(?:锟斤拷){4,}", "kunjinkao")):
        if any(len(match.group()) / length >= .3 for match in re.finditer(pattern, text)):
            return "repeated_mojibake:" + name
    if any(len(match.group()) / length >= .5 for match in HAN_REPEAT.finditer(text)):
        return "excessive_repetition:han_character"
    for match in PHRASE_REPEAT.finditer(text):
        unit = match[1]
        if (CHINESE_PHRASE.fullmatch(unit) and len(set(re.findall("[" + HAN_RANGE + "]", unit))) >= 2
                and len(match.group()) >= 32 and len(match.group()) / length >= .5):
            return "excessive_repetition:short_phrase"
    return None


def clean_document(source, text, t2s):
    """Return (text, changed-operation IDs, quarantine reason or None).

    Raw documents remain the caller's responsibility. Quarantine never guesses
    replacement characters. Surrogates return empty text because they cannot be
    serialized safely as UTF-8; other quarantined text remains inspectable.
    This function does not reject long documents, foreign text or mixed script.
    """
    operations = []
    invalid = _invalid_unicode(text)
    if "surrogate_character" in invalid:
        return "", ["surrogate_text_not_serialized"], "invalid_unicode:" + ",".join(sorted(invalid))

    def change(value, operation):
        nonlocal text
        if value != text:
            operations.append(operation)
            text = value

    if source in RENDERED_SOURCES:
        change(decode_entities(text), "html_entities_decoded")
        change(_platform_links(text), "platform_link_wrappers_removed")
        change(BREAK.sub("\n", text), "html_breaks_to_newlines")
    if source == "fineweb":
        change(text.replace("$P$", "\n"), "fineweb_paragraph_markers_to_newlines")
    # Entity decoding may reveal damaged or private-use characters too.
    invalid.update(_invalid_unicode(text))
    # Remove the control code points while preserving their text boundaries.
    change(text.replace("\r\n", "\n").replace("\r", "\n"), "carriage_returns_removed")
    change(ASCII_CONTROLS.sub(" ", text), "ascii_controls_removed")
    change(re.sub(r"[\u200b\ufeff]", " ", text), "invisible_separators_to_spaces")
    # ASCII-only URL matching deliberately stops before adjacent Chinese text.
    change(URL.sub(" ", text), "urls_to_spaces")
    change(re.sub(r"[ \t　]+", " ", text), "horizontal_whitespace_collapsed")
    change(t2s.convert(text), "traditional_to_simplified")
    change(text.strip(), "outer_whitespace_trimmed")
    if invalid:
        return text, operations, "invalid_unicode:" + ",".join(sorted(invalid))
    return text, operations, _repetition_reason(text)
