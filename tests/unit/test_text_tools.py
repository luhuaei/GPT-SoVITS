from gpt_sovits_api.text_tools import (
    build_longform_chinese_text,
    build_longform_english_text,
    cer,
    normalize_asr_text,
    strip_sensevoice_tags,
    wer,
)


def test_strip_sensevoice_tags():
    assert strip_sensevoice_tags("<|zh|><|happy|>你好<|ccc|> world") == "你好 world"


def test_normalize_asr_text_for_chinese():
    actual = normalize_asr_text("<|zh|> 你好 ， 世界 。", language="zh")
    assert actual == "你好,世界."


def test_normalize_asr_text_for_english():
    actual = normalize_asr_text("<|en|> Hello   WORLD !", language="en")
    assert actual == "hello world !"


def test_error_rate_helpers():
    assert cer("你好世界", "你好世界") == 0
    assert wer("hello world", "hello world") == 0
    assert cer("你好世界", "你好啊界") > 0
    assert wer("hello world", "hello brave world") > 0


def test_longform_text_lengths():
    assert len(build_longform_chinese_text()) >= 800
    assert len(build_longform_english_text()) >= 800

