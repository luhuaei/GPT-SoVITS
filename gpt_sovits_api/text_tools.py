import re
import unicodedata
from typing import Iterable


SENSEVOICE_TAG_RE = re.compile(r"<\|[^|]+?\|>")
MULTISPACE_RE = re.compile(r"\s+")
PUNCT_TRANSLATION = str.maketrans(
    {
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "；": ";",
        "：": ":",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "、": ",",
    }
)


def strip_sensevoice_tags(text: str) -> str:
    return SENSEVOICE_TAG_RE.sub("", text or "")


def normalize_asr_text(text: str, language: str | None = None) -> str:
    normalized = strip_sensevoice_tags(text)
    normalized = unicodedata.normalize("NFKC", normalized).translate(PUNCT_TRANSLATION)
    normalized = normalized.replace("\n", " ").replace("\r", " ")
    normalized = MULTISPACE_RE.sub(" ", normalized).strip()
    lang = (language or "").lower()
    if lang.startswith("zh") or lang in {"yue", "all_zh", "all_yue", "auto_yue"}:
        normalized = normalized.replace(" ", "")
    else:
        normalized = normalized.lower()
    return normalized


def levenshtein_distance(source: Iterable[str], target: Iterable[str]) -> int:
    source = list(source)
    target = list(target)
    if not source:
        return len(target)
    if not target:
        return len(source)
    previous = list(range(len(target) + 1))
    for row_index, source_item in enumerate(source, start=1):
        current = [row_index]
        for col_index, target_item in enumerate(target, start=1):
            insert_cost = current[col_index - 1] + 1
            delete_cost = previous[col_index] + 1
            replace_cost = previous[col_index - 1] + (0 if source_item == target_item else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    reference = reference or ""
    hypothesis = hypothesis or ""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein_distance(reference, hypothesis) / len(reference)


def wer(reference: str, hypothesis: str) -> float:
    reference_tokens = (reference or "").split()
    hypothesis_tokens = (hypothesis or "").split()
    if not reference_tokens:
        return 0.0 if not hypothesis_tokens else 1.0
    return levenshtein_distance(reference_tokens, hypothesis_tokens) / len(reference_tokens)


def build_longform_chinese_text() -> str:
    return (
        "今天这段长文用于验证语音合成服务在持续输出时是否稳定、清晰、自然。"
        "我们希望模型在较长的篇幅里保持均匀的语速，不要忽快忽慢，也不要在句子中间出现奇怪的停顿。"
        "第一部分描述一个普通的业务场景：用户上传参考音频，系统保存音色信息，然后按照输入文本生成语音并返回结果。"
        "第二部分继续观察服务在较长段落中的表现，重点检查是否会漏字、重复、吞音，或者把上一句末尾的内容错误地带到下一句开头。"
        "第三部分补充一些常见的说明语句，例如平台需要记录请求时间、请求顺序、队列状态和生成耗时，但这些信息不应该破坏整体朗读的流畅度。"
        "如果模型在长文朗读时节奏稳定，听感自然，说明前端分句、声学推理和音频拼接大体处于可接受状态。"
        "如果模型在长文朗读时突然停住，或者把一句简单的话拆成很多细碎的片段，就说明当前配置还需要继续调整。"
        "为了让测试更加接近真实环境，我们再加入一段面向用户的说明：当系统收到新的文本后，会先完成文本处理，再进行语音生成，最后把结果交给上层接口返回。"
        "在这个过程中，服务需要保持输出连贯，不能在同一个自然段里频繁改变语气，也不能让结尾部分明显失真。"
        "我们还希望这段文本包含叙述句、说明句和轻度指令句，这样可以同时观察模型在不同语气下的稳定程度。"
        "例如，系统应当能够平稳地播报状态更新、任务说明、操作提示和结果总结，而不是只擅长朗读非常短的句子。"
        "当长文继续延伸时，模型最好仍然保持一致的音色和一致的响度，避免前半段和后半段像是由两个不同的说话人生成。"
        "如果出现明显的音高漂移、尾音拉长、词语重叠或者句尾断裂，那么这次测试就应该被视为需要继续优化。"
        "为了进一步补足篇幅，我们再加入一段更平实的描述：服务上线之后，用户每天都会遇到播报通知、内容朗读、问答回放和任务提示等常见场景。"
        "这些场景并不追求夸张的语气，而是追求稳定、可信和长时间收听之后依然舒适的听感。"
        "最后再补充一段总结性描述：这条长文并不追求极端复杂，而是追求稳定可复现，用来帮助我们判断服务在日常使用中的真实表现。"
    )


def build_longform_english_text() -> str:
    return (
        "This long test sample checks whether the speech service can stay clear and steady during a sustained request. "
        "The voice should remain natural from the first sentence to the last sentence. "
        "The service should avoid repeated words, missing words, and broken pauses. "
        "A listener should hear a calm and consistent result throughout the passage. "
        "This sample uses simple language because the goal is stable regression testing. "
        "The system receives text, prepares the request, and returns generated audio. "
        "The voice should keep the same identity while the text continues. "
        "The rhythm should remain even and the endings should sound complete. "
        "If the model becomes noisy or unstable, the benchmark should reveal that behavior. "
        "This long test sample checks whether the speech service can stay clear and steady during a sustained request. "
        "The voice should remain natural from the first sentence to the last sentence. "
        "The service should avoid repeated words, missing words, and broken pauses. "
        "A listener should hear a calm and consistent result throughout the passage. "
        "This sample uses simple language because the goal is stable regression testing. "
        "The system receives text, prepares the request, and returns generated audio. "
        "The voice should keep the same identity while the text continues. "
        "The rhythm should remain even and the endings should sound complete. "
        "If the model becomes noisy or unstable, the benchmark should reveal that behavior. "
        "This long test sample checks whether the speech service can stay clear and steady during a sustained request. "
        "The voice should remain natural from the first sentence to the last sentence. "
    )
