# Every chat template an organisation publishes on the Hub, grouped by template text, against what `models` lists.
import argparse
import hashlib
import json
import re

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from tokin.templates import TEMPLATES

SKIP = re.compile(r"AWQ|GPTQ|GGUF|FP8|Int4|Int8|MLX|Embedding|Reranker|Tokenizer|mmproj", re.I)


def chat_template(repo: str) -> str | None:
    try:
        with open(hf_hub_download(repo, "chat_template.jinja")) as f:
            return f.read()
    except EntryNotFoundError:
        pass
    try:
        with open(hf_hub_download(repo, "tokenizer_config.json")) as f:
            template = json.load(f).get("chat_template")
    except EntryNotFoundError:
        return None
    return json.dumps(template, sort_keys=True) if isinstance(template, list) else template


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("org", help="Hub organisation, such as Qwen or zai-org")
    org = parser.parse_args().org
    listed = {m for t in TEMPLATES.values() for m in t.models}
    groups: dict[str, list[str]] = {}
    texts: dict[str, str] = {}
    for model in HfApi().list_models(author=org, limit=None):
        if SKIP.search(model.id):
            continue
        text = chat_template(model.id)
        if text is None:
            continue
        digest = hashlib.sha1(text.encode()).hexdigest()[:8]
        groups.setdefault(digest, []).append(model.id)
        texts[digest] = text
    for digest, members in sorted(groups.items(), key=lambda item: -len(item[1])):
        text = texts[digest]
        tools = (
            "xml tools"
            if "<function=" in text
            else "arg_key tools"
            if "<arg_key>" in text
            else "json tools"
            if "<tool_call>" in text
            else "no tools"
        )
        think = "think" if "<think>" in text else "no think"
        print(f"\n{digest}  {tools}, {think}  listed {sum(m in listed for m in members)}/{len(members)}")
        for member in sorted(members):
            print(f"  {'*' if member in listed else ' '} {member}")


if __name__ == "__main__":
    main()
