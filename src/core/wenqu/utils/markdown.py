import re

COMMENTS_PATTERN = re.compile(r'(<!--.*?-->)', re.DOTALL)


def remove_comments(text: str) -> str:
    return COMMENTS_PATTERN.sub("", text)
