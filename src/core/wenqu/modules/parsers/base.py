import tempfile
from pathlib import Path
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional, List, Literal

from loguru import logger
from blinker import Namespace
from fast_langdetect import detect_language
from langchain.docstore.document import Document

PARSER_REGISTRY = {}
PARSER_REGISTRY_EXTENSIONS = defaultdict(list)
PARSER_SIGNALS = Namespace()


class BaseParser(ABC):
    """
    BaseParser 是一个抽象基类（ABC），作为所有解析器对象的模板。它包含每个解析器应实现的公共属性和方法。
    """
    doc_chunked = PARSER_SIGNALS.signal("doc-chunked")
    splitter_selected = PARSER_SIGNALS.signal("splitter-selected")
    markdown_generated = PARSER_SIGNALS.signal("markdown-generated")

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    async def get_markdown(
        self,
        filepath_or_content: str | bytes | Path,
        metadata: dict = None,
        file_extension: str = None, 
        *args, 
        **kwargs
    ) -> str:
        pass

    @abstractmethod
    async def get_chunks(
        self,
        filepath_or_content: str | bytes | Path,
        metadata: Optional[dict] = None,
        file_extension: str = None,
        *args,
        **kwargs,
    ) -> List[Document]:
        pass

    @staticmethod
    def get_language(text: str) -> Literal["EN", "ZH"]:
        """确定提供文本的语言。
        此函数分析输入文本并检测其是否为英语（EN）或中文（ZH）。如果文本超过500个字符，则截断为前500个字符进行分析。
        如果检测到的语言不受支持，则默认使用英语。 

        默认通过地址 https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz 下载检测模型到`/tmp/fasttext-langdetect`目录。
        可通过设置环境变量`FTLANG_CACHE`更改缓存目录。
        
        Args:
            text (str): 要检测语言的输入文本。
        
        Returns:
            Literal["EN", "ZH"]: 检测到的语言，“EN”表示英语或“ZH”表示中文。如果语言无法识别，则默认为“EN”。
        
        Raises:
            None: 此函数不抛出任何异常。
        """
        if len(text) > 500: text = text[:500]
        lang = detect_language(" ".join(text.split()))
        logger.info(f"* 检测到的文本语言: {lang}")

        # 暂时只按中文和英文方式进行处理，默认为英文
        if lang in ["EN", "ZH"]:
            return lang
        
        logger.info(f"* language not supported, default to EN")
        return "EN"

    @contextmanager
    def input_as_bytes(self, filepath_or_content: str | bytes | Path, file_extension: str = None) -> Path:
        if isinstance(filepath_or_content, bytes):
            yield filepath_or_content
            
        if isinstance(filepath_or_content, (str, Path)):
            if isinstance(filepath_or_content, str): filepath_or_content = Path(filepath_or_content)
            assert filepath_or_content.exists(), f"file not found: {filepath_or_content}"
            yield filepath_or_content.read_bytes()

    @contextmanager
    def input_as_path(self, filepath_or_content: str | bytes | Path, file_extension: str = None) -> Path:
        if isinstance(filepath_or_content, Path):
            assert filepath_or_content.exists(), f"file not found: {filepath_or_content}"
            yield filepath_or_content
        
        if isinstance(filepath_or_content, str):
            filepath_or_content = Path(filepath_or_content)
            assert filepath_or_content.exists(), f"file not found: {filepath_or_content}"
            yield filepath_or_content
        
        if isinstance(filepath_or_content, bytes):
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=True) as f:
                f.write(filepath_or_content)
                yield Path(f.name)

    @contextmanager
    def input_as_str(self, filepath_or_content: str | bytes | Path, file_extension: str = None) -> Path:
        if isinstance(filepath_or_content, Path):
            assert filepath_or_content.exists(), f"file not found: {filepath_or_content}"
            yield filepath_or_content.read_text()
        
        if isinstance(filepath_or_content, str):
            yield filepath_or_content
        
        if isinstance(filepath_or_content, bytes):
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=True) as f:
                f.write(filepath_or_content)
                yield Path(f.name).read_text()


def register_parser(name: str, cls):
    """
    Registers all the available parsers.
    """
    global PARSER_REGISTRY
    if name in PARSER_REGISTRY:
        raise ValueError(
            f"Error while registering class {cls.__name__} already taken by {PARSER_REGISTRY[name].__name__}"
        )
    PARSER_REGISTRY[name] = cls
    for extension in cls.supported_file_extensions:
        PARSER_REGISTRY_EXTENSIONS[extension].append(name)


def get_supported_extensions() -> List[str]:
    """
    Returns a list of all the supported file extensions.
    """
    global PARSER_REGISTRY_EXTENSIONS
    return list(PARSER_REGISTRY_EXTENSIONS.keys())


def get_parser_for_extension(
    file_extension, parsers_map=None, *args, **kwargs
) -> BaseParser:
    """
    During the indexing phase, given the file_extension and parsers mapping, return the appropriate mapper.
    If no mapping is given, use the default registry.
    """
    global PARSER_REGISTRY_EXTENSIONS
    global PARSER_REGISTRY

    if parsers_map is None:
        parsers_map = {}

    # We dont have a parser for this extension yet
    if file_extension not in PARSER_REGISTRY_EXTENSIONS:
        logger.error(f"Loaded doc with extension {file_extension} is not supported")
        return None
    # Extension not given in parser map use the default registry
    if file_extension not in parsers_map:
        # get the first parser name registered with the extension
        name = PARSER_REGISTRY_EXTENSIONS[file_extension][0]
        print(
            f"Parser map not found in the collection for extension {file_extension}. Hence, using parser {name}"
        )
        logger.debug(
            f"Parser map not found in the collection for extension {file_extension}. Hence, using parser {name}"
        )
    else:
        name = parsers_map[file_extension]
        print(
            f"Parser map found in the collection for extension {file_extension}. Hence, using parser {name}"
        )
        logger.debug(
            f"Parser map found in the collection for extension {file_extension}. Hence, using parser {name}"
        )

    if name not in PARSER_REGISTRY:
        raise ValueError(f"No parser registered with name {name}")

    return PARSER_REGISTRY[name](*args, **kwargs)


def list_parsers():
    """
    Returns a list of all the registered parsers.

    Returns:
        List[dict]: A list of all the registered parsers.
    """
    global PARSER_REGISTRY
    return [
        {
            "type": type,
            "class": cls.__name__,
        }
        for type, cls in PARSER_REGISTRY.items()
    ]
