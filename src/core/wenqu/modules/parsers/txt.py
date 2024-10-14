from pathlib import Path
from typing import List, Optional

from loguru import logger
from langchain.docstore.document import Document

from wenqu.modules.rw import ReaderWriter
from wenqu.constants import CHINESE_SEPARATOR
from wenqu.modules.parsers.base import BaseParser
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter


class TXTParser(BaseParser):

    supported_file_extensions = [
        ".txt",
    ]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 51, separators: Optional[List[str]] = None, is_separator_regex: bool = False,
                 *args, **kwargs): # TODO: 添加keep_separators参数
        super().__init__(*args, **kwargs)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.is_separator_regex = is_separator_regex
        logger.info(f"Initializing Parser with chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}, separators: {separators}, is_separator_regex: {is_separator_regex}")

    async def get_chunks(self, filepath_or_content: str | bytes | Path, metadata: dict = None, image_writer: ReaderWriter = None,
                         file_extension: str = None, *args, **kwargs) -> List[Document]:
        if metadata is None:
            metadata = {}

        # txt文件chunk metadata设置为默认值
        if "title" not in metadata: metadata["title"] = "" # TODO: 设置为None时，milvus报错
        if "type" not in metadata: metadata["type"] = "text"
        
        with self.input_as_str(filepath_or_content, file_extension) as content:
            splitter = await self.get_splitter(content)
            chunks = splitter.create_documents([content], metadatas=[metadata])

            # 发送信号
            self.doc_chunked.send(self, chunks=chunks, context={"metadata": metadata, "file_extension": file_extension})

            return chunks

    async def get_markdown(self, filepath_or_content: str | bytes | Path, metadata: dict = None, image_writer: ReaderWriter = None,
                           file_extension: str = None, *args, **kwargs) -> str:
        raise NotImplementedError("This method is not implemented for TXTParser")
    
    async def get_splitter(self, text) -> TextSplitter:
        lang = self.get_language(text)

        if lang not in ["EN", "ZH"]:
            raise ValueError(f"unsupported language: {lang}")
        elif lang == "EN":
            splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
                                             separators=self.separators, is_separator_regex=self.is_separator_regex)
        elif lang == "ZH":
            logger.debug(f"* using chinese separator: {self.separators or CHINESE_SEPARATOR}")
            splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
                                             separators=self.separators or CHINESE_SEPARATOR, 
                                             is_separator_regex=self.is_separator_regex if self.separators else True)
            
        self.splitter_selected.send(self, splitter=splitter, context={"lang": lang})
        return splitter