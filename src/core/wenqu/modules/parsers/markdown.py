from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter

from wenqu.modules.rw import ReaderWriter
from wenqu.constants import CHINESE_SEPARATOR
from wenqu.modules.parsers.txt import TXTParser
from wenqu.utils.markdown import remove_comments
from wenqu.modules.splitters.markdown import MarkdownTextSplitter


class MarkdownParser(TXTParser):

    supported_file_extensions = [
        ".md",
    ]

    async def get_dataframe(self, filepath_or_content: str | bytes | Path, metadata: dict = None, image_writer: ReaderWriter = None,
                           file_extension: str = None, *args, **kwargs) -> pd.DataFrame:
        chunks = await self.get_chunks(filepath_or_content, metadata, image_writer, file_extension, *args, **kwargs)
        return self.chunks_to_dataframe(chunks)

    async def get_chunks(self, filepath_or_content: str | bytes | Path, metadata: dict = None, image_writer: ReaderWriter = None,
                         file_extension: str = None, *args, **kwargs) -> List[Document]:
        if metadata is None:
            metadata = {}
            
        self._markdown = md_content = await self.get_markdown(filepath_or_content, metadata, image_writer=image_writer, file_extension=file_extension, *args, **kwargs)
        logger.debug(f"generate markdown: {md_content[:100]}...")

        splitter = await self.get_splitter(md_content)
        chunks = splitter.create_documents([md_content], metadatas=[metadata])
        # 发送信号
        self.doc_chunked.send(self, chunks=chunks, context={"metadata": metadata, "file_extension": file_extension})
        return chunks

    async def get_markdown(self, filepath_or_content: str | bytes | Path, metadata: dict = None, image_writer: ReaderWriter = None,
                           file_extension: str = None, *args, **kwargs) -> str:
        with self.input_as_str(filepath_or_content, file_extension) as content:
            self.markdown_generated.send(self, content=content, context={
                "metadata": metadata,
                "file_extension": file_extension,
            })
            return remove_comments(content)
        
    async def get_splitter(self, text) -> TextSplitter:
        lang = self.get_language(text)
        logger.info(f"* detect text language: {lang}")

        if lang not in ["EN", "ZH"]:
            raise ValueError(f"unsupported language: {lang}")
        elif lang == "EN":
            logger.info(f"* using separator: {self.separators}, is_separator_regex: {self.is_separator_regex},"
                        f" chunk_size: {self.chunk_size}, chunk_overlap: {self.chunk_overlap}")
            splitter = MarkdownTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
                                             separators=self.separators, is_separator_regex=self.is_separator_regex)
        elif lang == "ZH":
            separators = self.separators or CHINESE_SEPARATOR
            is_separator_regex = self.is_separator_regex if self.separators else True

            logger.info(f"* using separator: {separators}, is_separator_regex: {is_separator_regex},"
                        f" chunk_size: {self.chunk_size}, chunk_overlap: {self.chunk_overlap}")
            splitter = MarkdownTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, 
                                        separators=separators, is_separator_regex=is_separator_regex)

        self.splitter_selected.send(self, splitter=splitter, context={"lang": lang})
        return splitter
