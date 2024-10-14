import copy
from typing import (
    Dict, 
    List, 
    Tuple, 
    Generator, 
    Optional, 
    Any, 
    Callable,
)

import mdformat_tables
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from mdformat.renderer import MDRenderer
from markdown_it.tree import SyntaxTreeNode
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from more_itertools import split_before
from loguru import logger


class MarkdownTextSplitter(RecursiveCharacterTextSplitter):

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 51,
        length_function: Callable[[str], int] = len,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
        separators: Optional[List[str]] = None,
        keep_separator: bool | str = "end",
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=length_function,
                         add_start_index=add_start_index, strip_whitespace=strip_whitespace,
                         separators=separators, keep_separator=keep_separator, 
                         is_separator_regex=is_separator_regex, **kwargs)
        self.parser = MarkdownIt("gfm-like")
        self.renderer = MDRenderer()

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = (metadatas or [{}]) * len(texts)
        documents = []
        for i, text in enumerate(texts):
            index = 0
            previous_chunk_len = 0
            for chunk, metadata in self.split_text(text, metadata=copy.deepcopy(_metadatas[i])):
                # 图片和表格内容，不进行overlap处理
                if metadata.get("type") in ("image", "table"):
                    new_doc = Document(page_content=chunk, metadata=metadata)
                    documents.append(new_doc)
                    continue
                
                # 文本进行overlap处理
                if self._add_start_index:
                    offset = index + previous_chunk_len - self._chunk_overlap
                    index = text.find(chunk, max(0, offset))
                    metadata["start_index"] = index
                    previous_chunk_len = len(chunk)
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_text(
            self, text: str, *, metadata: Dict[str, str] = None
        ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        tokens = self.parser.parse(text)
        node = SyntaxTreeNode(tokens)

        for header_text, section_text, section_type, section_metadata in self._split_by_header(node):
            if not section_text: 
                continue
            
            # 标题下的表格和图片
            if section_type in ("image", "table"):
                yield section_text, {
                    "title": header_text,
                    "type": section_type,
                    **section_metadata,
                    **(metadata or {})
                }
            # 标题下的章节文本内容
            elif section_type == "text":
                for chunk in self._split_text(section_text.strip(), self._separators):
                    yield chunk, {
                        "title": header_text,
                        "type": section_type,
                        **section_metadata,
                        **(metadata or {})
                    }
            else:
                raise ValueError(f"invaild chunk type: {section_type}")
    
    def _split_by_header(
            self, nodes: SyntaxTreeNode, header_markup: str = "#", parent_title: str = None
        ) -> Generator[Tuple[str, str, str, Dict[str, Any]], None, None]:
        # 按指定标题层级拆分Markdown
        for splitted_nodes in split_before(nodes, lambda node: node.type == "heading" and node.markup == header_markup):
            # 完整的章节，包含章节标题和章节内容
            if splitted_nodes[0].type == "heading":
                # 提取标题内容
                header_text = self._get_treenode_text(splitted_nodes[0]).strip()
                # 提取出标题后，继续切分这个section中下一级标题
                # 比如：当前是二级标题下的内容，则继续拆分这个二级标题下三级标题
                yield from self._split_by_header(splitted_nodes[1:], 
                                                 header_markup=header_markup + "#", 
                                                 parent_title=" > ".join([parent_title, header_text]) if parent_title else header_text)
            # 一个章节下的全部内容（不包含标题）
            # TODO: 如果没有一级标题，反而有二级标题时，可能进入了错误的递归
            else:
                title = "" if parent_title is None else parent_title
                # yield result: title, section text, section type, section metadata
                yield title, self._get_treenodes_text(splitted_nodes), "text", {}
                yield from self._get_nodes_nontext_contents(splitted_nodes, title)

    def _get_treenodes_text(self, nodes: List[SyntaxTreeNode]) -> str:
        return "".join([self._get_treenode_text(node) for node in nodes if not self._is_contain_nontext(node)])
    
    def _get_treenode_text(self, node: SyntaxTreeNode) -> str:
        try:
            tokens = node.to_tokens()
            return self.renderer.render(tokens, {"parser_extension": [mdformat_tables]}, {})
        except Exception as e:
            # TODO: 考虑Markdown出现非法语法情况时的处理方式
            # 出现渲染异常的情况下，记录日志并返回空字符串
            logger.warning(f"Error when rendering node: {node.pretty(indent=2, show_text=True)}")
            logger.exception(e)
            return ""
    
    def _get_nodes_nontext_contents(
            self, nodes: List[SyntaxTreeNode], title: str
        ) -> Generator[Tuple[str, str, str, Dict[str, Any]], None, None]:
        for node in nodes:
            if self._is_contain_nontext(node):
                yield from self._get_nontext_content(node, title)

    def _get_nontext_content(self, node: SyntaxTreeNode, title: str) -> Generator[Tuple[str, Dict[str, str]], None, None]: # TODO: 添加除image和table外的其他非文本内容块支持
        # Markdown Image
        if node.type == "image":
            yield title, node.attrs["src"], "image", {"image_url": node.attrs["src"]}
        # Makrdown Table
        elif node.type == "table":
            yield title, self._get_treenode_text(node), "table", {}
        # HTML Table
        elif node.type in ("html_block", "html_inline"):
            html_table = self._find_html_table(node)
            if html_table:
                yield title, str(html_table), "table", {}

            html_img = self._find_html_img(node)
            if html_img:
                yield title, str(html_img), "image", {"image_url": html_img.attrs["src"]}
        # Loop children
        else:
            for child in node.children:
                if not self._is_contain_nontext(child):
                    continue
                yield from self._get_nontext_content(child, title)
    
    def _is_contain_nontext(self, node: SyntaxTreeNode) -> bool: # TODO: 添加除image和table外的其他非文本内容块支持
        # Markdown Image & Table
        if node.type in ("image", "table"):
            return True
        # HTML Table/Image
        if node.type in ("html_block", "html_inline") and (self._find_html_table(node) or self._find_html_img(node)):
            return True
        # Loop Children
        if node.children and any(self._is_contain_nontext(c) for c in node.children):
            return True
        return False
    
    def _find_html_table(self, node: SyntaxTreeNode) -> BeautifulSoup:
        soup = BeautifulSoup(node.content)
        return soup.find("table")
    
    def _find_html_img(self, node: SyntaxTreeNode) -> BeautifulSoup:
        soup = BeautifulSoup(node.content)
        return soup.find("img")
