from wenqu.modules.parsers.txt import TXTParser
from wenqu.modules.parsers.markdown import MarkdownParser
from wenqu.modules.parsers.base import register_parser, get_parser_for_extension, get_supported_extensions

# The order of registry defines the order of precedence
register_parser("TXTParser", TXTParser)
register_parser("MarkdownParser", MarkdownParser)

__all__ = ["TXTParser", "MarkdownParser", "get_parser_for_extension", "get_supported_extensions", "register_parser"]


