"""
Format detection and handling for muwave.
Supports Markdown, HTML, JSON, XML, and other text formats.
"""

import re
import json
from enum import Enum
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


class ContentFormat(Enum):
    """Supported content formats."""
    PLAIN_TEXT = "text/plain"
    MARKDOWN = "text/markdown"
    HTML = "text/html"
    JSON = "application/json"
    XML = "application/xml"
    CODE = "text/code"
    YAML = "text/yaml"


@dataclass
class FormatMetadata:
    """Metadata about formatted content."""
    format_type: ContentFormat
    language: Optional[str] = None  # For code blocks
    confidence: float = 1.0
    
    def to_bytes(self) -> bytes:
        """Convert metadata to bytes for transmission."""
        # Format: [format_code(1 byte)][language_length(1 byte)][language(0-255 bytes)]
        format_code = self._format_to_code(self.format_type)
        
        if self.language:
            lang_bytes = self.language.encode('utf-8')[:255]
            return bytes([format_code, len(lang_bytes)]) + lang_bytes
        else:
            return bytes([format_code, 0])
    
    @staticmethod
    def from_bytes(data: bytes) -> Optional['FormatMetadata']:
        """Parse metadata from bytes."""
        if len(data) < 2:
            return None
        
        format_code = data[0]
        lang_length = data[1]
        
        format_type = FormatMetadata._code_to_format(format_code)
        if format_type is None:
            return None
        
        language = None
        if lang_length > 0 and len(data) >= 2 + lang_length:
            language = data[2:2+lang_length].decode('utf-8', errors='ignore')
        
        return FormatMetadata(format_type=format_type, language=language)
    
    @staticmethod
    def _format_to_code(fmt: ContentFormat) -> int:
        """Convert format to byte code."""
        mapping = {
            ContentFormat.PLAIN_TEXT: 0x00,
            ContentFormat.MARKDOWN: 0x01,
            ContentFormat.HTML: 0x02,
            ContentFormat.JSON: 0x03,
            ContentFormat.XML: 0x04,
            ContentFormat.CODE: 0x05,
            ContentFormat.YAML: 0x06,
        }
        return mapping.get(fmt, 0x00)
    
    @staticmethod
    def _code_to_format(code: int) -> Optional[ContentFormat]:
        """Convert byte code to format."""
        mapping = {
            0x00: ContentFormat.PLAIN_TEXT,
            0x01: ContentFormat.MARKDOWN,
            0x02: ContentFormat.HTML,
            0x03: ContentFormat.JSON,
            0x04: ContentFormat.XML,
            0x05: ContentFormat.CODE,
            0x06: ContentFormat.YAML,
        }
        return mapping.get(code)


class FormatDetector:
    """Detects content format from text."""
    
    @staticmethod
    def detect(content: str) -> FormatMetadata:
        """
        Detect format of content.
        
        Args:
            content: Text content to analyze
            
        Returns:
            FormatMetadata with detected format and confidence
        """
        # Try JSON
        if FormatDetector._is_json(content):
            return FormatMetadata(ContentFormat.JSON, confidence=0.95)
        
        # Try XML
        if FormatDetector._is_xml(content):
            return FormatMetadata(ContentFormat.XML, confidence=0.90)
        
        # Try YAML
        if FormatDetector._is_yaml(content):
            return FormatMetadata(ContentFormat.YAML, confidence=0.85)
        
        # Try HTML
        if FormatDetector._is_html(content):
            return FormatMetadata(ContentFormat.HTML, confidence=0.90)
        
        # Try Markdown
        md_confidence = FormatDetector._markdown_confidence(content)
        if md_confidence > 0.3:
            return FormatMetadata(ContentFormat.MARKDOWN, confidence=md_confidence)
        
        # Try code
        lang, confidence = FormatDetector._detect_code(content)
        if confidence > 0.5:
            return FormatMetadata(ContentFormat.CODE, language=lang, confidence=confidence)
        
        # Default to plain text
        return FormatMetadata(ContentFormat.PLAIN_TEXT, confidence=0.5)
    
    @staticmethod
    def _is_json(content: str) -> bool:
        """Check if content is JSON."""
        content = content.strip()
        if not (content.startswith('{') or content.startswith('[')):
            return False
        try:
            json.loads(content)
            return True
        except (json.JSONDecodeError, ValueError):
            return False
    
    @staticmethod
    def _is_xml(content: str) -> bool:
        """Check if content is XML."""
        content = content.strip()
        # Simple check for XML structure
        return (
            content.startswith('<?xml') or
            (content.startswith('<') and content.endswith('>') and 
             '</' in content and content.count('<') > 2)
        )
    
    @staticmethod
    def _is_yaml(content: str) -> bool:
        """Check if content is YAML."""
        # Check for YAML-specific patterns
        lines = content.strip().split('\n')
        yaml_indicators = 0
        
        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Key-value pairs with colon
            if ':' in line and not line.startswith('-'):
                yaml_indicators += 1
            # Lists with dash
            elif line.startswith('- '):
                yaml_indicators += 1
        
        return yaml_indicators >= 3
    
    @staticmethod
    def _is_html(content: str) -> bool:
        """Check if content is HTML."""
        content_lower = content.lower().strip()
        html_tags = ['<html', '<head', '<body', '<div', '<p>', '<span', '<!doctype']
        return any(tag in content_lower for tag in html_tags)
    
    @staticmethod
    def _markdown_confidence(content: str) -> float:
        """Calculate confidence that content is Markdown."""
        indicators = 0
        total_checks = 0
        
        lines = content.split('\n')
        
        # Check for headers
        total_checks += 1
        if any(line.strip().startswith('#') for line in lines):
            indicators += 1
        
        # Check for bold/italic
        total_checks += 1
        if re.search(r'(\*\*|__).+?\1', content) or re.search(r'(\*|_).+?\1', content):
            indicators += 1
        
        # Check for links
        total_checks += 1
        if re.search(r'\[.+?\]\(.+?\)', content):
            indicators += 1
        
        # Check for code blocks
        total_checks += 1
        if '```' in content or '`' in content:
            indicators += 1
        
        # Check for lists
        total_checks += 1
        if any(re.match(r'^[\s]*[-*+]\s', line) for line in lines):
            indicators += 1
        
        # Check for blockquotes
        total_checks += 1
        if any(line.strip().startswith('>') for line in lines):
            indicators += 1
        
        return indicators / total_checks if total_checks > 0 else 0.0
    
    @staticmethod
    def _detect_code(content: str) -> Tuple[Optional[str], float]:
        """Detect if content is code and identify language."""
        indicators = {
            'python': [r'def \w+\(', r'import \w+', r'from \w+ import', r'class \w+:'],
            'javascript': [r'function \w+\(', r'const \w+ =', r'let \w+ =', r'=>'],
            'java': [r'public class', r'private \w+', r'void \w+\(', r'System\.out'],
            'c': [r'#include <', r'int main\(', r'printf\(', r'void \w+\('],
            'go': [r'package \w+', r'func \w+\(', r'import \(', r':='],
            'rust': [r'fn \w+\(', r'let mut', r'impl \w+', r'use \w+::'],
        }
        
        best_lang = None
        best_score = 0.0
        
        for lang, patterns in indicators.items():
            matches = sum(1 for pattern in patterns if re.search(pattern, content))
            score = matches / len(patterns)
            if score > best_score:
                best_score = score
                best_lang = lang
        
        return best_lang, best_score


class FormatEncoder:
    """Encodes formatted content for transmission."""
    
    @staticmethod
    def encode(content: str, format_meta: Optional[FormatMetadata] = None) -> bytes:
        """
        Encode content with format metadata.
        
        Args:
            content: Content to encode
            format_meta: Optional format metadata (auto-detected if None)
            
        Returns:
            Encoded bytes with format prefix
        """
        if format_meta is None:
            format_meta = FormatDetector.detect(content)
        
        # Encode: [metadata_length(1 byte)][metadata][content_bytes]
        metadata_bytes = format_meta.to_bytes()
        metadata_length = len(metadata_bytes)
        
        if metadata_length > 255:
            metadata_length = 255
            metadata_bytes = metadata_bytes[:255]
        
        content_bytes = content.encode('utf-8')
        
        return bytes([metadata_length]) + metadata_bytes + content_bytes
    
    @staticmethod
    def decode(data: bytes) -> Tuple[str, Optional[FormatMetadata]]:
        """
        Decode content with format metadata.
        
        Args:
            data: Encoded bytes
            
        Returns:
            Tuple of (content string, format metadata)
        """
        if len(data) < 1:
            return "", None
        
        metadata_length = data[0]
        
        if len(data) < 1 + metadata_length:
            # No metadata, treat as plain text
            try:
                return data.decode('utf-8'), FormatMetadata(ContentFormat.PLAIN_TEXT)
            except UnicodeDecodeError:
                return "", None
        
        metadata_bytes = data[1:1+metadata_length]
        content_bytes = data[1+metadata_length:]
        
        format_meta = FormatMetadata.from_bytes(metadata_bytes)
        
        try:
            content = content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            return "", format_meta
        
        return content, format_meta


def format_content_for_display(content: str, format_meta: Optional[FormatMetadata] = None) -> str:
    """
    Format content for display based on its format.
    
    Args:
        content: Content to format
        format_meta: Optional format metadata
        
    Returns:
        Formatted content string
    """
    if format_meta is None:
        format_meta = FormatDetector.detect(content)
    
    # Add format indicator
    format_name = format_meta.format_type.value.split('/')[-1].upper()
    
    if format_meta.format_type == ContentFormat.CODE and format_meta.language:
        header = f"[{format_name}: {format_meta.language.upper()}]"
    else:
        header = f"[{format_name}]"
    
    # For terminal display, add visual separator
    if format_meta.format_type != ContentFormat.PLAIN_TEXT:
        return f"{header}\n{'─' * 40}\n{content}\n{'─' * 40}"
    
    return content
