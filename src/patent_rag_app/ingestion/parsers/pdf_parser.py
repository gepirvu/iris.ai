"""Patent PDF parser for front-matter extraction."""

from __future__ import annotations

from collections import OrderedDict
import statistics
import re
from dataclasses import dataclass, field
import statistics
from pathlib import Path
from typing import Any, Iterable, Iterator, List

import pdfplumber


from patent_rag_app.config.logging import get_logger
from patent_rag_app.ingestion.models import PatentClaim, PatentDocument, PatentSection, PatentTable

LOGGER = get_logger(__name__)

FIELD_SPLIT_PATTERN = re.compile(r"\((\d{2})\)")
PARAGRAPH_PATTERN = re.compile(r"^(?:\[(\d{4})\]|(\d{4}))\s*(.*)")
UPPER_DIGIT_PUNCT = re.compile(r"[A-Z0-9,.;:!?%/]")
TABLE_STREAM_SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "text_tolerance": 2,
    "intersection_x_tolerance": 3,
    "intersection_y_tolerance": 3,
    "min_words_vertical": 1,
    "min_words_horizontal": 1,
}

TABLE_LATTICE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "intersection_tolerance": 3,
    "edge_min_length": 3,
    "min_words_vertical": 1,
    "min_words_horizontal": 1,
}
CLAIMS_HEADING_PATTERN = re.compile(r"^\s*(claims?|we\s+claim|what\s+is\s+claimed|patentansprüche|revendications)\b", re.IGNORECASE)
CLAIM_NUMBER_PATTERN = re.compile(r"^(?:claim\s+)?(\d{1,4})[\.)]\s*(.*)", re.IGNORECASE)

# Multi-language section patterns
GERMAN_CLAIMS_PATTERN = re.compile(r"^\s*patentansprüche\s*$", re.IGNORECASE)
FRENCH_CLAIMS_PATTERN = re.compile(r"^\s*revendications\s*$", re.IGNORECASE)
ENGLISH_CLAIMS_PATTERN = re.compile(r"^\s*(claims?|we\s+claim|what\s+is\s+claimed)\s*$", re.IGNORECASE)

# Page header cleanup patterns
PAGE_HEADER_PATTERN = re.compile(r"^(EP|US|WO)\s+\d+\s+\d+\s+[A-Z]\d+(\s+\d+)*$")
PAGE_NUMBER_PATTERN = re.compile(r"^\d+$")


@dataclass
class PageSummary:
    page_number: int
    text: str
    words: list[dict[str, Any]]
    tables: list[dict[str, Any]] = field(default_factory=list)


class PatentParser:
    """Extract patent metadata from patent PDFs."""

    WORD_TOLERANCE_X = 1.5
    WORD_TOLERANCE_Y = 1.0
    COLUMN_GAP_THRESHOLD = 30.0
    LINE_Y_TOLERANCE = 2.0
    BASE_GAP_FACTOR = 0.30
    MODERATE_GAP_FACTOR = 0.18
    PARAGRAPH_GAP_THRESHOLD = 6.0

    def parse(self, path: Path) -> PatentDocument:
        LOGGER.info("Parsing patent PDF", extra={"path": str(path)})
        pages = list(self._extract_pages(path))
        total_pages = len(pages)
        first_page = pages[0] if pages else None

        front_page_fields = self._extract_front_page_fields(first_page) if first_page else {}
        front_page_text = first_page.text if first_page else ""

        description_sections = self._extract_description_sections(pages[1:]) if len(pages) > 1 else []
        tables = self._extract_tables(pages)
        claims = self._extract_claims(pages)
        raw_text = "\n\n".join(page.text for page in pages if page.text)

        return PatentDocument(
            patent_id=self._derive_patent_id(path),
            total_pages=total_pages,
            front_page_text=front_page_text,
            front_page_fields=front_page_fields,
            sections=description_sections,
            tables=tables,
            claims=claims,
            raw_text=raw_text,
            source_path=str(path),
        )

    def _extract_pages(self, path: Path) -> Iterator[PageSummary]:
        with pdfplumber.open(path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                reconstructed_text = self._reconstruct_page_text(page)
                try:
                    words = page.extract_words(
                        x_tolerance=self.WORD_TOLERANCE_X,
                        y_tolerance=self.WORD_TOLERANCE_Y,
                        keep_blank_chars=False,
                        use_text_flow=True,
                    )
                except TypeError:
                    words = page.extract_words() or []
                except Exception:
                    words = []

                raw_tables: list[dict[str, Any]] = []
                try:
                    lattice_tables = page.extract_tables(TABLE_LATTICE_SETTINGS) or []
                except Exception:
                    lattice_tables = []
                for tbl in lattice_tables:
                    raw_tables.append({"strategy": "lattice", "data": tbl})

                try:
                    stream_tables = page.extract_tables(TABLE_STREAM_SETTINGS) or []
                except Exception:
                    stream_tables = []
                if not lattice_tables:
                    for tbl in stream_tables:
                        raw_tables.append({"strategy": "stream", "data": tbl})

                yield PageSummary(
                    page_number=idx,
                    text=reconstructed_text,
                    words=words,
                    tables=raw_tables,
                )

    def _reconstruct_page_text(self, page: pdfplumber.page.Page) -> str:
        if not page.chars:
            return page.extract_text() or ""

        lines: list[list[dict[str, Any]]] = []
        for char in sorted(page.chars, key=lambda c: (round(c["top"], 1), c["x0"])):
            top = char["top"]
            matched_line = None
            for line in lines:
                if abs(line[0]["top"] - top) <= self.LINE_Y_TOLERANCE:
                    matched_line = line
                    break
            if matched_line is None:
                lines.append([char])
            else:
                matched_line.append(char)

        reconstructed_lines: list[str] = []
        last_bottom = None
        for line in lines:
            line.sort(key=lambda c: c["x0"])
            text = self._build_line_from_chars(line)
            if not text:
                continue

            if last_bottom is not None:
                vertical_gap = line[0]["top"] - last_bottom
                if vertical_gap > self.PARAGRAPH_GAP_THRESHOLD:
                    reconstructed_lines.append("")

            reconstructed_lines.append(text)
            last_bottom = max(char["bottom"] for char in line)

        return "\n".join(reconstructed_lines).strip()

    def _build_line_from_chars(self, chars: list[dict[str, Any]]) -> str:
        if not chars:
            return ""

        widths = [char["x1"] - char["x0"] for char in chars]
        if not widths:
            return ""

        try:
            median_width = statistics.median(widths)
        except statistics.StatisticsError:
            median_width = widths[0]

        parts: list[str] = []
        previous_char = None
        for idx, char in enumerate(chars):
            text = char.get("text", "")
            if not text:
                continue
            if previous_char is not None:
                gap = char["x0"] - previous_char["x1"]
                if gap > median_width * self.BASE_GAP_FACTOR:
                    parts.append(" ")
                elif gap > median_width * self.MODERATE_GAP_FACTOR:
                    next_char = text
                    if UPPER_DIGIT_PUNCT.match(next_char):
                        parts.append(" ")
            if not text:
                continue
            if text.isspace():
                parts.append(" ")
            else:
                parts.append(text)
            previous_char = char

        line_text = "".join(parts)
        line_text = re.sub(r"\s+", " ", line_text)
        return line_text.strip()

    def _extract_front_page_fields(self, page: PageSummary | None) -> dict[str, str]:
        if page is None:
            return {}

        fallback_fields = self._extract_front_page_fields_from_text(page.text)
        if not page.words:
            return fallback_fields

        code_entries = []
        for word in page.words:
            text = word.get("text", "")
            match = FIELD_SPLIT_PATTERN.fullmatch(text)
            if match:
                code_entries.append(
                    {
                        "code": match.group(1),
                        "x0": float(word.get("x0", 0.0)),
                        "x1": float(word.get("x1", 0.0)),
                        "top": float(word.get("top", 0.0)),
                    }
                )

        if not code_entries:
            return fallback_fields

        threshold = self._infer_column_threshold(code_entries)
        columns = {
            "left": [w for w in page.words if float(w.get("x0", 0.0)) <= threshold],
            "right": [w for w in page.words if float(w.get("x0", 0.0)) > threshold],
        }

        column_fields: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
        for entry in code_entries:
            column = "left" if entry["x0"] <= threshold else "right"
            entry["column"] = column
            column_fields[column].append(entry)

        for field_list in column_fields.values():
            field_list.sort(key=lambda item: item["top"])

        column_values: dict[str, str] = {}
        for column_name, field_list in column_fields.items():
            words = columns[column_name]
            for idx, field in enumerate(field_list):
                next_top = field_list[idx + 1]["top"] if idx + 1 < len(field_list) else float("inf")
                value = self._assemble_column_value(words, field["top"], next_top)
                if value:
                    column_values[field["code"]] = value

        merged_fields: OrderedDict[str, str] = OrderedDict(fallback_fields)
        for code, value in column_values.items():
            merged_fields[code] = value

        for code, value in merged_fields.items():
            merged_fields[code] = re.sub(r"\s+", " ", value).strip()

        return dict(merged_fields)

    def _extract_front_page_fields_from_text(self, text: str) -> OrderedDict[str, str]:
        fields: "OrderedDict[str, str]" = OrderedDict()
        if not text:
            return fields

        parts = FIELD_SPLIT_PATTERN.split(text)
        if len(parts) < 3:
            return fields

        for code, value_block in zip(parts[1::2], parts[2::2]):
            value = value_block.replace("\n", " ").strip()
            fields[code] = re.sub(r"\s+", " ", value)

        return fields

    def _infer_column_threshold(self, code_entries: list[dict[str, Any]]) -> float:
        unique_x = sorted({entry["x0"] for entry in code_entries})
        if len(unique_x) >= 2:
            gaps = [unique_x[i + 1] - unique_x[i] for i in range(len(unique_x) - 1)]
            for idx, gap in enumerate(gaps):
                if gap > self.COLUMN_GAP_THRESHOLD:
                    return (unique_x[idx] + unique_x[idx + 1]) / 2
            return (unique_x[0] + unique_x[1]) / 2
        return unique_x[0] + 50.0

    @staticmethod
    def _assemble_column_value(words: list[dict[str, Any]], start_top: float, end_top: float) -> str:
        selected = []
        for word in words:
            top = float(word.get("top", 0.0))
            if top < start_top - 0.5 or top >= end_top - 0.5:
                continue
            text = word.get("text", "")
            if FIELD_SPLIT_PATTERN.fullmatch(text):
                continue
            selected.append(word)

        if not selected:
            return ""

        selected.sort(key=lambda w: (round(float(w.get("top", 0.0)), 1), float(w.get("x0", 0.0))))
        lines: "OrderedDict[float, list[str]]" = OrderedDict()
        for word in selected:
            line_key = round(float(word.get("top", 0.0)), 1)
            lines.setdefault(line_key, []).append(word.get("text", ""))

        line_texts = [" ".join(line).strip() for line in lines.values() if any(part.strip() for part in line)]
        return " ".join(line_texts).strip()

    @staticmethod
    def _derive_patent_id(path: Path) -> str:
        return path.stem

    def _extract_front_page_only(self, text: str) -> str:
        """Extract only front page content, stopping at Description section."""
        # Stop at common section headers (EN/DE/FR)
        stop_patterns = [
            r'\bDescription\b',
            r'\bTechnisches\s+Gebiet\b',
            r'\bBeschreibung\b',
            r'\bClaims\b',
            r'\bPatentansprüche\b',
            r'\bRevendications\b'
        ]

        stop_pattern = '|'.join(f'(?:{p})' for p in stop_patterns)
        match = re.search(stop_pattern, text, re.IGNORECASE)

        if match:
            return text[:match.start()].strip()
        return text

    def _clean_inventors_field_content(self, raw_inventors: str, full_front_page_text: str) -> str:
        """Clean inventors field content using bullet point splitting."""
        # If already clean, return as-is
        if "•" in raw_inventors and raw_inventors.count("•") >= 6:
            # Remove any trailing junk (page numbers, notes, etc.)
            clean_inventors = re.sub(r'\s+\d+\s+[A-Z]?\s+\d+.*$', '', raw_inventors)
            clean_inventors = re.sub(r'\s*Note:.*$', '', clean_inventors, flags=re.IGNORECASE | re.DOTALL)
            return clean_inventors.strip()

        # Extract all bullet points from full front page text
        bullet_points = re.findall(r'•[^•]*', full_front_page_text)

        # Filter for inventor-like entries (contain names with comma)
        inventor_bullets = []
        for bullet in bullet_points:
            if ',' in bullet and re.search(r'[A-Z]{2,}', bullet):  # Has comma and uppercase surnames
                # Clean the bullet point
                clean_bullet = re.sub(r'\s+', ' ', bullet.strip())
                # Remove trailing junk
                clean_bullet = re.sub(r'\s*\(\d+\).*$', '', clean_bullet)
                clean_bullet = re.sub(r'\s*Note:.*$', '', clean_bullet, flags=re.IGNORECASE | re.DOTALL)
                if clean_bullet.strip():
                    inventor_bullets.append(clean_bullet.strip())

        return ' '.join(inventor_bullets) if inventor_bullets else raw_inventors

    def _parse_inventors(self, block: str) -> str:
        """Parse inventors by collecting everything between (72) and (73)."""
        # For inventors, we need to get everything between (72) and (73)
        # because of the two-column layout issue
        return self._extract_inventors_between_fields(block)

    def _extract_inventors_between_fields(self, full_text: str) -> str:
        """Extract all content between (72) Inventors and (73) Proprietor."""
        # Find the section between (72) and (73)
        inventors_match = re.search(r'\(72\).*?(?=\(73\)|$)', full_text, re.DOTALL)
        if not inventors_match:
            return ""

        inventors_section = inventors_match.group(0)

        # Split on bullets and filter
        bullet_entries = re.split(r'(?=•)', inventors_section)
        inventors = []

        for entry in bullet_entries:
            entry = entry.strip()
            if not entry or not entry.startswith('•'):
                continue

            # Clean up the entry
            entry = re.sub(r'\s+', ' ', entry)
            # Remove any field codes that leaked in
            entry = re.sub(r'\(\d{2}\)[^•]*', '', entry).strip()

            if entry and ',' in entry:  # Must have comma for name pattern
                inventors.append(entry)

        return ' '.join(inventors) if inventors else ""

    def _parse_references(self, block: str) -> str:
        """Parse references following example_code.txt approach."""
        # Normalize and split on spaces/breaks
        norm = block.replace("\n", " ")
        refs = re.split(r"\s{2,}|\s*(?=EP-|JP-|US-|WO-)", norm)

        cleaned = []
        for r in refs:
            r = re.sub(r'\s+', ' ', r.strip())
            if not r:
                continue
            # Fix split numbers like "EP-A1- 0 678 878" -> "EP-A1-0678878"
            r = re.sub(r"([A-Z]{2}-[A-Z0-9]+-)\s*([0-9 ]+)",
                      lambda m: m.group(1) + re.sub(r"\s+", "", m.group(2)), r)
            if re.match(r"^(EP|JP|US|WO)-", r):
                cleaned.append(r)

        return ' '.join(cleaned) if cleaned else block

    def _strip_field_label(self, content: str) -> str:
        """Strip field labels like 'Application number:' from content."""
        # Remove leading labels
        cleaned = re.sub(r"^[A-Za-z \(\)/-]+:\s*", "", content).strip()
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned

    def _extract_inventors_from_full_text(self, front_page_text: str) -> str:
        """Extract clean inventor entries from full front page text."""
        # Find all bullet points that look like inventors (contain comma for name pattern)
        bullet_pattern = r'•\s+[^•]*?,'
        potential_inventors = re.findall(bullet_pattern, front_page_text)

        clean_inventors = []
        for inventor_text in potential_inventors:
            # Expand to get full inventor entry until next bullet or field code
            start = front_page_text.find(inventor_text)
            if start == -1:
                continue

            # Find end - either next bullet point or field code
            end_match = re.search(r'(?=•|\(\d{2}\))', front_page_text[start + len(inventor_text):])
            if end_match:
                end = start + len(inventor_text) + end_match.start()
                full_inventor = front_page_text[start:end]
            else:
                full_inventor = inventor_text

            # Basic cleanup
            full_inventor = re.sub(r'\s+', ' ', full_inventor.strip())

            # Must have surname, firstname pattern and reasonable length
            if ',' in full_inventor and len(full_inventor) > 20:
                clean_inventors.append(full_inventor)

        return ' '.join(clean_inventors)

    def _clean_mixed_field_content(self, raw_content: str, field_code: str) -> str:
        """Clean mixed field content by removing obvious artifacts."""
        cleaned = raw_content

        # Remove bullet points (these belong to other fields)
        cleaned = re.sub(r'•[^•]*', '', cleaned)

        # Remove field codes that got mixed in
        cleaned = re.sub(r'\(\d{2}\)[^(]*(?=\(|$)', '', cleaned)

        # Remove common artifacts
        cleaned = re.sub(r'\s*\d+\s+[A-Z]\s+\d+.*', '', cleaned)  # Page numbers

        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned

    def _find_field_boundaries(self, text: str) -> dict[str, list[str]]:
        """Find field boundaries using improved parsing logic."""
        fields = {}

        lines = text.split('\n')
        current_field = None
        current_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_field and current_lines:
                    current_lines.append('')  # Preserve empty lines within fields
                continue

            # Check for field code at start of line
            field_match = re.match(r'^\((\d{2})\)\s*(.*)', stripped)
            if field_match:
                # Save previous field
                if current_field and current_lines:
                    fields[current_field] = [l for l in current_lines if l.strip()]  # Remove empty lines

                # Start new field
                current_field = field_match.group(1)
                first_content = field_match.group(2).strip()
                current_lines = [first_content] if first_content else []
            elif current_field:
                # Continue current field - but be careful about boundaries
                if self._looks_like_field_continuation(stripped, current_field):
                    current_lines.append(stripped)
                else:
                    # This might be spillover from adjacent fields, stop here
                    if current_lines:
                        fields[current_field] = [l for l in current_lines if l.strip()]
                    current_field = None
                    current_lines = []

        # Save final field
        if current_field and current_lines:
            fields[current_field] = [l for l in current_lines if l.strip()]

        return fields

    def _looks_like_field_continuation(self, line: str, field_code: str) -> bool:
        """Determine if line belongs to current field or is spillover."""

        # Stop at obvious field boundaries
        if re.match(r'^\(\d{2}\)', line):
            return False

        # Field-specific logic
        if field_code == "72":  # Inventors
            # Continue if it's a bullet point or reasonable inventor content
            if line.startswith('•') or (',' in line and len(line.split(',')) <= 3):
                return True
            # Stop at patent references or long technical content
            if re.search(r'JP-A-|EP-A-|US-B1-|Patent|Bulletin|opposition', line, re.IGNORECASE):
                return False
            return len(line) < 100  # Stop at very long lines

        elif field_code == "56":  # References
            # Continue if it looks like patent references
            return bool(re.search(r'(EP|JP|US|WO)-[A-Z0-9-]+', line))

        elif field_code == "57":  # Abstract
            # Stop at Description/Claims headers
            if re.search(r'\b(Description|Claims|Patentansprüche|Beschreibung)\b', line, re.IGNORECASE):
                return False
            return True

        elif field_code in ["43", "12", "54"]:  # Single-line fields
            # These should not continue beyond first line
            return False

        # Default: continue unless it's obviously wrong
        return not re.search(r'Printed by|Jouve|PARIS|Within nine months', line, re.IGNORECASE)

    def _extract_inventors_by_bullets(self, content_lines: list[str]) -> str:
        """Extract inventors using bullet point pattern matching."""
        inventors = []

        for line in content_lines:
            if line.strip().startswith('•'):
                # Extract inventor name (should have surname, firstname pattern)
                inventor_text = line.strip()

                # Clean the bullet point
                inventor_text = re.sub(r'^•\s*', '', inventor_text)

                # Must have comma for surname, firstname pattern
                if ',' in inventor_text:
                    # Clean up common artifacts
                    inventor_text = re.sub(r'\s*c/o.*$', '', inventor_text)  # Remove c/o addresses
                    inventor_text = re.sub(r'\s*\d+\s*\([^)]+\)\s*$', '', inventor_text)  # Remove zip codes

                    # Extract just the name part (until location/address)
                    name_match = re.match(r'^([^,]+,\s*[^,\d]+)', inventor_text)
                    if name_match:
                        clean_name = name_match.group(1).strip()
                        inventors.append(f'• {clean_name}')

        return ' '.join(inventors)

    def _extract_front_page_fields_with_coordinates(self, words: list[dict], text: str) -> OrderedDict[str, str]:
        """Extract front page fields using coordinate information for better boundary detection."""
        fields = OrderedDict()

        if not words:
            return fields

        # Group words by approximate line (y-coordinate)
        lines = self._group_words_by_line(words)

        # Find field codes and their positions
        field_positions = {}
        for line_y, line_words in lines.items():
            line_text = ' '.join(word['text'] for word in line_words)
            field_matches = list(re.finditer(r'\((\d{2})\)', line_text))
            for match in field_matches:
                code = match.group(1)
                # Find the approximate x-coordinate of this field code
                char_pos = match.start()
                word_pos = 0
                x_coord = None
                for word in line_words:
                    word_len = len(word['text'])
                    if word_pos <= char_pos < word_pos + word_len:
                        x_coord = word['x0']
                        break
                    word_pos += word_len + 1  # +1 for space
                if x_coord is not None:
                    field_positions[code] = {'y': line_y, 'x': x_coord}

        # Extract content for each field using spatial awareness
        for code, position in field_positions.items():
            content_words = self._extract_field_content_by_position(lines, code, position, field_positions)
            if content_words:
                content_text = ' '.join(content_words)

                # Apply field-specific cleaning with content validation
                if code == "72":  # Inventors
                    cleaned = self._clean_inventors_field_spatial(content_text)
                elif code == "56":  # References
                    cleaned = self._parse_references(content_text)
                elif code == "84":  # Designated states - should be just country codes
                    cleaned = self._extract_country_codes_only(content_text)
                elif code in ["12", "54", "43", "45", "21", "22"]:  # Single/short fields
                    cleaned = self._extract_field_value_precisely(content_text, code)
                else:
                    cleaned = self._strip_field_label(content_text)

                if cleaned:
                    fields[code] = cleaned

        return fields

    def _group_words_by_line(self, words: list[dict]) -> dict[float, list[dict]]:
        """Group words by their y-coordinate (line)."""
        lines = {}

        for word in words:
            y = round(word['top'], 1)  # Round to avoid floating point issues
            if y not in lines:
                lines[y] = []
            lines[y].append(word)

        # Sort words within each line by x-coordinate
        for line_words in lines.values():
            line_words.sort(key=lambda w: w['x0'])

        return lines

    def _extract_field_content_by_position(self, lines: dict, code: str, position: dict, all_positions: dict) -> list[str]:
        """Extract field content using spatial positioning."""
        content_words = []
        start_y = position['y']
        start_x = position['x']

        # Find the next field position to know where to stop
        next_field_y = None
        for other_code, other_pos in all_positions.items():
            if other_code != code and other_pos['y'] > start_y:
                if next_field_y is None or other_pos['y'] < next_field_y:
                    next_field_y = other_pos['y']

        # Extract content from current line and subsequent lines until next field
        for line_y in sorted(lines.keys()):
            if line_y < start_y:
                continue
            if next_field_y is not None and line_y >= next_field_y:
                break

            line_words = lines[line_y]

            if line_y == start_y:
                # On the same line as field code, take words after the field code
                for word in line_words:
                    if word['x0'] > start_x + 50:  # Some offset to skip the field code itself
                        content_words.append(word['text'])
            else:
                # On subsequent lines, take words that are reasonably positioned
                if self._line_belongs_to_field(line_words, code, start_x):
                    content_words.extend(word['text'] for word in line_words)

        return content_words

    def _line_belongs_to_field(self, line_words: list[dict], field_code: str, field_x: float) -> bool:
        """Determine if a line belongs to a specific field based on positioning."""
        if not line_words:
            return False

        line_text = ' '.join(word['text'] for word in line_words)

        # Skip lines that start with other field codes
        if re.match(r'^\(\d{2}\)', line_text):
            return False

        # Skip common footer text
        if re.search(r'Note:|Within nine months|Printed by|Jouve|PARIS', line_text):
            return False

        # Field-specific logic
        if field_code == "72":  # Inventors
            # Include lines with bullet points or inventor names
            if line_text.strip().startswith('•'):
                return True
            # Include lines that look like inventor names or addresses (but not patent references)
            if ',' in line_text and not re.search(r'JP-A-|EP-A-|US-B1-|Patent|Bulletin|opposition', line_text):
                return True
            # Include location lines for inventors (company addresses)
            if re.search(r'Nippon Steel|Corporation|Futtsu-shi|Chiba|Hyogo|Kitakyushu|Fukuoka', line_text):
                return True
            return False

        elif field_code == "56":  # References
            # Continue if it looks like patent references
            return bool(re.search(r'(EP|JP|US|WO)-[A-Z0-9-]+', line_text))

        # For other fields, use simple heuristics
        return len(line_text) < 150  # Avoid very long technical content

    def _clean_inventors_field_spatial(self, content: str) -> str:
        """Clean inventors field using spatial extraction results."""
        if not content:
            return ""

        # Process the content to extract clean inventor entries
        lines = content.replace('•', '\n•').split('\n')
        inventors = []
        current_inventor = None
        current_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('•'):
                # Save previous inventor if exists
                if current_inventor and current_lines:
                    inventor_text = self._format_inventor_entry(current_inventor, current_lines)
                    if inventor_text:
                        inventors.append(inventor_text)

                # Start new inventor
                name_part = line[1:].strip()  # Remove bullet
                if ',' in name_part:
                    current_inventor = name_part
                    current_lines = []
                else:
                    current_inventor = None
                    current_lines = []
            elif current_inventor:
                # Add address/company line for current inventor
                if re.search(r'Nippon Steel|Corporation|Futtsu-shi|Chiba|Hyogo|Kitakyushu|Fukuoka|c/o', line):
                    current_lines.append(line)

        # Save final inventor
        if current_inventor and current_lines:
            inventor_text = self._format_inventor_entry(current_inventor, current_lines)
            if inventor_text:
                inventors.append(inventor_text)

        return '\n'.join(inventors)

    def _format_inventor_entry(self, name: str, address_lines: list[str]) -> str:
        """Format a single inventor entry with name and address."""
        # Clean the name
        name = re.sub(r'\s+', ' ', name).strip()

        # Format the entry
        result = f'• {name}'

        # Add address lines
        for addr_line in address_lines:
            addr_line = addr_line.strip()
            if addr_line:
                result += f'\n{addr_line}'

        return result

    def _extract_country_codes_only(self, content: str) -> str:
        """Extract only country codes from field (84), avoiding spillover."""
        if not content:
            return ""

        # Look for 2-letter country codes pattern
        country_codes = re.findall(r'\b[A-Z]{2}\b', content)

        # Stop at first inventor bullet point or other obvious spillover
        words = content.split()
        clean_words = []

        for word in words:
            # Stop at bullet points (inventors)
            if word.startswith('•'):
                break
            # Stop at obvious inventor names (surname, firstname pattern)
            if ',' in word and len(word) > 3:
                break
            # Include country codes and common labels
            if re.match(r'^[A-Z]{2}$', word) or word in ['Designated', 'Contracting', 'States:']:
                clean_words.append(word)

        result = ' '.join(clean_words)
        # Remove label if present and return just country codes
        result = re.sub(r'^.*States:\s*', '', result)
        return result.strip()

    def _extract_field_value_precisely(self, content: str, field_code: str) -> str:
        """Extract field value with precise boundaries to avoid spillover."""
        if not content:
            return ""

        # Split into lines and words for analysis
        lines = content.strip().split('\n')
        first_line = lines[0] if lines else ""

        if field_code == "45":  # Publication date
            # Should be date format, stop at obvious boundaries
            match = re.search(r'(\d{2}\.\d{2}\.\d{4}\s+Bulletin\s+\d{4}/\d{2})', content)
            if match:
                return f"Date of publication and mention of the grant of the patent:\n{match.group(1)}"
            # Fallback: take first line and clean
            return self._strip_field_label(first_line)

        elif field_code == "21":  # Application number
            # Extract just the application number
            match = re.search(r'(\d+\.\d+(?:\s+PCT/[A-Z0-9/]+)?)', content)
            if match:
                return f"Application number: {match.group(1)}"
            return self._strip_field_label(first_line)

        elif field_code == "22":  # Filing date
            # Extract just the date
            match = re.search(r'(\d{2}\.\d{2}\.\d{4})', content)
            if match:
                return f"Date of filing: {match.group(1)}"
            return self._strip_field_label(first_line)

        elif field_code == "43":  # Publication of application
            # Should be date and bulletin
            match = re.search(r'(\d{2}\.\d{2}\.\d{4}\s+Bulletin\s+\d{4}/\d{2})', content)
            if match:
                return f"Date of publication of application:\n{match.group(1)}"
            return self._strip_field_label(first_line)

        else:
            # For other fields, take first line and clean
            return self._strip_field_label(first_line)

    def _clean_two_column_field(self, field_content: str) -> str:
        """Clean two-column field content by removing common artifacts."""
        if not field_content:
            return field_content

        # Remove common junk patterns that leak from column mixing
        cleaned = field_content

        # Remove field codes that got mixed in
        cleaned = re.sub(r'\s*\(\d{2}\)\s*', ' ', cleaned)

        # Remove page numbers and references
        cleaned = re.sub(r'\s*\d+\s+[A-Z]?\s+\d+', '', cleaned)

        # Remove bulletin references
        cleaned = re.sub(r'\s*Bulletin\s*\d+/\d+', '', cleaned)

        # Remove PCT/publication patterns
        cleaned = re.sub(r'\s*PCT/[A-Z]+\d+/\d+', '', cleaned)
        cleaned = re.sub(r'\s*WO\s*\d+/\d+', '', cleaned)

        # Remove dates in DD.MM.YYYY format when mixed with other content
        cleaned = re.sub(r'\s*\d{2}\.\d{2}\.\d{4}\s*', ' ', cleaned)

        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned


    @staticmethod
    def _derive_patent_id(path: Path) -> str:
        return path.stem

    def _clean_patent_text(self, text: str) -> str:
        """Remove page headers, page numbers, and normalize patent text."""
        if not text:
            return text

        lines = text.splitlines()
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()

            # Skip page headers like "EP 1 816 226 B1 13 5 10 15 20 25 30 35 40 45 50 55"
            if PAGE_HEADER_PATTERN.match(stripped):
                continue

            # Skip standalone page numbers
            if PAGE_NUMBER_PATTERN.match(stripped):
                continue

            # Skip lines with only numbers and spaces (page continuation indicators)
            if re.match(r"^[\d\s]+$", stripped) and len(stripped.split()) > 3:
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _detect_claims_language(self, text: str) -> str:
        """Detect the language of claims section."""
        text_lower = text.lower().strip()

        # Exact matches first
        if text_lower == "patentansprüche":
            return "german"
        elif text_lower == "revendications":
            return "french"
        elif text_lower == "claims":
            return "english"

        # Partial matches for patterns like "Claims 1. ..."
        elif text_lower.startswith("patentansprüche"):
            return "german"
        elif text_lower.startswith("revendications"):
            return "french"
        elif text_lower.startswith("claims"):
            return "english"
        elif "patentansprüche" in text_lower:
            return "german"
        elif "revendications" in text_lower:
            return "french"
        else:
            return "english"  # Default

    def _extract_description_sections(self, pages: List[PageSummary]) -> list[PatentSection]:
        sections: list[PatentSection] = []
        current_paragraph_id: str | None = None
        current_text: list[str] = []
        start_page = end_page = None

        for page in pages:
            lines = [line.rstrip() for line in (page.text or "").splitlines()]
            for line in lines:
                base_line = line.strip()
                base_line = re.sub(r'^\d+\s+', '', base_line)
                match = PARAGRAPH_PATTERN.match(base_line)
                if match:
                    if current_paragraph_id and current_text:
                        sections.append(
                            PatentSection(
                                section_id=current_paragraph_id,
                                title=f"Paragraph {current_paragraph_id.split('_')[-1]}",
                                text=self._normalise_paragraph_text(current_text),
                                page_start=start_page or page.page_number,
                                page_end=end_page or page.page_number,
                                section_path=["description"],
                            )
                        )
                    para_number = match.group(1) or match.group(2)
                    current_paragraph_id = f"para_{para_number}"
                    current_text = [match.group(3).strip()]
                    start_page = page.page_number
                    end_page = page.page_number
                elif current_paragraph_id:
                    stripped = re.sub(r'^\d+\s+', '', line).strip()
                    if stripped and not stripped.isdigit():
                        current_text.append(stripped)
                    end_page = page.page_number

        if current_paragraph_id and current_text:
            sections.append(
                PatentSection(
                    section_id=current_paragraph_id,
                    title=f"Paragraph {current_paragraph_id.split('_')[-1]}",
                    text=self._normalise_paragraph_text(current_text),
                    page_start=start_page or pages[-1].page_number,
                    page_end=end_page or pages[-1].page_number,
                    section_path=["description"],
                )
            )

        return sections

    def _extract_tables(self, pages: List[PageSummary]) -> list[PatentTable]:
        tables: list[PatentTable] = []
        counter = 1
        for page in pages:
            if not page.tables:
                continue
            for table_entry in page.tables:
                raw_data = table_entry.get("data") if isinstance(table_entry, dict) else table_entry
                cleaned_rows = self._normalise_table_rows(raw_data)
                if not cleaned_rows:
                    continue

                caption, headers, body_rows = self._split_table_header(cleaned_rows)
                if not body_rows:
                    continue

                if not self._is_table_candidate(headers, body_rows):
                    continue

                caption_clean = caption.title() if caption else None
                # DEBUG: print caption detection
                # print(f"Table candidate on page {page.page_number}: {caption_clean}")
                table_id = f"table_{counter:04d}"
                counter += 1
                tables.append(
                    PatentTable(
                        table_id=table_id,
                        caption=caption_clean,
                        page_number=page.page_number,
                        headers=headers,
                        rows=body_rows,
                    )
                )
        return tables

    def _find_table_caption(self, page_text: str) -> str | None:
        if not page_text:
            return None
        match = re.search(r"Table\s+(\d+)", page_text, flags=re.IGNORECASE)
        if match:
            return f"Table {match.group(1)}"
        return None

    @staticmethod
    def _normalise_table_rows(raw_table: list[list[str | None]] | None) -> list[list[str]]:
        if not raw_table:
            return []
        cleaned: list[list[str]] = []
        max_cols = 0
        for raw_row in raw_table:
            if raw_row is None:
                continue
            row = [PatentParser._clean_table_cell(cell) for cell in raw_row]
            if any(cell for cell in row):
                cleaned.append(row)
                max_cols = max(max_cols, len(row))
        if not cleaned or max_cols == 0:
            return []
        normalised: list[list[str]] = []
        for row in cleaned:
            padded = row + [""] * (max_cols - len(row))
            normalised.append(padded)
        return normalised

    @staticmethod
    def _clean_table_cell(value: str | None) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        text = text.replace('(cid:176)', 'deg')
        text = re.sub(r"\(cid:[^\)]+\)", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _split_table_header(self, rows: list[list[str]]) -> tuple[str | None, list[str], list[list[str]]]:
        if not rows:
            return None, [], []
        max_cols = max(len(row) for row in rows if row)
        if max_cols == 0:
            return None, [], []
        normalised = [row + [""] * (max_cols - len(row)) for row in rows]

        caption = None
        for row in normalised[:3]:
            for col_idx, cell in enumerate(row):
                cell_text = cell.strip()
                if cell_text and re.match(r"^table", cell_text, flags=re.IGNORECASE):
                    caption = caption or cell_text
                    row[col_idx] = ""
        while normalised and not any(cell.strip() for cell in normalised[0]):
            normalised = normalised[1:]
        if not normalised:
            return caption, [], []

        header_rows = [normalised[0]]
        next_idx = 1
        while next_idx < len(normalised) and self._looks_like_header_row(normalised[next_idx]):
            header_rows.append(normalised[next_idx])
            next_idx += 1

        parent_texts: list[str] = []
        last_parent = ""
        for cell in header_rows[0]:
            cell_text = cell.strip()
            if cell_text:
                last_parent = cell_text
            parent_texts.append(last_parent)

        headers: list[str] = []
        for col_idx in range(max_cols):
            parts: list[str] = []
            parent = parent_texts[col_idx].strip()
            if parent:
                parts.append(parent)
            for layer_idx in range(1, len(header_rows)):
                cell = header_rows[layer_idx][col_idx].strip()
                if cell:
                    parts.append(cell)
            header_text = " / ".join(dict.fromkeys(part for part in parts if part))
            header_text = re.sub(r"^table[^/]*\/\s*", "", header_text, flags=re.IGNORECASE)
            header_text = re.sub(r"^table[^/]*$", "", header_text, flags=re.IGNORECASE)
            header_text = header_text.strip()
            if not header_text:
                header_text = f"Column {col_idx + 1}"
            headers.append(header_text)

        body_rows = normalised[len(header_rows):]
        filtered_rows: list[list[str]] = []
        for row in body_rows:
            if self._looks_like_header_row(row):
                continue
            if any(cell.strip() for cell in row):
                filtered_rows.append(row)

        return caption, headers, filtered_rows

    @staticmethod
    def _looks_like_header_row(row: list[str]) -> bool:
        tokens = [cell.strip() for cell in row if cell and cell.strip()]
        if not tokens:
            return False
        alpha_tokens = sum(1 for token in tokens if any(ch.isalpha() for ch in token))
        digit_tokens = sum(1 for token in tokens if token.isdigit())
        return alpha_tokens >= max(1, len(tokens) // 2) and digit_tokens < len(tokens)

    @staticmethod
    def _is_table_candidate(headers: list[str], rows: list[list[str]]) -> bool:
        if not headers or not rows:
            return False
        max_cols = len(headers)
        if max_cols < 3 or max_cols > 12:
            return False

        total_cells = sum(len(row) for row in rows if row)
        if total_cells == 0:
            return False
        non_empty = sum(1 for row in rows for cell in row if cell and cell.strip())
        if non_empty / total_cells < 0.4:
            return False

        short_value_rows = 0
        numeric_like_rows = 0
        for row in rows[: min(6, len(rows))]:
            compact_cells = [cell for cell in row if cell and len(cell.strip()) <= 60]
            if len(compact_cells) >= 2:
                short_value_rows += 1
            numeric_cells = [cell for cell in row if cell and any(ch.isdigit() for ch in cell)]
            if numeric_cells and len(numeric_cells) >= max(1, max_cols // 3):
                numeric_like_rows += 1
        if short_value_rows < 2 and numeric_like_rows == 0:
            return False

        lengths = [len(cell.strip()) for row in rows for cell in row if cell and cell.strip()]
        if lengths and statistics.median(lengths) > 40:
            return False

        return True

    def _extract_claims(self, pages: List[PageSummary]) -> list[PatentClaim]:
        claims: list[PatentClaim] = []
        in_claims = False
        current_number: str | None = None
        current_lines: list[str] = []
        start_page: int | None = None
        end_page: int | None = None
        current_language = "english"  # Track current claims language

        for page in pages:
            # Clean page text to remove headers and page numbers
            cleaned_text = self._clean_patent_text(page.text or "")
            lines = [line.rstrip() for line in cleaned_text.splitlines()]

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    if current_lines:
                        current_lines.append("")
                    continue

                # Check for claims section headers and detect language
                if CLAIMS_HEADING_PATTERN.match(stripped):
                    # If we were already in claims, finalize the previous claim before switching languages
                    if in_claims and current_number and any(part.strip() for part in current_lines):
                        claim = self._finalize_claim(
                            number=current_number,
                            lines=current_lines,
                            start_page=start_page or page.page_number,
                            end_page=end_page or page.page_number,
                            language=current_language,
                        )
                        if claim:
                            claims.append(claim)

                        # Reset for next language section
                        current_number = None
                        current_lines = []
                        start_page = end_page = None

                    in_claims = True
                    current_language = self._detect_claims_language(stripped)
                    LOGGER.debug(f"Detected claims language: {current_language}")
                    continue

                if not in_claims:
                    continue

                match = CLAIM_NUMBER_PATTERN.match(stripped)
                if match:
                    number = match.group(1).strip()
                    separator_index = match.end(1)
                    if (
                        separator_index < len(stripped)
                        and stripped[separator_index] == "."
                        and separator_index + 1 < len(stripped)
                        and stripped[separator_index + 1].isdigit()
                    ):
                        continuation = self._clean_claim_line(stripped)
                        if continuation:
                            if current_lines and current_lines[-1].endswith("-"):
                                current_lines[-1] = current_lines[-1][:-1] + continuation.lstrip()
                            else:
                                current_lines.append(continuation)
                            end_page = page.page_number
                        continue

                    if current_number and any(part.strip() for part in current_lines):
                        claim = self._finalize_claim(
                            number=current_number,
                            lines=current_lines,
                            start_page=start_page or page.page_number,
                            end_page=end_page or page.page_number,
                            language=current_language,
                        )
                        if claim:
                            claims.append(claim)

                    current_number = number
                    remainder = self._clean_claim_line(match.group(2))
                    current_lines = []
                    if remainder:
                        current_lines.append(remainder)
                    start_page = page.page_number
                    end_page = page.page_number
                    continue

                if current_number:
                    cleaned_line = self._clean_claim_line(stripped)
                    if cleaned_line:
                        if current_lines and current_lines[-1].endswith('-'):
                            current_lines[-1] = current_lines[-1][:-1] + cleaned_line.lstrip()
                        else:
                            current_lines.append(cleaned_line)
                        end_page = page.page_number

        if current_number and any(part.strip() for part in current_lines):
            claim = self._finalize_claim(
                number=current_number,
                lines=current_lines,
                start_page=start_page or (pages[-1].page_number if pages else 1),
                end_page=end_page or (pages[-1].page_number if pages else 1),
                language=current_language,
            )
            if claim:
                claims.append(claim)

        return claims

    @staticmethod
    def _clean_claim_line(text: str) -> str:
        cleaned = (text or "").strip()
        cleaned = re.sub(
            r"EP\s*\d+(?:\s*\d+)*\s*[A-Z]\s*\d+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\bpage\s*\d+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\d+\s+", "", cleaned)
        cleaned = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", cleaned)
        cleaned = re.sub(r"(\w+)-\s+\d+\s+(\w+)", r"\1\2", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned.isdigit():
            return ""
        return cleaned

    def _finalize_claim(
        self,
        *,
        number: str,
        lines: list[str],
        start_page: int,
        end_page: int,
        language: str = "english",
    ) -> PatentClaim | None:
        text = self._normalise_claim_text(lines)
        if not text:
            return None
        clean_number = number.strip()
        padded = clean_number.zfill(4) if clean_number.isdigit() else clean_number
        return PatentClaim(
            claim_id=f"claim_{language}_{padded}",
            number=clean_number,
            text=text,
            page_start=start_page,
            page_end=end_page,
            language=language,
        )

    @staticmethod
    def _normalise_paragraph_text(lines: list[str]) -> str:
        text = " ".join(line.strip() for line in lines if line.strip())
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def _normalise_claim_text(lines: list[str]) -> str:
        text = " ".join(part.strip() for part in lines if part.strip())
        return re.sub(r"\s+", " ", text)


def parse_patent_pdf(path: Path) -> PatentDocument:
    parser = PatentParser()
    return parser.parse(path)





