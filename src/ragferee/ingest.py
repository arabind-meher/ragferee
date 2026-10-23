"""Parse the Laws of the Game PDF into citable chunks."""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter

from ragferee.models import Chunk

LAWS_PAGES = range(44, 152)  # PDF pages holding Laws 1-17; equal to the printed page numbers

_LAW_DIVIDER = re.compile(r"^Law\s*(\d+)$")
_SECTION = re.compile(r"^\d+\.\s+\S")
_MIN_CHUNK_CHARS = 20  # shorter text is a stray diagram label
_CHUNK_LIST = TypeAdapter(list[Chunk])


@dataclass(frozen=True)
class Line:
    """One line of text on a PDF page, with the font details the parser needs."""

    page: int
    text: str
    size: float
    bold: bool


def read_lines(path: Path, pages: range = LAWS_PAGES) -> Iterator[Line]:
    """Yield the text lines of the given pages in reading order."""
    import pymupdf  # imported here: it takes about two seconds to load

    with pymupdf.open(path) as document:
        for number in pages:
            page_lines: list[tuple[float, float, Line]] = []
            for block in document[number - 1].get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    text = _clean("".join(span["text"] for span in line["spans"]))
                    if not text:
                        continue
                    first = line["spans"][0]
                    bold = "Bold" in first["font"]
                    item = Line(page=number, text=text, size=round(first["size"], 1), bold=bold)
                    page_lines.append((first["origin"][1], line["bbox"][0], item))
            page_lines.sort(key=lambda entry: (entry[0], entry[1]))
            for *_, item in page_lines:
                yield item


def _clean(text: str) -> str:
    return " ".join(text.replace("\x07", " ").split())


class LawsParser:
    """Turns the lines of the Laws pages into one chunk per section per page."""

    def parse(self, lines: Iterable[Line]) -> list[Chunk]:
        """Group lines into chunks tagged with their Law, section and page."""
        self._chunks: list[Chunk] = []
        self._law_number: int | None = None
        self._law_title = ""
        self._section = ""
        self._page = 0
        self._text: list[str] = []

        for line in lines:
            self._handle(line)
        self._flush()
        return self._chunks

    def _handle(self, line: Line) -> None:
        divider = _LAW_DIVIDER.match(line.text)
        if divider and line.size >= 40:  # Law divider page: "Law" and a huge number
            self._flush()
            self._law_number = int(divider.group(1))
            self._law_title = ""
            self._section = ""
        elif 30 <= line.size < 40:  # Law title, possibly over several lines
            self._law_title = f"{self._law_title} {line.text}".strip()
        elif not 8 <= line.size <= 10:  # footers, diagram labels and figure captions
            return
        elif line.bold and _SECTION.match(line.text):
            self._flush()
            self._section = line.text
        else:
            if line.page != self._page:
                self._flush()
                self._page = line.page
            self._text.append(line.text)

    def _flush(self) -> None:
        text = "\n".join(self._text)
        self._text = []
        if self._law_number is None or len(text) < _MIN_CHUNK_CHARS:
            return
        self._chunks.append(
            Chunk(
                law_number=self._law_number,
                law_title=self._law_title,
                section=self._section or self._law_title,
                page=self._page,
                text=text,
            )
        )


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    """Write chunks to a JSON file, creating its folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_CHUNK_LIST.dump_json(chunks, indent=2))


def load_chunks(path: Path) -> list[Chunk]:
    """Read chunks saved by save_chunks, validating every one."""
    return _CHUNK_LIST.validate_json(path.read_bytes())
