"""
rag/ingest.py

PDF ingestion pipeline for Multimodal Q&A Pro.

Flow (per PRD Section 4 / Section 6.1):
    PDF upload
      -> extract text WITH page numbers (pypdf)
      -> RecursiveCharacterTextSplitter (chunk_size=500, overlap=100)
      -> embed with sentence-transformers/all-MiniLM-L6-v2 (HuggingFaceEmbeddings)
      -> store in ChromaDB with page-level metadata (filename + page number)

Page-level metadata is not optional: citations in both Chat Mode and
Report Mode depend on knowing exactly which page a chunk came from.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import List, Optional

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = "documents"


@dataclass
class PageText:
    """Raw text extracted from a single PDF page, before chunking."""
    page_number: int  # 1-indexed, matches what a human would cite
    text: str


@dataclass
class IngestResult:
    filename: str
    chunk_count: int
    page_count: int


class DocumentIngestor:
    """
    Owns the PDF -> ChromaDB pipeline. One instance can be reused across
    uploads within a session; the ChromaDB client is created once and
    persists to disk so documents survive across requests within the same
    container lifetime (NOTE: HF Spaces containers are ephemeral on
    restart -- documented as a known limitation in README.md).
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self._collection = self._client.get_or_create_collection(name=collection_name)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    # ---- extraction ----------------------------------------------------

    @staticmethod
    def extract_pages(pdf_path: str) -> List[PageText]:
        """
        Extract text per page, 1-indexed. This is the step that makes
        page-level citation possible later -- do not collapse this into a
        single blob of text for the whole document.
        """
        reader = PdfReader(pdf_path)
        pages: List[PageText] = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append(PageText(page_number=i, text=text))
        return pages

    # ---- chunking --------------------------------------------------------

    def _chunk_pages(self, filename: str, pages: List[PageText]):
        """
        Chunk each page's text independently (rather than chunking the
        whole document as one string) so every chunk can be tagged with
        exactly one page number. A chunk that straddled two pages would
        make citation ambiguous.
        """
        ids, documents, metadatas = [], [], []
        for page in pages:
            page_chunks = self._splitter.split_text(page.text)
            for idx, chunk in enumerate(page_chunks):
                chunk_id = hashlib.sha1(
                    f"{filename}:{page.page_number}:{idx}:{chunk[:50]}".encode("utf-8")
                ).hexdigest()
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append(
                    {
                        "filename": filename,
                        "page_number": page.page_number,
                    }
                )
        return ids, documents, metadatas

    # ---- public API --------------------------------------------------------

    def ingest_pdf(self, pdf_path: str, filename: Optional[str] = None) -> IngestResult:
        """
        Ingest a single PDF file into ChromaDB.

        Args:
            pdf_path: local filesystem path to the uploaded PDF (temp path
                is fine, we only read from it here).
            filename: display name to store in metadata / show in citations.
                Defaults to the basename of pdf_path.

        Returns:
            IngestResult with chunk_count, used by the
            POST /api/upload/pdf endpoint's confirmation response.
        """
        filename = filename or os.path.basename(pdf_path)

        pages = self.extract_pages(pdf_path)
        if not pages:
            raise ValueError(
                f"No extractable text found in '{filename}'. "
                "It may be a scanned/image-only PDF (OCR is out of scope for this build)."
            )

        ids, documents, metadatas = self._chunk_pages(filename, pages)
        if not documents:
            raise ValueError(f"'{filename}' produced no chunks after splitting.")

        embeddings = self._embeddings.embed_documents(documents)

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        return IngestResult(
            filename=filename,
            chunk_count=len(documents),
            page_count=len(pages),
        )

    @property
    def collection(self):
        """Expose the raw collection for retrieve.py to query against."""
        return self._collection

    @property
    def embeddings(self):
        """Expose the embedding function so retrieve.py embeds queries the same way."""
        return self._embeddings


# Module-level singleton so main.py and retrieve.py share one Chroma client
# instead of each opening their own (avoids file-lock contention on the
# persisted directory).
_ingestor: Optional[DocumentIngestor] = None


def get_ingestor() -> DocumentIngestor:
    global _ingestor
    if _ingestor is None:
        _ingestor = DocumentIngestor()
    return _ingestor

def ingest_pdf(pdf_path: str, session_id: Optional[str] = None) -> int:
    """
    Adapter for main.py (Vanshi's module), which expects:
        ingest_pdf(pdf_path, session_id=...) -> int   (chunk count)

    main.py saves uploads as "{session_id}_{original_filename}" (see
    UPLOAD_DIR handling in main.py's upload_pdf endpoint), so strip that
    prefix back off here to get a clean, citation-friendly filename before
    handing off to the real DocumentIngestor.
    """
    filename = os.path.basename(pdf_path)
    if session_id:
        prefix = f"{session_id}_"
        if filename.startswith(prefix):
            filename = filename[len(prefix):]

    result = get_ingestor().ingest_pdf(pdf_path, filename=filename)
    return result.chunk_count