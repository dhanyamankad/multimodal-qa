"""
rag/ingest.py

PDF ingestion pipeline for Multimodal Q&A Pro.

Flow (per PRD Section 4 / Section 6.1):
    PDF upload
      -> extract text WITH page numbers (pypdf)
      -> for any page with no/near-no extractable text, render that page to
         an image (PyMuPDF) and OCR it via the Groq vision model instead
         (see _ocr_page below) -- this is what makes PPT-exported PDFs
         (flattened slide images, no real text layer) actually work
      -> RecursiveCharacterTextSplitter (chunk_size=500, overlap=100)
      -> embed with sentence-transformers/all-MiniLM-L6-v2 (HuggingFaceEmbeddings)
      -> store in ChromaDB with page-level metadata (filename + page number)

Page-level metadata is not optional: citations in both Chat Mode and
Report Mode depend on knowing exactly which page a chunk came from.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional

# stage, current, total, detail -> None. Called throughout ingest_pdf() so
# callers (main.py) can surface real progress instead of an indeterminate
# spinner. Stages: "text", "ocr_start", "ocr_done", "ocr_failed", "chunking",
# "embedding", "done".
ProgressCallback = Callable[[str, int, int, str], None]

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings

try:
    import fitz  # PyMuPDF — used only to rasterize image-only pages for OCR
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore

from agent.vision import vision_call

logger = logging.getLogger("rag.ingest")

# A page with fewer than this many extracted characters is treated as
# "effectively no text layer" and sent through OCR instead. Not 0, because
# pypdf sometimes pulls a stray header/footer/page-number off an otherwise
# fully-image slide, which would wrongly skip OCR for real content.
OCR_TEXT_THRESHOLD_CHARS = 20

# Resolution multiplier for rendering a PDF page to an image before OCR.
# 1.0 = 72 DPI (PDF native), which is too blurry for small slide text to
# OCR reliably; 2.5 ~= 180 DPI, a reasonable quality/size/latency tradeoff.
# If the resulting JPEG is still too large at this zoom (see
# OCR_MAX_B64_BYTES below), we step down through these zoom levels before
# giving up.
OCR_RENDER_ZOOM_LEVELS = (2.5, 1.8, 1.3)

# JPEG quality levels tried at each zoom level, highest quality first.
OCR_JPEG_QUALITIES = (85, 70, 55, 40)

# Groq's meta-llama/llama-4-scout-17b-16e-instruct enforces a hard 4MB limit
# on base64-encoded image request bodies (console.groq.com/docs/vision,
# "Request Size Limit (Base64 Encoded Images)") -- this is what produced the
# 413 "Request Entity Too Large" errors. A full page rendered at 2.5x zoom
# (~180 DPI) as PNG easily exceeds that for image-heavy slides (photos,
# gradients, embedded charts), since PNG doesn't compress that kind of
# content well. JPEG does much better, so we render as JPEG and, if still
# too big, step down quality and then resolution until the base64 payload
# comfortably fits. Target is 3.8MB base64 (~2.85MB raw), leaving headroom
# under the 4MB cap for the rest of the JSON request body.
OCR_MAX_B64_BYTES = 3_800_000

# SEPARATE from the byte-size cap above: Groq's vision pipeline also rejects
# images over ~33.2 megapixels ("images can contain at most 33177600 pixels"
# -- confirmed by the 400 errors seen in testing on a slide deck with
# unusually large source page dimensions). JPEG quality does NOT reduce
# pixel count, only file size, so the byte-size stepdown loop alone cannot
# fix this -- a highly-compressed but still-oversized-in-pixels JPEG will
# still get a 400. This has to be enforced as a hard cap on render zoom,
# computed per-page from its actual point dimensions, before quality is
# even considered.
OCR_MAX_PIXELS = 33_000_000  # stay a hair under Groq's exact 33,177,600 cap

OCR_PROMPT = (
    "This image is a page from a document (it may be a slide, scanned page, "
    "chart, or diagram). Transcribe ALL visible text exactly as written, "
    "preserving reading order (e.g. title, then body, then captions/labels). "
    "If there are charts, tables, or diagrams, also describe their content "
    "and any data values shown in words, after the transcribed text. Do not "
    "add commentary, summarization, or anything not visibly present on the "
    "page — this transcription is used for search, so it must reflect only "
    "what is actually on the page."
)


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
    ocr: bool = False  # True if this page's text came from vision OCR,
    # not pypdf -- surfaced in chunk metadata so the UI/citations can be
    # honest about it (OCR text is a transcription, not the literal PDF
    # text layer, and can occasionally misread a character).


@dataclass
class IngestResult:
    filename: str
    chunk_count: int
    page_count: int
    ocr_page_count: int = 0


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
    def _render_page_jpeg(pdf_path: str, page_number: int) -> bytes:
        """
        Render a single 1-indexed PDF page to JPEG bytes that satisfy BOTH
        of Groq's independent limits:
          1. Pixel count (OCR_MAX_PIXELS) -- fixed only by rendering at a
             lower zoom; JPEG quality cannot fix this, since quality only
             changes compression, not width x height.
          2. Base64 payload size (OCR_MAX_B64_BYTES) -- fixed by lowering
             JPEG quality (cheap) and, if that's not enough, zoom too.

        The pixel cap is computed per-page from its actual point dimensions
        (page sizes vary -- a wide slide-exported page hits the pixel cap
        at a much lower zoom than a standard A4/Letter page would), and is
        applied as a hard ceiling on every zoom candidate before any
        encoding is attempted. Only once a candidate is pixel-safe does the
        quality stepdown loop run to also satisfy the byte-size cap.

        If every combination is still over the byte-size limit (extremely
        dense page even at the pixel-safe zoom), returns the smallest one
        produced anyway -- vision_call will get a 413 from Groq for that
        page, which extract_pages already catches and logs as a skipped
        page rather than letting it crash the whole upload.
        """
        if fitz is None:
            raise RuntimeError(
                "PyMuPDF (fitz) is not installed — required to render "
                "image-only PDF pages for OCR. Add 'PyMuPDF' to requirements.txt."
            )
        doc = fitz.open(pdf_path)
        try:
            page = doc[page_number - 1]
            page_w, page_h = page.rect.width, page.rect.height  # points, 72/inch

            # Highest zoom that keeps this specific page's rendered pixel
            # count under the cap. Every configured zoom level is clamped
            # to this ceiling -- for most normal pages the ceiling is above
            # 2.5 and changes nothing; for oversized pages (like the one
            # that triggered the 400s) it pulls every candidate down.
            max_zoom_for_pixels = (OCR_MAX_PIXELS / (page_w * page_h)) ** 0.5

            zoom_candidates = sorted(
                {min(z, max_zoom_for_pixels) for z in OCR_RENDER_ZOOM_LEVELS},
                reverse=True,
            )

            smallest: Optional[bytes] = None
            for zoom in zoom_candidates:
                if zoom <= 0:
                    continue
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                for quality in OCR_JPEG_QUALITIES:
                    data = pix.tobytes("jpeg", jpg_quality=quality)
                    if smallest is None or len(data) < len(smallest):
                        smallest = data
                    # base64 inflates raw size by ~4/3; check against that
                    # estimate rather than actually encoding every attempt.
                    estimated_b64_len = (len(data) + 2) // 3 * 4
                    if estimated_b64_len <= OCR_MAX_B64_BYTES:
                        return data
            return smallest  # best effort; caller/Groq will reject if still too big
        finally:
            doc.close()

    @classmethod
    def _ocr_page(cls, pdf_path: str, page_number: int) -> str:
        """
        Render a single 1-indexed PDF page to an image and OCR it via the
        Groq vision model (agent/vision.py). Used when pypdf's text layer
        for that page is empty/near-empty -- the classic case being a
        PPT-exported PDF where each slide is one flattened image.

        Raises on failure (no fallback masking here) -- the caller
        (extract_pages) decides whether one bad page should sink the whole
        document or just be skipped with a logged warning.
        """
        image_bytes = cls._render_page_jpeg(pdf_path, page_number)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        text = vision_call(image_b64, OCR_PROMPT, image_format="jpeg")
        return (text or "").strip()

    @classmethod
    def extract_pages(
        cls, pdf_path: str, progress_cb: Optional[ProgressCallback] = None
    ) -> List[PageText]:
        """
        Extract text per page, 1-indexed. This is the step that makes
        page-level citation possible later -- do not collapse this into a
        single blob of text for the whole document.

        Any page where pypdf's text layer is empty/near-empty (see
        OCR_TEXT_THRESHOLD_CHARS) is rendered to an image and OCR'd via the
        vision model instead, so image-only pages (scans, PPT-exported
        slides) still produce searchable text. A page that fails OCR is
        logged and skipped rather than aborting the whole document.

        progress_cb, if given, is called as progress_cb(stage, current,
        total, detail) after every page so callers (main.py) can drive a
        real progress bar instead of an indeterminate spinner.
        """
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        pages: List[PageText] = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if len(text) >= OCR_TEXT_THRESHOLD_CHARS:
                pages.append(PageText(page_number=i, text=text, ocr=False))
                if progress_cb:
                    progress_cb("text", i, total_pages, f"page {i}/{total_pages}")
                continue

            # This is a real network round-trip to Groq and can take
            # several seconds per page; log before/after so a multi-page
            # OCR run shows visible progress in the terminal instead of
            # long silent gaps that look identical to a hang. See also the
            # 60s client timeout in agent/vision.py, which bounds how long
            # any single page can block this loop.
            logger.info("OCR: page %d/%d of '%s' -- starting", i, total_pages, pdf_path)
            if progress_cb:
                progress_cb("ocr_start", i, total_pages, f"OCR: page {i}/{total_pages}")
            try:
                ocr_text = cls._ocr_page(pdf_path, i)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "OCR failed for page %d/%d of '%s': %s", i, total_pages, pdf_path, exc
                )
                # Fall back to whatever tiny scrap of text pypdf did find
                # (could be nothing) rather than losing the page silently.
                if text:
                    pages.append(PageText(page_number=i, text=text, ocr=False))
                if progress_cb:
                    progress_cb("ocr_failed", i, total_pages, f"page {i}/{total_pages} failed, skipped")
                continue
            logger.info("OCR: page %d/%d of '%s' -- done (%d chars)", i, total_pages, pdf_path, len(ocr_text or ""))

            if ocr_text:
                pages.append(PageText(page_number=i, text=ocr_text, ocr=True))
            elif text:
                pages.append(PageText(page_number=i, text=text, ocr=False))
            # else: genuinely blank page (e.g. a section divider) -- skip.
            if progress_cb:
                progress_cb("ocr_done", i, total_pages, f"OCR: page {i}/{total_pages} done")

        return pages

    # ---- chunking --------------------------------------------------------

    def _chunk_pages(self, filename: str, pages: List[PageText], session_id: str):
        """
        Chunk each page's text independently (rather than chunking the
        whole document as one string) so every chunk can be tagged with
        exactly one page number. A chunk that straddled two pages would
        make citation ambiguous.

        Every chunk is also tagged with `session_id` so retrieve.py can
        filter strictly to the requesting session's own uploads. Without
        this, every session shares one global collection and users see
        each other's documents.
        """
        ids, documents, metadatas = [], [], []
        for page in pages:
            page_chunks = self._splitter.split_text(page.text)
            for idx, chunk in enumerate(page_chunks):
                # session_id folded into the id too, so the same PDF
                # re-uploaded in a different session gets distinct chunk ids
                # instead of upserting over (and leaking into) another
                # session's copy.
                chunk_id = hashlib.sha1(
                    f"{session_id}:{filename}:{page.page_number}:{idx}:{chunk[:50]}".encode("utf-8")
                ).hexdigest()
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append(
                    {
                        "filename": filename,
                        "page_number": page.page_number,
                        "session_id": session_id,
                        "ocr": page.ocr,
                    }
                )
        return ids, documents, metadatas

    # ---- public API --------------------------------------------------------

    def ingest_pdf(
        self,
        pdf_path: str,
        filename: Optional[str] = None,
        session_id: str = "default",
        progress_cb: Optional[ProgressCallback] = None,
    ) -> IngestResult:
        """
        Ingest a single PDF file into ChromaDB.

        Args:
            pdf_path: local filesystem path to the uploaded PDF (temp path
                is fine, we only read from it here).
            filename: display name to store in metadata / show in citations.
                Defaults to the basename of pdf_path.
            session_id: isolates this document's chunks so only the
                uploading session's own retrieve() calls can ever see them.
                Required for real multi-user isolation — see the SESSION
                ISOLATION note in retrieve.py.
            progress_cb: optional real-time progress callback (see
                ProgressCallback above), called through extraction/OCR,
                chunking, and embedding so callers can drive a real progress
                bar instead of an indeterminate spinner.

        Returns:
            IngestResult with chunk_count, used by the
            POST /api/upload/pdf endpoint's confirmation response.
        """
        filename = filename or os.path.basename(pdf_path)

        pages = self.extract_pages(pdf_path, progress_cb=progress_cb)
        if not pages:
            raise ValueError(
                f"No extractable text or OCR-able content found in '{filename}'. "
                "It may be entirely blank pages, or OCR failed on every page "
                "(check GROQ_API_KEY / vision model availability)."
            )

        if progress_cb:
            progress_cb("chunking", 0, 1, "splitting extracted text into chunks")
        ids, documents, metadatas = self._chunk_pages(filename, pages, session_id)
        if not documents:
            raise ValueError(f"'{filename}' produced no chunks after splitting.")

        if progress_cb:
            progress_cb("embedding", 0, 1, f"embedding {len(documents)} chunks")
        embeddings = self._embeddings.embed_documents(documents)

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        if progress_cb:
            progress_cb("done", 1, 1, "indexing complete")

        return IngestResult(
            filename=filename,
            chunk_count=len(documents),
            page_count=len(pages),
            ocr_page_count=sum(1 for p in pages if p.ocr),
        )

    @property
    def collection(self):
        """Expose the raw collection for retrieve.py to query against."""
        return self._collection

    @property
    def embeddings(self):
        """Expose the embedding function so retrieve.py embeds queries the same way."""
        return self._embeddings

    def delete_session(self, session_id: str) -> None:
        """Purge every chunk tagged with this session_id. Called when a
        session ends (page refresh / explicit reset) so chunks don't sit
        around forever and so a reused session_id can never accidentally
        inherit stale data."""
        self._collection.delete(where={"session_id": session_id})


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

    SESSION ISOLATION: session_id is now also stored as per-chunk metadata
    (not just used to namespace the filename) so retrieve.py can filter
    strictly by session. Every uploaded doc must be tagged with a real
    session_id or it becomes visible to every other session.

    Kept as an int-returning adapter for backward compatibility with any
    existing caller of this exact signature. Use ingest_pdf_result() below
    if you also need ocr_page_count (main.py's /api/upload/pdf does).
    """
    return _ingest_pdf_impl(pdf_path, session_id).chunk_count


def ingest_pdf_result(
    pdf_path: str,
    session_id: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> IngestResult:
    """Same as ingest_pdf() but returns the full IngestResult (chunk_count,
    page_count, ocr_page_count) so callers can tell the user how many pages
    needed OCR."""
    return _ingest_pdf_impl(pdf_path, session_id, progress_cb=progress_cb)


def _ingest_pdf_impl(
    pdf_path: str,
    session_id: Optional[str],
    progress_cb: Optional[ProgressCallback] = None,
) -> IngestResult:
    filename = os.path.basename(pdf_path)
    if session_id:
        prefix = f"{session_id}_"
        if filename.startswith(prefix):
            filename = filename[len(prefix):]

    return get_ingestor().ingest_pdf(
        pdf_path,
        filename=filename,
        session_id=session_id or "default",
        progress_cb=progress_cb,
    )


def delete_session_documents(session_id: str) -> None:
    """Adapter for main.py's session-cleanup endpoint."""
    get_ingestor().delete_session(session_id)