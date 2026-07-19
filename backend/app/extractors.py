async def extract_upload(upload) -> str:
    """Extract useful text without persisting the upload."""
    name = (upload.filename or "").lower()
    content = await upload.read()
    if name.endswith(".pdf"):
        from io import BytesIO
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if name.endswith(".docx"):
        from io import BytesIO
        from docx import Document

        document = Document(BytesIO(content))
        return "\n".join(p.text for p in document.paragraphs if p.text.strip()).strip()
    if name.endswith((".txt", ".md", ".csv")):
        return content.decode("utf-8", errors="replace").strip()
    raise ValueError("Unsupported file type. Use PDF, DOCX, TXT, MD, or CSV.")
