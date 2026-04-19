from typing import Literal

ImageMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]

MEDIA_TYPES: dict[str, ImageMediaType] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

IMAGE_EXTENSIONS = set(MEDIA_TYPES.keys())

VISION_EXTRACTION_PROMPT = (
    "You are a technical document analyst. Extract ALL text and information from this image thoroughly.\n\n"
    "Instructions:\n"
    "- Transcribe all visible text exactly as shown\n"
    "- Describe diagrams, schematics, and drawings in detail (components, connections, flow direction)\n"
    "- Note part numbers, model numbers, measurements, and specifications\n"
    "- Describe the layout and spatial relationships between elements\n"
    "- For tables, preserve the row/column structure\n"
    "- Mark any text that is unclear or partially visible as [illegible]\n"
    "- Do NOT add interpretation or opinions — only describe what is visible\n\n"
    "Output the extracted content as plain text."
)

MAX_VISION_FILE_SIZE = 20 * 1024 * 1024
