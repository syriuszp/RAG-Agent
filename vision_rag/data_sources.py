"""Data source management for VisionRAG."""

from typing import List, Dict, Callable
from PIL import Image
from utils import pdf_to_images

Item = Dict[str, object]

class SourceManager:
    """Manage loading of different content sources."""

    def __init__(self) -> None:
        self.loaders: Dict[str, Callable] = {}

    def register_loader(self, mime_prefix: str, loader: Callable) -> None:
        """Register a loader for a mime type prefix."""
        self.loaders[mime_prefix] = loader

    def load(self, uploaded_files) -> List[Item]:
        """Load items from uploaded files using registered loaders."""
        items: List[Item] = []
        for file in uploaded_files:
            for prefix, loader in self.loaders.items():
                if file.type.startswith(prefix):
                    items.extend(loader(file))
                    break
        return items


def _image_loader(file) -> List[Item]:
    img = Image.open(file).convert("RGB")
    return [{"type": "image", "name": file.name, "img": img, "emb": None}]


def _pdf_loader(file) -> List[Item]:
    pdf_bytes = file.read()
    pages = pdf_to_images(pdf_bytes)
    items: List[Item] = []
    for i, page_img in enumerate(pages):
        name = f"{file.name} - Page {i + 1}"
        items.append({"type": "pdf_page", "name": name, "img": page_img, "emb": None})
    return items


source_manager = SourceManager()
source_manager.register_loader("image/", _image_loader)
source_manager.register_loader("application/pdf", _pdf_loader)
