"""Embedding and question-answer processing for VisionRAG."""

from typing import List, Dict, Tuple
from utils import get_cohere_embedding, gemini_vqa, image_to_bytes, find_most_similar

Item = Dict[str, object]


def compute_embeddings(items: List[Item], cohere_api: str) -> None:
    """Compute embeddings for items if not already computed."""
    for item in items:
        if item.get("emb") is None and item["type"] in {"image", "pdf_page"}:
            img_bytes = image_to_bytes(item["img"])
            item["emb"] = get_cohere_embedding(cohere_api, img_bytes, input_type="image")


def answer_question(question: str, items: List[Item], cohere_api: str, gemini_api: str) -> Tuple[str, Item, float]:
    """Find best matching item and generate answer using Gemini."""
    q_emb = get_cohere_embedding(cohere_api, question, input_type="text")
    emb_list = [item["emb"] for item in items]
    idx, sim = find_most_similar(q_emb, emb_list)
    best_item = items[idx]
    img_bytes = image_to_bytes(best_item["img"])
    answer = gemini_vqa(gemini_api, img_bytes, question)
    return answer, best_item, sim
