"""Node registration for MangaPanelizer."""

from .node_templates import CR_ComicPanelTemplates
from .image_transform import MangaImageTransform
from .manga_speech_bubbles import NODE_CLASS_MAPPINGS as SPEECH_BUBBLE_CLASSES
from .manga_speech_bubbles import NODE_DISPLAY_NAME_MAPPINGS as SPEECH_BUBBLE_DISPLAY_NAMES

NODE_CLASS_MAPPINGS = {
    "CR_ComicPanelTemplates": CR_ComicPanelTemplates,
    "MangaImageTransform": MangaImageTransform,
    **SPEECH_BUBBLE_CLASSES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CR_ComicPanelTemplates": "MangaPanelizer",
    "MangaImageTransform": "Manga Image Transform",
    **SPEECH_BUBBLE_DISPLAY_NAMES,
}
