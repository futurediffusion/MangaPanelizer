"""Node registration for MangaPanelizer."""

from .node_templates import CR_ComicPanelTemplates
from .manga_speech_bubbles import NODE_CLASS_MAPPINGS as SPEECH_BUBBLE_CLASSES
from .manga_speech_bubbles import NODE_DISPLAY_NAME_MAPPINGS as SPEECH_BUBBLE_DISPLAY_NAMES

NODE_CLASS_MAPPINGS = {
    "CR_ComicPanelTemplates": CR_ComicPanelTemplates,
    **SPEECH_BUBBLE_CLASSES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CR_ComicPanelTemplates": "MangaPanelizer Panel Templates",
    **SPEECH_BUBBLE_DISPLAY_NAMES,
}
