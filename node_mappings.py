"""Node registration for MangaPanelizer."""

from .node_templates import CR_ComicPanelTemplates

NODE_CLASS_MAPPINGS = {
    "CR_ComicPanelTemplates": CR_ComicPanelTemplates,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CR_ComicPanelTemplates": "MangaPanelizer Panel Templates",
}
