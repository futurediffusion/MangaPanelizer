"""Node registration for MangaPanelizer."""

from .nodes.comic_panel_templates import CR_ComicPanelTemplates

NODE_CLASS_MAPPINGS = {
    "CR_ComicPanelTemplates": CR_ComicPanelTemplates,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CR_ComicPanelTemplates": "MangaPanelizer Panel Templates",
}
