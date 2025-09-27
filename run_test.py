from node_templates import CR_ComicPanelTemplates
from PIL import Image

node = CR_ComicPanelTemplates()
image, help_text = node.layout(
    page_width=1024,
    page_height=1920,
    template='H1/2',
    reading_direction='left to right',
    border_thickness=56,
    outline_thickness=5,
    outline_color='black',
    panel_color='white',
    background_color='white',
    custom_panel_layout='H1/2',
    images=None,
    internal_padding=20,
    division_height_offset=0,
    division_horizontal_offset=0,
    outline_color_hex='#000000',
    panel_color_hex='#000000',
    bg_color_hex='#FFFFFF',
)
print(image.shape)

