# MangaPanelizer

MangaPanelizer is a focused set of custom nodes for ComfyUI that help you design clean manga and comic page layouts. The project currently concentrates on a single, streamlined node: **CR_ComicPanelTemplates**. It offers fast presets, supports both traditional left-to-right and manga-style right-to-left reading directions, and can auto-place your rendered panels inside the layout.

## Features
- Prebuilt grid, horizontal, and vertical panel templates.
- Optional custom layout strings for fine control over complex pages.
- Batch image support with automatic cropping to fit each panel.
- Flexible colour controls for borders, panel fills, and page backgrounds.

## Installation
1. Navigate to your `ComfyUI/custom_nodes` directory.
2. Clone this repository: `git clone https://github.com/your-user/MangaPanelizer.git`.
3. Restart ComfyUI. The MangaPanelizer node will appear under **🧩 MangaPanelizer/Templates**.

## Usage
Add the **CR_ComicPanelTemplates** node to your workflow, choose a preset (or provide a custom layout code), and optionally connect a batch of images. The node returns a ready-to-use page tensor along with a short description string.

### Custom layout strings and diagonal cuts
Selecting the **custom** template unlocks fine-grained control over the panel arrangement:

- Start the string with `H` to describe rows (top to bottom) or `V` to describe columns (left to right).
- Each digit that follows represents how many panels appear in that row or column. For example, `H123` renders one wide panel on the first row, two panels on the second row, and three on the third row.
- Insert a slash (`/`) between two digits to replace the straight separator with a diagonal cut between those sections. The diagonal leans across the full width (for `H`) or height (for `V`) of the page.

Examples:

- `H1/23` &rarr; One panel on top, a diagonal split, then two panels in the middle row and three on the bottom row.
- `V2/3` &rarr; Two columns on the left, a diagonal divider, and three columns stacked on the right.

You can chain multiple `/` markers in a single layout code to add several diagonal transitions across the page.

## Roadmap
We trimmed the original project down to this single node so we can focus on making it exceptional. Expect refinements to template options, smarter automatic placement, and better tooling for mangaka-style pages.

---
MangaPanelizer is based on the foundations laid by the original Comfyroll Studio project. We thank the previous maintainers for their work and plan to expand in our own direction from here.
