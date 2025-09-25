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

## Roadmap
We trimmed the original project down to this single node so we can focus on making it exceptional. Expect refinements to template options, smarter automatic placement, and better tooling for mangaka-style pages.

---
MangaPanelizer is based on the foundations laid by the original Comfyroll Studio project. We thank the previous maintainers for their work and plan to expand in our own direction from here.
