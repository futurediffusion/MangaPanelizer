# MangaPanelizer

MangaPanelizer es un paquete de nodos personalizados para ComfyUI enfocado en diseñar páginas de manga y cómic listas para imprimir. El nodo principal **CR_ComicPanelTemplates** genera composiciones limpias a partir de plantillas predefinidas o cadenas personalizadas y puede rellenar los paneles con tus imágenes renderizadas.

## Características clave
- Plantillas G/H/V para cuadr?culas, filas y columnas con ?ngulos controlados ?nicamente mediante `first_division_angle` y `second_division_angle`.
- Controles separados para márgenes externos (`border_thickness`) y separación interna entre paneles (`internal_padding`).
- Desplazamientos ajustables para diagonales (`first_division_angle`, `diagonal_slant_offset (second_division_angle)`) que permiten mover el punto de encuentro en ambos ejes.
- Trazos uniformes con antialiasing para bordes y diagonales sin “sierra”.
- Colores personalizables para contorno, panel y fondo; compatibilidad con lotes de imágenes.
- Lectura izquierda?derecha o derecha?izquierda con un solo cambio de parámetro.

## Instalación
1. Entra en tu carpeta `ComfyUI/custom_nodes`.
2. Clona el repositorio o copia esta carpeta: `git clone https://github.com/your-user/MangaPanelizer.git`.
3. Reinicia ComfyUI. Encontrarás el nodo bajo **MangaPanelizer / Templates**.

## Nodo disponible
### CR_ComicPanelTemplates
Genera una página completa y devuelve:
- `image`: tensor de imagen listo para ComfyUI.
- `show_help`: texto con recordatorios rápidos del uso.

#### Entradas obligatorias
- `page_width`, `page_height`: tamaño del lienzo interior (sin contar `border_thickness`).
- `template`: plantilla predefinida (ver lista). El valor `custom` habilita la cadena personalizada.
- `reading_direction`: `left to right` o `right to left` (invierte horizontalmente la página al final).
- `border_thickness`: margen exterior que rodea toda la composición.
- `outline_thickness`: grosor del trazo de paneles.
- `outline_color`, `panel_color`, `background_color`: paleta predefinida compartida.
- `custom_panel_layout`: cadena usada cuando `template = custom` (se ignora en otros casos, pero siempre está visible para que puedas editarla rápido).
- `internal_padding`: separación entre paneles sucesivos.
- `first_division_angle`: desplaza la unión vertical de diagonales (afecta `H` y columnas dentro de `V`).
- `diagonal_slant_offset (second_division_angle)`: desplaza la unión horizontal de diagonales (afecta `H` y `V`).

#### Entradas opcionales
- `images`: lote de tensores. Cada panel recibe la siguiente imagen disponible; si faltan, se rellenan con el color del panel.

## Plantillas incluidas
```
G22  G33
H2   H3   H12  H13  H21  H23  H31  H32
V2   V3   V12  V13  V21  V23  V31  V32
```
Las cadenas solo describen la estructura: letras `H` o `V` seguidas de d?gitos. Controla los ?ngulos desde los par?metros del nodo.

## Sintaxis de cadenas personalizadas
1. Comienza con `H` (filas de arriba a abajo) o `V` (columnas de izquierda a derecha).
2. Cada d?gito representa cu?ntos paneles contiene ese bloque.
3. Mant?n `first_division_angle = 0` y `second_division_angle = 0` para divisiones rectas. Ajusta cualquiera de los dos par?metros para inclinar las l?neas cuando lo necesites.

### Ejemplos
- `H12` - Primer bloque a p?gina completa y segundo bloque dividido en dos columnas.
- `H21` - Dos bloques arriba y uno abajo.
- `V12` - Primera columna completa y segunda columna con dos paneles apilados.
- `V21` - Dos columnas a la izquierda y una columna amplia a la derecha.
- `V32` - Tres columnas principales y una columna con dos paneles apilados.

## Consejos de uso
- **Separaciones**: `internal_padding` añade espacio entre paneles; `border_thickness` envuelve el lienzo final.
- **Ajustes de diagonales**: combina `first_division_angle` y `diagonal_slant_offset (second_division_angle)` para mover el punto de encuentro de las líneas. Valores positivos empujan hacia abajo/derecha, negativos hacia arriba/izquierda.
- **Antialiasing**: los bordes se trazan en alta resolución y se reducen, evitando “sierra” incluso con diagonales gruesas.
- **Lectura oriental**: selecciona `right to left` para espejar toda la página sin cambiar la cadena de layout.
- **Relleno de imágenes**: conecta una lista de tensores (por ejemplo, con `LoadImage` + `ImageBatch`). El nodo recorta cada imagen al aspecto del panel antes de pegarla.

## Salidas
- **image**: página final como tensor ComfyUI (RGB).
- **show_help**: descripción corta con recordatorios de comandos.

## Roadmap
- Herramientas opcionales para numeración de paneles y texto auxiliar.
- Layouts compatibles con márgenes interiores asimétricos.
- Exportadores directos a PDF/PNG en lote.

---
Basado en el trabajo previo de Comfyroll Studio. Agradecemos la base original y seguimos expandiendo la herramienta con la comunidad.


