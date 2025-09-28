# MangaPanelizer

MangaPanelizer es un paquete de nodos personalizados para ComfyUI enfocado en diseñar páginas de manga y cómic listas para imprimir. El nodo principal **CR_ComicPanelTemplates** genera composiciones limpias a partir de plantillas predefinidas o cadenas personalizadas y puede rellenar los paneles con tus imágenes renderizadas.

## Características clave
- Plantillas G/H/V para cuadr?culas, filas y columnas con variaciones diagonales (`/`, `*`, `:angulo`).
- Controles separados para márgenes externos (`border_thickness`) y separación interna entre paneles (`internal_padding`).
- Desplazamientos ajustables para diagonales (`division_height_offset`, `diagonal_slant_offset (division_horizontal_offset)`) que permiten mover el punto de encuentro en ambos ejes.
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
- `division_height_offset`: desplaza la unión vertical de diagonales (afecta `H` y columnas dentro de `V`).
- `diagonal_slant_offset (division_horizontal_offset)`: desplaza la unión horizontal de diagonales (afecta `H` y `V`).

#### Entradas opcionales
- `images`: lote de tensores. Cada panel recibe la siguiente imagen disponible; si faltan, se rellenan con el color del panel.

## Plantillas incluidas
```
G22  G33
H2   H3   H12  H13  H21  H23  H31  H32  H1*2  H1/2  H2*1  H2/1
V2   V3   V12  V13  V21  V23  V31  V32  V1*2  V1/2  V2*1  V2/1  V1/*2
```
`*` y `/` indican versiones diagonales frecuentes. Usa `custom` para combinaciones más complejas.

## Sintaxis de cadenas personalizadas
1. Comienza con `H` (filas de arriba a abajo) o `V` (columnas de izquierda a derecha).
2. Cada dígito representa cuántos paneles contiene ese bloque.
3. Usa separadores para diagonales y offsets:
   - `/` (horizontal):
     - En plantillas `H`: diagonal entre filas adyacentes (panel inferior se inclina).
     - En plantillas `V`: diagonal vertical entre columnas adyacentes.
   - `*` (vertical dentro del bloque):
     - En `H`: diagonal dentro de una fila, inclinando la línea que separa columnas.
     - En `V`: diagonal horizontal dentro de una columna, inclinando la separación entre paneles apilados.
   - `:ángulo`: opcional al final de la cadena para fijar el ángulo de las diagonales (0–90). Por defecto 18° aprox (`0.2`).

4. Se admiten múltiples diagonales encadenadas: cada `/` o `*` afecta a la transición siguiente.

### Ejemplos
- `H1*2:30` ? Panel superior completo y dos paneles inferiores separados por una diagonal inclinada 30°.
- `H2/1` ? Dos filas; diagonal entre la primera y segunda fila.
- `V1*2` ? Columna izquierda completa y columna derecha con dos paneles separados por una diagonal horizontal.
- `V1/2` ? Columna izquierda completa y diagonal vertical hacia una columna con dos paneles rectos.
- `V2/*2` ? Dos columnas a la izquierda, diagonal vertical hacia una columna derecha con dos paneles y diagonal horizontal interna.

## Consejos de uso
- **Separaciones**: `internal_padding` añade espacio entre paneles; `border_thickness` envuelve el lienzo final.
- **Ajustes de diagonales**: combina `division_height_offset` y `diagonal_slant_offset (division_horizontal_offset)` para mover el punto de encuentro de las líneas. Valores positivos empujan hacia abajo/derecha, negativos hacia arriba/izquierda.
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


