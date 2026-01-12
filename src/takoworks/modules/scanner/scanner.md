# Scanner (TakoWorks) - Notas actuales

Resumen del comportamiento actual del modulo Scanner.

## Flujo
- Vista previa simplificada: permite borrar zonas (seleccion libre) y marcar un recorte rectangular (arriba/abajo/izq/der).
- No hay rotacion ni seleccion por rectangulos.
- Las zonas a borrar se guardan por pagina como poligonos en ratios (0..1).
- Opcion: omitir paginas vacias (sin tinta despues de cortes y borrados).
- El core procesa todas las paginas con el flujo legacy:
  - render a imagen
  - aplicar borrados
  - recorte rectangular
  - deteccion de columnas (derecha a izquierda)
  - OCR con YomiToku

## Datos guardados
- skip_polys_by_page: dict pagina -> lista de poligonos (ratios)
- crop_rect: recorte global (left/right/top/bottom)
- crop_rect_by_page: recortes por pagina

## Archivos principales
- src/takoworks/modules/scanner/panel.py
- src/takoworks/modules/scanner/core.py
