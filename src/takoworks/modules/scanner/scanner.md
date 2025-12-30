# Scanner (TakoWorks) — Resumen de este chat (para Codex en VS Code)

Este documento resume **lo acordado y desarrollado** en este chat sobre el módulo **Scanner**, incluyendo decisiones de UX, atajos de teclado, rotación por página y el problema de la vista previa “negra”.

---

## 1) Objetivo general del cambio

Pasar de seleccionar “columnas” con clics a un flujo más natural:

- **Selección por rectángulo** (clic + arrastrar) en la vista previa.
- Ese **rectángulo** es exactamente el área que se exporta como imagen para OCR.

Además, añadir:
- Navegación de páginas con teclado (**A/D**).
- **Rotación por página** con teclado (**Q/E**) para enderezar escaneos mal alineados.
- Robustez de render en PDFs “raros” (vista previa negra).

---

## 2) Evolución de requisitos (cronología)

1) **Cambio de selección de columnas**
- Antes: marcar columnas con clics (bordes / separadores).
- Pedido: “clic y arrastrar formando un rectángulo; el área delimitada es lo que se genera como imagen”.
- Se implementó primero como rectángulo con altura fijada por `upper/lower` (modo mixto).

2) **Rectángulo total (X/Y libre)**
- Confirmación del usuario: quiere **rectángulo total**, no solo X.
- Diseño final: almacenar por página una lista de rectángulos `rects_by_page[page]`.
- Los rectángulos se guardan en **ratios 0..1** respecto al tamaño renderizado para ser independientes del zoom.

3) **Atajos de navegación**
- Cambio solicitado: flechas → **A** (izquierda / página anterior), **D** (derecha / página siguiente).

4) **Rotación manual por página**
- Problema: PDFs escaneados torcidos ⇒ rectángulos poco fiables.
- Solicitud: rotación fina por página:
  - **Q** rota ligeramente a la izquierda
  - **E** rota ligeramente a la derecha
- (Opcional propuesto): Shift para paso grande, `0` para reset, `R` para auto-deskew.

5) **Problemas encontrados**
- Error inicial: `AttributeError: '_PreviewWindow' object has no attribute '_prev_page'`
  - Causa: métodos quedaron fuera de la clase por error de indentación.
  - Solución: corregir estructura/indentación para que `_prev_page/_next_page` y bindings existan.

- Luego: **Vista previa negra**
  - Se ve la ventana y controles, pero el canvas es negro, sin traceback.
  - Primera hipótesis/solución: forzar render RGB (`fitz.csRGB`).
  - El usuario confirma: **sigue negro**.
  - Conclusión probable: PDF con **CMYK / DeviceN / alfa** (o stride/padding) ⇒ los `samples` se estaban interpretando mal.
  - Próximo paso: implementar conversión robusta `pixmap_to_bgr()` que:
    - convierta pixmap a RGB cuando no sea 1/3 canales,
    - componga alfa sobre blanco si hay transparencia,
    - respete `pix.stride` (padding por fila).

---

## 3) Funcionalidad final deseada (definitiva)

### 3.1 Selección por rectángulos libres (por página)
- En la vista previa:
  - `ButtonPress-1`: inicio del drag
  - `B1-Motion`: dibuja rectángulo temporal
  - `ButtonRelease-1`: normaliza coords y guarda rectángulo persistente
- Guardado:
  - `rects_by_page: Dict[int, List[Tuple[x0r,y0r,x1r,y1r]]]` (ratios 0..1)
- UI:
  - Overlay verde para rects persistentes.
  - Botones útiles (según versión que ya existe): reset, deshacer, copiar áreas de pág previa, etc.

### 3.2 Navegación de páginas con teclado
- **A**: página anterior
- **D**: página siguiente

### 3.3 Rotación fina por página
- **Q**: rotar -delta grados
- **E**: rotar +delta grados
- Guardado:
  - `rot_deg_by_page: Dict[int, float]`
- Aplicación:
  - Se aplica en la vista previa (para seleccionar con precisión).
  - Se aplica en el core al generar imágenes para OCR.

### 3.4 Core: prioridad de rectángulos
- Si la página tiene `rects_by_page[page]`:
  - render → aplicar rotación → recortar cada rect → exportar imágenes → OCR
- Si no hay rects:
  - fallback al modo legacy (upper/lower y/o columnas automáticas si existen).

---

## 4) Problema actual: vista previa negra (estado)

**Estado:** incluso tras forzar RGB, la vista previa sigue completamente negra.

**Hipótesis más probable:**
- Pixmap en **CMYK / DeviceN**, o con **alfa**,
- O lectura incorrecta del buffer por **stride** (padding por fila),
- Resultado: el array final se interpreta mal y termina viéndose negro.

**Solución propuesta (para Codex):**
- Implementar un helper único y compartido (o duplicado idéntico) en `panel.py` y `core.py`:

### 4.1 `pixmap_to_bgr(pix) -> np.ndarray` (robusto)
Requisitos:
- Convertir el pixmap a un formato conocido antes de leer `samples`.
- Manejar:
  - `pix.n` (número de canales),
  - `pix.alpha`,
  - `pix.stride` (bytes por fila).

Estrategia (alto nivel):
1. Obtener pixmap (idealmente con `alpha=True` para poder componer correctamente).
2. Si colorspace/canales no son RGB/RGBA:
   - convertir a RGB (o a RGBA y luego componer).
3. Construir `np.ndarray` respetando `stride`:
   - reshape usando altura y `stride`, luego recortar a `w * n_channels`.
4. Si hay alfa:
   - componer sobre fondo blanco.
5. Convertir a **BGR** (OpenCV) para el resto del pipeline.

### 4.2 Debug recomendado (si aún falla)
- Log por página:
  - `pix.n`, `pix.alpha`, `pix.stride`, `pix.colorspace`
- Guardar PNG temporal y abrirlo fuera:
  - si el PNG ya está negro → render/conversión
  - si PNG está bien pero canvas negro → problema Tk/PhotoImage

---

## 5) Archivos a tocar (en el repo real)
Según el traceback del usuario:
- `C:\TakoWorks\src\takoworks\modules\scanner\panel.py`
- `C:\TakoWorks\src\takoworks\modules\scanner\core.py`

---

## 6) Checklist de validación (manual)

1) Vista previa:
- El PDF **se ve** (no negro).
- Arrastrar crea rectángulo y se guarda (overlay).
- A/D cambia de página.
- Q/E rota y el render se actualiza (rects se mantienen en la página correcta).

2) Ejecución Scanner:
- Si hay rects en una página, se generan imágenes solo de esas áreas.
- Rotación se aplica antes del recorte.
- OCR procesa esas imágenes.
- Fallback legacy funciona si no hay rects.

---

## 7) Prompt listo para Codex (copiar/pegar)

"""  
En `src/takoworks/modules/scanner/`, implementa/ajusta:

1) Vista previa: selección por rectángulo libre (clic+arrastrar) guardada por página en ratios `rects_by_page[page]`.
2) Teclas: A=prev, D=next.  
3) Rotación por página: Q/E ajusta `rot_deg_by_page[page]` y re-renderiza; la rotación también debe aplicarse en el core al exportar imágenes para OCR.
4) Arreglar vista previa negra: implementa `pixmap_to_bgr(pix)` robusto (CMYK/DeviceN/alfa/stride), úsalo tanto en `panel.py` como en `core.py`.
   - Respetar `pix.stride`.
   - Si hay alfa, componer sobre blanco.
   - Asegurar salida BGR para OpenCV.
5) Si una página tiene rectángulos, el core debe priorizarlos sobre el modo legacy.

Incluye una lista corta de pruebas manuales y añade logs mínimos solo si es imprescindible (idealmente sin prints finales).  
"""  

