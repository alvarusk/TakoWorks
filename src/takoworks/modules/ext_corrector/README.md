# TakoWorks - Corrector (v0.471 repack)

## Uso rápido
```bat
py corrector.py "archivo.ass"
```

### Modo servidor (todo en un comando)
```bat
py corrector.py "archivo.ass" --serve
```

Esto:
1) Genera el HTML
2) Abre el navegador en `http://127.0.0.1:8765/<reporte.html>`
3) Arranca un servidor local que sirve el HTML y expone `/api/add-word`
4) En el HTML, el botón **➕ Diccionario** añade palabras directamente al diccionario de LT

Ctrl+C para cerrar el servidor.

## Omitir errores por defecto
- Para desactivar desde LT: `disabled_rules` / `disabled_categories`
- Para ocultar solo en el HTML: `suppress_rules` / `suppress_categories`

## Diccionario desde el HTML
En errores de ortografía (misspelling), aparece **➕ Diccionario**.
- Si el servidor está activo (`--serve` o `lt_bridge_server.py`), se añade a LT al instante.
- Si no, se guarda en cola (copiar/descargar .txt).

## Scripts extra
- `lt_bridge_server.py`: servidor bridge independiente (si quieres usar el HTML como file:// y aun así añadir a LT).
- `lt_words_apply.py`: añade a LT desde un TXT (una palabra por línea).

## Cierre automático del servidor al cerrar el HTML
En modo `--serve`, el HTML hace un *heartbeat* (ping) al servidor.
Cuando cierras la pestaña/ventana, el ping se detiene y el servidor se cierra automáticamente tras un timeout.

- `--serve-idle 45` (por defecto 45s)
Ejemplo:
```bat
py corrector.py "archivo.ass" --serve --serve-idle 20
```
