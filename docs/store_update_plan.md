# MS Store y boton "Actualizar" (plantilla VoiceX)

## Publicar en Microsoft Store (modo VoiceX)
- Reservar nombre y crear app en Partner Center. (Datos actuales)
  - Name: `KingdomM.TakoWorks`
  - Publisher: `CN=CCDE0167-EDA3-4A6F-B196-577B92A14DEC`
  - PublisherDisplayName: `Kingdom M`
  - Package Family Name: `KingdomM.TakoWorks_dhxzbhgcx0x2c`
  - Store ID: `9P5HF1G89F57`
- Versionado: usa la versi?n de `src/takoworks/__init__.py` (bump con `python bin/bump_version.py ...`) y fija en el manifest formato 1.x.y.0 sin sufijos.
- Empaquetar como MSIX desde el binario PyInstaller (`dist/TakoWorks`) usando MSIX Packaging Tool o `makeappx`. Incluir AppxManifest con:
  - DisplayName, Description, Versión (sin prerelease), iconos (44/50/150/310 px).
  - Capabilities solo las necesarias (InternetClient, etc.).
  - Publisher que coincide con el certificado.
- Firmar el MSIX con `signtool sign /f <cert.pfx> /p <pass> /fd sha256 /a <TakoWorks.msix>`.
- Subir el MSIX a Partner Center -> Submission -> Packages. Completar Store listing, pricing y destinos. Guardar y enviar (el Store se encarga de updates automáticos).
- CI opcional: añadir job en `.github/workflows/release.yml` que:
  - Construya PyInstaller.
  - Empaquete MSIX (script de empaquetado) y lo firme con secretos `PFX_CERT`, `PFX_PASS`.
  - Use StoreBroker (`Submit-ApplicationPackage`) o API de Partner Center para subir el MSIX al canal que elijas.

## Botón "Actualizar" via GitHub Actions (build más reciente)
- Workflow actual `release.yml` genera el artefacto `takoworks-windows` con el zip `TakoWorks_win64.zip`.
- La app consultará la rama `main` en GitHub Actions, obtendrá el último run `success` del workflow `release.yml`, y si la versión remota (`src/takoworks/__init__.py` en main) es mayor que la local, descargará ese artefacto y sustituirá la instalación.
- Requisitos de entorno para que funcione en clientes:
  - Variable `GITHUB_TOKEN` con scope `repo` (para descargar el artefacto).
  - Carpeta de instalación con permisos de escritura (ideal: `%LOCALAPPDATA%\\TakoWorks` o MS Store que ya gestiona updates).
- Flujo en cliente (similar a VoiceX):
  1) Al pulsar "Actualizar", compara versión local vs `main`.
  2) Si hay nueva, baja el artefacto de Actions, prepara la copia y relanza TakoWorks tras aplicar.
  3) Si no hay cambios, muestra "Ya estás al día".