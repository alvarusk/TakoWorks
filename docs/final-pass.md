# Final Pass

La casilla **Final Pass** esta disponible en Transcriber y Translator y recuerda
su estado por separado. Tambien puede activarse con `--final-pass` en ambas CLI.
Necesita la clave OpenAI de la configuracion de TakoWorks o `OPENAI_API_KEY`.

Despues de la ultima traduccion, revisa los subtitulos con el mismo modelo GPT
que usa Transcriber. Translator traduce con DeepL y utiliza ese GPT para la
revision. Si solo se han seleccionado otros motores en Transcriber, se utiliza
igualmente GPT para Final Pass.

Transcriber envia el ASS intermedio `_ja_asr` o `_zh_asr`, incluidas las
explicaciones gramaticales y semanticas, junto con las traducciones de todos los
motores disponibles. Translator envia el ASS original ingles o frances y la
traduccion de DeepL. No busca archivos vecinos que puedan pertenecer a otra
version del episodio. Las instrucciones de terminologia de la serie tambien se
aplican a esta revision.

El proceso escribe `<nombre>_final.ass` y `<nombre>_final.txt` junto a los archivos
de salida, conservando originales y traducciones intermedias. En Translator,
`<nombre>` es el nombre del archivo traducido. El TXT se abre con Bloc de notas
en Windows, incluso si la revision falla o se cancela.

Las reglas que recibe el modelo estan en
`src/takoworks/shared/final_pass.md`. Incluyen los cortes sintacticos, los turnos
de dialogo, la preferencia por una segunda linea mas larga y los limites de
20,00 CPS, 40 CPL y dos lineas. Los tiempos, estilos y demas metadatos del ASS
se conservan. CPS cuenta espacios y puntuacion visibles, excluye tags y saltos;
CPL cuenta caracteres visibles de cada linea. Los dibujos ASS se conservan sin
traduccion y quedan anotados como exentos de limites de texto.

La revision trabaja en bloques de 20 subtitulos con contexto contiguo. Tras cada
respuesta se comprueban los limites, los IDs y los tags, y se piden correcciones
hasta tres intentos por bloque. La sintaxis y la fidelidad se revisan mediante
el modelo; la comprobacion local de cortes solo detecta palabras funcionales al
final de las lineas internas, sin sustituir un analisis gramatical completo.

Si no se pueden cumplir los limites conservando el sentido y los tiempos, se
guarda la mejor propuesta y el TXT indica **REQUIERE REVISION**, con los
incumplimientos concretos. No se recorta texto automaticamente. Si no hay
traducciones, la API falla sin dar ninguna propuesta valida o no se obtiene
texto ASS seguro, no se publica un nuevo final; se conserva cualquier final
anterior y se explica el fallo en el TXT. Los tokens de Final Pass se registran
en el informe y, en Transcriber, se suman al consumo de GPT.
