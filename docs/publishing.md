# Publicación en PyPI

El nombre de distribución propuesto es `rpca-kit` y el módulo importable es
`rpca_kit`. Que un nombre no tenga actualmente una página pública no garantiza
que PyPI permita registrarlo. Comprueba el resultado de la publicación antes de
anunciar una versión como disponible.

## Preparar y revisar los archivos

```bash
python -m pip install -e '.[dev]'
python scripts/release.py
```

El script ejecuta las pruebas, verifica estilo, construye las distribuciones en
un directorio temporal y revisa los metadatos con Twine. Después copia los archivos
validados a `dist/`. No sube archivos sin `--publish`.

Para probar la instalación de una rueda fuera del directorio fuente:

```bash
python -m venv /tmp/rpca-wheel-test
/tmp/rpca-wheel-test/bin/python -m pip install dist/rpca_kit-0.1.0-py3-none-any.whl
cd /tmp
/tmp/rpca-wheel-test/bin/python -c 'import rpca_kit; print(rpca_kit.__version__)'
```

La distribución fuente `.tar.gz` se instala con `pip install archivo.tar.gz`.
Para compartir también todo el proyecto como ZIP, ejecuta desde el directorio
superior, excluyendo credenciales, cachés y entornos virtuales:

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
root = Path('rpca-kit')
excluded = {'.git', '.venv', '__pycache__', '.pytest_cache', '.ruff_cache',
            'build', 'dist'}
with ZipFile('rpca-kit-source.zip', 'w', ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if any(part in excluded or part.endswith('.egg-info') for part in relative.parts):
            continue
        if path.is_file() and path.name not in {'.env', '.pypirc'}:
            archive.write(path, Path(root.name) / relative)
PY
```

## Opción recomendada: Trusted Publishing desde GitHub

Este repositorio incluye `.github/workflows/publish.yml`. El archivo por sí solo
no configura una cuenta de PyPI ni crea un publicador autorizado.

1. Crea un repositorio de GitHub bajo tu cuenta y sube el código revisado.
2. En GitHub, crea un entorno de Actions llamado `pypi`; opcionalmente limita
   las ramas o configura revisores para el despliegue.
3. En tu cuenta de [PyPI](https://pypi.org/), configura un publicador de confianza
   pendiente para un proyecto nuevo, o uno autorizado en el proyecto existente.
   Selecciona GitHub y registra propietario, repositorio, archivo de workflow
   `publish.yml` y entorno `pypi` exactamente como aparecen en GitHub.
4. Ejecuta manualmente **Publish to PyPI** en Actions sobre la revisión que deseas
   distribuir. Primero se ejecutan pruebas de NumPy y PyTorch en CPU. El trabajo
   de publicación obtiene una identidad OIDC temporal;
   no se guarda un token de PyPI en el repositorio.
5. Verifica la versión y ambos archivos en `https://pypi.org/project/rpca-kit/`.
   Instala la versión desde un entorno limpio antes de anunciarla.

Guía oficial: [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/).

## Alternativa: token local seguro

Usa un token de API de una cuenta autorizada para publicar el proyecto. Para
un proyecto nuevo puede ser necesario un token con el alcance inicial adecuado;
después usa un token limitado al proyecto. No pegues tokens en chats, código,
commits ni comandos con valores literales.

Desde una terminal de confianza, puedes capturarlo sin eco y sin guardarlo en
el historial:

```bash
read -r -s -p 'PyPI API token: ' TWINE_PASSWORD
echo
export TWINE_PASSWORD
export TWINE_USERNAME='__token__'
python scripts/release.py --publish
unset TWINE_PASSWORD TWINE_USERNAME
```

El token permanece temporalmente en el entorno del proceso. También puedes
configurar un gestor de credenciales compatible con Twine. El script usa
`--non-interactive`, para fallar de forma clara cuando falta autenticación.

Para probar el flujo en TestPyPI, configura sus propias credenciales y ejecuta:

```bash
python scripts/release.py --publish --repository testpypi
```

PyPI y TestPyPI son servicios y credenciales diferentes. La publicación es una
acción externa: cada número de versión debe identificar archivos finales;
PyPI no permite reemplazar una distribución publicada con el mismo nombre.
