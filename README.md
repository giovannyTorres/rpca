# rpca-kit

Descomposición robusta de matrices en componentes de bajo rango y dispersa, con
NumPy para CPU y PyTorch opcional para CPU o GPU CUDA. Python 3.10 o posterior.

```text
D = L + S + E
```

`L` representa la estructura de bajo rango; `S`, las entradas o columnas atípicas;
`E`, el residuo. La biblioteca entrega diagnósticos de convergencia junto con las
matrices. La versión inicial es experimental y utiliza SVD exacta.

## Instalación

Desde una distribución publicada en PyPI:

```bash
python -m pip install rpca-kit
```

Desde este código fuente, antes de publicar:

```bash
python -m pip install .
```

Para usar PyTorch:

```bash
python -m pip install 'rpca-kit[gpu]'
# Desde el código fuente: python -m pip install '.[gpu]'
```

El extra instala PyTorch, pero la disponibilidad de CUDA depende del hardware,
el controlador y la distribución de PyTorch instalada. Selecciona la distribución
adecuada en [la guía oficial de PyTorch](https://pytorch.org/get-started/locally/).
La instalación básica requiere solamente NumPy.

## Uso en CPU

```python
import numpy as np
from rpca_kit import pcp

rng = np.random.default_rng(7)
left = rng.standard_normal((80, 3))
right = rng.standard_normal((3, 60))
low_rank = left @ right
sparse = np.zeros_like(low_rank)
indices = rng.choice(sparse.size, size=100, replace=False)
sparse.flat[indices] = rng.choice([-20.0, 20.0], size=indices.size)
data = low_rank + sparse

result = pcp(data, tol=1e-7, max_iter=1000)
print(result.converged, result.n_iter, result.error)
L, S = result
assert np.allclose(data, L + S + result.residual)
```

`result.error` depende del criterio de parada del método; consulta
[los métodos y criterios](https://github.com/giovannyTorres/rpca/blob/main/docs/algorithms.md). Reconstruir la entrada no demuestra
por sí solo que se hayan recuperado las componentes reales. Las pruebas sintéticas
también comparan las componentes estimadas con la verdad conocida.

## Uso en GPU

```python
import torch
from rpca_kit import pcp

data = torch.randn(100, 80, dtype=torch.float64, device="cuda")
result = pcp(data, backend="torch", device="cuda")
print(result.low_rank.device)       # cuda:0
cpu_result = result.to_numpy()      # copia explícita de resultados a CPU
```

Un tensor de PyTorch se conserva en su dispositivo con `backend="auto"`.
Para transferir una matriz NumPy a GPU, usa `device="cuda"`. Solicitar CUDA sin
una GPU disponible produce un error; no se cambia silenciosamente a CPU.
Las operaciones de los algoritmos se ejecutan sin construir grafos de autograd.
Los escalares de diagnóstico se transfieren al host en cada iteración.

## Variantes

| Función | Modelo | Cuándo usarla |
| --- | --- | --- |
| `pcp` | PCP por IALM | Corrupciones dispersas en entradas, sin ruido denso explícito. |
| `admm` | PCP por ADMM con penalización fija | El mismo modelo convexo, con control de la penalización. |
| `stable_pcp` | PCP penalizado por FISTA | Residuo denso, con regularización ajustable. |
| `outlier_pursuit` | Bajo rango más columnas atípicas | Corrupción concentrada por columnas. |
| `godec` | Proyecciones alternadas de rango y cardinalidad | Rango y número de entradas dispersas conocidos o seleccionados externamente. |

```python
from rpca_kit import RPCA, decompose, stable_pcp, godec

stable = stable_pcp(data, tau=0.5, lambda_=0.1)
fixed_rank = godec(data, rank=3, cardinality=100)
model = RPCA(method="ialm", tol=1e-7).fit(data)
same_api = decompose(data, method="admm")
```

`lambda_=None` selecciona `1 / sqrt(max(m, n))`; es una referencia inicial,
especialmente para PCP, y no un ajuste universal para todas las variantes.
IALM incrementa la penalización cuando domina el error primal; ADMM la mantiene fija.
GoDec requiere `rank` y `cardinality` explícitos. Los parámetros y diferencias
entre modelos se explican en [docs/algorithms.md](https://github.com/giovannyTorres/rpca/blob/main/docs/algorithms.md).

## Resultados y límites

- `low_rank`, `sparse` y `residual` conservan el backend de cálculo.
- `converged`, `n_iter`, `error`, `objective` e `history` permiten revisar la parada.
- `rank` calcula el rango numérico de la componente de bajo rango.
- `.to_numpy()` devuelve un resultado independiente en CPU.
- Las entradas deben ser matrices densas, reales, finitas y no vacías. Se
  preservan `float32` y `float64`; los enteros se convierten a `float64`.

Con `float32`, usa inicialmente `tol=1e-5`; la tolerancia por defecto `1e-7`
puede quedar por debajo de la precisión alcanzable y producir una advertencia
de falta de convergencia. La biblioteca conserva la tolerancia que solicites.

PCP necesita supuestos de identificabilidad: una matriz de bajo rango también
dispersa puede admitir varias descomposiciones. GoDec es no convexo y puede llegar
a un mínimo local. Una bandera de convergencia no constituye una garantía de
recuperación estadística.

La SVD completa cuesta aproximadamente `O(m*n*min(m,n))` por iteración y requiere
memoria proporcional al tamaño de la matriz. Esta versión no implementa flujo de
datos, matrices dispersas, SVD aleatoria, datos faltantes ni estimación automática
del rango. En matrices pequeñas, la sincronización y transferencia pueden hacer
que GPU sea más lenta que CPU.

## Desarrollo y distribución

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check src tests examples scripts
python scripts/release.py
```

La última orden valida pruebas, construye una rueda y una distribución fuente y
revisa los metadatos sin subir archivos. Las pruebas CUDA requieren hardware
real; omitirlas por falta de hardware no valida la ejecución en GPU.

Consulta [docs/publishing.md](https://github.com/giovannyTorres/rpca/blob/main/docs/publishing.md) para publicar en PyPI y
[examples/synthetic.py](https://github.com/giovannyTorres/rpca/blob/main/examples/synthetic.py) para ejecutar un ejemplo reproducible.

## Referencias

- Candès, Li, Ma y Wright. *Robust Principal Component Analysis?* JACM, 2011.
  [DOI: 10.1145/1970392.1970395](https://doi.org/10.1145/1970392.1970395).
- Lin, Chen y Ma. *The Augmented Lagrange Multiplier Method for Exact Recovery
  of Corrupted Low-Rank Matrices*, 2010. [arXiv:1009.5055](https://arxiv.org/abs/1009.5055).
- Zhou, Li, Wright, Candès y Ma. *Stable Principal Component Pursuit*, 2010.
  [arXiv:1001.2363](https://arxiv.org/abs/1001.2363). La implementación usa la
  formulación penalizada descrita en la documentación, no la restricción explícita
  de una cota de ruido de ese artículo.
- Xu, Caramanis y Sanghavi. *Robust PCA via Outlier Pursuit*, 2010.
  [arXiv:1010.4237](https://arxiv.org/abs/1010.4237).
- Zhou y Tao. *GoDec: Randomized Low-rank & Sparse Matrix Decomposition in Noisy
  Case*, ICML, 2011. Se implementa el núcleo determinista de proyecciones
  alternadas con SVD exacta, sin la aceleración aleatoria del artículo.

Licencia MIT. Véase [LICENSE](https://github.com/giovannyTorres/rpca/blob/main/LICENSE).
