# Modelos, parámetros y convergencia

Para una entrada `D` de tamaño `m × n`, todos los métodos devuelven `L`, `S` y
`E = D - L - S`. Las normas y criterios se calculan sobre las matrices sin
normalización previa de los datos. No se centra ni estandariza automáticamente.

## PCP: IALM y ADMM

El problema convexo es:

```text
min ||L||_* + lambda ||S||_1    sujeto a D = L + S
```

`||L||_*` suma los valores singulares y `||S||_1` suma valores absolutos por entrada.
`pcp(D, lambda_=None, ...)` usa IALM y actualiza la penalización; `admm(D,
lambda_=None, mu=None, ...)` mantiene fija la penalización `mu`. Ambas variantes
alternan umbralización de valores singulares, umbralización escalar y una
actualización de la variable dual.

`lambda_=None` usa `1/sqrt(max(m,n))`. Una `lambda_` mayor penaliza más las
entradas dispersas. La penalización `mu` controla el procedimiento numérico,
no el peso relativo del modelo. `tol` debe ser positivo; `max_iter`, un entero
positivo. Revisa `converged` después de ajustar límites de iteración.

La penalización inicial por defecto es `1.25 / ||D||_2`. IALM la multiplica por
`rho=1.5` sólo cuando el error primal supera al dual, hasta un máximo de
`mu_inicial * 1e7`; `rho` debe ser al menos 1. Esta continuación permite avanzar
en el objetivo cuando aún domina el cambio dual. El historial incluye `penalty`.
El criterio final es el máximo de:

```text
primal = ||D-L-S||_F / ||D||_F
dual   = mu * ||S_actual-S_anterior||_F / max(1, ||Y_actual||_F)
```

Aquí `Y` es el multiplicador de Lagrange actualizado. Ambos errores deben ser
menores o iguales que `tol`; el historial conserva cada uno por separado.

## PCP estable penalizado

`stable_pcp(D, tau=1.0, lambda_=None, ...)` minimiza:

```text
0.5 ||D - L - S||_F² + tau ||L||_* + tau lambda ||S||_1
```

La implementación usa gradiente proximal acelerado (FISTA). El gradiente del
término cuadrático respecto al par `(L,S)` tiene constante de Lipschitz 2.
Una `tau` mayor aumenta la regularización; una `tau` muy pequeña puede exigir
más iteraciones. La solución admite residuo denso no nulo. La tolerancia se
aplica a un criterio de estacionariedad/cambio y no exige residuo cero.

Se usa paso `1/2` con reinicios cuando la extrapolación aumenta el objetivo.
Para el punto extrapolado `Z` y la siguiente pareja proximal `P(Z)`, el error es
`2*||P(Z)-Z||_F / max(1,||D||_F)`, donde la norma conjunta suma los cuadrados de
las normas de ambas componentes. Es una medida de gradiente proximal.

Este objetivo penalizado no equivale, para cualquier `tau` fija, a especificar
directamente una cota `||E||_F <= delta`. La biblioteca no estima automáticamente
`tau` ni el nivel de ruido.

## Outlier Pursuit

`outlier_pursuit(D, lambda_=None, mu=None, ...)` resuelve:

```text
min ||L||_* + lambda sum_j ||S[:,j]||_2    sujeto a D = L + S
```

La proximidad sobre `S` reduce normas de columnas completas. El resultado tiene
columnas dispersas, no necesariamente entradas dispersas. Orienta los datos de
modo que cada muestra potencialmente atípica sea una columna. La selección
teórica de `lambda` depende de supuestos sobre el número de columnas atípicas;
el valor por defecto compartido con PCP es sólo un punto inicial de ajuste.

El criterio de parada primal/dual coincide con el de ADMM descrito arriba.
En columnas arbitrariamente corrompidas no se identifican, en general, los valores
originales de `L`; el objetivo principal es recuperar el subespacio limpio y
localizar columnas atípicas.

## GoDec determinista

`godec(D, rank=r, cardinality=k, ...)` alterna una aproximación de rango `r`
mediante SVD exacta y la retención de las `k` entradas de mayor valor absoluto
del residuo. Corresponde al núcleo de proyecciones alternadas del modelo:

```text
min 0.5 ||D - L - S||_F²
sujeto a rank(L) <= r, número de entradas no nulas de S <= k
```

`r` y `k` son cotas de complejidad, no valores inferidos por la biblioteca. No
se incorpora la proyección bilateral aleatoria del algoritmo acelerado original.
El problema es no convexo: puede estancarse o converger a una solución local.

El criterio de parada es la norma conjunta de los cambios de `(L,S)` entre dos
iteraciones, dividida entre `||D||_F`; no exige residuo nulo. `rank` puede variar
de 0 a `min(m,n)` y `cardinality` de 0 a `m*n`, ambos como enteros.

## Diagnósticos y precisión

`objective` corresponde al objetivo del método elegido; valores entre métodos
distintos no deben compararse sin atender a sus definiciones. `history` registra
las iteraciones y `error` el criterio de parada final. Consulta las docstrings de
la función para las fórmulas exactas del criterio implementado.

La entrada cero devuelve matrices cero con `n_iter=0`, `converged=True` e historial
vacío. Agotar `max_iter` devuelve el último resultado y emite `ConvergenceWarning`.

Las operaciones sobre `float32` pueden estabilizarse antes de tolerancias muy
pequeñas; empieza con `tol=1e-5` al usar esa precisión, también en PyTorch.
La biblioteca no relaja automáticamente la tolerancia solicitada. Para pruebas
de recuperación usa `float64`, datos reproducibles y
errores relativos tanto de reconstrucción como de componentes conocidas. Al
medir GPU, sincroniza CUDA antes y después de la región cronometrada.
