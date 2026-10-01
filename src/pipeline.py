"""Pipelines de producción, encapsulados y libres de fuga.

Secuencia obligatoria (estándar RA1 de la guía):
    StandardScaler -> [PCA(0.95)] -> modelo
Todo transformador con estado (media, desviación, componentes) se ajusta
únicamente con los datos de entrenamiento de cada pliegue, porque vive dentro
del Pipeline que GridSearchCV / cross_validate clonan y reajustan en cada fold.

El StandardScaler se incluye en TODOS los modelos para una comparación
homogenea (los árboles son invariantes a la escala; el costo del escalador
queda contabilizado en el tiempo y el tamaño serializado).
"""
from dataclasses import dataclass, field
from typing import Callable

from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from .config import SEED

PCA_VARIANCE = 0.95


@dataclass(frozen=True)
class ModelSpec:
    label: str
    group: str                       # baseline | lineal | svm | ensemble
    factory: Callable[[], object]
    grid: dict = field(default_factory=dict)   # claves con prefijo clf__
    use_pca: bool = False


MODEL_SPECS = {
    "dummy": ModelSpec(
        "DummyClassifier", "baseline",
        lambda: DummyClassifier(strategy="most_frequent")),
    "logreg": ModelSpec(
        "LogisticRegression", "baseline",
        lambda: LogisticRegression(max_iter=2000, random_state=SEED),
        {"clf__C": [1, 10, 100, 1000]}),
    "linearsvc": ModelSpec(
        "LinearSVC", "svm",
        lambda: LinearSVC(max_iter=20000, random_state=SEED),
        {"clf__C": [1, 10, 100, 1000]}),
    "svm_linear": ModelSpec(
        "SVC lineal (OvO)", "svm",
        lambda: SVC(kernel="linear", random_state=SEED),
        {"clf__C": [1, 10, 100]}),
    "svm_rbf": ModelSpec(
        "SVC RBF", "svm",
        lambda: SVC(kernel="rbf", random_state=SEED),
        {"clf__C": [10, 100, 1000, 10000], "clf__gamma": [0.0001, 0.0003, 0.001, 0.003]}),
    "svm_rbf_pca": ModelSpec(
        "PCA + SVC RBF", "svm",
        lambda: SVC(kernel="rbf", random_state=SEED),
        {"clf__C": [10, 100, 1000, 10000], "clf__gamma": [0.0001, 0.0003, 0.001, 0.003]},
        use_pca=True),
    "rf": ModelSpec(
        "RandomForest", "ensemble",
        lambda: RandomForestClassifier(random_state=SEED, n_jobs=1),
        {"clf__n_estimators": [100, 300, 600], "clf__max_depth": [8, 12, None]}),
    "hgb": ModelSpec(
        "HistGradientBoosting", "ensemble",
        lambda: HistGradientBoostingClassifier(random_state=SEED, early_stopping=False),
        {"clf__learning_rate": [0.1, 0.2, 0.3], "clf__max_iter": [200, 400, 800]}),
}


def build_pipeline(name: str, params: dict | None = None) -> Pipeline:
    """Construye el Pipeline (sin ajustar) del modelo `name`.

    `params` acepta claves 'clf__*' (p. ej. las de best_params_ de GridSearchCV).
    """
    spec = MODEL_SPECS[name]
    steps = [("scaler", StandardScaler())]
    if spec.use_pca:
        steps.append(("pca", PCA(n_components=PCA_VARIANCE, random_state=SEED)))
    steps.append(("clf", spec.factory()))
    pipe = Pipeline(steps)
    if params:
        pipe.set_params(**params)
    return pipe
