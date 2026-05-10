# Sales Forecasting for Mars: Classical Time Series Models and Machine Learning Approaches

## Описание проекта

Этот проект посвящён анализу и прогнозированию временных рядов продаж.
Основная цель — исследовать структуру данных, выявить паттерны и построить baseline-модель для прогноза спроса.

Проект включает:

- разведочный анализ данных (EDA)
- кластеризацию временных рядов
- анализ сезонности и структуры
- построение базовой модели прогнозирования
- применение автокорреляционных моделей для прогнозирования рядов
- применение методов машинного обучения для прогнозирования рядов
- реализация production-like платформы для прогнозирования рядов

---

## Структура проекта

```
.
├── LICENSE
├── README.md
├── data # исходные данные и данные обработанные в течение экспериментов
│   ├── Mars 2 Data Chanels.xlsx
│   ├── arima_best_per_series.csv
│   ├── arima_summary.csv
│   ├── best_by_ts.csv
│   ├── data_4_forecast.csv
│   ├── ml_results.csv
│   └── ts_meta.csv
├── notebooks # эксперименты
│   ├── EDA.ipynb
│   ├── advanced_models.ipynb
│   ├── baseline.ipynb
│   └── bible.ipynb
├── plots     # папка, где собраны графики из экспериментов 
├── requirements.txt
└── service
    ├── Dockerfile.api
    ├── Dockerfile.dashboard
    ├── Dockerfile.training
    ├── Makefile
    ├── app
    │   ├── api
    │   │   └── routes.py
    │   ├── main.py
    │   └── services
    │       ├── aggregation_service.py
    │       ├── data_loader.py
    │       ├── forecasting
    │       │   └── forecast_service.py
    │       ├── ml_model_service.py
    │       ├── model_selector.py
    │       ├── s3_service.py
    │       └── storage_service.py
    ├── dashboard
    │   └── streamlit_app.py
    ├── data
    │   ├── outputs
    │   │   └── forecasts.parquet
    │   └── processed
    │       ├── best_by_ts.csv
    │       ├── business_data.csv
    │       ├── data_4_forecast.csv
    │       └── ml_experiment_results.csv
    ├── docker-compose.yml
    ├── models
    │   ├── LightGBM_horizon_12.pkl
    │   ├── LightGBM_horizon_3.pkl
    │   ├── LightGBM_horizon_6.pkl
    │   ├── XGBoost_horizon_12.pkl
    │   ├── XGBoost_horizon_3.pkl
    │   └── XGBoost_horizon_6.pkl
    ├── requirements.txt
    └── training
        ├── __init__.py
        ├── metrics.py
        └── train_ml_models.py
```

---

## Данные

В проекте используются следующие датасеты:

- `data_4_forecast.csv` — основной датасет для прогнозирования
- `ts_meta.csv` — мета-информация по временным рядам
- `Mars 2 Data Chanels.xlsx` — "сырые" данные
- `best_by_ts.csv` - мета-информация по лучшим моделям для ряда

---

## Анализ данных (EDA)

В ноутбуке `EDA.ipynb`:

- Получены графики продаж внутри категорий product, market, (product,market)
- Проведены декомпозиция рядов, adf, acf, pacf
- Получены порядки диффиренцирования
- Кластеризация для определения основных групп временных рядов
- Изучены корреляции с целевой переменной
- Введены дополнительные переменные
- - лаги: 1,3,6,12 месяцев
- rolling statistics mean, std

Основные визуализации:

- распределения (`histplot`, `boxplot`)
- корреляционная матрица
- кластеризация временных рядов
- временные графики продаж

---

## Моделирование

В `baseline.ipynb` реализована базовая модель прогнозирования:

- подготовка временных рядов
- разделение на train/test
- обучение модели
- оценка качества (сравнивали на MAPE)
  > - Ошибка прогнозирования (**MAPE**) увеличивается с горизонтом прогнозирования для всех моделей
  > - **Холт-Винтерс** демонстрирует наилучшие и наиболее стабильные результаты на всех горизонтах
  > - **Наивная** и **Сезонная наивная** модели служат базовыми и показывают худшие результаты
  > - Более высокая детализация данных (товар + магазин) → более высокая ошибка
  > - Агрегированные данные (общий объем продаж) → значительно более высокая точность прогнозирования

  > - **Вывод:** Холт-Винтерс — наиболее надежная модель для этой задачи.

В `advanced_models.ipynb` реализованы автокорреляционные модели прогнозирования временных рядов и методы машинного обучения

**Вывод:** в ходе проведения экспериментов были использованы такие модели как ARIMA, SARIMA, LightGBM, XGBoost
> - были получены датасеты `arima_best_per_series.csv` и `ml_results.csv` - результаты экспериментов с автокорреляционными моделями и методами машинного обучения, соответственно
> - для каждого временного ряда вида $(product \times market \times horizon)$ был подобран метод прогнозирования, дающий наилучший показатель по метрике `smape` 
---

## Сервис

В папке service лежит сервис, представляющий собой production-like платформу для прогнозирования продаж продуктов компании Mars

Сервис включает:

* API для прогнозирования временных рядов,
* интерактивный dashboard,
* хранение моделей и данных в S3,
* experiment tracking через MLflow,
* хранение истории прогнозов,
* Docker-based инфраструктуру.

Система поддерживает:

### Классические модели

* Holt-Winters
* ARIMA
* SARIMA

### ML-модели

* LightGBM
* XGBoost

---

# Архитектура системы

```text id="lt4rv4"
                     Streamlit Dashboard
                              ↓
                           FastAPI
                              ↓
         ┌────────────────────┼────────────────────┐
         ↓                    ↓                    ↓
      Postgres             MinIO/S3             MLflow
 история прогнозов    данные/модели/artifacts  tracking экспериментов

                              ↑
                              │
                        Training Pipeline
                 (LightGBM / XGBoost training)
```

---

# Структура проекта

```text id="ff9dql"
.
├── Dockerfile.api
├── Dockerfile.dashboard
├── Dockerfile.training
├── Makefile
├── app
├── dashboard
├── data
├── docker-compose.yml
├── models
├── requirements.txt
└── training
```

---

# Основные компоненты

## FastAPI

Основной backend сервиса.

Отвечает за:

* загрузку данных,
* выбор лучшей модели,
* построение прогнозов,
* выдачу данных для dashboard,
* сохранение истории прогнозов.

---

## Streamlit Dashboard

Интерактивный интерфейс для:

* выбора продукта и рынка,
* выбора горизонта прогнозирования,
* визуализации временных рядов,
* отображения прогнозов,
* отображения метрик качества,
* отображения бизнес-метрик.

---

## MinIO / S3

Используется как объектное хранилище для:

* исходных данных,
* обученных моделей,
* artifacts моделей и экспериментов.

---

## MLflow

Используется для:

* логирования экспериментов,
* хранения параметров моделей,
* хранения метрик,
* хранения artifacts моделей.

---

## Training Pipeline

Отдельный pipeline для обучения ML-моделей:

* обучение LightGBM/XGBoost,
* GridSearchCV,
* логирование экспериментов в MLflow,
* загрузка моделей в MinIO/S3.

---

# Workflow прогнозирования

```text id="0x5sy6"
Dashboard
    ↓
FastAPI /forecast
    ↓
Выбор лучшей модели
    ↓
Если модель классическая:
    использование statistical forecasting

Если модель ML:
    загрузка .pkl модели из MinIO
    ↓
    построение future features
    ↓
    model.predict()
```

---

# Запуск проекта

## 1. Запуск всех сервисов

Из корня проекта:

```bash id="ppw77r"
docker compose up --build
```

или через Makefile:

```bash id="s0p3ub"
make up
```

---

# Доступные сервисы

| Сервис    | URL                                                      |
| --------- | -------------------------------------------------------- |
| FastAPI   | [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) |
| Dashboard | [http://127.0.0.1:8501](http://127.0.0.1:8501)           |
| MLflow    | [http://127.0.0.1:5050](http://127.0.0.1:5050)           |
| MinIO     | [http://127.0.0.1:9001](http://127.0.0.1:9001)           |

---

# Данные для входа в MinIO

```text id="pn6zfa"
login: minio
password: minio123
```

---

# Загрузка исходных данных в S3

После запуска проекта необходимо загрузить данные в MinIO.

Открыть Swagger:

```text id="n44l9w"
http://127.0.0.1:8000/docs
```

Выполнить endpoint:

```text id="xf1ksm"
POST /s3/upload-raw-data
```

Будут загружены:

```text id="9gy0cl"
raw/sales_history.csv
raw/best_models.csv
raw/business_metrics.csv
```

---

# Обучение ML-моделей

## Запуск training pipeline

```bash id="t6e7h3"
docker compose run --rm training
```

или:

```bash id="jq79l8"
make train-ml-docker
```

---

# Что делает training pipeline

```text id="y2rmrr"
1. Загружает df_ml.csv
2. Обучает LightGBM/XGBoost
3. Выполняет GridSearchCV
4. Считает метрики
5. Логирует эксперименты в MLflow
6. Сохраняет .pkl модели
7. Загружает модели в MinIO
```

---

# Обученные модели

Модели сохраняются в:

```text id="cl42k7"
models/
├── LightGBM_horizon_3.pkl
├── LightGBM_horizon_6.pkl
├── LightGBM_horizon_12.pkl
├── XGBoost_horizon_3.pkl
├── XGBoost_horizon_6.pkl
└── XGBoost_horizon_12.pkl
```

и загружаются в:

```text id="btovxg"
s3://mars-forecasting/models/
```

---

# Команды Makefile

## Запуск проекта

```bash id="92ywwn"
make up
```

## Остановка проекта

```bash id="9j23g0"
make down
```

## Перезапуск проекта

```bash id="ph4fx9"
make restart
```

## Запуск API локально

```bash id="5bn7n2"
make api
```

## Запуск dashboard локально

```bash id="bg0f5c"
make dashboard
```

## Локальный запуск ML training

```bash id="l5qf82"
make train-ml
```

## Запуск ML training в Docker

```bash id="42w0jm"
make train-ml-docker
```

---

# Полный workflow запуска

## 1. Поднять инфраструктуру

```bash id="p7hhlh"
docker compose up --build
```

---

## 2. Загрузить исходные данные в MinIO

В Swagger выполнить:

```text id="5j00pv"
POST /s3/upload-raw-data
```

---

## 3. Запустить обучение ML-моделей

```bash id="rd9u9g"
docker compose run --rm training
```

---

## 4. Открыть dashboard

```text id="n0qmyr"
http://127.0.0.1:8501
```

---

# Используемые технологии

## Backend

* FastAPI
* Pandas
* NumPy
* Statsmodels

## Machine Learning

* Scikit-learn
* LightGBM
* XGBoost

## Хранилища

* MinIO / S3

## Experiment Tracking

* MLflow

## Frontend

* Streamlit
* Plotly

## Infrastructure

* Docker
* Docker Compose

---

# Возможные дальнейшие улучшения

* Airflow orchestration
* автоматический retraining
* model registry
* caching layer
* monitoring & alerting
* Kubernetes deployment


