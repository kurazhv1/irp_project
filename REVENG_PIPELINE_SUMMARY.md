# IRP Cloth Manipulation — Архитектурный конспект (реверс)

Кратко фиксирует базовую (оригинальную) архитектуру, её проблемы и обновлённый пайплайн, который мы собрали при реверсе.

---

## Оригинальная архитектура (RSS 2022, legacy mujoco-py)

- Данные: `irp_cloth.zarr` (тренировка), `irp_cloth.ckpt` (предобученная модель).
- Тренировка: `train_irp_cloth.py` + Hydra конфиг `config/train_irp_cloth.yaml` → датамодуль `datasets/cloth_delta_gaussian_dataset.py` → сеть `networks/cloth_delta_deeplab.py` (DeepLab v3+ на ResNet) → ckpt + wandb.
- Оценка: `eval_irp_cloth_sim.py` + `config/eval_irp_cloth_sim.yaml` → загрузка ckpt → среда `environments/table_cloth_sim_environment.py` (MuJoCo 2.1.2.14, mujoco-py) → семплер `real_ur5/delta_action_sampler.py` + селектор `real_ur5/delta_action_selector.py` → 5 rope конфигураций × 11 goals × 16 шагов → метрики/логи (wandb, log.pkl).
- Алгоритм цикла eval: obs рендер → предсказание keypoints → Gaussian семплинг N действий → lookahead выбор лучшего → step в среде → расчёт loss → логирование.
- Визуализация: встроенный mujoco-py viewer (interactive), вызывается внутри step; rending/offscreen не отделён.

```mermaid
flowchart LR
    subgraph Train
        ZARR[data/irp_cloth.zarr] --> DM[datasets/cloth_delta_gaussian_dataset.py]
        DM --> TRAIN[train_irp_cloth.py]
        TRAIN --> NET[networks/cloth_delta_deeplab.py]
        NET --> CKPT[irp_cloth.ckpt]
    end
    subgraph Eval (legacy, mujoco-py)
        CKPT --> EVAL[eval_irp_cloth_sim.py]
        CFG[config/eval_irp_cloth_sim.yaml] --> EVAL
        EVAL --> ENV[table_cloth_sim_environment.py]
        EVAL --> SAMPLER[delta_action_sampler.py]
        EVAL --> SELECTOR[delta_action_selector.py]
        ENV --> VIEWER[mujoco-py viewer]
        EVAL --> WANDB[wandb/log.pkl]
    end
```

---

## Основные проблемы legacy пайплайна

- Segfault в live визуализации (mujoco-py viewer) во время cloth симуляции; `--show-vis` приводит к крашу.
- Нестабильность cloth в MuJoCo 2.x (NaN/QACC), требуется try/except; влияет на живой просмотр, но headless прогон tolerable.
- Сохранение кадров в legacy (`--save-images`) ломается: пустые obs → OpenCV error.
- Зависимость от устаревшего mujoco-py; нет контейнеров/CI, ручное управление окружением.
- В MuJoCo 3 нет порта flex cloth: имеется только упрощённый rigid placeholder, интерактивный режим не проверен.

---

## Обновлённая архитектура (реверс-инж, текущее состояние)

- Две среды (разделение задач):
  - `irp_legacy` (Py3.8 + mujoco-py 2.1.2.14) — продакшн eval и точная cloth физика; запускать только headless.
  - `irp` (Py3.10 + MuJoCo 3.3.6) — визуализация/видео без segfaultов; пока cloth = rigid заглушка.
- Логирование действий: `eval_irp_cloth_sim.py` сохраняет `outputs/.../action_logs/*.json` (дата, rope/goal, 16 шагов) + wandb; 55 эпизодов собраны (см. `EVAL_COMPLETE_RESULTS.md`).
- Реплей без падений:
  - Полная cloth физика headless: `replay_cloth_full.py action_log.json` (loss ≈ логам, 15/16 шагов типично).
  - Современная виза: `replay_viewer_mj3.py action_log.json --headless` (без segfault), может писать видео.
- Адаптация под MuJoCo 3: среда `environments/table_cloth_sim_environment_mj3.py`, модель `assets/mujoco/cloth/cloth_mj3.xml`, скрипты `replay_viewer_mj3.py`, `visualize_mujoco3.py` (см. `MUJOCO3_VISUALIZATION_GUIDE.md`, `MUJOCO3_SUCCESS.md`).
- Документация/диаграммы: `ARCHITECTURE_DIAGRAM.md`, `CLOTH_PIPELINE_ARCHITECTURE.md`, `SESSION_SUMMARY.md` фиксируют блок-схемы, псевдокод, зависимости.

```mermaid
flowchart LR
    subgraph Legacy (prod physics, headless)
        CKPT2[irp_cloth.ckpt] --> EVAL2[eval_irp_cloth_sim.py]
        EVAL2 --> ENV2[table_cloth_sim_environment.py]
        EVAL2 --> LOGS[action_logs/*.json]
        LOGS --> REPLAY_FULL[replay_cloth_full.py (headless)]
        EVAL2 --> WANDB2[wandb/log.pkl]
    end
    subgraph Modern (viz, MJ3)
        LOGS --> MJENV[table_cloth_sim_environment_mj3.py]
        MJENV --> MJVIEW[replay_viewer_mj3.py --headless/--video]
        MJVIEW --> VIDEO[mp4/frames]
        MJMODEL[assets/mujoco/cloth/cloth_mj3.xml] --> MJENV
    end
```

---

## Как обновлённый пайплайн закрывает проблемы

- Segfault viewer → обход: строгий headless в legacy + отдельный MJ3 renderer для видео; интерактив переведён в MJ3 (без падений).
- Нестабильность cloth → допускается в headless (try/except), метрики воспроизводимы; для визы используем стабильный rigid MJ3.
- Отсутствие воспроизводимости → action_logs JSON + headless реплей дают точную сверку потерь.
- Устаревший стек → введён modern стек (MuJoCo 3) как параллельный трек; можно постепенно заменить физику (flex) без риска для eval.

---

## Датасет IRP Cloth: структура и привязка к среде

### Схема `data/irp_cloth.zarr`

```
irp_cloth.zarr/
├── dim_keys                   # ['cloth_size','cloth_density','duration','gy1','gz1','gy2']
├── dim_samples/
│   ├── cloth_size             # (4,) [0.3, 0.4, 0.5, 0.6]
│   ├── cloth_density          # (4,) [0.2, 0.6, 1.0, 1.4]
│   ├── duration               # (8,) 0..1
│   ├── gy1                    # (16,) 0..1
│   ├── gz1                    # (16,) 0..1
│   └── gy2                    # (16,) 0..1
├── traj_occu                  # (4,4,8,16,16,16,9,256,256) bool
├── is_valid                   # (4,4,8,16,16,16) bool (~93% valid)
└── split/
    ├── is_train               # (4,4) bool
    └── is_val                 # (4,4) bool
```

**Что внутри:**
- `traj_occu` — 9 каналов occupancy-карт траекторий (по 3x3 keypoint на ткани).
- Действия нормализованы [0,1] и мапятся в физические параметры через `ActionMapper` (`duration, gy1, gz1, gy2`).
- `is_valid` — маска допустимых комбинаций (используется для rejection sampling).

### Почему любое изменение требует нового датасета

1. **Меняется физика/геометрия** → траектории keypoints другие.  
   Любые правки `cloth_spacing`, `cloth_density`, XML модели или сетки ткани приводят к новому распределению `traj_occu`.

2. **Меняется смысл действий** → сетка `duration/gy1/gz1/gy2` больше не соответствует реальным движениям.  
   `ActionMapper` определяет физические диапазоны; если они изменяются, старые `is_valid` и `traj_occu` становятся некорректными.

3. **Меняется проекция/камера** → occupancy-карты другие.  
   `GridCoordTransformer` фиксирует рамку и масштаб; при изменениях нужна перегенерация.

**Вывод:** датасет — это полностью "зашитый" снимок конкретной среды. Любая модификация среды требует нового `irp_cloth.zarr`, иначе сравнение/обучение некорректно.

---

## Что добавить дальше (приоритеты)

- Порт flex cloth в MuJoCo 3 (`<flexcomp>`) для совпадения физики и визы.
- Скрипт пакетной генерации видео по всем action_logs (обвязка над `replay_viewer_mj3.py --video`).
- Docker/CI для двух окружений (legacy eval, modern viz) + Make/README с краткими командами.
- (Опционально) Автотесты на быстрый smoke: загрузка ckpt, один step в обеих средах, проверка логов.
