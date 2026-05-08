# 🎬 Визуализация с MuJoCo 3 - Руководство

## ✅ Что было сделано

### Созданные файлы:

1. **`environments/table_cloth_sim_environment_mj3.py`**
   - Адаптированный environment для MuJoCo 3
   - Совместим с modern API (mujoco, не mujoco-py)
   - Поддерживает создание из action log metadata

2. **`assets/mujoco/cloth/cloth_mj3.xml`**
   - MuJoCo 3 совместимая модель
   - Упрощённая версия (rigid body вместо flex cloth)
   - Работает без segfault

3. **`replay_viewer_mj3.py`**
   - Интерактивный viewer для action logs
   - Headless режим для video export
   - Keyboard controls

4. **`visualize_mujoco3.py`**
   - Полнофункциональный визуализатор (более сложный)
   - Пока не протестирован

---

## 🚀 Использование

### Активация окружения:

```bash
# Для MuJoCo 3 визуализации
conda activate irp

# Для legacy eval/replay
conda activate irp_legacy
```

---

## 📺 Интерактивная визуализация

### Запуск с GUI (если есть дисплей):

```bash
conda activate irp
cd /home/aalto-robotics/IRP_Project/original/irp_project

# Replay одного эпизода
python replay_viewer_mj3.py outputs/2025-10-26/16-05-10/action_logs/action_log_20251026_160517_rope0_goal0.json

# С настройкой скорости
python replay_viewer_mj3.py action_log.json --pause 0.5
```

### Keyboard controls:
- **Мышь**: Вращение камеры
- **Scroll**: Zoom
- **Close window**: Выход

---

## 🎥 Headless режим (без GUI)

### Просто replay:

```bash
python replay_viewer_mj3.py action_log.json --headless
```

### С сохранением видео:

```bash
python replay_viewer_mj3.py action_log.json --headless --video output.mp4
```

---

## 🧪 Тестирование

### 1. Проверить, что MuJoCo 3 установлен:

```bash
conda activate irp
python -c "import mujoco; print(f'MuJoCo: {mujoco.__version__}')"
```

Должно вывести: `MuJoCo: 3.3.6` (или новее)

### 2. Тест environment:

```bash
conda activate irp
cd /home/aalto-robotics/IRP_Project/original/irp_project
python environments/table_cloth_sim_environment_mj3.py
```

Должно вывести:
```
Testing TableClothSimEnvironmentMJ3...
Using MuJoCo 3 compatible XML: assets/mujoco/cloth/cloth_mj3.xml
Loading MuJoCo model: assets/mujoco/cloth/cloth_mj3.xml
✓ Environment initialized
✓ Reset successful, obs shape: (256, 256, 3)
✓ Step successful, loss: ...
✓ All tests passed!
```

### 3. Тест headless replay:

```bash
python replay_viewer_mj3.py \
    outputs/2025-10-26/16-05-10/action_logs/action_log_20251026_160517_rope0_goal0.json \
    --headless
```

Должен показать progress bar и успешно завершиться.

### 4. Тест интерактивной визуализации:

```bash
python replay_viewer_mj3.py \
    outputs/2025-10-26/16-05-10/action_logs/action_log_20251026_160517_rope0_goal0.json
```

Должно открыться окно с 3D визуализацией.

---

## 📊 Результаты тестирования

### ✅ Работает:
- [x] MuJoCo 3 environment инициализация
- [x] XML загрузка (cloth_mj3.xml)
- [x] Headless replay
- [x] Step simulation
- [ ] Интерактивная визуализация (нужно тестировать с дисплеем)
- [ ] Video export (нужно тестировать)

### ⚠️ Ограничения:
- Используется rigid body вместо flexible cloth
- Нет полной физики деформируемых объектов
- Для полного cloth simulation нужно портировать на MuJoCo 3 flex API

---

## 🔧 Окружения

### E0 - Legacy (irp_legacy):
```
Python: 3.8
mujoco-py: 2.1.2.14
PyTorch: 1.9.0+cu111
```

**Использовать для:**
- Evaluation (`eval_irp_cloth_sim.py`)
- Training (`train_irp_cloth.py`)
- Legacy replay (`replay_actions_legacy.py`)

### E1 - Modern (irp):
```
Python: 3.10 (предположительно)
MuJoCo: 3.3.6
PyTorch: 2.8 (предположительно)
```

**Использовать для:**
- Visualization (`replay_viewer_mj3.py`)
- Modern replay (`visualize_mujoco3.py`)
- Future development

---

## 📝 Примеры команд

### Replay всех эпизодов с сохранением видео:

```bash
conda activate irp
cd /home/aalto-robotics/IRP_Project/original/irp_project

for log in outputs/2025-10-26/16-05-10/action_logs/*.json; do
    basename=$(basename "$log" .json)
    echo "Processing: $basename"
    python replay_viewer_mj3.py "$log" --headless --video "videos/${basename}.mp4"
done
```

### Сравнить несколько эпизодов:

```bash
# Rope 0, goals 0-5
for goal in {0..5}; do
    python replay_viewer_mj3.py \
        outputs/2025-10-26/16-05-10/action_logs/action_log_20251026_160517_rope0_goal${goal}.json
done
```

---

## 🐛 Troubleshooting

### Проблема: `ImportError: No module named 'mujoco'`

**Решение:**
```bash
conda activate irp
pip install mujoco
```

### Проблема: `XML Error: Schema violation`

**Причина:** Старый XML использует MuJoCo 2 синтаксис

**Решение:** Используй `cloth_mj3.xml` (уже настроено по умолчанию)

### Проблема: Segmentation fault в interactive mode

**Решение:** Используй headless режим:
```bash
python replay_viewer_mj3.py action_log.json --headless
```

### Проблема: `AttributeError: 'TableClothSimEnvironmentMJ3' object has no attribute 'viewer'`

**Причина:** Ошибка при инициализации environment

**Решение:** Проверь XML файл, убедись что он корректный

---

## 🎯 Следующие шаги

### Для diploma:

1. ✅ **Документация** - создана
2. ✅ **Action logging** - работает
3. ✅ **Eval complete** - 55 episodes
4. ✅ **MuJoCo 3 adaptation** - базовая версия работает
5. ⏳ **Video generation** - нужно тестировать
6. ⏳ **Full cloth physics** - опционально, сложно

### Для улучшения:

- [ ] Портировать flex cloth на MuJoCo 3
- [ ] Добавить запись траекторий
- [ ] Создать comparison tool (legacy vs modern)
- [ ] Docker containerization

---

## 📚 Документация

- **ARCHITECTURE_DIAGRAM.md** - полная архитектура проекта
- **ACTION_LOGGING_README.md** - система логирования
- **QUICKSTART.md** - быстрый старт eval
- **TESTING_REPORT.md** - отчёт о тестировании
- **MUJOCO3_VISUALIZATION_GUIDE.md** (этот файл)

---

**Создано**: 2025-11-01  
**Статус**: ✅ Базовая визуализация работает  
**Тестировано**: Headless replay ✅, Interactive GUI ⏳

