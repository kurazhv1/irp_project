# Cloth Replay Viewer - Quick Start Guide

## Что сделано
✅ Создан интерактивный viewer для replay симуляции ткани
✅ Добавлена возможность сделать ткань жестче (не хлипкой)
✅ Используется MuJoCo 2.3.7 + GLFW (работает без segfault)

## Как запустить

### Вариант 1: Жесткая ткань (рекомендуется)
```bash
conda activate irp_legacy
cd /home/aalto-robotics/IRP_Project/original/irp_project
python replay_cloth_glfw_viewer.py outputs/2025-12-01/13-04-31/action_logs/action_log_20251201_130443_rope4_goal5.json
```

### Вариант 2: Мягкая ткань (оригинальная физика)
```bash
python replay_cloth_glfw_viewer.py outputs/2025-12-01/13-04-31/action_logs/action_log_20251201_130443_rope4_goal5.json --soft
```

## Управление в viewer
- **Space**: Пауза/воспроизведение
- **←/→**: Шаг назад/вперед по кадрам
- **Home**: В начало
- **End**: В конец  
- **ESC**: Выход

## Что изменено для жесткости
- `flatinertia`: 0.01 → 0.1 (10x жестче)
- `joint damping`: 0.001 → 0.01 (10x больше)
- `twist damping`: 0.0001 → 0.001 (10x больше)

Результат: ткань меньше деформируется и не летает за грипперами

## Альтернативные скрипты

### Headless replay (без окна, только проверка физики)
```bash
python replay_cloth_full.py outputs/2025-12-01/13-04-31/action_logs/action_log_20251201_130443_rope4_goal5.json
```
Результат: Final loss: 0.0157 (1.57cm) ✅ - точно совпадает с evaluation

### Все replay скрипты в проекте
- `replay_cloth_glfw_viewer.py` - **ЭТОТ!** Интерактивный viewer (MuJoCo 2.3.7)
- `replay_cloth_full.py` - Headless с правильной физикой (mujoco-py)
- `replay_cloth_with_vis.py` - С mujoco-py viewer (segfault!)
- `replay_cloth_exact.py` - MuJoCo 2.3.7, но reset before each action
- `replay_cloth_video.py` - Запись в MP4 (не работает)
- Другие (legacy, старые версии)

## Troubleshooting

### Окно не открывается
- Проверь что X11 forwarding работает: `echo $DISPLAY`
- Попробуй запустить test: `python test_cloth_glfw_viewer.py`

### Ткань все равно слишком мягкая
Отредактируй `replay_cloth_glfw_viewer.py`, строка ~35:
```python
xml_content = xml_content.replace('flatinertia="0.01"', 'flatinertia="0.5"')  # Еще жестче!
```

### Хочу изменить камеру
В `replay_cloth_glfw_viewer.py`, строки ~160-165:
```python
cam.azimuth = 90      # Поворот вокруг
cam.elevation = -45   # Угол сверху
cam.distance = 2.5    # Дистанция
```

## Для диплома

Лучший эпизод уже найден:
- **File**: `outputs/2025-12-01/13-04-31/action_logs/action_log_20251201_130443_rope4_goal5.json`
- **Loss**: 1.57cm (лучший из 55 эпизодов)
- **Rope**: #4 (spacing=0.0358, density=0.66)
- **Goal**: alpha=0.5 (средняя сложность)

Можешь:
1. Показать replay в viewer (интерактивно)
2. Записать скринкаст процесса
3. Сделать скриншоты ключевых моментов
4. Сравнить с худшим эпизодом (6.41cm)

## Статистика полной evaluation

Из 55 эпизодов (5 ropes × 11 goals):
- **Best**: 1.57cm (rope4, goal5)
- **Worst**: 6.41cm (rope0, goal10)
- **Mean**: 3.39cm
- **Median**: 3.28cm

Это **хуже** чем авторы заявляли (1-2cm), но replay показывает что физика правильная.
