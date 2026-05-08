# 🚀 Quick Start Guide - Running Cloth Evaluation

## Что сделано:

✅ **Данные распакованы**: `irp_cloth.zarr/` готов к использованию  
✅ **Action logging добавлен**: `eval_irp_cloth_sim.py` теперь сохраняет все действия в JSON  
✅ **Replay скрипт создан**: `replay_actions_mujoco3.py` для воспроизведения на MuJoCo 3  
✅ **Документация**: `ACTION_LOGGING_README.md` с полными инструкциями

## Запуск эксперимента:

### Шаг 1: Активируйте legacy окружение
```bash
conda activate irp_legacy
cd /home/aalto-robotics/IRP_Project/original/irp_project
```

### Шаг 2: Запустите evaluation (БЕЗ визуализации, чтобы избежать segfault)
```bash
# Вариант 1: Запуск с offline wandb (без интернета)
WANDB_MODE=offline python eval_irp_cloth_sim.py

# Вариант 2: Если хотите логировать в wandb онлайн
python eval_irp_cloth_sim.py
```

### Шаг 3: Проверьте результаты
```bash
# Найдите output директорию (создаётся автоматически с timestamp)
ls -lh outputs/

# Например: outputs/2025-10-26/15-00-00/
cd outputs/2025-10-26/15-00-00/

# Посмотрите action logs
ls -lh action_logs/

# Проверьте содержимое одного файла
cat action_logs/action_log_*_rope0_goal0.json | head -50
```

## Что получите:

1. **action_logs/*.json** - все действия в JSON формате (для replay и анализа)
2. **log.pkl** - оригинальные логи с траекториями
3. **config.yaml** - параметры запуска
4. **wandb/** (опционально) - метрики и графики

## Следующие шаги:

### Для диплома:
1. Соберите несколько action logs из разных экспериментов
2. Проанализируйте структуру JSON файлов
3. Создайте визуализации/графики из логов

### Для replay на MuJoCo 3:
```bash
conda activate irp_modern  # или другое окружение с MuJoCo 3
python replay_actions_mujoco3.py --model assets/mujoco/cloth/cloth.xml outputs/.../action_log_....json
```

## Примечания:

⚠️ **Visualization отключена**: Пока skip `show_vis=True` чтобы избежать segfault  
📊 **Wandb**: Можно использовать `WANDB_MODE=offline` для работы без интернета  
💾 **Storage**: Action logs легковесные (~KB на эпизод), можно хранить все  

## Если что-то пошло не так:

### Error: "No such file or directory: data/irp_cloth.zarr"
```bash
# Убедитесь что данные в правильном месте
cd /home/aalto-robotics/IRP_ReverseEngineering
ls -lh data/irp_cloth.zarr/
```

### Error: "ModuleNotFoundError"
```bash
# Проверьте что окружение активировано
conda activate irp_legacy
which python
```

### Segmentation fault
```bash
# Убедитесь что НЕ используете show_vis=True
# Если в конфиге есть show_vis, установите в false:
# В config/eval_irp_cloth_sim.yaml: show_vis: false
```

---

**Готово к запуску!** Просто выполните команды из Шага 1-3 выше.
