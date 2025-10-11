# IRP (Iterative Residual Policy) - Reverse Engineering Log

**Дата начала:** 21 сентября 2025
**Цель:** Reverse engineering проекта IRP для понимания архитектуры и воссоздания функциональности

## Обзор проекта

### Исходная информация
- **Название:** Iterative Residual Policy for Goal-Conditioned Dynamic Manipulation of Deformable Objects
- **Статус:** Окружение E1 полностью функционально!

## Этап 6: Создание окружения E0 (Legacy) ⏳
**Время:** 23:30-
**Стратегия:** Двухветочный подход для максимальной совместимости

### 🎯 ПЛАН ДЕЙСТВИЙ:
#### Ветка 1: Legacy (E0) - Полная совместимость с оригиналом
- **Python 3.8** + **PyTorch 1.9.0** + **CUDA 11.2** 
- **MuJoCo 2.1.0** + **mujoco-py 2.0.2** (с лицензией)
- **abr_control** как в оригинальном проекте
- Цель: Запустить проект **ТОЧНО КАК У АВТОРОВ**

#### Ветка 2: Modern (E1) - Современные технологии  
- **Python 3.10** + **PyTorch 2.8.0** + современные библиотеки
- **MuJoCo 3.3.6** (без лицензии, встроенный Python API)
- Модернизированный код под новые API
- Цель: Обновить проект под **СОВРЕМЕННЫЕ СТАНДАРТЫ**

### 📋 Текущие действия:
1. **Пользователь ищет лицензию MuJoCo** для mujoco-py 2.0.2
2. **Создаём E0 окружение** с оригинальными версиями
3. **Тестируем оба окружения** на eval_irp_rope_dataset.py
4. **Создаём git ветки** для разделения legacy/modern кода

### 💡 Почему такой подход:
- **E0**: Гарантированная работа как у авторов
- **E1**: Возможность модернизации и развития
- **Git ветки**: Чистое разделение подходов
- **Документация**: Полная трассируемость изменений

### 🔧 Файлы окружений:
- `environment_e0.yml` - Legacy окружение (Python 3.8, PyTorch 1.9)
- `environment_e1.yml` - Modern окружение (Python 3.10, PyTorch 2.8)
- **Исходный код:** `/home/aalto-robotics/IRP_Project/original/irp_project`

### Основные характеристики
- **Язык:** Python 3.8
- **Основные фреймворки:** PyTorch 1.9.0, PyTorch Lightning 1.4.4
- **CUDA:** 11.2
- **Симуляция:** MuJoCo 2.1.0
- **Контроль камеры:** OpenCV, ZED Camera SDK
- **Логирование:** Weights & Biases (wandb)

## Архитектурный анализ

### Структура проекта
```
irp_project/
├── environment.yml              # Conda окружение
├── eval_*.py                   # Скрипты оценки (5 шт)
├── train_*.py                  # Скрипты обучения (3 шт)
├── ur5_camera_calibration_app.py # Калибровка камеры
├── video_labeler.py            # Разметка видео
├── abr_control_mod/            # Модификации ABR control
├── assets/                     # MuJoCo модели (cloth, ur5)
├── common/                     # Общие утилиты (9 модулей)
├── components/                 # Компоненты сети (3 модуля)
├── config/                     # Конфигурационные файлы (8 YAML)
├── datasets/                   # Датасеты и аугментация (3 модуля)
├── environments/               # Симуляционные среды (5 модулей)
├── networks/                   # Нейросетевые архитектуры (3 модуля)
├── pl_vis/                     # PyTorch Lightning визуализация (3 модуля)
└── real_ur5/                   # Интеграция с реальным роботом (6 модулей)
```

### Ключевые компоненты

#### 1. Нейросетевая архитектура
- **DeepLab v3+** как основа для сегментации
- **ResNet** backbone
- **Tracking** компоненты для отслеживания объектов

#### 2. Датасеты
- **IRP Rope Dataset** (7.63GB) - основной для eval
- **IRP Cloth Dataset** (938MB) - для тренировки ткани
- Поддержка **zarr** формата для эффективного хранения

#### 3. Среды симуляции
- **MuJoCo** для физической симуляции
- **UR5** робот манипулятор
- **Cloth** и **Rope** объекты

## Этапы Reverse Engineering

### Этап 1: Анализ зависимостей [ВЫПОЛНЕНО]
- Изучен `environment.yml` (255 зависимостей)
- Ключевые пакеты: pytorch, torchvision, pytorch-lightning, opencv, wandb, zarr
- CUDA 11.2 и связанные библиотеки

### Этап 2: Создание рабочего окружения [ВЫПОЛНЕНО]
- Создана директория `/home/aalto-robotics/IRP_ReverseEngineering`
- Скопирован исходный код в `irp_source/`
- Инициализирован git репозиторий
- Создан .gitignore и начальный commit

### Этап 3: Установка conda окружения [В ПРОЦЕССЕ]
- Попытка установки с оригинальным environment.yml не удалась
- Обнаружены конфликты зависимостей (типично для проектов 2022 года)
- Требуется создание упрощенного окружения

## Проблемы и их решения

### Проблема 1: Понимание архитектуры
**Описание:** Сложная структура с множеством компонентов
**Решение:** Поэтапный анализ каждого модуля

### Проблема 2: Конфликты зависимостей в conda
**Описание:** Оригинальный environment.yml (255 зависимостей) вызывает ошибки при установке
**Анализ:** 
- Конфликты между libopencv, tifffile, zarr, pytorch и другими пакетами
- Устаревшие версии пакетов (pytorch 1.9.0, python 3.8 из 2022 года)
- Проблемы совместимости с современными системами
**Решение:** Создать два отдельных окружения (E1 современное, E0 архивное)

### Проблема 3: Анализ места на диске
**Текущее состояние диска:**
- Общий размер раздела: 183GB
- Использовано: 166GB  
- Свободно: 7.0GB (96% заполнения)
- Conda окружения: 23GB (/home/aalto-robotics/anaconda3/envs/)
- Кэш пакетов: 31GB (/home/aalto-robotics/anaconda3/pkgs/)
- Текущий проект IRP: 36MB

**Оценка требований:**
- E1 окружение (Python 3.10 + PyTorch): ~2-3GB
- E0 окружение (Python 3.8 + PyTorch 1.9): ~2-3GB
- IRP Rope Dataset: 7.63GB
- IRP Cloth Dataset: 938MB
- Pretrained Models: ~1.4GB
- **Общая потребность: ~15-16GB**

**Проблема:** Недостаточно места (только 7GB свободно)
**Решение:** 
1. ✅ Очистка кэша conda: `conda clean --all` - освобождено 16GB
2. ✅ Сохранение YAML конфигураций существующих окружений
3. ✅ Удаление неиспользуемых окружений (orbit, pybulletenv, voxposer-env) - освобождено 23GB
4. Создание двух специализированных окружений (E1 и E0)

**Результат очистки диска:**
- Было: 7GB свободно (96% заполнения)
- После очистки кэша: 25GB свободно (86% заполнения)  
- После удаления окружений: 39GB свободно (78% заполнения)
- Освобождено всего: 32GB
- Достаточно для создания двух окружений и всех данных проекта

**Сохранённые окружения:**
- Созданы YAML файлы для восстановления в `saved_environments/`
- Удалены окружения общим размером ~23GB
- Оставлено: base + mujoco_py_test (может пригодиться для IRP)

### Следующие шаги
1. Копирование исходного кода
2. Создание conda окружения
3. Анализ основных модулей
4. Тестирование базовой функциональности
5. Документирование архитектурных решений

---

## Детальные заметки

### О датасетах
- Используется формат **zarr** для больших массивов данных
- Google Drive как основной источник данных
- Требуется ~8.5GB места для полного датасета

### О конфигурации
- **Hydra** для управления конфигурациями
- YAML файлы для каждого типа eval/train
- Поддержка multi-GPU через `action.gpu_id`

### О реальном роботе
- **UR5-CB3/UR5e** с RTDE интерфейсом
- **ZED 2i** стереокамера
- Кастомное крепление для деревянной палки
- Калибровка через homography

## Git Repository Setup - СИНХРОНИЗАЦИЯ ЗАВЕРШЕНА ✅

### Статус: УСПЕШНО СИНХРОНИЗИРОВАНО
- [x] Git repository initialized
- [x] Remote origin configured: https://github.com/kurazhv1/irp_project.git
- [x] User configuration: Vladimir Kurazhev / vladimir.kurazhev@aalto.fi
- [x] Merge conflicts resolved in .gitignore
- [x] All branches pushed to GitHub

### GitHub Synchronization: COMPLETED ✅
**Время:** 00:45 - Все ветки успешно загружены на GitHub!

Результат синхронизации:
```bash
git branch -a
  legacy                     ✅ PUSHED
* master                     ✅ PUSHED  
  modern                     ✅ PUSHED
  remotes/origin/legacy      ✅ SYNCED
  remotes/origin/master      ✅ SYNCED
  remotes/origin/modern      ✅ SYNCED
```

### Трехветочная стратегия РЕАЛИЗОВАНА:
1. **master** - Комбинированная версия с reverse engineering документацией
2. **legacy** - Python 3.8 + mujoco-py окружение (E0)  
3. **modern** - Python 3.10 + MuJoCo 3.x окружение (E1)

**URL:** https://github.com/kurazhv1/irp_project
**Все reverse engineering изменения сохранены и готовы к работе!**

## 🐳 NEXT PHASE: Docker Containerization Plan

### 📋 Docker Strategy (После тестирования обеих environment):

#### Phase 1: Legacy Docker Container (E0)
```dockerfile
# irp-legacy:latest
FROM nvidia/cuda:11.1-devel-ubuntu20.04
# Python 3.8 + PyTorch 1.9.0+CUDA + mujoco-py 2.1.2.14
# Полная совместимость с оригинальным проектом
```

#### Phase 2: Modern Docker Container (E1)  
```dockerfile
# irp-modern:latest
FROM nvidia/cuda:12.2-devel-ubuntu22.04
# Python 3.10 + PyTorch 2.8 + MuJoCo 3.3.6
# Современная версия с улучшенной производительностью
```

### 🎯 Docker Features:
- **GPU Support**: NVIDIA runtime для CUDA
- **Volume Mounting**: Для данных и результатов
- **Jupyter Integration**: Для интерактивной работы
- **SSH Access**: Для удаленного доступа
- **Port Forwarding**: Для wandb и веб-интерфейсов

### 📦 Deliverables:
1. `Dockerfile.legacy` - Legacy окружение
2. `Dockerfile.modern` - Modern окружение  
3. `docker-compose.yml` - Оркестрация контейнеров
4. `run_remote.sh` - Скрипт для запуска на удаленной машине
5. **Docker Hub Images**: Готовые к использованию контейнеры

### 🚀 Remote Deployment:
```bash
# На удаленной машине:
docker pull kurazhv1/irp-legacy:latest
docker pull kurazhv1/irp-modern:latest
docker-compose up -d
# Готово к работе!
```

**Цель**: Полная портируемость и воспроизводимость на любой машине с Docker + GPU

## 📊 UPDATED ROADMAP (11 октября 2025)

### ✅ COMPLETED:
- [x] Legacy Environment (E0) - FULLY FUNCTIONAL
- [x] Data Pipeline - ALL DATASETS READY
- [x] Git Strategy - IMPLEMENTED
- [x] Basic Evaluation - WORKING

### 🔄 IN PROGRESS:
- [ ] Visualization Fix (segfault issue)
- [ ] Modern Environment Testing (E1)

### 📋 TODO (приоритизировано):
1. **Fix MuJoCo Visualization** (High)
2. **Test Modern Environment** (Medium)  
3. **🐳 Docker Containerization** (Medium-High)
4. **Performance Comparison** (Low)
5. **Documentation Cleanup** (Low)

### 🎯 NEXT SESSION GOALS:
1. Commit current changes to legacy branch
2. Fix visualization or implement offscreen rendering
3. Begin Docker strategy planning

**PROGRESS: ~85% Complete** 🚀
