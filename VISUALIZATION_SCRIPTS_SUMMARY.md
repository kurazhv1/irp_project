# 🎯 VISUALIZATION SCRIPTS SUMMARY

## ✅ ПРАВИЛЬНЫЕ СКРИПТЫ (используют cloth.xml или template)

### 1. **replay_cloth_native.py** ⭐ РЕКОМЕНДУЕТСЯ
**Что делает:** Запускает нативный MuJoCo simulate viewer  
**XML:** `assets/mujoco/cloth/cloth.xml` ✅  
**Environment:** Native MuJoCo  
**Pros:** Нативный viewer, полная визуализация  
**Cons:** Нет replay actions, только показывает модель

```bash
python replay_cloth_native.py outputs/.../action_log.json
```

---

### 2. **replay_cloth_full.py** ⭐ ДЛЯ ТОЧНОЙ ФИЗИКИ
**Что делает:** Headless replay с ТОЧНОЙ cloth физикой  
**XML:** `table_cloth_template.xml.jinja2` (через TableClothSimEnvironment) ✅  
**Environment:** TableClothSimEnvironment (legacy)  
**Pros:** Точная физика, воспроизводит loss values  
**Cons:** Headless (нет визуализации), segfault если show_vis=True

```bash
conda activate irp_legacy
python replay_cloth_full.py outputs/.../action_log.json
```

---

### 3. **replay_actions_mujoco3.py** 
**Что делает:** Replay с MuJoCo 3  
**XML:** `assets/mujoco/cloth/cloth.xml` ✅  
**Environment:** MuJoCo 3 (python bindings)  
**Pros:** Modern API, возможна визуализация  
**Cons:** Может быть несовместимость с cloth

```bash
conda activate irp
python replay_actions_mujoco3.py outputs/.../action_log.json
```

---

### 4. **replay_cloth_with_vis.py** ⚠️ SEGFAULT
**Что делает:** Попытка replay С визуализацией (legacy)  
**XML:** `table_cloth_template.xml.jinja2` ✅  
**Environment:** TableClothSimEnvironment(show_vis=True)  
**Pros:** Полная физика  
**Cons:** ❌ SEGFAULT при env.step()

```bash
# НЕ РАБОТАЕТ - segfault
conda activate irp_legacy
python replay_cloth_with_vis.py outputs/.../action_log.json
```

---

### 5. **replay_cloth_original_viz.py**
**Что делает:** Попытка оригинальной визуализации  
**XML:** `table_cloth_template.xml.jinja2` ✅  
**Environment:** TableClothSimEnvironment(show_vis=True)  
**Status:** Вероятно segfault

---

### 6. **replay_with_frames.py**
**Что делает:** Сохранение кадров из симуляции  
**XML:** `table_cloth_template.xml.jinja2` ✅  
**Environment:** TableClothSimEnvironment (headless)  
**Pros:** Сохраняет PNG кадры  
**Cons:** Headless

---

### 7. **replay_cloth_video.py**
**Что делает:** Запись видео через offscreen rendering  
**XML:** `table_cloth_template.xml.jinja2` ✅  
**Environment:** TableClothSimEnvironment (headless)  
**Status:** OpenGL errors

---

## 🗑️ УДАЛЁННЫЕ (мусор с cloth_mj3.xml)

- ❌ `replay_viewer_mj3.py` - использовал cloth_mj3.xml
- ❌ `replay_cloth_hybrid.py` - использовал cloth_mj3.xml  
- ❌ `visualize_mujoco3.py` - использовал cloth_mj3.xml
- ❌ `cloth_mj3.xml` - упрощённая модель (rigid body)

---

## 📋 ПРАВИЛЬНЫЕ XML ФАЙЛЫ

### ✅ `assets/mujoco/cloth/cloth.xml`
**Статус:** ИСПОЛЬЗУЕМ ДЛЯ ВИЗУАЛИЗАЦИИ  
**Тип:** Полная cloth модель (composite)  
**Параметры:** Фиксированные  
**Используется в:**
- `replay_cloth_native.py`
- `replay_actions_mujoco3.py`
- Native MuJoCo simulate

### ✅ `assets/mujoco/cloth/table_cloth_template.xml.jinja2`
**Статус:** ИСПОЛЬЗУЕМ ДЛЯ PYTHON КОДА  
**Тип:** Jinja2 template  
**Параметры:** Динамические (cloth_spacing, cloth_density, etc.)  
**Используется в:**
- `TableClothSimEnvironment` (все replay скрипты на Python)
- Генерирует XML с правильными параметрами для каждого эпизода

---

## 🎯 РЕКОМЕНДАЦИИ ДЛЯ ДИПЛОМА

### Для визуализации:
```bash
# Вариант 1: Нативный viewer (лучше всего для демонстрации)
~/.mujoco/mujoco210/bin/simulate assets/mujoco/cloth/cloth.xml

# Вариант 2: Python скрипт
python replay_cloth_native.py outputs/.../action_log.json
```

### Для проверки физики:
```bash
# Headless replay (точные loss values)
conda activate irp_legacy
python replay_cloth_full.py outputs/.../action_log.json
```

### Что показать:
1. **cloth.xml** в nativeсимуляторе - полная деформация ткани
2. **Headless результаты** - точность воспроизведения loss
3. **Action logs** - 55 эпизодов собраны

---

## 📊 Статус файлов

| Файл | XML | Статус | Использование |
|------|-----|--------|---------------|
| `cloth.xml` | ✅ | Production | Визуализация |
| `table_cloth_template.xml.jinja2` | ✅ | Production | Python evaluation |
| `cloth_mj3.xml` | ❌ | Удалён | Мусор |
| `replay_cloth_native.py` | ✅ | Working | Запуск viewer |
| `replay_cloth_full.py` | ✅ | Working | Headless physics |
| `replay_actions_mujoco3.py` | ✅ | Working | MuJoCo 3 replay |
| `replay_viewer_mj3.py` | ❌ | Удалён | Использовал мусорный XML |

---

**Дата обновления:** 30 ноября 2025  
**Статус:** ✅ Все скрипты используют правильные XML файлы
