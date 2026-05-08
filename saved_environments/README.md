# Сохранённые conda окружения

**Дата сохранения:** 21 сентября 2025  
**Цель:** Освобождение места для проекта IRP Reverse Engineering

## Сохранённые окружения

### 1. mujoco_py_test.yml (423MB)
- Тестовое окружение для MuJoCo Python bindings
- Восстановление: `conda env create -f mujoco_py_test.yml`

### 2. orbit.yml (13GB) 
- ORBIT simulation framework окружение
- Самое большое окружение, содержит Isaac Sim компоненты
- Восстановление: `conda env create -f orbit.yml`

### 3. pybulletenv.yml (6.1GB)
- PyBullet физическая симуляция
- Восстановление: `conda env create -f pybulletenv.yml`

### 4. voxposer-env.yml (3.7GB)
- VoxPoser окружение для манипуляции объектов
- Восстановление: `conda env create -f voxposer-env.yml`

## Освобождение места

Общий размер удаляемых окружений: **~23GB**
После удаления общее свободное место: **~48GB**

## Команды для восстановления

```bash
# Восстановить конкретное окружение
conda env create -f saved_environments/<environment_name>.yml

# Восстановить все окружения
for env in saved_environments/*.yml; do
    conda env create -f "$env"
done
```

## Примечания
- Окружения можно восстановить в любое время
- YAML файлы содержат точные версии всех пакетов
- Рекомендуется восстанавливать только при необходимости