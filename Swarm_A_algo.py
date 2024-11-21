#!/usr/bin/env python
# coding: utf-8

# In[49]:


# Файл one.py

# =============================================================================
# Импорты
# =============================================================================

import os
import sys
import time
import random
import logging
import heapq
import enum
from math import ceil, radians
from collections import deque
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional, Dict, Set
from functools import lru_cache
from threading import Thread

import numpy as np
import pygame

# =============================================================================
# Константы
# =============================================================================

# ----------------------
# Параметры модели
# ----------------------
LEARNING_RATE = 0.001
GAMMA = 0.99
CLIP_RANGE = 0.2
N_STEPS = 4096
COEF = 0.001
VF_COEF = 0.6
CLIP_RANGE_VF = 0.2
N_EPOCHS = 1
BATCH_SIZE = 64

# ----------------------
# Параметры экрана и сетки
# ----------------------
SCREEN_SIZE = 900
STATUS_PANEL_WIDTH = 300  # Ширина панели статуса справа
BAR_HEIGHT = int(SCREEN_SIZE * 0.13)
GRID_SIZE = 20
MARGIN_SIZE = 1

# ----------------------
# Параметры игры
# ----------------------
NUM_AGENTS = 3
COUNT_TARGETS = 100  # ceil((GRID_SIZE ** 2) * 0.4)
COUNT_OBSTACLES = 12  # ceil((GRID_SIZE ** 2) * 0.03)
STATION_SIZE = 2
MAX_STEPS_GAME = (GRID_SIZE ** 2) * 10
VIEW_RANGE = 1  # Область зрения 3x3
ON_TARGET_CONSUMPTION = 10  # Расход воды
TANK_CAPACITY = (COUNT_TARGETS * ON_TARGET_CONSUMPTION) / 2  # Максимальный запас воды
ENERGY_CAPACITY = 1000  # Максимальный запас топлива
ENERGY_CONSUMPTION_MOVE = 1
ENERGY_CONSUMPTION_DONE = 2
COUNT_ACTIONS = 4
MIN_GAME_STEPS = (GRID_SIZE * GRID_SIZE // NUM_AGENTS) * 2

# ----------------------
# Награды
# ----------------------
REWARD_EXPLORE = 5  # Вознаграждение за исследование новых клеток
REWARD_DONE = 3
REWARD_COMPLETION = (REWARD_DONE * COUNT_TARGETS) * 10
PENALTY_LOOP = 1
PENALTY_OUT_FIELD = 2
PENALTY_OBSTACLE = 2
PENALTY_CRASH = 3

# ----------------------
# Позиции
# ----------------------
PLACEMENT_MODE = 'random'

FIXED_TARGET_POSITIONS = [
    (2, 2), (2, 8), (4, 3), (4, 7), (6, 2),
    (6, 8), (8, 4), (8, 6), (3, 5), (7, 5)
]

FIXED_OBSTACLE_POSITIONS = [
    (1, 1), (1, 9), (3, 3), (7, 7), (9, 5)
]

# ----------------------
# Цвета
# ----------------------
WHITE = (200, 200, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)
BLUE = (0, 0, 255)
RED = (255, 69, 0)
GRAY = (30, 30, 30)
YELLOW = (255, 255, 0)
LIGHT_BLUE = (173, 216, 230)

# ----------------------
# Изображения
# ----------------------
AGENT_IMG = "images/drone.png"
TARGET_IMG = "images/bad_plant.png"
DONE_TARGET_IMG = "images/healthy_plant.png"
OBSTACLES_DIR = "./images/obstacles"
STATION_IMG = "images/robdocst.png"
FIELD_IMG = "images/field.png"
FIELD_BACKGROUND_IMG = "images/forest.jpg"

# ----------------------
# Логирование
# ----------------------
LOG_DIR = "./logs"
LOG_FILE = os.path.join(LOG_DIR, 'logging.log')

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,  # Уровень логирования изменен на INFO для более подробного отслеживания
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# =============================================================================
# Перечисления
# =============================================================================

class PointStatus(enum.Enum):
    """Статусы точек на карте."""
    EMPTY = 0
    VIEWED = 1
    VISITED = 2

class ObjectStatus(enum.Enum):
    """Статусы объектов на карте."""
    EMPTY = 0
    OBSTACLE = 1
    TARGET = 2

# =============================================================================
# Утилиты
# =============================================================================

def load_image(filename: str, cell_size: int) -> pygame.Surface:
    """
    Загружает и масштабирует изображение для объектов.
    Используется многопоточность для предварительной загрузки изображений.
    """
    try:
        image = pygame.image.load(filename).convert_alpha()
        return pygame.transform.scale(image, (cell_size, cell_size))
    except pygame.error as e:
        logging.error(f"Не удалось загрузить изображение {filename}: {e}")
        return pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)

def load_obstacles(directory: str, cell_size: int, count: int) -> List[pygame.Surface]:
    """
    Загружает случайные изображения препятствий из указанного каталога.
    Использует многопоточность для ускорения загрузки.
    """
    try:
        all_files = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        selected_files = random.sample(all_files, min(count, len(all_files)))
        return [load_image(f, cell_size) for f in selected_files]
    except Exception as e:
        logging.error(f"Ошибка загрузки препятствий из {directory}: {e}")
        return []

def render_text(
    screen: pygame.Surface,
    text: Any,
    font: pygame.font.Font,
    color: Tuple[int, int, int],
    x: int,
    y: int
) -> pygame.Rect:
    """
    Отображает текст на экране pygame и возвращает область отрисовки.
    """
    try:
        text_surface = font.render(str(text), True, color)
        rect = text_surface.get_rect(topleft=(x, y))
        screen.blit(text_surface, rect)
        return rect
    except Exception as e:
        logging.error(f"Ошибка при отображении текста: {e}")
        return pygame.Rect(0, 0, 0, 0)

# =============================================================================
# Менеджер Pygame
# =============================================================================

class PygameManager:
    """
    Класс для централизованного управления инициализацией и ресурсами pygame.
    """

    def __init__(self, width: int, height: int):
        pygame.init()
        self.screen = None
        self.width = width
        self.height = height

    def create_screen(self) -> None:
        """
        Создает окно pygame.
        """
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.width, self.height))
            logging.info(f"Создано окно pygame размером {self.width}x{self.height}")

    def quit(self) -> None:
        """Завершает работу pygame."""
        pygame.quit()
        logging.info("pygame завершил работу")

# =============================================================================
# Классы
# =============================================================================

class AStarPathfinder:
    """
    Класс для реализации алгоритма A* поиска пути с оптимизациями.
    """

    def __init__(self):
        self.cache: Dict[Tuple[Tuple[int, int], Tuple[Tuple[int, int], ...]], Optional[List[Tuple[int, int]]]] = {}

    @staticmethod
    def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Манхэттенское расстояние между двумя точками."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(
        self,
        start: Tuple[int, int],
        goals: Set[Tuple[int, int]],
        known_map: np.ndarray,
        grid_size: int,
        occupied_positions: Set[Tuple[int, int]]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Ищет путь от старта до одной из целей с использованием A* и кэшированием.
        """
        try:
            # Сортировка целей по эвристике для более быстрой остановки
            sorted_goals = sorted(goals, key=lambda goal: self.heuristic(start, goal))
            
            if not sorted_goals:
                logging.warning("No goals provided for pathfinding")
                return None

            goals_tuple = tuple(sorted_goals)

            cache_key = (start, goals_tuple)
            if cache_key in self.cache:
                return self.cache[cache_key]

            open_heap = []
            heapq.heappush(open_heap, (0, start))
            came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
            g_score = {start: 0}
            f_score = {start: self.heuristic(start, sorted_goals[0])}

            open_set = {start}
            closed_set = set()

            while open_heap:
                current_f, current = heapq.heappop(open_heap)
                open_set.remove(current)

                if current in goals:
                    # Восстанавливаем путь
                    path = []
                    while current in came_from:
                        path.append(current)
                        current = came_from[current]
                    path.append(start)
                    path.reverse()
                    self.cache[cache_key] = path
                    logging.debug(f"Найден путь: {path}")
                    return path

                closed_set.add(current)

                neighbors = [
                    (current[0] + dx, current[1] + dy)
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                ]

                for neighbor in neighbors:
                    if not (0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size):
                        continue
                    if known_map[neighbor[0]][neighbor[1]][1] == ObjectStatus.OBSTACLE.value:
                        continue
                    if neighbor in occupied_positions:
                        continue
                    if neighbor in closed_set:
                        continue

                    tentative_g_score = g_score[current] + 1

                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        # Улучшенная эвристика: сортировка целей позволяет использовать минимальную эвристику
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, sorted_goals[0])
                        if neighbor not in open_set:
                            heapq.heappush(open_heap, (f_score[neighbor], neighbor))
                            open_set.add(neighbor)

            logging.debug("Путь не найден")
            self.cache[cache_key] = None
            return None  # Путь не найден
        except Exception as e:
            logging.error(f"Ошибка в AStarPathfinder.find_path: {e}")
            return None

class Renderer:
    """
    Класс для управления процессом рендеринга с оптимизациями.
    """

    def __init__(self, pygame_manager: PygameManager, grid_size: int, cell_size: int, status_panel_width: int):
        """
        Инициализирует Renderer.
        """
        self.pygame_manager = pygame_manager
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.status_panel_width = status_panel_width
        self.screen = pygame_manager.screen
        self.base_icons: Dict[str, pygame.Surface] = {}
        self.text_cache: Dict[str, pygame.Rect] = {}
        self.load_icons()

    def load_icons(self) -> None:
        """Загружает все необходимые иконки."""
        try:
            # Использование многопоточности для загрузки изображений
            def load_all():
                self.base_icons['agent'] = load_image(AGENT_IMG, self.cell_size)
                self.base_icons['target'] = load_image(TARGET_IMG, self.cell_size)
                self.base_icons['done_target'] = load_image(DONE_TARGET_IMG, self.cell_size)
                self.base_icons['base'] = load_image(STATION_IMG, self.cell_size * STATION_SIZE)
                self.base_icons['field_bg'] = pygame.transform.smoothscale(
                    load_image(FIELD_BACKGROUND_IMG, self.grid_size * self.cell_size),
                    (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
                )
                self.base_icons['field'] = pygame.transform.smoothscale(
                    load_image(FIELD_IMG, self.grid_size * self.cell_size),
                    (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
                )

            load_thread = Thread(target=load_all)
            load_thread.start()
            load_thread.join()
            logging.info("Иконки успешно загружены")
        except Exception as e:
            logging.error(f"Ошибка загрузки иконок: {e}")

    def draw_grid(self) -> None:
        """Отрисовывает сетку на экране."""
        try:
            grid_surface = pygame.Surface((self.grid_size * self.cell_size, self.grid_size * self.cell_size), pygame.SRCALPHA)
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    pygame.draw.rect(
                        grid_surface, BLACK,
                        (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size), 1
                    )
            self.screen.blit(grid_surface, (0, 0))
        except Exception as e:
            logging.error(f"Ошибка при отрисовке сетки: {e}")

    def draw_base(self, base_position: Tuple[int, int]) -> pygame.Rect:
        """Отрисовывает базу на экране и возвращает область обновления."""
        try:
            base_size = STATION_SIZE * self.cell_size
            base_icon_scaled = pygame.transform.smoothscale(self.base_icons['base'], (base_size, base_size))
            rect = self.screen.blit(
                base_icon_scaled,
                (base_position[1] * self.cell_size, base_position[0] * self.cell_size)
            )
            return rect
        except Exception as e:
            logging.error(f"Ошибка при отрисовке базы: {e}")
            return pygame.Rect(0, 0, 0, 0)

    def draw_objects(
        self,
        target_positions: List[Tuple[int, int]],
        done_status: np.ndarray,
        obstacle_positions: Set[Tuple[int, int]],
        current_map: np.ndarray,
        obstacle_icons: List[pygame.Surface]
    ) -> List[pygame.Rect]:
        """
        Отрисовывает цели и препятствия на экране.
        Возвращает список обновлённых областей.
        """
        try:
            rects = []
            for index, target in enumerate(target_positions):
                x, y = target
                if current_map[x, y, 0] != PointStatus.EMPTY.value:
                    if index < len(done_status):
                        icon = self.base_icons['done_target'] if done_status[index] else self.base_icons['target']
                    else:
                        icon = self.base_icons['target']  # По умолчанию, если индекс выходит за пределы
                    rect = self.screen.blit(icon, (y * self.cell_size, x * self.cell_size))
                    rects.append(rect)

            for obstacle in obstacle_positions:
                x, y = obstacle
                if current_map[x, y, 0] != PointStatus.EMPTY.value:
                    obstacle_icon = obstacle_icons[hash(obstacle) % len(obstacle_icons)]
                    rect = self.screen.blit(obstacle_icon, (y * self.cell_size, x * self.cell_size))
                    rects.append(rect)
            return rects
        except Exception as e:
            logging.error(f"Ошибка при отрисовке объектов: {e}")
            return []

    def draw_agents(self, agents: List['Agent']) -> List[pygame.Rect]:
        """Отрисовывает агентов на экране и возвращает список обновлённых областей."""
        try:
            rects = []
            for agent in agents:
                if agent.position:
                    agent_image = self.base_icons['agent']
                    rect = self.screen.blit(
                        agent_image,
                        (agent.position[1] * self.cell_size, agent.position[0] * self.cell_size)
                    )
                    rects.append(rect)

                    # Рисуем прогресс-бары для топлива и воды на панели статуса
                    # Эти прогресс-бары будут рисоваться на панели, а не рядом с агентом
                    # Поэтому здесь ничего не делаем для прогресс-баров

            return rects
        except Exception as e:
            logging.error(f"Ошибка при отрисовке агентов: {e}")
            return []

    def draw_progress_bars(self, agents: List['Agent'], font: pygame.font.Font) -> List[pygame.Rect]:
        """
        Отрисовывает прогресс-бары для топлива и воды каждого агента на панели статуса.
        Возвращает список обновлённых областей.
        """
        try:
            rects = []
            bar_width = self.status_panel_width - 40
            bar_height = 20
            for idx, agent in enumerate(agents):
                # Определяем позиции для каждой строки информации
                base_x = self.grid_size * self.cell_size + 20
                base_y = 20 + idx * 100

                # Заголовок агента
                text = f"Дрон {idx + 1}"
                rect = render_text(
                    self.screen, text, font, BLACK, base_x, base_y
                )
                rects.append(rect)

                # Топливо
                fuel_text = "Топливо:"
                rect = render_text(
                    self.screen, fuel_text, font, BLACK, base_x, base_y + 30
                )
                rects.append(rect)

                fuel_percentage = max(0, min(100, (agent.fuel_level / 100.0)))
                fuel_bar_rect = pygame.Rect(base_x + 80, base_y + 30, bar_width, bar_height)
                pygame.draw.rect(self.screen, BLACK, fuel_bar_rect, 2)  # Рамка
                pygame.draw.rect(self.screen, YELLOW, (base_x + 82, base_y + 32, (fuel_percentage / 100.0) * (bar_width - 4), bar_height - 4))
                rects.append(fuel_bar_rect)

                fuel_percent_text = f"{int(fuel_percentage)}%"
                rect = render_text(
                    self.screen, fuel_percent_text, font, BLACK, base_x + bar_width + 90, base_y + 30
                )
                rects.append(rect)

                # Вода
                water_text = "Вода:"
                rect = render_text(
                    self.screen, water_text, font, BLACK, base_x, base_y + 60
                )
                rects.append(rect)

                water_percentage = max(0, min(100, (agent.water_level / 100.0)))
                water_bar_rect = pygame.Rect(base_x + 80, base_y + 60, bar_width, bar_height)
                pygame.draw.rect(self.screen, BLACK, water_bar_rect, 2)  # Рамка
                pygame.draw.rect(self.screen, BLUE, (base_x + 82, base_y + 62, (water_percentage / 100.0) * (bar_width - 4), bar_height - 4))
                rects.append(water_bar_rect)

                water_percent_text = f"{int(water_percentage)}%"
                rect = render_text(
                    self.screen, water_percent_text, font, BLACK, base_x + bar_width + 90, base_y + 60
                )
                rects.append(rect)

            return rects
        except Exception as e:
            logging.error(f"Ошибка при отрисовке прогресс-баров: {e}")
            return []

    def draw_status_panel(
        self,
        agents: List['Agent'],
        total_steps: int,
        total_points: int,
        detected_flowers: int,
        sprayed_flowers: int
    ) -> List[pygame.Rect]:
        """
        Отрисовывает горизонтальную панель статуса справа от игрового поля.
        Возвращает список обновлённых областей.
        """
        try:
            rects = []
            base_x = self.grid_size * self.cell_size + 20
            base_y = self.grid_size * self.cell_size + 20

            # Заголовок панели
            font = pygame.font.SysFont('Arial', 24)
            text = "Статус"
            rect = render_text(
                self.screen, text, font, BLACK, base_x, 10
            )
            rects.append(rect)

            # Общая статистика
            stats = [
                f"Шаги: {total_steps}",
                f"Очки: {total_points}",
                f"Обнаружено цветков: {detected_flowers}",
                f"Полито цветков: {sprayed_flowers}"
            ]
            for idx, stat in enumerate(stats):
                rect = render_text(
                    self.screen, stat, font, BLACK, base_x, 50 + idx * 30
                )
                rects.append(rect)

            # Прогресс-бары для каждого агента
            progress_rects = self.draw_progress_bars(agents, font)
            rects.extend(progress_rects)

            return rects
        except Exception as e:
            logging.error(f"Ошибка при отрисовке панели статуса: {e}")
            return []

    def draw_overlay(self, current_map: np.ndarray) -> List[pygame.Rect]:
        """Накладывает тёмный оверлей на неизведанные области и возвращает список обновлённых областей."""
        try:
            rects = []
            dark_overlay = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            dark_overlay.fill((0, 0, 0, 200))  # Непрозрачный

            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    if current_map[x, y, 0] == PointStatus.EMPTY.value:
                        rect = self.screen.blit(dark_overlay, (y * self.cell_size, x * self.cell_size))
                        rects.append(rect)
            return rects
        except Exception as e:
            logging.error(f"Ошибка при наложении оверлея: {e}")
            return []

    def render_message(self, message: str, color_title: Tuple[int, int, int] = RED, color_body: Tuple[int, int, int] = GREEN) -> pygame.Rect:
        """
        Отображает сообщение в центре экрана и возвращает область обновления.
        """
        try:
            self.screen.fill(GRAY)

            lines = message.split('\n')
            screen_width, screen_height = self.screen.get_size()

            # Настройка шрифтов
            font_title = pygame.font.SysFont('Arial', ceil(SCREEN_SIZE * 0.05))
            font_body = pygame.font.SysFont('Arial', ceil(SCREEN_SIZE * 0.03))

            total_height = 0
            rects = []
            for i, line in enumerate(lines):
                if i == 0:
                    font = font_title
                    color = color_title
                else:
                    font = font_body
                    color = color_body
                text_surface = font.render(line, True, color)
                text_width, text_height = font.size(line)
                x = (screen_width - text_width) // 2
                y = (screen_height - len(lines) * text_height) // 2 + total_height
                rect = self.screen.blit(text_surface, (x, y))
                rects.append(rect)
                total_height += text_height + 5

            pygame.display.update(rects)
            pygame.time.wait(10)
            return pygame.Rect(0, 0, screen_width, screen_height)
        except Exception as e:
            logging.error(f"Ошибка при отображении сообщения: {e}")
            return pygame.Rect(0, 0, 0, 0)

class BaseScenario(ABC):
    """
    Базовый абстрактный класс для сценариев игры.
    """

    @abstractmethod
    def step(self) -> Tuple[Dict[str, Any], int, bool, bool, Dict[str, Any]]:
        """Выполняет один шаг сценария."""
        pass

    @abstractmethod
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Сбрасывает сценарий в исходное состояние."""
        pass

    @abstractmethod
    def render(self) -> None:
        """Отображает текущее состояние сценария."""
        pass

class Agent:
    """
    Класс, представляющий агента в сценарии.
    """

    def __init__(self, scenario: 'SprayingScenario', name: str = None):
        """
        Инициализирует агента.
        """
        self.name = name or f'agent_{id(self)}'
        self.env = scenario
        self.position: Optional[Tuple[int, int]] = None
        self.tank: Optional[int] = None  # Вода
        self.energy: Optional[int] = None  # Топливо
        self.position_history: Optional[deque] = None
        self.action_space = list(range(COUNT_ACTIONS))
        self.observation_space: Optional[Dict[str, Any]] = None
        self.fuel_level: float = 100.0  # Уровень топлива в процентах
        self.water_level: float = 100.0  # Уровень воды в процентах
        self.known_targets: Set[Tuple[int, int]] = set()  # Добавлено

    def reset(self) -> Dict[str, Any]:
        """
        Сбрасывает состояние агента.
        """
        try:
            self.position = random.choice(list(self.env.base_positions))
            logging.info(f"Позиция {self.name} стартовая {self.position}")
            self.position_history = deque(maxlen=10)
            self.tank = TANK_CAPACITY
            self.energy = ENERGY_CAPACITY
            self.fuel_level = 100.0
            self.water_level = 100.0
            coords = np.zeros((self.env.grid_size, self.env.grid_size, 2), dtype=np.int8)  # Изменен тип данных
            self.observation_space = {
                'pos': self.position,
                'coords': coords
            }
            return self.observation_space
        except Exception as e:
            logging.error(f"Ошибка в методе reset агента {self.name}: {e}")
            return {}

    def refuel(self) -> None:
        """
        Пополняет танк и энергию агента до максимальных значений.
        """
        try:
            self.tank = TANK_CAPACITY
            self.energy = ENERGY_CAPACITY
            self.fuel_level = 100.0
            self.water_level = 100.0
            logging.info(f"{self.name} заправился на базе. Танк и энергия восстановлены.")
        except Exception as e:
            logging.error(f"Ошибка в методе refuel агента {self.name}: {e}")
    
    def take_action(self) -> Tuple[Tuple[int, int], int, bool, bool, Dict[str, Any]]:
        """
        Определяет и выполняет действие агента.
        """
        reward = 0
        terminated = False
        truncated = False

        try:
            # Проверка уровней топлива и воды
            if self.fuel_level <= 10 or self.water_level <= 10:
                logging.info(f"{self.name} возвращается на базу из-за низкого уровня топлива или воды")
                path_to_base = self.env.pathfinder.find_path(
                    start=self.position,
                    goals=self.env.base_positions,
                    known_map=self.env.current_map,
                    grid_size=self.env.grid_size,
                    occupied_positions=self.env.occupied_positions - {self.position}
                )
                if path_to_base and len(path_to_base) > 1:
                    next_position = path_to_base[1]
                    action = self._determine_action(self.position, next_position)
                else:
                    action = random.choice(self.action_space)
            else:
                obs = self.get_observation()
                # Выбор действия на основе наличия целей или необходимости разведки
                #visible_targets = self.get_visible_targets(obs['coords'])
                # visible_targets = self.get_visible_targets(self.env.current_map)
                visible_targets = obs.get('visible_targets', [])
                unexplored_cells = self.get_unexplored_cells()
                explored_unsprayed = self.env.get_explored_unsprayed_cells()
                
                if visible_targets:
                    # Если цели видны, направляемся к ближайшей
                    closest_visible_target = min(
                        visible_targets,
                        key=lambda t: self.heuristic(self.position, t)
                    )
                    path = self.env.pathfinder.find_path(
                        start=self.position,
                        goals={closest_visible_target},
                        known_map=self.env.current_map,
                        grid_size=self.env.grid_size,
                        occupied_positions=self.env.occupied_positions - {self.position}
                    )
                elif explored_unsprayed:
                    # Стремимся к разведанным, но непролитым клеткам
                    closest_explored_unsprayed = min(
                        explored_unsprayed,
                        key=lambda t: self.heuristic(self.position, t)
                    )
                    path = self.env.pathfinder.find_path(
                        start=self.position,
                        goals={closest_explored_unsprayed},
                        known_map=self.env.current_map,
                        grid_size=self.env.grid_size,
                        occupied_positions=self.env.occupied_positions - {self.position}
                    )
                    logging.debug(f"{self.name} найден путь к разведанной непролитой клетке {closest_explored_unsprayed}: {path}")    
                elif unexplored_cells:
                    # Можно выбрать ближайшую неизведанную клетку
                    closest_unexplored = min(
                        unexplored_cells,
                        key=lambda t: self.heuristic(self.position, t)
                    )
                    path = self.env.pathfinder.find_path(
                        start=self.position,
                        goals={closest_unexplored},
                        known_map=self.env.current_map,
                        grid_size=self.env.grid_size,
                        occupied_positions=self.env.occupied_positions - {self.position}
                    )
                else:
                    # Если все клетки исследованы, выбираем случайное действие
                    path = None
                    logging.debug(f"{self.name} не находит видимых целей или неизведанных клеток")
                if path and len(path) > 1:
                    next_position = path[1]
                    action = self._determine_action(self.position, next_position)
                else:
                    action = random.choice(self.action_space)
                    logging.debug(f"{self.name} выбирает случайное действие {action}")
            # Выполнение действия
            new_position, reward_increment = self.execute_action(action)
            reward += reward_increment

            # Проверка на заправку, если агент находится на базе
            if new_position in self.env.base_positions:
                self.refuel()
            
            return new_position, reward, terminated, truncated, {}
        except Exception as e:
            logging.error(f"Ошибка в методе take_action агента {self.name}: {e}")
            return self.position, 0, False, False, {}

    def execute_action(self, action: int) -> Tuple[Tuple[int, int], int]:
        """
        Выполняет выбранное действие и обновляет состояние агента.
        """
        reward = 0
        try:
            # Выполнение действия
            if action == 0:  # Вверх
                new_position = (max(0, self.position[0] - 1), self.position[1])
                self.energy -= ENERGY_CONSUMPTION_MOVE
            elif action == 1:  # Вниз
                new_position = (min(self.env.grid_size - 1, self.position[0] + 1), self.position[1])
                self.energy -= ENERGY_CONSUMPTION_MOVE
            elif action == 2:  # Влево
                new_position = (self.position[0], max(0, self.position[1] - 1))
                self.energy -= ENERGY_CONSUMPTION_MOVE
            elif action == 3:  # Вправо
                new_position = (self.position[0], min(self.env.grid_size - 1, self.position[1] + 1))
                self.energy -= ENERGY_CONSUMPTION_MOVE
            else:
                new_position = self.position

            # Обновление уровней топлива и воды
            self.fuel_level = max(0.0, min(100.0, (self.energy / ENERGY_CAPACITY) * 100))
            self.water_level = max(0.0, min(100.0, (self.tank / TANK_CAPACITY) * 100))

            obs = self.get_observation()
            value_new_position = obs['coords'][new_position[0]][new_position[1]]
            new_position, agent_reward = self.get_agent_rewards(new_position, value_new_position)
            self.position = new_position
            logging.info(f"Действие: {action} - позиция: {self.position} - {self.name}")

            return new_position, agent_reward
        except Exception as e:
            logging.error(f"Ошибка в методе execute_action агента {self.name}: {e}")
            return self.position, 0

    def _determine_action(self, current: Tuple[int, int], next_pos: Tuple[int, int]) -> int:
        """
        Определяет действие на основе текущей и следующей позиции.
        """
        dx = next_pos[0] - current[0]
        dy = next_pos[1] - current[1]
        if dx == -1:
            return 0  # Вверх
        elif dx == 1:
            return 1  # Вниз
        elif dy == -1:
            return 2  # Влево
        elif dy == 1:
            return 3  # Вправо
        else:
            return random.choice(self.action_space)

    def get_observation(self) -> Dict[str, Any]:
        """
        Получает наблюдения агента.
        """
        try:
            coords = np.zeros((self.env.grid_size, self.env.grid_size, 2), dtype=np.int8)
            x, y = self.position
            # Векторизация области зрения
            x_min = max(x - VIEW_RANGE, 0)
            x_max = min(x + VIEW_RANGE + 1, self.env.grid_size)
            y_min = max(y - VIEW_RANGE, 0)
            y_max = min(y + VIEW_RANGE + 1, self.env.grid_size)
    
            coords[x_min:x_max, y_min:y_max, 0] = PointStatus.VIEWED.value
    
            # Обновление статусов объектов с использованием Numpy
            obstacle_indices = np.array(list(self.env.obstacle_positions))
            target_indices = np.array(self.env.target_positions)
    
            if obstacle_indices.size > 0:
                coords[obstacle_indices[:,0], obstacle_indices[:,1], 1] = ObjectStatus.OBSTACLE.value
            if target_indices.size > 0:
                coords[target_indices[:,0], target_indices[:,1], 1] = ObjectStatus.TARGET.value
    
            # Теперь вызываем get_visible_targets после обновления информации о целях
            visible_targets = self.get_visible_targets(coords)
    
            # Объединение наблюдений
            observation = {
                'pos': self.position,
                'coords': coords,
                'energy': self.energy,
                'tank': self.tank,
                'visible_targets': visible_targets
            }
            self.observation_space = observation
    
            # Обновление известных целей
            self.known_targets.update(visible_targets)
    
            return observation
        except Exception as e:
            logging.error(f"Ошибка в методе get_observation агента {self.name}: {e}")
            return {}


    def get_agent_rewards(
        self,
        new_position: Tuple[int, int],
        value: np.ndarray
    ) -> Tuple[Tuple[int, int], int]:
        """
        Обновляет позиции, обрабатывает препятствия и цели, начисляет награды и штрафы.
        """
        agent_reward = 0
        try:
            self.position_history.append(new_position)

            if not self.is_within_bounds(new_position):
                agent_reward -= PENALTY_OUT_FIELD
                logging.warning(f"Агент {self.name} вышел за границы поля: {new_position}")
                new_position = self.position
            else:
                if value[1] == ObjectStatus.OBSTACLE.value:
                    agent_reward -= PENALTY_OBSTACLE
                    new_position = self.position
                    logging.info(
                        f"Упс, препятствие! {self.name} - штраф {PENALTY_OBSTACLE}, вернулся на {new_position}"
                    )
                elif value[1] == ObjectStatus.TARGET.value:
                    if new_position in self.env.target_positions:
                        index = self.env.target_positions.index(new_position)
                        if self.env.done_status[index] == 0:
                            # Уменьшаем уровень воды при поливе
                            self.water_level = max(0.0, self.water_level - (ON_TARGET_CONSUMPTION / TANK_CAPACITY) * 100)
                            self.tank -= ON_TARGET_CONSUMPTION
                            # Уменьшаем уровень топлива при поливе
                            self.energy = max(0, self.energy - ENERGY_CONSUMPTION_DONE)
                            self.fuel_level = max(0.0, (self.energy / ENERGY_CAPACITY) * 100)
                            self.env.done_status[index] = 1
                            agent_reward += REWARD_DONE
                            logging.info(f"Опрыскал растение {self.name} + награда {REWARD_DONE}")
                            # Увеличиваем счетчик политых цветков
                            self.env.sprayed_flowers += 1
                else:
                    if len(self.position_history) > 3:
                        pos_counter = self.position_history.count(new_position)
                        if pos_counter >= 2:
                            if new_position == self.position_history[-2]:
                                agent_reward -= PENALTY_LOOP * 2
                                logging.info(
                                    f"Штраф {self.name} за второй раз в одну клетку {self.position_history[-2]}"
                                )
                            elif 2 < pos_counter < 4:
                                agent_reward -= PENALTY_LOOP * 3
                                logging.info(
                                    f"Штраф {self.name} за вторичное посещение {new_position} в последние 10 шагов"
                                )
                            elif pos_counter >= 4:
                                agent_reward -= PENALTY_LOOP * 5
                                logging.info(
                                    f"Штраф {self.name} за многократное посещение {new_position} в последние 10 шагов"
                                )
            return new_position, agent_reward
        except Exception as e:
            logging.error(f"Ошибка в методе get_agent_rewards агента {self.name}: {e}")
            return self.position, 0

    def is_within_bounds(self, position: Tuple[int, int]) -> bool:
        """
        Проверяет, находится ли позиция внутри границ поля.
        """
        return 0 <= position[0] < self.env.grid_size and 0 <= position[1] < self.env.grid_size

    def find_path_to_position(
        self,
        target_position: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Ищет путь к заданной позиции с использованием A*.
        """
        try:
            return self.env.pathfinder.find_path(
                start=self.position,
                goals={target_position},
                known_map=self.env.current_map,
                grid_size=self.env.grid_size,
                occupied_positions=self.env.occupied_positions - {self.position}
            )
        except Exception as e:
            logging.error(f"Ошибка в методе find_path_to_position агента {self.name}: {e}")
            return None

    def find_path_to_nearest_target(self) -> Optional[List[Tuple[int, int]]]:
        """
        Ищет путь к ближайшей цели с использованием A*.
        """
        try:
            return self.env.pathfinder.find_path(
                start=self.position,
                goals=set(self.get_remaining_targets()),
                known_map=self.env.current_map,
                grid_size=self.env.grid_size,
                occupied_positions=self.env.occupied_positions - {self.position}
            )
        except Exception as e:
            logging.error(f"Ошибка в методе find_path_to_nearest_target агента {self.name}: {e}")
            return None

    def find_path_to_nearest_base(self) -> Optional[List[Tuple[int, int]]]:
        """
        Ищет путь к ближайшей базе с использованием A*.
        """
        try:
            return self.env.pathfinder.find_path(
                start=self.position,
                goals=self.env.base_positions,
                known_map=self.env.current_map,
                grid_size=self.env.grid_size,
                occupied_positions=self.env.occupied_positions - {self.position}
            )
        except Exception as e:
            logging.error(f"Ошибка в методе find_path_to_nearest_base агента {self.name}: {e}")
            return None

    def get_unexplored_cells(self) -> Set[Tuple[int, int]]:
        """
        Возвращает множество координат неизведанных клеток (EMPTY = 0).
        """
        try:
            unexplored = set()
            for x in range(self.env.grid_size):
                for y in range(self.env.grid_size):
                    if self.env.current_map[x, y, 0] == PointStatus.EMPTY.value:
                        unexplored.add((x, y))
            return unexplored
        except Exception as e:
            logging.error(f"Ошибка в методе get_unexplored_cells агента {self.name}: {e}")
            return set()
    
    # def get_visible_targets(self, coords: np.ndarray) -> List[Tuple[int, int]]:
    #     """
    #     Возвращает список неполитых целей, находящихся в области зрения агента.
    #     """
    #     try:
    #         visible_targets = []
    #         x, y = self.position
    #         x_min = max(x - VIEW_RANGE, 0)
    #         x_max = min(x + VIEW_RANGE + 1, self.env.grid_size)
    #         y_min = max(y - VIEW_RANGE, 0)
    #         y_max = min(y + VIEW_RANGE + 1, self.env.grid_size)

    #         # Использование векторизации для ускорения
    #         visible_coords = coords[x_min:x_max, y_min:y_max, 1]
    #         target_indices = np.argwhere(visible_coords == ObjectStatus.TARGET.value)
    #         for dx, dy in target_indices:
    #             target_pos = (x_min + dx, y_min + dy)
    #             if target_pos in self.get_remaining_targets():
    #                 visible_targets.append(target_pos)
    #         logging.debug(f"{self.name} видит цели: {visible_targets}")        
    #         return visible_targets
    #     except Exception as e:
    #         logging.error(f"Ошибка в методе get_visible_targets агента {self.name}: {e}")
    #         return []
    def get_visible_targets(self, coords: np.ndarray) -> List[Tuple[int, int]]:
        """
        Возвращает список неполитых целей, находящихся в области зрения агента,
        а также уже известные (разведанные) цели.
        """
        try:
            visible_targets = []
            x, y = self.position
            x_min = max(x - VIEW_RANGE, 0)
            x_max = min(x + VIEW_RANGE + 1, self.env.grid_size)
            y_min = max(y - VIEW_RANGE, 0)
            y_max = min(y + VIEW_RANGE + 1, self.env.grid_size)
    
            # Использование векторизации для ускорения
            visible_coords = coords[x_min:x_max, y_min:y_max, 1]
            target_indices = np.argwhere(visible_coords == ObjectStatus.TARGET.value)
            for dx, dy in target_indices:
                target_pos = (x_min + dx, y_min + dy)
                if target_pos in self.get_remaining_targets():
                    visible_targets.append(target_pos)
            
            logging.debug(f"{self.name} видит цели в зоне обзора: {visible_targets}")
    
            # Получаем известные цели, которые не находятся в текущей зоне обзора
            known_remaining_targets = self.known_targets.intersection(
                {pos for pos, done in zip(self.env.target_positions, self.env.done_status) if done == 0}
            )
            # Исключаем из известных уже видимые цели, чтобы избежать дублирования
            known_remaining_targets = known_remaining_targets.difference(set(visible_targets))
            
            logging.debug(f"{self.name} известные цели вне зоны обзора: {known_remaining_targets}")
    
            # Объединяем видимые цели с известными
            all_targets = visible_targets + list(known_remaining_targets)
            
            logging.debug(f"{self.name} все цели для рассмотрения: {all_targets}")
            
            return all_targets
        except Exception as e:
            logging.error(f"Ошибка в методе get_visible_targets агента {self.name}: {e}")
            return []

    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Манхэттенское расстояние между двумя точками.
        """
        return AStarPathfinder.heuristic(a, b)


    def get_remaining_targets(self) -> Set[Tuple[int, int]]:
        """Возвращает известные агенту неполитые цели."""
        try:
            # Извлекаем неполитые цели из глобального состояния
            remaining_targets = {
                pos for pos, done in zip(self.env.target_positions, self.env.done_status) if done == 0
            }
    
            # Пересечение с известными агенту целями
            remaining_known_targets = self.known_targets & remaining_targets
    
            logging.debug(
                f"{self.name} известные цели: {self.known_targets}, "
                f"неполитые цели: {remaining_targets}, "
                f"известные неполитые цели: {remaining_known_targets}"
            )
    
            return remaining_known_targets
        except Exception as e:
            logging.error(f"Ошибка в методе get_remaining_targets агента {self.name}: {e}")
            return set()

    def __repr__(self) -> str:
        return f'<Agent {self.name}>'

class SprayingScenario(BaseScenario):
    """
    Класс сценария опрыскивания с оптимизациями.
    """

    def __init__(self, num_agents: int, grid_size: int, renderer: Renderer):
        """
        Инициализирует сценарий опрыскивания.
        """
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.cell_size = renderer.cell_size
        self.margin = MARGIN_SIZE
        self.inner_grid_size = self.grid_size - self.margin * 2
        base_coords = (self.margin + 1, self.grid_size // 2 - STATION_SIZE // 2)
        self.base_positions: Set[Tuple[int, int]] = {
            (base_coords[0] + i, base_coords[1] + j)
            for i in range(STATION_SIZE)
            for j in range(STATION_SIZE)
        }
        self.agents: List[Agent] = [
            Agent(self, name=f'agent_{i}') for i in range(self.num_agents)
        ]
        self.done_status = np.zeros(COUNT_TARGETS, dtype=int)
        self.start_time: Optional[float] = None
        self.total_reward: float = 0.0
        self.step_reward: float = 0.0
        self.target_positions: List[Tuple[int, int]] = []
        self.obstacle_positions: Set[Tuple[int, int]] = set()
        self.current_map: np.ndarray = np.zeros((self.grid_size, self.grid_size, 2), dtype=np.int8)
        self.step_count: int = 0
        self.obstacle_icons: List[pygame.Surface] = []
        self.renderer = renderer
        self.pathfinder = AStarPathfinder()
        self.occupied_positions: Set[Tuple[int, int]] = set()
        # Дополнительная статистика
        self.detected_flowers: int = 0
        self.sprayed_flowers: int = 0
        self.total_points: int = 0

    def reset_objects_positions(self) -> None:
        """Сбрасывает позиции объектов в сценарии."""
        if PLACEMENT_MODE == 'random':
            self._randomize_positions()
        elif PLACEMENT_MODE == 'fixed':
            self._fixed_positions()
        else:
            logging.error("Invalid PLACEMENT_MODE. Choose 'random' or 'fixed'.")
            raise ValueError("Invalid PLACEMENT_MODE. Choose 'random' or 'fixed'.")

    def _randomize_positions(self) -> None:
        """Генерирует случайные позиции объектов."""
        try:
            unavailable_positions = set(self.base_positions)
            self.target_positions = self._get_objects_positions(unavailable_positions, COUNT_TARGETS)
            unavailable_positions.update(self.target_positions)
            self.obstacle_positions = set(self._get_objects_positions(unavailable_positions, COUNT_OBSTACLES))
            # Проверка на окружение целей препятствиями с использованием множеств
            while any(self._is_surrounded_by_obstacles(target) for target in self.target_positions):
                self.obstacle_positions = set(self._get_objects_positions(unavailable_positions, COUNT_OBSTACLES))
            logging.info("Случайные позиции объектов успешно сгенерированы")
        except Exception as e:
            logging.error(f"Ошибка в методе _randomize_positions: {e}")

    def _fixed_positions(self) -> None:
        """Устанавливает фиксированные позиции объектов."""
        try:
            # Проверка, что фиксированные позиции находятся в пределах grid_size
            self.target_positions = [pos for pos in FIXED_TARGET_POSITIONS if pos[0] < self.grid_size and pos[1] < self.grid_size]
            self.obstacle_positions = set([pos for pos in FIXED_OBSTACLE_POSITIONS if pos[0] < self.grid_size and pos[1] < self.grid_size])
            logging.info("Фиксированные позиции объектов установлены")
        except Exception as e:
            logging.error(f"Ошибка в методе _fixed_positions: {e}")

    def _is_surrounded_by_obstacles(self, target_position: Tuple[int, int]) -> bool:
        """
        Проверяет, окружена ли цель препятствиями.
        """
        try:
            x, y = target_position
            step = 1
            surrounding_positions = {
                (x - step, y), (x + step, y), (x, y - step), (x, y + step)
            }

            obstacle_count = len(surrounding_positions & self.obstacle_positions)
            return obstacle_count == 4
        except Exception as e:
            logging.error(f"Ошибка в методе _is_surrounded_by_obstacles: {e}")
            return False

    def _get_available_positions(self, unavailable: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Получает доступные позиции для размещения объектов.
        """
        try:
            all_positions = [
                (i, j) for i in range(self.margin, self.inner_grid_size + 1)
                for j in range(self.margin, self.inner_grid_size + 1)
            ]
            available = [pos for pos in all_positions if pos not in unavailable]
            return available
        except Exception as e:
            logging.error(f"Ошибка в методе _get_available_positions: {e}")
            return []

    def _get_objects_positions(
        self,
        unavailable: Set[Tuple[int, int]],
        size: int
    ) -> List[Tuple[int, int]]:
        """
        Получает список позиций объектов с учётом недоступных позиций.
        """
        try:
            available_positions = self._get_available_positions(unavailable)
            if len(available_positions) < size:
                raise ValueError("Недостаточно доступных позиций для размещения объектов.")
            selected_positions = random.sample(available_positions, size)
            return selected_positions
        except Exception as e:
            logging.error(f"Ошибка в методе _get_objects_positions: {e}")
            return []

    # def get_remaining_targets(self) -> List[Tuple[int, int]]:
    #     """Возвращает список оставшихся целей для опрыскивания."""
    #     # Удаляем предоставление полного списка целей
    #     # Теперь агенты используют свои собственные известные цели
    #     logging.warning("Метод get_remaining_targets больше не используется агентами.")
    #     return []  # Возвращаем пустой список или полностью удаляем этот метод

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Сбрасывает сценарий в исходное состояние.
        """
        try:
            self.reset_objects_positions()
            self.start_time = time.time()
            self.step_count = 1
            self.done_status = np.zeros(COUNT_TARGETS, dtype=int)
            self.total_reward = 0.0
            self.step_reward = 0.0
            self.current_map = np.zeros((self.grid_size, self.grid_size, 2), dtype=np.int8)
            self.obstacle_icons = load_obstacles(OBSTACLES_DIR, self.cell_size, COUNT_OBSTACLES)
            for agent in self.agents:
                agent.reset()
            obs = self.get_observation()
            logging.info("Перезагрузка среды")
            return obs, {}
        except Exception as e:
            logging.error(f"Ошибка в методе reset: {e}")
            return {}, {}

    def get_observation(self) -> Dict[str, Any]:
        """
        Получает объединённые наблюдения от всех агентов.
        """
        try:
            combined_coords = np.zeros((self.grid_size, self.grid_size, 2), dtype=np.int8)
            combined_energy = []
            combined_tank = []
            for agent in self.agents:
                obs = agent.get_observation()
                combined_coords = np.maximum(combined_coords, obs['coords'])
                combined_energy.append(obs['energy'])
                combined_tank.append(obs['tank'])

            self.current_map = np.maximum(self.current_map, combined_coords)

            # Обновление занятых позиций
            self.env_update_occupied_positions()

            obs = {
                'pos': np.array([agent.position for agent in self.agents]),
                'coords': self.current_map,
                'energy': combined_energy,
                'tank': combined_tank
            }
            return obs
        except Exception as e:
            logging.error(f"Ошибка в методе get_observation: {e}")
            return {}

    def get_explored_unsprayed_cells(self) -> Set[Tuple[int, int]]:
        """
        Возвращает множество координат разведанных клеток, содержащих непролитыми цветками.
        
        Разведанные клетки — это те, которые были помечены как VIEWED или VISITED.
        Непролитые цветки — это цели с done_status == 0.
        
        Returns:
            Set[Tuple[int, int]]: Множество координат (x, y).
        """
        try:
            explored_unsprayed = set()
            for idx, target_pos in enumerate(self.target_positions):
                if self.done_status[idx] == 0:
                    x, y = target_pos
                    point_status = self.current_map[x, y, 0]
                    object_status = self.current_map[x, y, 1]
                    
                    # Проверяем, что клетка разведана и содержит цель
                    if (point_status in {PointStatus.VIEWED.value, PointStatus.VISITED.value} and
                        object_status == ObjectStatus.TARGET.value):
                        explored_unsprayed.add(target_pos)
            
            logging.debug(f"Разведанные и непролитые клетки: {explored_unsprayed}")
            return explored_unsprayed
        except Exception as e:
            logging.error(f"Ошибка в методе get_explored_unsprayed_cells: {e}")
            return set()
    
    def env_update_occupied_positions(self):
        """Обновляет множество занятых позиций для быстрого доступа."""
        self.occupied_positions = {agent.position for agent in self.agents if agent.position is not None}

    def step(self) -> Tuple[Dict[str, Any], int, bool, bool, Dict[str, Any]]:
        """
        Выполняет один шаг сценария.
        """
        try:
            logging.info(f"Шаг: {self.step_count}")
            obs = self.get_observation()
            
            self.step_reward = 0.0

            # Пакетная обработка действий агентов
            actions = []
            desired_positions: Dict[Agent, Tuple[int, int]] = {}
            for agent in self.agents:
                new_position, agent_reward, terminated, truncated, info = agent.take_action()
                desired_positions[agent] = new_position
                actions.append((agent, new_position, agent_reward))

            # Обработка действий агентов
            # Проверяем на наложение позиций
            position_counts: Dict[Tuple[int, int], int] = {}
            for _, pos, _ in actions:
                if pos in position_counts:
                    position_counts[pos] += 1
                else:
                    position_counts[pos] = 1

            for agent, new_position, agent_reward in actions:
                if position_counts[new_position] > 1:
                    # Если несколько агентов пытаются попасть в одну клетку, предотвращаем движение
                    logging.warning(f"Дрон {agent.name} не может переместиться на занятую клетку {new_position}")
                    new_position = agent.position  # Остаётся на месте
                    agent_reward -= PENALTY_CRASH
                else:
                    value_position = obs['coords'][new_position[0]][new_position[1]]
                    if value_position[0] in (PointStatus.EMPTY.value, PointStatus.VIEWED.value):
                        if value_position[1] != ObjectStatus.TARGET.value:
                            self.step_reward += REWARD_EXPLORE
                            self.detected_flowers += 1  # Увеличиваем счетчик обнаруженных цветков
                            self.current_map[new_position[0], new_position[1], 0] = PointStatus.VISITED.value
                            logging.info(
                                f"{agent.name} исследовал новую клетку {new_position} + {REWARD_EXPLORE}"
                            )
                        #self.current_map[new_position[0], new_position[1], 0] = PointStatus.VISITED.value
                    agent.position = new_position
                    self.step_reward += agent_reward

            # Обновляем общие очки
            self.total_reward += self.step_reward

            # Проверяем условия завершения игры
            reward, terminated, truncated, info = self._check_termination_conditions()
            self.step_count += 1
            logging.info(
                f"Награда: {self.total_reward}, "
                f"Завершено: {terminated}, "
                f"Прервано: {truncated}"
            )

            return obs, reward, terminated, truncated, {}
        except Exception as e:
            logging.error(f"Ошибка в методе step: {e}")
            return {}, 0, False, False, {}

    def _check_termination_conditions(self) -> Tuple[int, bool, bool, Dict[str, Any]]:
        """
        Проверяет условия завершения игры: количество шагов и обработка всех целей.
        """
        try:
            terminated = False
            truncated = False
            total_reward = 0

            if self.step_count >= MAX_STEPS_GAME:
                logging.info("Достигнуто максимальное количество шагов")
                truncated = True
                total_reward = 0

            elif np.all(self.done_status == 1):
                terminated = True
                logging.info("Все растения опрысканы")
                for agent in self.agents:
                    agent.position = random.choice(list(self.base_positions))
                logging.info("Агенты вернулись на базу")

                if self.step_count <= MIN_GAME_STEPS:
                    total_reward = self.total_reward + REWARD_COMPLETION * 3
                    logging.info(
                        f"Увеличенная награда: {total_reward} за шагов меньше, чем {MIN_GAME_STEPS}"
                    )
                else:
                    total_reward = self.total_reward + REWARD_COMPLETION
                    logging.info(f"Награда: {total_reward}")
                self.total_reward = 0.0
            else:
                total_reward = 0

            return total_reward, terminated, truncated, {}
        except Exception as e:
            logging.error(f"Ошибка в методе _check_termination_conditions: {e}")
            return 0, False, False, {}

    def check_crash(
        self,
        obs: Dict[str, Any],
        agent: Agent,
        new_position: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Проверяет на столкновение агентов.
        """
        try:
            collision_count = 0
            # Использование множеств для быстрого поиска
            if new_position in self.occupied_positions:
                collision_count += 1
            if collision_count > 0:
                self.total_reward -= PENALTY_CRASH * collision_count
                logging.warning(
                    f"Столкновение {collision_count} агентов в позиции {new_position}"
                )
                new_position = agent.position
            return new_position
        except Exception as e:
            logging.error(f"Ошибка в методе check_crash: {e}")
            return agent.position

    def render(self) -> None:
        """Отображает текущее состояние игры с оптимизациями."""
        try:
            rects_to_update = []

            # Отрисовка фона
            self.renderer.screen.blit(self.renderer.base_icons['field_bg'], (0, 0))
            self.renderer.screen.blit(self.renderer.base_icons['field'], (0, 0))
            rects_to_update.extend([
                self.renderer.base_icons['field_bg'].get_rect(),
                self.renderer.base_icons['field'].get_rect()
            ])

            # Отрисовка сетки
            self.renderer.draw_grid()

            # Отрисовка базы
            rect = self.renderer.draw_base(next(iter(self.base_positions)))
            rects_to_update.append(rect)

            # Отрисовка целей и препятствий
            rects_to_update.extend(self.renderer.draw_objects(
                target_positions=self.target_positions,
                done_status=self.done_status,
                obstacle_positions=self.obstacle_positions,
                current_map=self.current_map,
                obstacle_icons=self.obstacle_icons
            ))

            # Накладываем оверлей на неизведанные области
            rects_to_update.extend(self.renderer.draw_overlay(self.current_map))

            # Отрисовка агентов
            rects_to_update.extend(self.renderer.draw_agents(self.agents))

            # Отрисовка панели статуса
            rects_to_update.extend(self.renderer.draw_status_panel(
                agents=self.agents,
                total_steps=self.step_count,
                total_points=int(self.total_reward),
                detected_flowers=self.detected_flowers,
                sprayed_flowers=self.sprayed_flowers
            ))

            # Обновление только изменённых областей
            pygame.display.update(rects_to_update)
        except Exception as e:
            logging.error(f"Ошибка в методе render: {e}")

    def render_full_screen(self) -> None:
        """
        Рендерит начальное сообщение и настраивает экран.
        """
        try:
            self.renderer.render_message(
                "Начало выполнения сценария\n\n\n" +
                f"Гиперпараметры модели:\n\n"
                f"Темп: {LEARNING_RATE}\n"
                f"Гамма: {GAMMA}\n"
                f"Диапазон обрезки: {CLIP_RANGE}\n"
                f"Длина эпизода: {N_STEPS}\n"
                f"Энтропия: {COEF}\n"
                f"Баланс ценности: {VF_COEF}\n"
                f"Эпох: {N_EPOCHS}\n"
                f"Размер батча: {BATCH_SIZE}\n"
            )
            pygame.display.set_caption("OS SWARM OF DRONES")
            logging.info("Начало выполнения сценария")
        except Exception as e:
            logging.error(f"Ошибка в методе render_full_screen: {e}")

class PygameHandler:
    """
    Класс для централизованного управления pygame и основным циклом игры с оптимизациями.
    """

    def __init__(self):
        self.total_steps: int = 0
        self.total_points: int = 0
        self.detected_flowers: int = 0
        self.sprayed_flowers: int = 0
        # Создаем менеджер Pygame с новым размером экрана
        self.pygame_manager = PygameManager(width=SCREEN_SIZE + STATUS_PANEL_WIDTH, height=SCREEN_SIZE + BAR_HEIGHT)
        self.pygame_manager.create_screen()
        self.renderer: Optional[Renderer] = None  # Инициализируем позже
        self.scenario: Optional[SprayingScenario] = None

    def run(self) -> None:
        """Запускает основной цикл игры."""
        try:
            num_agents, grid_size, selected = self.input_screen()
            cell_size = SCREEN_SIZE // grid_size

            # Теперь инициализируем Renderer с пользовательскими параметрами
            self.renderer = Renderer(
                pygame_manager=self.pygame_manager,
                grid_size=grid_size,
                cell_size=cell_size,
                status_panel_width=STATUS_PANEL_WIDTH
            )

            # Инициализируем сценарий с актуальными параметрами
            self.scenario = SprayingScenario(num_agents, grid_size, self.renderer)
            self.scenario.reset_objects_positions()
            self.scenario.render_full_screen()

            clock = pygame.time.Clock()
            pygame.display.set_caption("Pesticide Spraying Scenario")

            obs, info = self.scenario.reset()
            self.total_steps = 0
            self.total_points = 0
            self.detected_flowers = 0
            self.sprayed_flowers = 0

            while True:
                self.handle_events()

                # Выполняем шаг сценария
                obs, reward, terminated, truncated, info = self.scenario.step()
                self.total_steps = self.scenario.step_count
                self.total_points = int(self.scenario.total_reward)
                self.sprayed_flowers = self.scenario.sprayed_flowers
                self.detected_flowers = self.scenario.detected_flowers
                self.scenario.render()
                self.renderer.draw_status_panel(
                    agents=self.scenario.agents,
                    total_steps=self.total_steps,
                    total_points=self.total_points,
                    detected_flowers=self.detected_flowers,
                    sprayed_flowers=self.sprayed_flowers
                )
                pygame.display.flip()

                if truncated:
                    obs, info = self.scenario.reset()
                    self.renderer.render_message("Новая миссия")
                    pygame.time.wait(5000)
                    self.total_steps = 0
                    self.total_points = 0
                    self.detected_flowers = 0
                    self.sprayed_flowers = 0

                if terminated:
                    message = f"Конец миссии, награда: {int(reward)}, шагов: {self.total_steps}"
                    self.renderer.render_message(message)
                    pygame.display.flip()
                    pygame.time.wait(5000)
                    break

                clock.tick(60)  # Поддерживаем частоту кадров
        except KeyboardInterrupt:
            logging.info("Прервано пользователем")
        except Exception as e:
            logging.error(f"Произошла ошибка в основном цикле: {e}")
            raise
        finally:
            self.pygame_manager.quit()

    def handle_events(self) -> None:
        """Обрабатывает события pygame."""
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
        except Exception as e:
            logging.error(f"Ошибка при обработке событий: {e}")

    def input_screen(self) -> Tuple[int, int, int]:
        """
        Отображает окно ввода для выбора количества агентов, размера поля и сценария.
        """
        try:
            # Используем уже созданный screen из PygameManager
            screen = self.pygame_manager.screen
            font = pygame.font.Font(None, 36)
            small_font = pygame.font.Font(None, 24)
            clock = pygame.time.Clock()

            inputs = [
                "Введите количество агентов:",
                "Введите размер поля (минимум):",
                "Выберите сценарий (1 - spraying):"
            ]
            input_boxes = [pygame.Rect(150, 150 + i * 80, 300, 40) for i in range(len(inputs))]
            input_values = ["", "", ""]

            active_box = 0
            finished = False

            while not finished:
                screen.fill(GRAY)
                grid_size_min = 0  # Минимальный размер поля, обновляется динамически
                try:
                    num_agents = int(input_values[0]) if input_values[0].isdigit() else NUM_AGENTS
                    grid_size_min = ceil(
                        (COUNT_TARGETS + COUNT_OBSTACLES + int(num_agents)) ** 0.5
                    ) + STATION_SIZE
                except ValueError:
                    grid_size_min = 0  # Если ввод некорректный, не рассчитываем

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        for i, box in enumerate(input_boxes):
                            if box.collidepoint(event.pos):
                                active_box = i
                    if event.type == pygame.KEYDOWN:
                        if active_box < len(inputs):
                            if event.key == pygame.K_BACKSPACE:
                                input_values[active_box] = input_values[active_box][:-1]
                            elif event.key == pygame.K_RETURN:
                                active_box += 1
                                if active_box >= len(inputs):
                                    finished = True
                            else:
                                input_values[active_box] += event.unicode

                # Отрисовка подсказок и полей ввода
                for i, text in enumerate(inputs):
                    # Отображаем текст с выравниванием
                    if i == 1 and grid_size_min > 0:
                        display_text = f"Введите размер поля (минимум: {grid_size_min}):"
                    else:
                        display_text = text

                    render_text(
                        screen, display_text, small_font, (200, 200, 200), 150,
                        120 + i * 80
                    )
                    color = WHITE if i == active_box else BLACK
                    pygame.draw.rect(screen, color, input_boxes[i], 2)
                    render_text(
                        screen, input_values[i], font, BLACK, input_boxes[i].x + 5,
                        input_boxes[i].y + 5
                    )

                pygame.display.flip()
                clock.tick(30)

            # Преобразование и проверка введённых данных
            try:
                num_agents = int(input_values[0]) if input_values[0] else NUM_AGENTS
                grid_size = int(input_values[1]) if input_values[1] else GRID_SIZE
                if grid_size < grid_size_min:
                    raise ValueError(f"Размер поля должен быть больше, чем {grid_size_min}")
                selected_scenario = int(input_values[2]) if input_values[2] else 1
            except ValueError as e:
                logging.error(f"Ошибка ввода: {e}")
                # Используем значения по умолчанию, так как renderer еще не инициализирован
                pygame.display.set_caption("Ошибка ввода")
                self.pygame_manager.screen.fill(GRAY)
                message = f"Ошибка ввода:\n{e}"
                font = pygame.font.SysFont('Arial', 36)
                lines = message.split('\n')
                total_height = 0
                for i, line in enumerate(lines):
                    text_surface = font.render(line, True, RED)
                    text_width, text_height = font.size(line)
                    x = (SCREEN_SIZE + STATUS_PANEL_WIDTH - text_width) // 2
                    y = (SCREEN_SIZE + BAR_HEIGHT - len(lines) * text_height) // 2 + total_height
                    self.pygame_manager.screen.blit(text_surface, (x, y))
                    total_height += text_height + 5
                pygame.display.flip()
                pygame.time.wait(3000)
                pygame.quit()
                sys.exit()

            return num_agents, grid_size, selected_scenario
        except Exception as e:
            logging.error(f"Ошибка в методе input_screen: {e}")
            # Показать сообщение об ошибке, если Renderer не инициализирован
            if self.renderer:
                self.renderer.render_message("Ошибка ввода. Использованы значения по умолчанию.")
                pygame.time.wait(3000)
            pygame.quit()
            return NUM_AGENTS, GRID_SIZE, 1

# =============================================================================
# Главная функция
# =============================================================================

def run() -> None:
    """
    Главная функция для запуска сценария.
    """
    try:
        pygame_handler = PygameHandler()
        pygame_handler.run()
    except Exception as e:
        logging.error(f"Ошибка при запуске сценария: {e}")
        pygame_handler.pygame_manager.quit()
        sys.exit()

if __name__ == '__main__':
    run()


# In[ ]:





# In[ ]:




