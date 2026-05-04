"""
Simulación de Robots AGV (Kiva-style) - Amazon Warehouse
=========================================================
Proyecto: Lab Robótica - UCT
Autor: Sebastián

4 robots AGV navegan por una bodega, recogen carritos asignados
aleatoriamente, los llevan a un destino y vuelven a sus bases a cargar.
Las rutas se calculan usando Dijkstra sobre un grafo 2D.

Controles:
    ESC / Cerrar ventana -> Salir
    SPACE                -> Pausar/reanudar
    R                    -> Reiniciar ciclo
"""

import pygame
import random
import heapq
import math

# =======================================================
# CONFIGURACIÓN
# =======================================================

GRID_COLS = 12
GRID_ROWS = 9
CELL = 60
PANEL_W = 280

SCREEN_W = GRID_COLS * CELL + PANEL_W
SCREEN_H = GRID_ROWS * CELL

FPS = 60

# Velocidades (en píxeles por frame y grados por frame)
MOVE_SPEED = 2.5
ROTATE_SPEED = 4.0

# Batería
BATTERY_DRAIN_PER_STEP = 1.5
BATTERY_CHARGE_PER_FRAME = 0.6

# Colores
BG = (28, 30, 38)
GRID_LINE = (45, 48, 58)
FLOOR = (38, 42, 52)
SHELF = (110, 78, 48)
SHELF_TOP = (140, 100, 62)
PANEL_BG = (22, 24, 30)
PANEL_BORDER = (60, 64, 76)
TEXT = (220, 224, 232)
TEXT_DIM = (140, 148, 160)
BASE_COLOR = (55, 60, 72)
BASE_BORDER = (90, 95, 110)
CART_BASE = (200, 200, 210)
DEST_COLOR = (80, 200, 120)
QR_COLOR = (70, 75, 88)

ROBOT_COLORS = [
    (235, 80, 80),    # rojo
    (70, 140, 245),   # azul
    (80, 200, 110),   # verde
    (245, 195, 60),   # amarillo
]

# =======================================================
# MAPA
# =======================================================

# 'S' = estantería, '.' = pasillo, 'B' = base de robot, 'D' = destino entrega
# Diseñado para que TODO sea conexo (pasillos verticales y horizontales)
LAYOUT = [
    "B..........B",
    "............",
    ".SSS.SSS.SS.",
    "............",
    ".SSS.SSS.SS.",
    "............",
    ".SSS.SSS.SS.",
    "............",
    "B....DD....B",
]

assert len(LAYOUT) == GRID_ROWS
for row in LAYOUT:
    assert len(row) == GRID_COLS

# =======================================================
# GRAFO
# =======================================================

class Graph:
    def __init__(self):
        self.edges = {}

    def add_edge(self, a, b):
        self.edges.setdefault(a, []).append(b)
        self.edges.setdefault(b, []).append(a)

    def neighbors(self, node):
        return self.edges.get(node, [])

    def dijkstra(self, start, goal, blocked=None):
        if blocked is None:
            blocked = set()
        if start == goal:
            return [start]
        # Permitir start aunque esté "bloqueado" (estamos parados ahí)
        pq = [(0, start, [start])]
        seen = set()
        while pq:
            cost, node, path = heapq.heappop(pq)
            if node in seen:
                continue
            seen.add(node)
            if node == goal:
                return path
            for nxt in self.neighbors(node):
                if nxt in seen:
                    continue
                if nxt in blocked and nxt != goal:
                    continue
                heapq.heappush(pq, (cost + 1, nxt, path + [nxt]))
        return None


def build_graph(walkable):
    g = Graph()
    for (x, y) in walkable:
        for dx, dy in [(1, 0), (0, 1)]:
            n = (x + dx, y + dy)
            if n in walkable:
                g.add_edge((x, y), n)
    return g

# =======================================================
# ENTIDADES
# =======================================================

class Cart:
    def __init__(self, cid, pos):
        self.id = cid
        self.pos = pos          # celda (x, y)
        self.home = pos         # posición original
        self.original_color = CART_BASE
        self.color = CART_BASE
        self.carried_by = None  # robot id o None


# Estados del robot
S_IDLE = "IDLE"
S_TO_CART = "→ CARRITO"
S_CARRYING = "↗ DESTINO"
S_RETURNING = "← BASE"
S_CHARGING = "CARGANDO"
S_WAITING = "ESPERANDO"   # bloqueado por otro robot, esperando paso
S_REROUTING = "REPLANEA"  # recalculando ruta

# Sub-estado: el "trabajo" actual (lo que estaba haciendo antes de bloquearse)
# Se almacena aparte para poder retomar después de WAITING/REROUTING

MAX_WAIT_FRAMES = 90        # tras esperar tanto, fuerza replanificación
REROUTE_COOLDOWN = 30       # frames mínimos entre replanificaciones


class Robot:
    def __init__(self, rid, base_pos, color):
        self.id = rid
        self.base = base_pos
        self.cell = base_pos          # celda lógica actual
        self.px = base_pos[0] * CELL  # píxel x (esquina sup izq de la celda)
        self.py = base_pos[1] * CELL
        self.color = color
        self.angle = 0.0              # grados, 0 = mirando a la derecha (+x)
        self.target_angle = 0.0
        self.battery = 100.0
        self.cart = None              # Cart asignado
        self.path = []                # lista de celdas
        self.path_idx = 0             # próximo nodo a alcanzar dentro de path
        self.state = S_IDLE
        self.task_state = S_IDLE      # tarea real (cuando está esperando/replaneando)
        self.delivery_dest = None     # celda donde soltar el carrito
        self.goal = None              # destino lógico actual (carrito/destino/base)
        self.wait_counter = 0         # frames bloqueado
        self.reroute_cooldown = 0     # frames hasta poder volver a replanificar

    # ---------- utilidades ----------
    def center_px(self):
        return (self.px + CELL / 2, self.py + CELL / 2)

    def is_moving(self):
        return self.path and self.path_idx < len(self.path)

    def in_transit(self):
        """True si el robot está en medio de una celda (entre celdas)."""
        target_px = self.cell[0] * CELL
        target_py = self.cell[1] * CELL
        return abs(self.px - target_px) > 0.5 or abs(self.py - target_py) > 0.5

    def next_cell(self):
        """La celda hacia la que se está moviendo (o None si no se mueve)."""
        if self.is_moving():
            return self.path[self.path_idx]
        return None

    def claimed_cells(self):
        """Celdas que este robot 'ocupa': la actual + la siguiente si está en tránsito."""
        cells = {self.cell}
        nxt = self.next_cell()
        if nxt is not None and self.in_transit():
            cells.add(nxt)
        return cells

    def assign_path(self, path):
        if not path or len(path) < 2:
            self.path = []
            self.path_idx = 0
            return
        self.path = path
        self.path_idx = 1  # path[0] es la celda actual

    # ---------- update ----------
    def update(self, blocked_cells=None):
        """blocked_cells: set de celdas que otros robots están ocupando/reclamando."""
        if blocked_cells is None:
            blocked_cells = set()

        if self.reroute_cooldown > 0:
            self.reroute_cooldown -= 1

        if self.state == S_CHARGING:
            self.battery = min(100.0, self.battery + BATTERY_CHARGE_PER_FRAME)
            return

        if not self.is_moving():
            # Sin path = sin acción de movimiento; estado lo maneja Simulation
            if self.state == S_WAITING:
                # Si estaba esperando y ya no tiene path, su tarea está completa o
                # debe ser replanificada por la simulación
                pass
            return

        next_cell = self.path[self.path_idx]

        # === DETECCIÓN DE COLISIÓN ===
        # Si NO estamos ya en tránsito hacia esa celda y está bloqueada -> esperar
        if not self.in_transit() and next_cell in blocked_cells:
            # Robot bloqueado: esperar y eventualmente replanificar
            if self.state != S_WAITING:
                self.task_state = self.state  # guardar tarea actual
                self.state = S_WAITING
            self.wait_counter += 1
            return

        # Si llegamos aquí, podemos avanzar -> volver al estado de tarea si veníamos esperando
        if self.state == S_WAITING:
            self.state = self.task_state
            self.wait_counter = 0

        target_px = next_cell[0] * CELL
        target_py = next_cell[1] * CELL

        # Calcular ángulo deseado
        dx = target_px - self.px
        dy = target_py - self.py

        if abs(dx) > 0.5 or abs(dy) > 0.5:
            desired = math.degrees(math.atan2(dy, dx))
            self.target_angle = desired % 360

        # Rotar primero
        if not self._angles_close(self.angle, self.target_angle):
            self._rotate_step()
            return

        # Mover
        if abs(dx) > MOVE_SPEED:
            self.px += MOVE_SPEED if dx > 0 else -MOVE_SPEED
        else:
            self.px = target_px

        if abs(dy) > MOVE_SPEED:
            self.py += MOVE_SPEED if dy > 0 else -MOVE_SPEED
        else:
            self.py = target_py

        # ¿Llegamos a la celda?
        if self.px == target_px and self.py == target_py:
            self.cell = next_cell
            self.path_idx += 1
            self.battery = max(0.0, self.battery - BATTERY_DRAIN_PER_STEP)
            self.wait_counter = 0
            # Solo arrastrar el carrito si REALMENTE lo está cargando
            # (no durante la fase S_TO_CART en la que solo va camino a recogerlo)
            if self.cart is not None and self.cart.carried_by == self.id:
                self.cart.pos = self.cell

    def _angles_close(self, a, b, tol=1.0):
        diff = abs((a - b + 180) % 360 - 180)
        return diff <= tol

    def _rotate_step(self):
        # Diferencia más corta en círculo
        diff = (self.target_angle - self.angle + 180) % 360 - 180
        if abs(diff) <= ROTATE_SPEED:
            self.angle = self.target_angle
        else:
            self.angle += ROTATE_SPEED if diff > 0 else -ROTATE_SPEED
            self.angle %= 360

    # ---------- dibujo ----------
    def draw(self, surf):
        cx, cy = self.center_px()
        size = CELL - 14
        half = size / 2

        # Sombra
        shadow = pygame.Surface((size + 6, size + 6), pygame.SRCALPHA)
        pygame.draw.rect(shadow, (0, 0, 0, 80), shadow.get_rect(), border_radius=8)
        surf.blit(shadow, (cx - size / 2 - 1, cy - size / 2 + 3))

        # Cuerpo del robot (cuadrado redondeado tipo Kiva)
        body = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(body, self.color, body.get_rect(), border_radius=8)
        # Borde más oscuro
        darker = tuple(max(0, c - 60) for c in self.color)
        pygame.draw.rect(body, darker, body.get_rect(), width=2, border_radius=8)

        # Indicador de dirección (triángulo / flecha)
        tip = (size * 0.85, size / 2)
        left = (size * 0.55, size * 0.30)
        right = (size * 0.55, size * 0.70)
        pygame.draw.polygon(body, (255, 255, 255), [tip, left, right])

        # Pequeño "ojo" / cámara QR en el centro
        pygame.draw.circle(body, (30, 30, 40), (size / 2, size / 2), 4)

        # Rotar el cuerpo
        rotated = pygame.transform.rotate(body, -self.angle)  # pygame Y va hacia abajo
        rect = rotated.get_rect(center=(cx, cy))
        surf.blit(rotated, rect)

        # ID del robot
        font = pygame.font.SysFont("consolas", 14, bold=True)
        label = font.render(str(self.id), True, (255, 255, 255))
        lbl_bg = pygame.Surface((16, 16), pygame.SRCALPHA)
        pygame.draw.circle(lbl_bg, (0, 0, 0, 160), (8, 8), 8)
        surf.blit(lbl_bg, (cx - 8, cy - half - 10))
        surf.blit(label, label.get_rect(center=(cx, cy - half - 2)))

        # Indicador de espera: signo de exclamación naranja sobre el robot
        if self.state == S_WAITING:
            warn_bg = pygame.Surface((20, 20), pygame.SRCALPHA)
            pygame.draw.circle(warn_bg, (255, 165, 60), (10, 10), 10)
            pygame.draw.circle(warn_bg, (180, 100, 30), (10, 10), 10, 2)
            surf.blit(warn_bg, (cx + half - 10, cy - half - 10))
            warn_font = pygame.font.SysFont("consolas", 14, bold=True)
            warn_txt = warn_font.render("!", True, (40, 30, 10))
            surf.blit(warn_txt, warn_txt.get_rect(center=(cx + half, cy - half)))


# =======================================================
# SIMULACIÓN
# =======================================================

class Simulation:
    def __init__(self):
        self.walkable = set()
        self.shelves = set()
        self.bases = []
        self.destinations = []

        self._parse_layout()
        self.graph = build_graph(self.walkable)

        # Crear robots en sus bases
        self.robots = []
        for i, base in enumerate(self.bases[:4]):
            self.robots.append(Robot(i + 1, base, ROBOT_COLORS[i]))

        # Crear carritos en posiciones libres alejadas de bases
        self.carts = self._spawn_carts()

        self.cycle = 0
        self.phase = "ASSIGN"  # ASSIGN, EXECUTING, CHARGING_WAIT
        self.paused = False
        self.message = ""

        self._start_new_cycle()

    def _parse_layout(self):
        for y, row in enumerate(LAYOUT):
            for x, ch in enumerate(row):
                if ch == 'S':
                    self.shelves.add((x, y))
                else:
                    self.walkable.add((x, y))
                    if ch == 'B':
                        self.bases.append((x, y))
                    elif ch == 'D':
                        self.destinations.append((x, y))

    def _spawn_carts(self):
        # Posiciones candidatas: pasillos no ocupados por bases ni destinos
        forbidden = set(self.bases) | set(self.destinations)
        # También evitar la fila inferior y superior para que se vea ordenado
        candidates = [
            p for p in self.walkable
            if p not in forbidden and 1 <= p[1] <= GRID_ROWS - 2
        ]
        # Distribuir uno por "banda" entre filas de estanterías
        chosen = []
        bands = [(1, 2), (3, 4), (5, 6), (7, 7)]
        random.shuffle(bands)
        for i, (lo, hi) in enumerate(bands[:4]):
            band_cells = [c for c in candidates if lo <= c[1] <= hi and c not in chosen]
            if band_cells:
                chosen.append(random.choice(band_cells))

        # Si por alguna razón no se llenaron 4, completar
        while len(chosen) < 4:
            extra = random.choice([c for c in candidates if c not in chosen])
            chosen.append(extra)

        return [Cart(i + 1, pos) for i, pos in enumerate(chosen)]

    # --------- ciclo lógico ---------

    def _start_new_cycle(self):
        self.cycle += 1
        self.message = f"Ciclo {self.cycle}: asignando carritos…"

        # Devolver carritos a su home si quedaron sueltos
        for cart in self.carts:
            cart.color = cart.original_color
            cart.carried_by = None

        # Asignación aleatoria robot -> carrito.
        # El carrito MANTIENE su color original (gris) hasta que el robot lo
        # recoja físicamente. Recién ahí cambia al color del robot.
        shuffled_carts = self.carts[:]
        random.shuffle(shuffled_carts)
        for robot, cart in zip(self.robots, shuffled_carts):
            robot.cart = cart
            cart.color = cart.original_color   # gris hasta ser recogido
            cart.carried_by = None             # aún no lo está cargando
            robot.state = S_TO_CART
            robot.task_state = S_TO_CART
            robot.goal = cart.pos
            # Calcular ruta hacia el carrito
            self._plan_path(robot, cart.pos)

        self.phase = "EXECUTING"

    def _plan_path(self, robot, goal):
        """Planifica una ruta desde robot.cell hasta goal evitando otros robots."""
        robot.goal = goal
        blocked = self._dynamic_blocked(exclude_robot=robot, exclude_target=goal)
        path = self.graph.dijkstra(robot.cell, goal, blocked=blocked)
        if path is None:
            # Sin alternativa: probar sin bloqueos dinámicos (los otros eventualmente
            # se moverán y este robot tomará la ruta cuando esté libre)
            path = self.graph.dijkstra(robot.cell, goal, blocked=set())
        robot.assign_path(path)
        robot.reroute_cooldown = REROUTE_COOLDOWN

    def _dynamic_blocked(self, exclude_robot=None, exclude_target=None):
        """Celdas reclamadas por otros robots (actual + próxima si está en tránsito)."""
        blocked = set()
        for r in self.robots:
            if r is exclude_robot:
                continue
            blocked |= r.claimed_cells()
        if exclude_target in blocked:
            blocked.discard(exclude_target)
        return blocked

    def _advance_robot_logic(self, robot):
        """Cuando un robot termina su path, decidir el siguiente paso."""
        if robot.is_moving() or robot.state in (S_CHARGING, S_WAITING):
            return

        if robot.state == S_TO_CART:
            # Llegó al carrito -> AHORA lo levanta, le pinta el color del robot
            # y se dirige al destino de entrega
            robot.state = S_CARRYING
            robot.task_state = S_CARRYING
            if robot.cart is not None:
                robot.cart.carried_by = robot.id     # ahora sí lo está cargando
                robot.cart.color = robot.color       # adopta el color del robot
                robot.cart.pos = robot.cell
            dest = random.choice(self.destinations)
            robot.delivery_dest = dest
            self._plan_path(robot, dest)

        elif robot.state == S_CARRYING:
            # Llegó al destino -> soltar carrito y volver a base
            if robot.cart is not None:
                robot.cart.pos = robot.cell
                robot.cart.color = robot.cart.original_color
                robot.cart.carried_by = None
                robot.cart = None
            robot.state = S_RETURNING
            robot.task_state = S_RETURNING
            self._plan_path(robot, robot.base)

        elif robot.state == S_RETURNING:
            # Llegó a base -> cargar
            robot.state = S_CHARGING
            robot.task_state = S_CHARGING
            robot.goal = None

    def _handle_blocked_robots(self):
        """Si un robot lleva mucho esperando, replanifica su ruta esquivando."""
        for r in self.robots:
            if r.state != S_WAITING:
                continue
            if r.wait_counter < MAX_WAIT_FRAMES:
                continue
            if r.reroute_cooldown > 0:
                continue
            if r.goal is None:
                continue

            # Replanificar evitando los otros robots
            old_path_len = len(r.path)
            self._plan_path(r, r.goal)
            r.wait_counter = 0
            # Volver al estado de tarea
            r.state = r.task_state

    def update(self):
        if self.paused:
            return

        # === Calcular celdas reclamadas por cada robot ===
        # Para evitar deadlocks frente-a-frente: si dos robots quieren la celda del
        # otro, el de mayor ID cede el paso (espera) y el de menor ID avanza.
        deferring = self._compute_deferring_robots()

        # Avanzar cada robot pasándole las celdas bloqueadas para él
        for r in self.robots:
            blocked = set()
            for other in self.robots:
                if other is r:
                    continue
                blocked |= other.claimed_cells()
            # Si este robot está cediendo el paso, también bloquearle su próxima celda
            # para forzarlo a esperar.
            if r.id in deferring:
                nxt = r.next_cell()
                if nxt is not None:
                    blocked.add(nxt)
            r.update(blocked_cells=blocked)

        # Resolver bloqueos prolongados con replanificación
        self._handle_blocked_robots()

        # Lógica de transición de estados
        for r in self.robots:
            self._advance_robot_logic(r)

        # ¿Todos cargados? -> reiniciar ciclo
        if all(r.state == S_CHARGING and r.battery >= 99.9 for r in self.robots):
            for cart in self.carts:
                cart.pos = cart.home
            self._start_new_cycle()

        # Mensaje de estado
        states = [r.state for r in self.robots]
        if all(s == S_CHARGING for s in states):
            self.message = f"Ciclo {self.cycle}: cargando baterías…"
        elif any(s == S_WAITING for s in states):
            n_waiting = sum(1 for s in states if s == S_WAITING)
            self.message = f"Ciclo {self.cycle}: {n_waiting} robot(s) cediendo paso"
        elif any(s == S_CARRYING for s in states):
            self.message = f"Ciclo {self.cycle}: transportando carritos"
        else:
            self.message = f"Ciclo {self.cycle}: en operación"

    def _compute_deferring_robots(self):
        """
        Detecta swaps (A→B y B→A) y devuelve los IDs de los robots que deben ceder.
        Regla: en un conflicto de swap, cede el de MAYOR ID (prioridad por ID menor).
        """
        deferring = set()
        for a in self.robots:
            a_next = a.next_cell()
            if a_next is None:
                continue
            for b in self.robots:
                if a is b:
                    continue
                b_next = b.next_cell()
                if b_next is None:
                    continue
                # Conflicto de swap: a quiere ir a b.cell y b quiere ir a a.cell
                if a_next == b.cell and b_next == a.cell:
                    deferring.add(max(a.id, b.id))
        return deferring

    def reset(self):
        for r in self.robots:
            r.cell = r.base
            r.px = r.base[0] * CELL
            r.py = r.base[1] * CELL
            r.angle = 0
            r.target_angle = 0
            r.battery = 100
            r.path = []
            r.path_idx = 0
            r.cart = None
            r.state = S_IDLE
            r.task_state = S_IDLE
            r.goal = None
            r.wait_counter = 0
            r.reroute_cooldown = 0
        for c in self.carts:
            c.pos = c.home
            c.color = c.original_color
            c.carried_by = None
        self.cycle = 0
        self._start_new_cycle()


# =======================================================
# RENDER
# =======================================================

class Renderer:
    def __init__(self, sim):
        self.sim = sim
        pygame.font.init()
        self.font_sm = pygame.font.SysFont("consolas", 12)
        self.font_md = pygame.font.SysFont("consolas", 14, bold=True)
        self.font_lg = pygame.font.SysFont("consolas", 18, bold=True)
        self.font_xl = pygame.font.SysFont("consolas", 22, bold=True)

    def draw(self, screen):
        screen.fill(BG)
        self._draw_floor(screen)
        self._draw_grid(screen)
        self._draw_qr_codes(screen)
        self._draw_destinations(screen)
        self._draw_bases(screen)
        self._draw_shelves(screen)
        self._draw_paths(screen)
        self._draw_carts(screen)
        for r in self.sim.robots:
            r.draw(screen)
        self._draw_panel(screen)

    def _draw_floor(self, screen):
        rect = pygame.Rect(0, 0, GRID_COLS * CELL, GRID_ROWS * CELL)
        pygame.draw.rect(screen, FLOOR, rect)

    def _draw_grid(self, screen):
        for x in range(GRID_COLS + 1):
            pygame.draw.line(screen, GRID_LINE, (x * CELL, 0),
                             (x * CELL, GRID_ROWS * CELL))
        for y in range(GRID_ROWS + 1):
            pygame.draw.line(screen, GRID_LINE, (0, y * CELL),
                             (GRID_COLS * CELL, y * CELL))

    def _draw_qr_codes(self, screen):
        # Pequeño cuadrado QR en cada celda transitable
        for (x, y) in self.sim.walkable:
            cx = x * CELL + CELL // 2
            cy = y * CELL + CELL // 2
            pygame.draw.rect(screen, QR_COLOR, (cx - 3, cy - 3, 6, 6))

    def _draw_destinations(self, screen):
        for (x, y) in self.sim.destinations:
            rect = pygame.Rect(x * CELL + 6, y * CELL + 6, CELL - 12, CELL - 12)
            pygame.draw.rect(screen, (40, 80, 55), rect, border_radius=6)
            pygame.draw.rect(screen, DEST_COLOR, rect, width=2, border_radius=6)
            txt = self.font_sm.render("OUT", True, DEST_COLOR)
            screen.blit(txt, txt.get_rect(center=rect.center))

    def _draw_bases(self, screen):
        for i, (x, y) in enumerate(self.sim.bases[:4]):
            rect = pygame.Rect(x * CELL + 4, y * CELL + 4, CELL - 8, CELL - 8)
            pygame.draw.rect(screen, BASE_COLOR, rect, border_radius=6)
            color = ROBOT_COLORS[i] if i < len(ROBOT_COLORS) else BASE_BORDER
            pygame.draw.rect(screen, color, rect, width=2, border_radius=6)
            # Esquinas marca de base
            for cx, cy in [(rect.left, rect.top), (rect.right, rect.top),
                           (rect.left, rect.bottom), (rect.right, rect.bottom)]:
                pygame.draw.circle(screen, color, (cx, cy), 3)

    def _draw_shelves(self, screen):
        for (x, y) in self.sim.shelves:
            rect = pygame.Rect(x * CELL + 2, y * CELL + 2, CELL - 4, CELL - 4)
            pygame.draw.rect(screen, SHELF, rect, border_radius=4)
            top = pygame.Rect(rect.x, rect.y, rect.w, 8)
            pygame.draw.rect(screen, SHELF_TOP, top, border_radius=4)
            # líneas de detalle
            for i in range(1, 3):
                ly = rect.y + i * (rect.h // 3)
                pygame.draw.line(screen, (70, 50, 30),
                                 (rect.x + 4, ly), (rect.right - 4, ly), 1)

    def _draw_paths(self, screen):
        # Mostrar path planificado del robot tenuemente
        for r in self.sim.robots:
            if not r.is_moving():
                continue
            pts = []
            # incluir posición actual del robot
            pts.append(r.center_px())
            for cell in r.path[r.path_idx:]:
                pts.append((cell[0] * CELL + CELL / 2, cell[1] * CELL + CELL / 2))
            if len(pts) >= 2:
                color = (*r.color, 80)
                surf = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
                pygame.draw.lines(surf, color, False, pts, 3)
                screen.blit(surf, (0, 0))

    def _draw_carts(self, screen):
        """Dibuja los carritos. Los que están en el suelo van con tamaño normal.
        Los cargados por un robot se dibujan MÁS GRANDES (como una plataforma
        bajo el robot, estilo Kiva real) para que se vea su color asomando."""
        for c in self.sim.carts:
            if c.carried_by is not None:
                # El carrito lo carga un robot -> se dibuja como plataforma
                robot = next(
                    (r for r in self.sim.robots if r.id == c.carried_by), None
                )
                if robot is None:
                    continue
                rcx, rcy = robot.center_px()
                # Plataforma más grande que el robot para que se vea el color
                size = CELL - 4
                rect = pygame.Rect(rcx - size / 2, rcy - size / 2, size, size)
                # Sombra
                pygame.draw.rect(screen, (15, 18, 25),
                                 rect.move(2, 3), border_radius=6)
                # Plataforma (color del robot/carrito)
                pygame.draw.rect(screen, c.color, rect, border_radius=6)
                # Borde grueso del color más oscuro
                border = tuple(max(0, v - 60) for v in c.color)
                pygame.draw.rect(screen, border, rect, width=3, border_radius=6)
                # ID del carrito en las esquinas (visible aunque el robot tape el centro)
                small_font = self.font_sm
                txt = small_font.render(f"C{c.id}", True, (255, 255, 255))
                # esquina superior izquierda
                screen.blit(txt, (rect.x + 4, rect.y + 2))
            else:
                # Carrito en el suelo (esperando o ya entregado)
                cx = c.pos[0] * CELL
                cy = c.pos[1] * CELL

                # Halo de color si el carrito está ASIGNADO pero aún no recogido.
                # (el carrito tiene color != gris original mientras está asignado)
                is_assigned = c.color != c.original_color
                if is_assigned:
                    halo = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
                    halo_color = (*c.color, 60)
                    pygame.draw.rect(halo, halo_color,
                                     halo.get_rect(), border_radius=10)
                    screen.blit(halo, (cx, cy))

                rect = pygame.Rect(cx + 10, cy + 10, CELL - 20, CELL - 20)
                pygame.draw.rect(screen, (30, 32, 40),
                                 rect.move(2, 2), border_radius=4)
                pygame.draw.rect(screen, c.color, rect, border_radius=4)
                border = tuple(max(0, v - 70) for v in c.color)
                pygame.draw.rect(screen, border, rect, width=3, border_radius=4)
                txt = self.font_sm.render(f"C{c.id}", True, (30, 30, 40))
                screen.blit(txt, txt.get_rect(center=rect.center))

    def _draw_panel(self, screen):
        panel_x = GRID_COLS * CELL
        panel_rect = pygame.Rect(panel_x, 0, PANEL_W, SCREEN_H)
        pygame.draw.rect(screen, PANEL_BG, panel_rect)
        pygame.draw.line(screen, PANEL_BORDER,
                         (panel_x, 0), (panel_x, SCREEN_H), 2)

        pad = 16
        y = pad

        title = self.font_xl.render("KIVA AGV SIM", True, TEXT)
        screen.blit(title, (panel_x + pad, y))
        y += title.get_height() + 2

        sub = self.font_sm.render("Amazon Warehouse - UCT", True, TEXT_DIM)
        screen.blit(sub, (panel_x + pad, y))
        y += sub.get_height() + 14

        # Estado global
        msg = self.font_md.render(self.sim.message, True, (180, 200, 255))
        screen.blit(msg, (panel_x + pad, y))
        y += msg.get_height() + 14

        pygame.draw.line(screen, PANEL_BORDER,
                         (panel_x + pad, y),
                         (panel_x + PANEL_W - pad, y), 1)
        y += 10

        # Tarjeta por robot
        for r in self.sim.robots:
            self._draw_robot_card(screen, panel_x + pad, y, PANEL_W - 2 * pad, r)
            y += 78

        # Pie con controles
        y = SCREEN_H - 70
        pygame.draw.line(screen, PANEL_BORDER,
                         (panel_x + pad, y),
                         (panel_x + PANEL_W - pad, y), 1)
        y += 8
        for line in ["[SPACE] Pausar", "[R] Reiniciar", "[ESC] Salir"]:
            t = self.font_sm.render(line, True, TEXT_DIM)
            screen.blit(t, (panel_x + pad, y))
            y += 16

        if self.sim.paused:
            overlay = pygame.Surface((GRID_COLS * CELL, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            screen.blit(overlay, (0, 0))
            t = self.font_xl.render("PAUSA", True, (255, 255, 255))
            screen.blit(t, t.get_rect(center=(GRID_COLS * CELL // 2, SCREEN_H // 2)))

    def _draw_robot_card(self, screen, x, y, w, r):
        rect = pygame.Rect(x, y, w, 70)
        pygame.draw.rect(screen, (32, 36, 46), rect, border_radius=6)
        pygame.draw.rect(screen, r.color, rect, width=2, border_radius=6)

        # Cuadrito de color
        sw = pygame.Rect(x + 8, y + 8, 18, 18)
        pygame.draw.rect(screen, r.color, sw, border_radius=3)

        # ID y estado
        title = self.font_md.render(f"R{r.id}", True, TEXT)
        screen.blit(title, (x + 32, y + 6))

        # Color del estado: naranja si está esperando, gris normal si no
        state_color = (255, 165, 60) if r.state == S_WAITING else TEXT_DIM
        state_txt = self.font_sm.render(r.state, True, state_color)
        screen.blit(state_txt, (x + 32, y + 22))

        # Carrito asignado
        cart_id = f"C{r.cart.id}" if r.cart else "—"
        ct = self.font_sm.render(f"Carrito: {cart_id}", True, TEXT_DIM)
        screen.blit(ct, (x + 8, y + 40))

        # Barra de batería
        bx = x + 8
        by = y + 56
        bw = w - 16
        bh = 8
        pygame.draw.rect(screen, (50, 54, 64), (bx, by, bw, bh), border_radius=3)
        fill = int(bw * (r.battery / 100))
        if r.battery > 50:
            bcol = (80, 200, 110)
        elif r.battery > 20:
            bcol = (245, 195, 60)
        else:
            bcol = (235, 80, 80)
        pygame.draw.rect(screen, bcol, (bx, by, fill, bh), border_radius=3)
        bt = self.font_sm.render(f"{int(r.battery)}%", True, TEXT)
        screen.blit(bt, (x + w - 36, y + 40))


# =======================================================
# MAIN
# =======================================================

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Kiva AGV Simulation - UCT")
    clock = pygame.time.Clock()

    sim = Simulation()
    renderer = Renderer(sim)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    sim.paused = not sim.paused
                elif event.key == pygame.K_r:
                    sim.reset()

        sim.update()
        renderer.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()