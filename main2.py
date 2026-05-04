"""
Simulación de Robots AGV (estilo Kiva) - Almacén de Amazon
=========================================================
Proyecto: Lab Robótica - UCT (Universidad Católica de Temuco)
Autores: Sebastián Riquelme, Bryan Zapata 

Descripción: 
4 robots inteligentes navegan por una bodega, recogen carritos asignados
al azar, los llevan a una zona de descarga y vuelven a sus bases a cargar energía.
Las rutas se calculan usando el algoritmo de Dijkstra (como un GPS).
"""

import pygame  # Librería para crear la ventana y los gráficos
import random  # Para elegir carritos y destinos al azar
import heapq   # Herramienta para que el GPS (Dijkstra) sea muy rápido
import math    # Para cálculos matemáticos como los ángulos de giro

# =======================================================
# CONFIGURACIÓN (Los valores básicos del mundo)
# =======================================================

GRID_COLS = 12  # El mapa tiene 12 columnas de ancho
GRID_ROWS = 12  # El mapa tiene 12 filas de alto
CELL = 60       # Cada cuadro del mapa mide 60 píxeles
PANEL_W = 280   # Ancho del panel lateral donde se ve la información

SCREEN_W = GRID_COLS * CELL + PANEL_W  # Ancho total de la ventana
SCREEN_H = GRID_ROWS * CELL            # Alto total de la ventana

FPS = 60  # Cuántas veces por segundo se actualiza la imagen (fluidez)

# Velocidades de los robots
MOVE_SPEED = 2.5    # Píxeles que avanza el robot en cada paso
ROTATE_SPEED = 4.0  # Grados que gira el robot en cada paso

# Configuración de la batería 
BATTERY_DRAIN_PER_STEP = 1.5    # Energía que gasta el robot al moverse
BATTERY_CHARGE_PER_FRAME = 0.6  # Energía que recupera al estar en la base

# Colores (en formato Rojo, Verde, Azul)
BG = (28, 30, 38)            # Fondo oscuro de la ventana
GRID_LINE = (45, 48, 58)     # Color de las líneas de la cuadrícula
FLOOR = (38, 42, 52)         # Color del suelo de la bodega
SHELF = (110, 78, 48)        # Color de las estanterías (madera)
SHELF_TOP = (140, 100, 62)   # Color de la parte de arriba de la estantería
PANEL_BG = (22, 24, 30)      # Fondo del panel lateral
PANEL_BORDER = (60, 64, 76)  # Borde del panel
TEXT = (220, 224, 232)       # Color del texto principal
TEXT_DIM = (140, 148, 160)   # Color del texto secundario
BASE_COLOR = (55, 60, 72)    # Color de las bases de carga
BASE_BORDER = (90, 95, 110)  # Borde de las bases
CART_BASE = (200, 200, 210)  # Color del carrito cuando está libre (gris)
DEST_COLOR = (80, 200, 120)  # Color de la zona de salida (verde)
QR_COLOR = (70, 75, 88)      # Color de los puntitos QR en el suelo 

# Colores únicos para identificar a cada uno de los 4 robots 
ROBOT_COLORS = [
    (235, 80, 80),    # Rojo
    (70, 140, 245),   # Azul
    (80, 200, 110),   # Verde
    (245, 195, 60),   # Amarillo
]

# =======================================================
# EL MAPA (El diseño de la bodega) 
# =======================================================

# 'S' = estantería, '.' = pasillo, 'B' = base, 'D' = destino de entrega
LAYOUT = [
    "B..........B",
    "............",
    "............",
    ".SSS.SSS.SS.",
    "............",
    ".SSS.SSS.SS.",
    "............",
    ".SSS.SSS.SS.",
    "............",
    "............",
    "............",
    "B....DD....B",
]

# Comprobación de seguridad para que el mapa no tenga errores de tamaño
assert len(LAYOUT) == GRID_ROWS
for row in LAYOUT:
    assert len(row) == GRID_COLS

# =======================================================
# EL GRAFO (Las conexiones del mapa para el GPS) 
# =======================================================

class Graph:
    def __init__(self):
        self.edges = {}  # Diccionario que guarda qué cuadros están conectados

    def add_edge(self, a, b):
        # Conecta el cuadro A con el B para que el robot sepa que puede pasar
        self.edges.setdefault(a, []).append(b)
        self.edges.setdefault(b, []).append(a)

    def neighbors(self, node):
        # Devuelve los cuadros vecinos a los que se puede ir desde el cuadro actual
        return self.edges.get(node, [])

    def dijkstra(self, start, goal, blocked=None):
        # Algoritmo de Dijkstra para encontrar el camino más corto
        if blocked is None:
            blocked = set()  # Cuadros ocupados por otros robots
        if start == goal:
            return [start]   # Si ya estamos en el destino, la ruta es solo este punto
        
        # pq: Lista de prioridad (costo, cuadro actual, camino recorrido)
        pq = [(0, start, [start])]
        seen = set()  # Para no revisar dos veces el mismo lugar
        while pq:
            cost, node, path = heapq.heappop(pq)  # Saca el camino más corto actual
            if node in seen:
                continue
            seen.add(node)  # Marca el cuadro como visitado
            if node == goal:
                return path  # ¡Éxito! Encontramos la ruta más rápida
            for nxt in self.neighbors(node):
                if nxt in seen:
                    continue
                # Si el camino está bloqueado por otro robot, lo ignora (a menos que sea la meta)
                if nxt in blocked and nxt != goal:
                    continue
                # Agrega el vecino a la lista de exploración sumando 1 al costo
                heapq.heappush(pq, (cost + 1, nxt, path + [nxt]))
        return None  # Si no hay camino, devuelve nada


def build_graph(walkable):
    # Construye el mapa de conexiones recorriendo todos los pasillos libres
    g = Graph()
    for (x, y) in walkable:
        for dx, dy in [(1, 0), (0, 1)]:  # Revisa hacia la derecha y hacia abajo
            n = (x + dx, y + dy)
            if n in walkable:
                g.add_edge((x, y), n)  # Crea un camino entre el cuadro actual y el de al lado
    return g

# =======================================================
# ENTIDADES (Los objetos del mundo: Carritos y Robots)
# =======================================================

class Cart:
    def __init__(self, cid, pos):
        self.id = cid               # Número de identificación del carrito
        self.pos = pos              # Cuadro (x, y) donde está parado
        self.home = pos             # Posición original para cuando se reinicie
        self.original_color = CART_BASE  # Color gris original
        self.color = CART_BASE      # Color actual (cambiará al del robot que lo lleve)
        self.carried_by = None      # ID del robot que lo está cargando (si hay uno) 


# Estados o "modos" en los que puede estar el robot
S_IDLE = "IDLE"           # Sin tarea
S_TO_CART = "→ CARRITO"   # Yendo a buscar su carrito asignado 
S_CARRYING = "↗ DESTINO"  # Llevando el carrito a la zona verde 
S_RETURNING = "← BASE"    # Volviendo a la base tras entregar 
S_CHARGING = "CARGANDO"   # Recuperando batería en la base 
S_WAITING = "ESPERANDO"   # Pausado porque hay tráfico u otro robot en su camino
S_REROUTING = "REPLANEA"  # Recalculando ruta para evitar un bloqueo

# Ajustes de inteligencia para el tráfico
MAX_WAIT_FRAMES = 90        # Cuántos pasos espera antes de buscar otro camino (paciencia)
REROUTE_COOLDOWN = 30       # Tiempo mínimo entre cada vez que cambia de ruta


class Robot:
    def __init__(self, rid, base_pos, color):
        self.id = rid                  # ID del robot (1, 2, 3 o 4)
        self.base = base_pos           # Cuadro de su base de carga
        self.cell = base_pos           # Cuadro actual en el mapa lógico
        self.px = base_pos[0] * CELL   # Posición real en la pantalla (píxel X)
        self.py = base_pos[1] * CELL   # Posición real en la pantalla (píxel Y)
        self.color = color             # Color único del robot 
        self.angle = 0.0               # Hacia dónde mira actualmente (en grados)
        self.target_angle = 0.0        # Hacia dónde debe girar para avanzar
        self.battery = 100.0           # Nivel de batería (empieza lleno)
        self.cart = None               # Carrito que tiene asignado
        self.path = []                 # Lista de cuadros que forman su ruta actual
        self.path_idx = 0              # Cuál es el siguiente cuadro de la ruta
        self.state = S_IDLE            # Modo actual del robot
        self.task_state = S_IDLE       # Guarda lo que hacía antes de quedarse esperando
        self.delivery_dest = None      # Cuadro de destino para dejar el carrito
        self.goal = None               # El objetivo final (carrito, salida o base)
        self.wait_counter = 0          # Cuánto tiempo lleva bloqueado por tráfico
        self.reroute_cooldown = 0      # Tiempo que falta para poder replanificar de nuevo

    # --- Funciones de utilidad ---
    def center_px(self):
        # Calcula el centro exacto del robot en la pantalla
        return (self.px + CELL / 2, self.py + CELL / 2)

    def is_moving(self):
        # Indica si el robot todavía tiene cuadros pendientes en su ruta
        return self.path and self.path_idx < len(self.path)

    def in_transit(self):
        # Indica si el robot está "entre medio" de dos cuadros físicamente
        target_px = self.cell[0] * CELL
        target_py = self.cell[1] * CELL
        return abs(self.px - target_px) > 0.5 or abs(self.py - target_py) > 0.5

    def next_cell(self):
        # Mira cuál es el siguiente paso en la ruta
        if self.is_moving():
            return self.path[self.path_idx]
        return None

    def claimed_cells(self):
        # Indica qué cuadros ocupa el robot (el actual y el siguiente si se está moviendo)
        cells = {self.cell}
        nxt = self.next_cell()
        if nxt is not None and self.in_transit():
            cells.add(nxt)
        return cells

    def assign_path(self, path):
        # Recibe una nueva ruta del GPS y la configura
        if not path or len(path) < 2:
            self.path = []
            self.path_idx = 0
            return
        self.path = path
        self.path_idx = 1  # El punto 0 es donde ya está parado el robot

    # --- Actualización del Robot (Movimiento y Giro) ---
    def update(self, blocked_cells=None):
        if blocked_cells is None:
            blocked_cells = set()

        if self.reroute_cooldown > 0:
            self.reroute_cooldown -= 1 # Baja el tiempo de espera para replanificar

        # Lógica si el robot está cargando en la base
        if self.state == S_CHARGING:
            self.battery = min(100.0, self.battery + BATTERY_CHARGE_PER_FRAME)
            return

        # Si no tiene ruta, no hace nada físico
        if not self.is_moving():
            return

        next_cell = self.path[self.path_idx]

        # DETECCIÓN DE TRÁFICO: Si el siguiente cuadro está ocupado por otro robot
        if not self.in_transit() and next_cell in blocked_cells:
            if self.state != S_WAITING:
                self.task_state = self.state  # Guarda su tarea actual
                self.state = S_WAITING        # Se pone en modo pausa
            self.wait_counter += 1            # Aumenta su contador de espera
            return

        # Si el camino se despeja, vuelve a su tarea
        if self.state == S_WAITING:
            self.state = self.task_state
            self.wait_counter = 0

        # Posición a la que debe llegar físicamente
        target_px = next_cell[0] * CELL
        target_py = next_cell[1] * CELL

        # Calcular dirección y ángulo
        dx = target_px - self.px
        dy = target_py - self.py

        if abs(dx) > 0.5 or abs(dy) > 0.5:
            # Calcula el ángulo en grados usando trigonometría 
            desired = math.degrees(math.atan2(dy, dx))
            self.target_angle = desired % 360

        # Lógica de Giro: El robot primero gira y luego avanza
        if not self._angles_close(self.angle, self.target_angle):
            self._rotate_step()
            return

        # Lógica de Avance: Se mueve píxel a píxel hacia el destino
        if abs(dx) > MOVE_SPEED:
            self.px += MOVE_SPEED if dx > 0 else -MOVE_SPEED
        else:
            self.px = target_px

        if abs(dy) > MOVE_SPEED:
            self.py += MOVE_SPEED if dy > 0 else -MOVE_SPEED
        else:
            self.py = target_py

        # Si llegó al centro del siguiente cuadro
        if self.px == target_px and self.py == target_py:
            self.cell = next_cell # Actualiza su posición lógica
            self.path_idx += 1    # Pasa al siguiente punto de la ruta
            self.battery = max(0.0, self.battery - BATTERY_DRAIN_PER_STEP) # Gasta batería
            self.wait_counter = 0 # Reinicia su paciencia
            # Si lleva un carrito, el carrito se mueve con él
            if self.cart is not None and self.cart.carried_by == self.id:
                self.cart.pos = self.cell

    def _angles_close(self, a, b, tol=1.0):
        # Revisa si el ángulo actual es casi igual al deseado (con margen de error)
        diff = abs((a - b + 180) % 360 - 180)
        return diff <= tol

    def _rotate_step(self):
        # Gira el robot paso a paso hacia el ángulo correcto por el camino más corto 
        diff = (self.target_angle - self.angle + 180) % 360 - 180
        if abs(diff) <= ROTATE_SPEED:
            self.angle = self.target_angle
        else:
            self.angle += ROTATE_SPEED if diff > 0 else -ROTATE_SPEED
            self.angle %= 360

    # --- Dibujo del Robot en pantalla ---
    def draw(self, surf):
        cx, cy = self.center_px() # Centro del robot
        size = CELL - 14          # Tamaño visual del robot
        half = size / 2

        # Dibuja la sombra del robot para que se vea más profesional
        shadow = pygame.Surface((size + 6, size + 6), pygame.SRCALPHA)
        pygame.draw.rect(shadow, (0, 0, 0, 80), shadow.get_rect(), border_radius=8)
        surf.blit(shadow, (cx - size / 2 - 1, cy - size / 2 + 3))

        # Dibuja el cuerpo cuadrado redondeado (tipo robot de Amazon Kiva)
        body = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(body, self.color, body.get_rect(), border_radius=8)
        darker = tuple(max(0, c - 60) for c in self.color)
        pygame.draw.rect(body, darker, body.get_rect(), width=2, border_radius=8)

        # Dibuja un triángulo blanco que indica hacia dónde mira el robot
        tip = (size * 0.85, size / 2)
        left = (size * 0.55, size * 0.30)
        right = (size * 0.55, size * 0.70)
        pygame.draw.polygon(body, (255, 255, 255), [tip, left, right])

        # Pequeño círculo central (representa la cámara que lee los QR) 
        pygame.draw.circle(body, (30, 30, 40), (size / 2, size / 2), 4)

        # Aplica la rotación a la imagen del robot
        rotated = pygame.transform.rotate(body, -self.angle)
        rect = rotated.get_rect(center=(cx, cy))
        surf.blit(rotated, rect)

        # Dibuja el número (ID) del robot sobre su cabeza
        font = pygame.font.SysFont("consolas", 14, bold=True)
        label = font.render(str(self.id), True, (255, 255, 255))
        lbl_bg = pygame.Surface((16, 16), pygame.SRCALPHA)
        pygame.draw.circle(lbl_bg, (0, 0, 0, 160), (8, 8), 8)
        surf.blit(lbl_bg, (cx - 8, cy - half - 10))
        surf.blit(label, label.get_rect(center=(cx, cy - half - 2)))

        # Si está esperando por tráfico, muestra un signo de exclamación naranja
        if self.state == S_WAITING:
            warn_bg = pygame.Surface((20, 20), pygame.SRCALPHA)
            pygame.draw.circle(warn_bg, (255, 165, 60), (10, 10), 10)
            pygame.draw.circle(warn_bg, (180, 100, 30), (10, 10), 10, 2)
            surf.blit(warn_bg, (cx + half - 10, cy - half - 10))
            warn_font = pygame.font.SysFont("consolas", 14, bold=True)
            warn_txt = warn_font.render("!", True, (40, 30, 10))
            surf.blit(warn_txt, warn_txt.get_rect(center=(cx + half, cy - half)))


# =======================================================
# LA SIMULACIÓN (El director de orquesta) 
# =======================================================

class Simulation:
    def __init__(self):
        self.walkable = set()     # Conjunto de cuadros por donde se puede caminar
        self.shelves = set()      # Conjunto de cuadros con estanterías
        self.bases = []           # Lista de cuadros que son bases de carga
        self.destinations = []    # Lista de cuadros de la zona de salida

        self._parse_layout()                 # Lee el mapa LAYOUT y clasifica los cuadros
        self.graph = build_graph(self.walkable)  # Crea el grafo de navegación

        # Crea los 4 robots y los pone en sus bases 
        self.robots = []
        for i, base in enumerate(self.bases[:4]):
            self.robots.append(Robot(i + 1, base, ROBOT_COLORS[i]))

        # Crea los carritos y los pone en lugares al azar de los pasillos
        self.carts = self._spawn_carts()

        self.cycle = 0           # Contador de cuántas veces se ha completado el trabajo
        self.paused = False      # Para pausar el movimiento con espacio
        self.message = ""        # Texto de estado que aparece en el panel

        self._start_new_cycle()  # Inicia el primer ciclo de trabajo

    def _parse_layout(self):
        # Analiza el dibujo del mapa para saber qué hay en cada cuadro
        for y, row in enumerate(LAYOUT):
            for x, ch in enumerate(row):
                if ch == 'S':
                    self.shelves.add((x, y)) # Es una estantería
                else:
                    self.walkable.add((x, y)) # Es pasillo libre
                    if ch == 'B':
                        self.bases.append((x, y)) # Es una base
                    elif ch == 'D':
                        self.destinations.append((x, y)) # Es zona de salida

    def _spawn_carts(self):
        # Coloca 4 carritos en los pasillos evitando bases y salidas
        forbidden = set(self.bases) | set(self.destinations)
        candidates = [p for p in self.walkable if p not in forbidden and 1 <= p[1] <= GRID_ROWS - 2]
        chosen = []
        bands = [(1, 2), (3, 4), (5, 6), (7, 7)] # Divide el mapa en bandas para repartirlos
        random.shuffle(bands)
        for i, (lo, hi) in enumerate(bands[:4]):
            band_cells = [c for c in candidates if lo <= c[1] <= hi and c not in chosen]
            if band_cells:
                chosen.append(random.choice(band_cells))
        while len(chosen) < 4:
            extra = random.choice([c for c in candidates if c not in chosen])
            chosen.append(extra)
        return [Cart(i + 1, pos) for i, pos in enumerate(chosen)]

    def _start_new_cycle(self):
        # Reinicia la misión: asigna un carrito al azar a cada robot 
        self.cycle += 1
        self.message = f"Ciclo {self.cycle}: asignando carritos…"
        for cart in self.carts:
            cart.color = cart.original_color # Carrito vuelve a ser gris
            cart.carried_by = None
        shuffled_carts = self.carts[:]
        random.shuffle(shuffled_carts) # Mezcla los carritos
        for robot, cart in zip(self.robots, shuffled_carts):
            robot.cart = cart
            robot.state = S_TO_CART     # Robot ahora busca su carrito
            robot.goal = cart.pos
            self._plan_path(robot, cart.pos) # Calcula ruta GPS hasta el carrito

    def _plan_path(self, robot, goal):
        # Planifica una ruta evitando a otros robots
        robot.goal = goal
        blocked = self._dynamic_blocked(exclude_robot=robot, exclude_target=goal)
        path = self.graph.dijkstra(robot.cell, goal, blocked=blocked)
        if path is None:
            # Si no hay camino libre, busca uno ignorando robots (esperará en el camino)
            path = self.graph.dijkstra(robot.cell, goal, blocked=set())
        robot.assign_path(path)
        robot.reroute_cooldown = REROUTE_COOLDOWN

    def _dynamic_blocked(self, exclude_robot=None, exclude_target=None):
        # Devuelve qué cuadros están ocupados por todos los robots menos el actual
        blocked = set()
        for r in self.robots:
            if r is exclude_robot: continue
            blocked |= r.claimed_cells()
        if exclude_target in blocked:
            blocked.discard(exclude_target)
        return blocked

    def _advance_robot_logic(self, robot):
        # Controla el cambio de tareas del robot cuando llega a un destino
        if robot.is_moving() or robot.state in (S_CHARGING, S_WAITING):
            return

        if robot.state == S_TO_CART:
            # LLEGÓ AL CARRITO: Lo levanta y lo lleva a la salida 
            robot.state = S_CARRYING
            if robot.cart is not None:
                robot.cart.carried_by = robot.id
                robot.cart.color = robot.color # El carrito toma el color del robot
            dest = random.choice(self.destinations)
            self._plan_path(robot, dest)

        elif robot.state == S_CARRYING:
            # LLEGÓ A LA SALIDA: Suelta el carrito y vuelve a la base 
            if robot.cart is not None:
                robot.cart.color = robot.cart.original_color
                robot.cart.carried_by = None
                robot.cart = None
            robot.state = S_RETURNING
            self._plan_path(robot, robot.base)

        elif robot.state == S_RETURNING:
            # LLEGÓ A LA BASE: Se pone a cargar 
            robot.state = S_CHARGING
            robot.goal = None

    def _handle_blocked_robots(self):
        # Si un robot lleva mucho tiempo bloqueado, busca otra ruta para desatascarse
        for r in self.robots:
            if r.state != S_WAITING: continue
            # Tiempo de espera escalonado por ID para que no replaneen todos a la vez
            retraso_personalizado = MAX_WAIT_FRAMES + (r.id * 30)
            if r.wait_counter < retraso_personalizado: continue
            if r.reroute_cooldown > 0: continue
            if r.goal is None: continue

            self._plan_path(r, r.goal) # Recalcula ruta esquivando a los otros
            r.wait_counter = 0
            r.state = r.task_state # Vuelve a intentar moverse

    def update(self):
        # Se ejecuta 60 veces por segundo para mover todo el sistema
        if self.paused: return

        # Detecta conflictos de frente (Swap) y decide quién pasa por su ID
        deferring = self._compute_deferring_robots()

        for r in self.robots:
            blocked = set()
            for other in self.robots:
                if other is r: continue
                blocked |= other.claimed_cells()
            # Si tiene ID alto y hay conflicto, le prohibimos avanzar
            if r.id in deferring:
                nxt = r.next_cell()
                if nxt is not None: blocked.add(nxt)
            r.update(blocked_cells=blocked)

        self._handle_blocked_robots() # Maneja desvíos por tráfico

        for r in self.robots:
            self._advance_robot_logic(r) # Gestiona cambios de tarea

        # Si todos terminaron y cargaron batería, reinicia el ciclo de trabajo
        if all(r.state == S_CHARGING and r.battery >= 99.9 for r in self.robots):
            for cart in self.carts: cart.pos = cart.home
            self._start_new_cycle()

        # Actualiza el mensaje informativo del panel lateral
        states = [r.state for r in self.robots]
        if all(s == S_CHARGING for s in states):
            self.message = f"Ciclo {self.cycle}: cargando baterías…"
        elif any(s == S_WAITING for s in states):
            self.message = f"Ciclo {self.cycle}: robot(s) cediendo paso"
        else:
            self.message = f"Ciclo {self.cycle}: en operación"

    def _compute_deferring_robots(self):
        # Si dos robots se encuentran de frente, el de ID más grande cede el paso
        deferring = set()
        for a in self.robots:
            a_next = a.next_cell()
            if a_next is None: continue
            for b in self.robots:
                if a is b: continue
                b_next = b.next_cell()
                if b_next is None: continue
                # Conflicto: A quiere ir donde está B, y B donde está A
                if a_next == b.cell and b_next == a.cell:
                    deferring.add(max(a.id, b.id)) # El ID mayor debe esperar
        return deferring

    def reset(self):
        # Reinicia toda la simulación a su estado inicial
        for r in self.robots:
            r.cell = r.base
            r.px, r.py = r.base[0] * CELL, r.base[1] * CELL
            r.angle, r.target_angle = 0, 0
            r.battery, r.path, r.path_idx = 100, [], 0
            r.cart, r.state, r.goal, r.wait_counter = None, S_IDLE, None, 0
        for c in self.carts:
            c.pos, c.color, c.carried_by = c.home, c.original_color, None
        self.cycle = 0
        self._start_new_cycle()


# =======================================================
# EL RENDER (La parte visual: dibujos y panel)
# =======================================================

class Renderer:
    def __init__(self, sim):
        self.sim = sim
        pygame.font.init() # Inicializa el sistema de letras
        self.font_sm = pygame.font.SysFont("consolas", 12)
        self.font_md = pygame.font.SysFont("consolas", 14, bold=True)
        self.font_xl = pygame.font.SysFont("consolas", 22, bold=True)

    def draw(self, screen):
        # Dibuja cada elemento capa por capa
        screen.fill(BG)
        self._draw_floor(screen)
        self._draw_grid(screen)
        self._draw_qr_codes(screen)
        self._draw_destinations(screen)
        self._draw_bases(screen)
        self._draw_shelves(screen)
        self._draw_paths(screen) # Muestra las líneas de las rutas GPS
        self._draw_carts(screen)
        for r in self.sim.robots: r.draw(screen) # Dibuja los 4 robots
        self._draw_panel(screen) # Dibuja la barra de información lateral

    def _draw_floor(self, screen):
        # Dibuja el rectángulo del suelo
        rect = pygame.Rect(0, 0, GRID_COLS * CELL, GRID_ROWS * CELL)
        pygame.draw.rect(screen, FLOOR, rect)

    def _draw_grid(self, screen):
        # Dibuja las líneas de la cuadrícula
        for x in range(GRID_COLS + 1):
            pygame.draw.line(screen, GRID_LINE, (x * CELL, 0), (x * CELL, GRID_ROWS * CELL))
        for y in range(GRID_ROWS + 1):
            pygame.draw.line(screen, GRID_LINE, (0, y * CELL), (GRID_COLS * CELL, y * CELL))

    def _draw_qr_codes(self, screen):
        # Dibuja los puntos QR que usan los robots para ubicarse 
        for (x, y) in self.sim.walkable:
            cx, cy = x * CELL + CELL // 2, y * CELL + CELL // 2
            pygame.draw.rect(screen, QR_COLOR, (cx - 3, cy - 3, 6, 6))

    def _draw_destinations(self, screen):
        # Dibuja la zona verde donde se entregan los productos
        for (x, y) in self.sim.destinations:
            rect = pygame.Rect(x * CELL + 6, y * CELL + 6, CELL - 12, CELL - 12)
            pygame.draw.rect(screen, (40, 80, 55), rect, border_radius=6)
            pygame.draw.rect(screen, DEST_COLOR, rect, width=2, border_radius=6)

    def _draw_bases(self, screen):
        # Dibuja las estaciones de carga de los robots
        for i, (x, y) in enumerate(self.sim.bases[:4]):
            rect = pygame.Rect(x * CELL + 4, y * CELL + 4, CELL - 8, CELL - 8)
            pygame.draw.rect(screen, BASE_COLOR, rect, border_radius=6)
            pygame.draw.rect(screen, ROBOT_COLORS[i], rect, width=2, border_radius=6)

    def _draw_shelves(self, screen):
        # Dibuja las estanterías de madera
        for (x, y) in self.sim.shelves:
            rect = pygame.Rect(x * CELL + 2, y * CELL + 2, CELL - 4, CELL - 4)
            pygame.draw.rect(screen, SHELF, rect, border_radius=4)
            pygame.draw.rect(screen, SHELF_TOP, (rect.x, rect.y, rect.w, 8), border_radius=4)

    def _draw_paths(self, screen):
        # Dibuja una línea tenue que muestra el camino que el GPS le dio al robot
        for r in self.sim.robots:
            if not r.is_moving(): continue
            pts = [r.center_px()] + [(c[0] * CELL + CELL/2, c[1] * CELL + CELL/2) for c in r.path[r.path_idx:]]
            if len(pts) >= 2:
                surf = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
                pygame.draw.lines(surf, (*r.color, 80), False, pts, 3)
                screen.blit(surf, (0, 0))

    def _draw_carts(self, screen):
        # Dibuja los carritos (grandes si los lleva un robot, pequeños si están en el suelo)
        for c in self.sim.carts:
            if c.carried_by is not None:
                # Dibujo del carrito debajo del robot (estilo plataforma) 
                r = next(rob for rob in self.sim.robots if rob.id == c.carried_by)
                rect = pygame.Rect(0, 0, CELL - 4, CELL - 4)
                rect.center = r.center_px()
                pygame.draw.rect(screen, c.color, rect, border_radius=6)
                pygame.draw.rect(screen, (0, 0, 0, 100), rect, width=3, border_radius=6)
            else:
                # Carrito esperando en el suelo
                rect = pygame.Rect(c.pos[0] * CELL + 10, c.pos[1] * CELL + 10, CELL - 20, CELL - 20)
                pygame.draw.rect(screen, c.color, rect, border_radius=4)
                pygame.draw.rect(screen, (0, 0, 0, 50), rect, width=2, border_radius=4)

    def _draw_panel(self, screen):
        # Dibuja el panel lateral con la información de los robots
        panel_x = GRID_COLS * CELL
        pygame.draw.rect(screen, PANEL_BG, (panel_x, 0, PANEL_W, SCREEN_H))
        pygame.draw.line(screen, PANEL_BORDER, (panel_x, 0), (panel_x, SCREEN_H), 2)
        
        y = 16
        screen.blit(self.font_xl.render("KIVA AGV SIM", True, TEXT), (panel_x + 16, y))
        y += 40
        screen.blit(self.font_md.render(self.sim.message, True, (180, 200, 255)), (panel_x + 16, y))
        y += 40

        # Tarjeta de información para cada robot (Nombre, Estado, Batería)
        for r in self.sim.robots:
            rect = pygame.Rect(panel_x + 16, y, PANEL_W - 32, 70)
            pygame.draw.rect(screen, (32, 36, 46), rect, border_radius=6)
            pygame.draw.rect(screen, r.color, rect, width=2, border_radius=6)
            
            screen.blit(self.font_md.render(f"Robot {r.id}", True, TEXT), (rect.x + 10, rect.y + 10))
            col = (255, 165, 60) if r.state == S_WAITING else TEXT_DIM
            screen.blit(self.font_sm.render(r.state, True, col), (rect.x + 10, rect.y + 30))
            
            # Barra de batería visual
            pygame.draw.rect(screen, (50, 54, 64), (rect.x + 10, rect.y + 50, rect.w - 20, 8), border_radius=3)
            fill_w = (rect.w - 20) * (r.battery / 100)
            b_col = (80, 200, 110) if r.battery > 20 else (235, 80, 80)
            pygame.draw.rect(screen, b_col, (rect.x + 10, rect.y + 50, fill_w, 8), border_radius=3)
            y += 80

# =======================================================
# FUNCIÓN PRINCIPAL (El botón de inicio)
# =======================================================

def main():
    pygame.init() # Inicia Pygame
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Kiva AGV Simulation - UCT")
    clock = pygame.time.Clock() # Reloj para controlar los FPS

    sim = Simulation() # Crea la lógica de la bodega
    renderer = Renderer(sim) # Crea la parte visual

    running = True
    while running:
        # Revisa si el usuario hizo algo (teclado o cerrar ventana)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                if event.key == pygame.K_SPACE: sim.paused = not sim.paused # Pausa
                if event.key == pygame.K_r: sim.reset() # Reinicia todo

        sim.update()       # Actualiza posiciones y lógica
        renderer.draw(screen) # Dibuja todo en la ventana
        pygame.display.flip() # Muestra los cambios en pantalla
        clock.tick(FPS)    # Mantiene la velocidad constante a 60 FPS

    pygame.quit() # Cierra el programa correctamente

if __name__ == "__main__":
    main() # Llama a la función principal para empezar