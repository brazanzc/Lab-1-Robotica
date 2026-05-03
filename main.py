import pygame
import random
import sys
from collections import deque

# ---------------- CONFIG ----------------
COLS, ROWS = 12, 10
CELL = 60
PANEL_W = 260
WIDTH = COLS * CELL + PANEL_W
HEIGHT = ROWS * CELL
FPS = 80

# velocidades (frames por movimiento)
SPEEDS = {1:30, 2:20, 3:12, 4:8, 5:4}
speed_level = 3

# ---------------- COLORES ----------------
BG = (30,30,30)
GRID = (60,60,60)
NODE = (200,50,50)
SHELF = (70,40,40)
BASE = (40,80,40)
TEXT = (220,220,220)

ROBOT_COLORS = [
    (0,255,0),
    (0,120,255),
    (255,140,0),
    (255,0,200)
]

CART_DEFAULT = (120,120,120)

# ---------------- MAPA ----------------
SHELVES = {
    (2,2),(2,3),(3,2),
    (5,1),(5,2),
    (8,2),(8,3),
    (2,6),(2,7),(3,7),
    (6,5),(6,6),
    (9,5),(9,6)
}

BASES = [(0,0),(11,0),(0,9),(11,9)]
CARTS = [(4,1),(7,1),(4,8),(7,8),(2,5),(9,3)]
DESTS = [(5,4),(6,4),(5,5),(6,3)]

# ---------------- BFS ----------------
def neighbors(c,r):
    for dc,dr in [(1,0),(-1,0),(0,1),(0,-1)]:
        nc,nr=c+dc,r+dr
        if 0<=nc<COLS and 0<=nr<ROWS and (nc,nr) not in SHELVES:
            yield (nc,nr)

def bfs(start,goal,blocked):
    queue=deque([start])
    came={start:None}

    while queue:
        cur=queue.popleft()
        if cur==goal: break

        for nxt in neighbors(*cur):
            if nxt not in came and nxt not in blocked:
                came[nxt]=cur
                queue.append(nxt)

    if goal not in came:
        return [start]

    path=[]
    cur=goal
    while cur:
        path.append(cur)
        cur=came[cur]

    return path[::-1]

# ---------------- CLASES ----------------
class Cart:
    def __init__(self,id,c,r):
        self.id=id
        self.col=c
        self.row=r
        self.color=CART_DEFAULT
        self.held=False
        self.carrier=None

        self._px = c*CELL + CELL//2
        self._py = r*CELL + CELL//2

    def draw(self,screen):
        x=int(self._px)
        y=int(self._py)

        pygame.draw.rect(screen,self.color,(x-8,y-8,16,16),border_radius=3)
        pygame.draw.circle(screen,(20,20,20),(x-5,y+6),2)
        pygame.draw.circle(screen,(20,20,20),(x+5,y+6),2)

class Robot:
    def __init__(self,id,base,color):
        self.id=id
        self.base=base
        self.col,self.row=base
        self.color=color

        self.px = self.col*CELL + CELL//2
        self.py = self.row*CELL + CELL//2

        self.path=[]
        self.target_cart=None
        self.target_dest=None

        self.state="CHARGING"

        # 🔥 anti-loop
        self.last_pos=None
        self.stuck_counter=0

        # control velocidad
        self.move_tick=0

    def blocked(self,robots,carts):
        b=set()

        for r in robots:
            if r.id!=self.id:
                b.add((r.col,r.row))

        for c in carts:
            if not c.held or c.carrier!=self.id:
                b.add((c.col,c.row))

        return b

    def update(self,robots,carts):

        global speed_level

        self.move_tick += 1
        if self.move_tick < SPEEDS[speed_level]:
            return
        self.move_tick = 0

        if self.state=="CHARGING":
            free=[c for c in carts if not c.held]
            if free:
                cart=random.choice(free)

                # 🔥 FIX COLOR: asignar aquí también
                cart.color=self.color

                cart.held=True
                cart.carrier=self.id

                self.target_cart=cart
                self.target_dest=random.choice(DESTS)

                self.path=bfs((self.col,self.row),(cart.col,cart.row),
                              self.blocked(robots,carts))

                self.state="GO_CART"

        elif self.state=="GO_CART":
            self.move(robots,carts,"PICK")

        elif self.state=="PICK":
            # asegurar color SIEMPRE
            self.target_cart.color=self.color

            self.state="GO_DEST"

            self.path=bfs((self.col,self.row),self.target_dest,
                          self.blocked(robots,carts))

        elif self.state=="GO_DEST":
            self.move(robots,carts,"DROP")

        elif self.state=="DROP":
            c=self.target_cart
            c.col=self.col
            c.row=self.row
            c.color=CART_DEFAULT
            c.held=False
            c.carrier=None

            self.target_cart=None
            self.state="RETURN"

            self.path=bfs((self.col,self.row),self.base,
                          self.blocked(robots,carts))

        elif self.state=="RETURN":
            self.move(robots,carts,"CHARGING")

        # 🔥 ARRASTRE
        if self.target_cart and self.target_cart.carrier==self.id:
            self.target_cart._px=self.px
            self.target_cart._py=self.py+10

    def move(self,robots,carts,next_state):

        if len(self.path)<=1:
            self.state=next_state
            return

        next_node=self.path[1]

        blocked=self.blocked(robots,carts)

        # 🔥 anti loop detection
        if self.last_pos == next_node:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        if self.stuck_counter > 3:
            # forzar cambio: ignorar bloqueos momentáneamente
            self.path=bfs((self.col,self.row),self.path[-1],set())
            self.stuck_counter = 0
            return

        if next_node in blocked:
            self.path=bfs((self.col,self.row),self.path[-1],blocked)
            return

        self.last_pos = (self.col,self.row)

        self.col,self.row=next_node
        self.path=self.path[1:]

        self.px=self.col*CELL + CELL//2
        self.py=self.row*CELL + CELL//2

    def draw(self,screen):
        pygame.draw.rect(screen,self.color,
            (self.px-10,self.py-10,20,20),border_radius=4)

# ---------------- MAIN ----------------
pygame.init()
screen=pygame.display.set_mode((WIDTH,HEIGHT))
clock=pygame.time.Clock()
font=pygame.font.SysFont("Arial",14)

robots=[Robot(i,BASES[i],ROBOT_COLORS[i]) for i in range(4)]
carts=[Cart(i,*pos) for i,pos in enumerate(CARTS)]

def draw():
    screen.fill(BG)

    # grid
    for c in range(COLS):
        for r in range(ROWS):
            rect=(c*CELL,r*CELL,CELL,CELL)
            pygame.draw.rect(screen,GRID,rect,1)

            if (c,r) not in SHELVES:
                pygame.draw.circle(screen,NODE,
                    (c*CELL+CELL//2,r*CELL+CELL//2),3)

    # shelves
    for c,r in SHELVES:
        pygame.draw.rect(screen,SHELF,
            (c*CELL+5,r*CELL+5,CELL-10,CELL-10))

    # bases
    for i,(c,r) in enumerate(BASES):
        pygame.draw.rect(screen,BASE,
            (c*CELL+5,r*CELL+5,CELL-10,CELL-10),2)

    # carritos suelo
    for cart in carts:
        if not cart.held:
            cart._px=cart.col*CELL + CELL//2
            cart._py=cart.row*CELL + CELL//2

    for cart in carts:
        cart.draw(screen)

    for r in robots:
        r.draw(screen)

    # panel
    px=COLS*CELL
    pygame.draw.rect(screen,(20,20,20),(px,0,PANEL_W,HEIGHT))

    screen.blit(font.render(f"Velocidad: {speed_level} (1-5)",True,TEXT),(px+10,10))

    for i,r in enumerate(robots):
        txt=f"R{i} [{r.col},{r.row}] {r.state}"
        screen.blit(font.render(txt,True,TEXT),(px+10,40+i*30))

    pygame.display.flip()

# loop
running=True
while running:
    clock.tick(FPS)

    for e in pygame.event.get():
        if e.type==pygame.QUIT:
            running=False

        if e.type==pygame.KEYDOWN:
            if e.key in [pygame.K_1,pygame.K_2,pygame.K_3,pygame.K_4,pygame.K_5]:
                speed_level=int(e.unicode)

    for r in robots:
        r.update(robots,carts)

    draw()

pygame.quit()
sys.exit()