from itertools import combinations

OFFSETS = [(-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1)]


class Group:
    def __init__(self, mines, cells=None):
        self.mines = mines
        self.cells = set(cells) if cells else set()

    def __len__(self):
        return len(self.cells)

    def __bool__(self):
        return bool(self.cells)

    def __iter__(self):
        return iter(self.cells)

    def __repr__(self):
        return f"Group(mines={self.mines}, cells={sorted(self.cells)})"

    def __eq__(self, other):
        return (isinstance(other, Group)
                and self.mines == other.mines
                and self.cells == other.cells)

    def __hash__(self):
        return hash((self.mines, frozenset(self.cells)))

    def __contains__(self, cell):
        return cell in self.cells

    def __le__(self, other):
        return self.cells <= other.cells

    def __lt__(self, other):
        return self.cells < other.cells

    def __ge__(self, other):
        return self.cells >= other.cells

    def __gt__(self, other):
        return self.cells > other.cells

    def __and__(self, other):
        return self.cells & other.cells

    def __sub__(self, other):
        if not (other.cells <= self.cells):
            raise ValueError(f"Нельзя вычесть {other} из {self}: не подмножество")
        return Group(self.mines - other.mines, self.cells - other.cells)


class Board:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.board = [[0] * w for _ in range(h)]
        self.groups = []
        self.known_bombs = set()

    def fill(self, x, y, value):
        self.board[y][x] = value

    def make_groups(self):
        self.groups.clear()
        for y in range(self.h):
            for x in range(self.w):
                if type(self.board[y][x]) is int:
                    group = Group(self.board[y][x], [])
                    for offset in OFFSETS:
                        x1 = x + offset[0]
                        y1 = y + offset[1]
                        if self.w > x1 >= 0 and self.h > y1 >= 0:
                            if self.board[y1][x1] == "X":
                                group.mines -= 1
                            elif self.board[y1][x1] == "#":
                                group.cells.add((x1, y1))
                    # Если группа не пустая и её ещё нет в списке
                    if group.cells and group not in self.groups:
                        self.groups.append(group)

    def make_move(self):
        for (x, y) in self.known_bombs:
            self.fill(x, y, "X")
        self.make_groups()
        move = []
        while True:
            self.make_groups()
            while True:
                to_remove = set()
                to_add = set()
                for g1, g2 in combinations(self.groups, 2):
                    if g1 == g2:
                        to_remove.add(g2)
                        continue
                    common = g1 & g2
                    if not common:
                        continue
                    if g1 < g2:
                        to_remove.add(g2)
                        new_g = g2 - g1
                        if new_g.cells: to_add.add(new_g)
                    elif g2 < g1:
                        to_remove.add(g1)
                        new_g = g1 - g2
                        if new_g.cells: to_add.add(new_g)
                    else:
                        G1, G2 = (g1, g2) if g1.mines > g2.mines else (g2, g1)

                        # ПРАВИЛЬНАЯ МАТЕМАТИКА:
                        # Вычитаем из большей группы её же остаток вне пересечения.
                        min_mines = G1.mines - (len(G1) - len(common))

                        if min_mines == G2.mines:
                            new_g = Group(min_mines, common)
                            to_remove.add(g1)
                            to_remove.add(g2)
                            r1 = g1 - new_g
                            r2 = g2 - new_g
                            if r1.cells: to_add.add(r1)
                            if r2.cells: to_add.add(r2)
                            to_add.add(new_g)

                if not to_remove and not to_add:
                    break
                self.groups = [g for g in self.groups if g not in to_remove]
                self.groups.extend(to_add)

            step = set()
            flags_added = False
            for g in self.groups:
                if g.mines == 0:
                    for (x, y) in g.cells:
                        if self.board[y][x] == "#":
                            step.add((x, y))
                elif g.mines == len(g):
                    for (x, y) in g.cells:
                        if self.board[y][x] == "#":
                            self.board[y][x] = "X"
                            self.known_bombs.add((x, y))
                            flags_added = True

            if not step and not flags_added:
                break

            for x, y in step:
                if self.board[y][x] == "#":
                    self.board[y][x] = "."
                    move.append((x, y))
        return move