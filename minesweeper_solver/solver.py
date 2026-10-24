import keyboard
from PIL import Image

from minesweeper_solver.board_metrics import get_board_metrics
from minesweeper_solver.calculator import Board
from minesweeper_solver.scanner import MinesweeperScanner

import cv2
import numpy as np
import time
import pyautogui
import mss
import os
if not os.path.exists("debug_crash"):
    os.makedirs("debug_crash")


# Предполагается, что эти классы уже импортированы или написаны выше
# from scanner import MinesweeperScanner
# from board import Board

class MinesweeperSolver:
    def __init__(self):
        self.scanner = MinesweeperScanner("templates")
        self.sct = mss.MSS()

        self.is_solving = False
        self.stop_requested = False  # <-- Флаг для остановки

        self.bx = self.by = self.bw = self.bh = 0
        self.cw = self.ch = 0.0
        self.cols = self.rows = 0

    def stop_solving(self):
        """Метод для вызова по горячей клавише остановки"""
        if self.is_solving:
            print("\n[!] ПОЛУЧЕН СИГНАЛ ОСТАНОВКИ. Прерываем решение...")
            self.stop_requested = True
        else:
            print("\n[!] Солвер сейчас не работает (или уже остановлен).")

    def get_screenshot(self):
        monitor = self.sct.monitors[1]
        img = np.array(self.sct.grab(monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def auto_detect_board(self, padding=6):
        img = self.get_screenshot()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 50, 20), (180, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if (w * h) > 50000 and 0.6 < (w / h) < 1.4:
                valid_boxes.append((x, y, w, h, w * h))

        if not valid_boxes:
            return False

        valid_boxes.sort(key=lambda b: b[4], reverse=True)
        x, y, w, h, _ = valid_boxes[0]

        self.bx, self.by = x + padding, y + padding
        self.bw, self.bh = w - padding * 2, h - padding * 2

        board_img = img[self.by:self.by + self.bh, self.bx:self.bx + self.bw]
        gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray

        col_prof = np.sum(inv, axis=0).astype(np.float32)
        row_prof = np.sum(inv, axis=1).astype(np.float32)
        col_prof -= col_prof.mean()
        row_prof -= row_prof.mean()

        fft_x = np.abs(np.fft.rfft(col_prof))
        fft_y = np.abs(np.fft.rfft(row_prof))
        fft_x[:5] = 0;
        fft_x[30:] = 0
        fft_y[:5] = 0;
        fft_y[30:] = 0

        self.cols = int(np.argmax(fft_x))
        self.rows = int(np.argmax(fft_y))

        if self.cols == 0 or self.rows == 0:
            return False

        self.cw = self.bw / self.cols
        self.ch = self.bh / self.rows
        return True

    def click_cell(self, c, r):
        import random  # Импортируем внутри, если нет в начале файла

        x = int(self.bx + c * self.cw + self.cw / 2)
        y = int(self.by + r * self.ch + self.ch / 2)
        pyautogui.click(x, y)

        # Случайная задержка от 0.08 до 0.15 секунды.
        # Это выглядит более "по-человечески" и дает игре время обработать клик.
        time.sleep(random.uniform(0.08, 0.15))

    def is_cell_open(self, img, c, r):
        x1 = int(self.bx + c * self.cw)
        y1 = int(self.by + r * self.ch)
        x2 = int(self.bx + (c + 1) * self.cw)
        y2 = int(self.by + (r + 1) * self.ch)
        dx, dy = int(self.cw * 0.15), int(self.ch * 0.15)
        cell = img[y1 + dy: y2 - dy, x1 + dx: x2 - dx]
        if cell.size == 0: return False
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        corners_h = np.concatenate([
            hsv[:3, :3, 0].flatten(), hsv[:3, -3:, 0].flatten(),
            hsv[-3:, :3, 0].flatten(), hsv[-3:, -3:, 0].flatten()
        ])
        bg_h = np.median(corners_h)
        return not (35 <= bg_h <= 85)  # Если не зеленый - значит открыт

    def park_mouse(self):
        """Уводит курсор мыши в безопасный угол, чтобы он не перекрывал клетки и не вызывал подсветку"""
        # Берем размеры экрана
        screen_width, screen_height = pyautogui.size()
        # Уводим курсор в правый верхний угол (отступаем 50 пикселей от края, чтобы не нажать случайно крестик)
        safe_x = screen_width - 200
        safe_y = 200
        pyautogui.moveTo(safe_x, safe_y, duration=0.1)

    def wait_for_board_update(self, timeout=3.0):
        start_time = time.time()
        prev_img = self.get_screenshot()

        while time.time() - start_time < timeout:
            if self.stop_requested:  # <-- Проверка на остановку во время ожидания
                return False

            time.sleep(0.2)
            curr_img = self.get_screenshot()
            diff = cv2.absdiff(prev_img, curr_img)
            if np.sum(diff) < 100000:
                return True
            prev_img = curr_img
        return False

    def is_game_won(self, img):
        h, w = img.shape[:2]
        center_crop = img[h // 2 - 100: h // 2 + 100, w // 2 - 200: w // 2 + 200]
        hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (40, 150, 150), (80, 255, 255))
        return cv2.countNonZero(mask) > 1000

    def is_board_obstructed(self, img):
        """Проверяет, не перекрыто ли поле черным уведомлением"""
        board_img = img[self.by:self.by + self.bh, self.bx:self.bx + self.bw]
        hsv = cv2.cvtColor(board_img, cv2.COLOR_BGR2HSV)

        # Уведомление — это почти чистый черный цвет.
        # Ищем пиксели с яркостью (V) ниже 30.
        mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 30))

        # Считаем площадь черных пикселей
        black_area = cv2.countNonZero(mask)
        total_area = board_img.shape[0] * board_img.shape[1]

        # Если черного цвета больше 3% от всего поля, значит это уведомление
        print(black_area / total_area)
        if (black_area / total_area) > 0.013:
            return True
        return False

    def find_notification(self, img):
        """Ищет черное окно уведомления в верхней части экрана"""
        # Задаем зону поиска: от верха экрана (или на 150px выше поля) до 4-го ряда поля
        y1 = max(0, self.by - 150)
        y2 = self.by + int(self.ch * 4)
        x1 = self.bx
        x2 = self.bx + self.bw

        # Вырезаем зону для анализа
        zone = img[y1:y2, x1:x2]

        if zone.size == 0:
            return None

        # Ищем чисто черные пиксели (фон уведомления)
        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Уведомление — это широкий горизонтальный прямоугольник
            if w > 150 and h > 30 and (w / h) > 1.5:
                # Координаты уведомления на всем экране
                notif_x1 = x1 + x
                notif_y1 = y1 + y
                notif_x2 = notif_x1 + w
                notif_y2 = notif_y1 + h

                # Вырезаем картинку уведомления
                notif_img = img[notif_y1:notif_y2, notif_x1:notif_x2]
                return notif_img

        return None

    def check_notification(self, img):
        """
        Ищет уведомление строго НА ПОЛЕ (в верхних рядах).
        """
        # 1. Зона поиска — только верхняя часть поля (первые 5 рядов)
        x1 = self.bx
        x2 = self.bx + self.bw
        y1 = self.by
        y2 = self.by + int(self.ch * 1)  # Первые 1 рядов клеток

        roi = img[int(y1):int(y2), int(x1):int(x2)]

        if roi.size == 0:
            return False

        # 2. Ищем черный цвет
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Уведомление почти чисто черное. Обычный фон клеток (коричневый/зеленый) имеет яркость выше 50.
        _, mask = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY_INV)

        # 3. Считаем процент черного
        black_pixels = cv2.countNonZero(mask)
        total_pixels = roi.shape[0] * roi.shape[1]
        black_percentage = black_pixels / total_pixels
        cv2.imwrite("debug_notification.png", roi)


        # 4. Если черного больше 5%, это точно уведомление
        if black_percentage > 0.05:
            print(f"Уведомление НА ПОЛЕ! Черного цвета: {black_percentage * 100:.1f}%")

            # Сохраняем картинку, чтобы ты мог проверить, что именно поймалось
            cv2.imwrite("debug_notification.png", roi)
            return True


        return False

    def solve(self):
        if self.is_solving:
            print("\n[!] Солвер уже работает! Подожди...")
            return

        self.is_solving = True
        self.stop_requested = False

        try:
            print("\n=== ЗАПУСК РЕШЕНИЯ ===")

            if not self.auto_detect_board():
                print("[X] Ошибка: Поле не найдено.")
                return

            print(f"Поле найдено: {self.cols}x{self.rows}")
            print("Клик в центр для старта...")

            center_c, center_r = self.cols // 2, self.rows // 2
            self.click_cell(center_c, center_r)
            self.park_mouse()
            self.wait_for_board_update(timeout=5.0)

            board = Board(self.cols, self.rows)
            turn = 1

            while True:
                if self.stop_requested:
                    print("\n[STOP] Решение прервано.")
                    break

                img = self.get_screenshot()

                if self.check_notification(img):
                    print("Ждем исчезновения уведомления...")
                    time.sleep(1.5)
                    continue

                matrix = self.scanner.scan_board(img, self.bx, self.by, self.bw, self.bh, self.cw, self.ch)

                for r in range(self.rows):
                    for c in range(self.cols):
                        val = matrix[r][c]
                        if val.isdigit():
                            board.fill(c, r, int(val))
                        elif val == ".":
                            board.fill(c, r, ".")
                        else:
                            if (c, r) not in board.known_bombs:
                                board.fill(c, r, "#")

                old_bombs_count = len(board.known_bombs)
                moves = board.make_move()

                if not moves:
                    if len(board.known_bombs) > old_bombs_count:
                        print("Логика нашла новые мины, обновляем доску...")
                        continue
                    else:
                        # ПРОВЕРКА НА РЕАЛЬНУЮ ПОБЕДУ
                        # Если ходов нет, проверяем, остались ли на поле неоткрытые безопасные клетки
                        closed_safe_cells = 0
                        for r in range(self.rows):
                            for c in range(self.cols):
                                if matrix[r][c] == "#" and (c, r) not in board.known_bombs:
                                    closed_safe_cells += 1

                        if closed_safe_cells == 0:
                            print("\n[ПОБЕДА] Все безопасные клетки открыты! Игра пройдена.")
                        else:
                            print(f"\n[ТУПИК] Ходов нет. Осталось закрытых безопасных клеток: {closed_safe_cells}.")
                            cv2.imwrite("debug_stuck_board.png", img)
                            with open("debug_stuck_matrix.txt", "w", encoding="utf-8") as f:
                                for row in matrix:
                                    f.write(" ".join(f"{v:>2}" for v in row) + "\n")
                        break

                print(f"Ход {turn}: открываем {len(moves)} клеток...")

                # --- СИСТЕМА УПОРНЫХ КЛИКОВ ---
                pending_moves = list(moves)
                max_attempts = 3

                for attempt in range(max_attempts):
                    if not pending_moves or self.stop_requested:
                        break

                    # Кликаем по всем, что еще не открылись
                    for (c, r) in pending_moves:
                        if self.stop_requested:
                            break
                        self.click_cell(c, r)

                    self.park_mouse()
                    self.wait_for_board_update(timeout=2.5)

                    # Проверяем результат
                    check_img = self.get_screenshot()
                    next_pending = []
                    for (c, r) in pending_moves:
                        if not self.is_cell_open(check_img, c, r):
                            next_pending.append((c, r))

                    if next_pending:
                        print(f"[!] Не открылось клеток: {len(next_pending)}. Повторяем попытку {attempt + 2}...")

                    pending_moves = next_pending

                if pending_moves:
                    print(f"[X] Внимание! Некоторые клетки так и не открылись после 3 попыток.")

                turn += 1

        except Exception as e:
            print(f"\n[X] Произошла ошибка: {e}")
        finally:
            self.is_solving = False
            self.stop_requested = False
            print("=== РАБОТА ЗАВЕРШЕНА ===")


# ==========================================
# ИНИЦИАЛИЗАЦИЯ И ОЖИДАНИЕ КЛАВИШ
# ==========================================
if __name__ == "__main__":
    solver = MinesweeperSolver()

    # Включаем аварийную остановку: если резко завести мышь в левый верхний угол экрана, программа упадет.
    pyautogui.FAILSAFE = True

    print("==============================================")
    print("  Авто-Сапер запущен в фоновом режиме!")
    print("  [Ctrl+Shift+3] - Начать/перезапустить решение")
    print("  [F8] ИЛИ [Ctrl+Shift+4] - Экстренная остановка")
    print("  [Мышь в левый верхний угол] - Аварийный выход")
    print("  [Ctrl+C] в консоли - Выход из программы")
    print("==============================================\n")


    def run_solver():
        solver.solve()


    # Регистрируем обе горячие клавиши на всякий случай
    keyboard.add_hotkey('ctrl+shift+3', run_solver)
    keyboard.add_hotkey('ctrl+shift+4', solver.stop_solving)
    keyboard.add_hotkey('f8', solver.stop_solving)

    # Бесконечный цикл ожидания
    keyboard.wait()