import cv2
import numpy as np
import os
import random

from minesweeper_solver.board_metrics import get_board_metrics


class MinesweeperScanner:
    def __init__(self, templates_dir="templates"):
        self.templates = {}
        # Добавил "6" в список
        for digit in ["1", "2", "3", "4", "5", "6"]:
            path = os.path.join(templates_dir, f"{digit}.png")
            if not os.path.exists(path):
                continue

            img = cv2.imread(path)
            # ИСПРАВЛЕНИЕ: Передаем диапазоны красного цвета для шаблонов
            mask_red1 = self._get_content_mask(img, (0, 100, 100), (8, 255, 255))
            mask_red2 = self._get_content_mask(img, (170, 100, 100), (180, 255, 255))
            mask = cv2.bitwise_or(mask_red1, mask_red2)

            digit_img = self._extract_and_resize_digit(mask)
            if digit_img is not None:
                self.templates[digit] = digit_img

        if not os.path.exists("unknown_digits"):
            os.makedirs("unknown_digits")

    def _get_content_mask(self, cell, lower, upper):
        """Возвращает маску для конкретного цветового диапазона"""
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, lower, upper)

    def _extract_and_resize_digit(self, mask, size=32):
        coords = cv2.findNonZero(mask)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        crop = mask[y:y + h, x:x + w]

        pad = max(w, h)
        square = np.zeros((pad, pad), dtype=np.uint8)
        dx = (pad - w) // 2
        dy = (pad - h) // 2
        square[dy:dy + h, dx:dx + w] = crop
        return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)

    def classify_cell(self, cell):
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)

        # 1. Фон (по углам)
        corners_h = np.concatenate([
            hsv[:3, :3, 0].flatten(), hsv[:3, -3:, 0].flatten(),
            hsv[-3:, :3, 0].flatten(), hsv[-3:, -3:, 0].flatten()
        ])
        if 35 <= np.median(corners_h) <= 85:
            return "#"

        # 2. Считаем маски для ВСЕХ цифр
        mask_1 = self._get_content_mask(cell, (65, 50, 50), (120, 255, 255))
        mask_2 = self._get_content_mask(cell, (20, 100, 100), (45, 255, 255))
        mask_3 = self._get_content_mask(cell, (8, 150, 100), (20, 255, 255))
        mask_6 = self._get_content_mask(cell, (120, 50, 50), (165, 255, 255))

        mask_red1 = self._get_content_mask(cell, (0, 100, 100), (10, 255, 255))
        mask_red2 = self._get_content_mask(cell, (165, 100, 100), (180, 255, 255))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # 3. Считаем количество пикселей для каждой маски
        counts = {
            "1": cv2.countNonZero(mask_1),
            "2": cv2.countNonZero(mask_2),
            "3": cv2.countNonZero(mask_3),
            "6": cv2.countNonZero(mask_6),
            "red": cv2.countNonZero(mask_red)
        }

        # 4. Находим цвет, которого БОЛЬШЕ ВСЕГО
        best_color = max(counts, key=counts.get)
        max_pixels = counts[best_color]

        # Если пикселей слишком мало — это пустая клетка
        if max_pixels < 8:
            return "."

        # 5. Если победил красный, значит это 4 или 5 (сравниваем форму)
        if best_color == "red":
            digit_crop = self._extract_and_resize_digit(mask_red)
            if digit_crop is None: return "?"

            score_4 = 0
            score_5 = 0
            if "4" in self.templates:
                score_4 = cv2.matchTemplate(digit_crop, self.templates["4"], cv2.TM_CCOEFF_NORMED).max()
            if "5" in self.templates:
                score_5 = cv2.matchTemplate(digit_crop, self.templates["5"], cv2.TM_CCOEFF_NORMED).max()

            return "4" if score_4 > score_5 else "5"

        # Если победил один из цветов 1, 2, 3, 6 — возвращаем его
        return best_color

    def scan_board(self, image, board_x, board_y, board_w, board_h, cell_w, cell_h):
        """
        Сканирует всё поле и возвращает двумерный массив (матрицу) символов.
        Используется солвером в фоновом режиме (без вывода окон).
        """
        cols = int(round(board_w / cell_w))
        rows = int(round(board_h / cell_h))

        matrix = []

        for r in range(rows):
            row_data = []
            for c in range(cols):
                x1 = int(board_x + c * cell_w)
                y1 = int(board_y + r * cell_h)
                x2 = int(board_x + (c + 1) * cell_w)
                y2 = int(board_y + (r + 1) * cell_h)

                dx = int(cell_w * 0.15)
                dy = int(cell_h * 0.15)
                clean_cell = image[y1 + dy: y2 - dy, x1 + dx: x2 - dx]

                if clean_cell.size == 0:
                    row_data.append("?")
                    continue

                char = self.classify_cell(clean_cell)
                row_data.append(char)

            matrix.append(row_data)

        return matrix

    def debug_full_board(self, image, board_x, board_y, board_w, board_h, cell_w, cell_h):
        """
        Метод для отладки. Рисует поверх картинки, что распознал, и открывает окно.
        """
        cols = int(round(board_w / cell_w))
        rows = int(round(board_h / cell_h))

        debug_img = image.copy()
        matrix = []

        for r in range(rows):
            row_data = []
            for c in range(cols):
                x1 = int(board_x + c * cell_w)
                y1 = int(board_y + r * cell_h)
                x2 = int(board_x + (c + 1) * cell_w)
                y2 = int(board_y + (r + 1) * cell_h)

                dx = int(cell_w * 0.15)
                dy = int(cell_h * 0.15)
                clean_cell = image[y1 + dy: y2 - dy, x1 + dx: x2 - dx]

                if clean_cell.size == 0:
                    row_data.append("?")
                    continue

                result_char = self.classify_cell(clean_cell)
                row_data.append(result_char)

                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                if result_char != ".":
                    color = (0, 0, 255) if result_char != "?" else (0, 255, 255)
                    cv2.putText(debug_img, result_char, (x1 + 4, y1 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            matrix.append(row_data)

        print("\nМатрица символов:")
        for row in matrix:
            print(" ".join(f"{ch:>2}" for ch in row))

        cv2.namedWindow("Debug Full Board", cv2.WINDOW_NORMAL)
        cv2.imshow("Debug Full Board", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return matrix


if __name__ == "__main__":
    IMAGE_PATH = "pictures/qwer8.png"
    TEMPLATES_DIR = "templates"

    # Координаты (пример)
    BOARD_X, BOARD_Y = 675, 205
    BOARD_W, BOARD_H = 570, 742
    CELL_W, CELL_H = 57.0, 57.08
    BOARD_X, BOARD_Y, BOARD_W, BOARD_H, CELL_W, CELL_H = get_board_metrics("pictures/empty2.png")

    scanner = MinesweeperScanner(TEMPLATES_DIR)
    img = cv2.imread(IMAGE_PATH)

    if img is not None:
        # Запускаем отладку всего поля
        matrix = scanner.debug_full_board(img, BOARD_X, BOARD_Y, BOARD_W, BOARD_H, CELL_W, CELL_H)
    else:
        print(f"Картинка {IMAGE_PATH} не найдена.")