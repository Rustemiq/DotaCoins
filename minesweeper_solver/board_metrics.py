import cv2
import numpy as np
import os


def get_board_metrics(image_path: str, padding: int = 6) -> tuple[int, int, int, int, float, float]:
    """
    Анализирует скриншот поля Сапера.

    :param image_path: Путь к исходному изображению.
    :param padding: Отступ от краев (чтобы срезать зеленую рамку).
    :return: (x, y, w, h, size_w, size_h) —
             координаты левого верхнего угла, ширина/высота поля,
             ширина/высота одной клетки.
    """
    # 1. Загрузка и проверка
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

    # 2. Поиск внешних границ поля
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 50, 20), (180, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_boxes = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        aspect_ratio = bw / bh
        if area > 50000 and 0.6 < aspect_ratio < 1.4:
            valid_boxes.append((bx, by, bw, bh, area))

    if not valid_boxes:
        raise ValueError("На изображении не найдено игровое поле.")

    # Берем самый большой подходящий контур
    valid_boxes.sort(key=lambda b: b[4], reverse=True)
    x, y, w, h, _ = valid_boxes[0]

    # Обрезаем зеленую рамку (padding) и сдвигаем координаты
    x += padding
    y += padding
    w -= padding * 2
    h -= padding * 2

    # Вырезаем область поля для анализа
    board_img = img[y:y + h, x:x + w]

    # 3. Анализ сетки через FFT
    gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray  # Инвертируем, чтобы линии сетки стали светлыми

    col_prof = np.sum(inv, axis=0).astype(np.float32)
    row_prof = np.sum(inv, axis=1).astype(np.float32)

    col_prof -= col_prof.mean()
    row_prof -= row_prof.mean()

    fft_x = np.abs(np.fft.rfft(col_prof))
    fft_y = np.abs(np.fft.rfft(row_prof))

    # Ограничиваем поиск (от 5 до 30 клеток)
    fft_x[:5] = 0
    fft_x[30:] = 0
    fft_y[:5] = 0
    fft_y[30:] = 0

    cols = int(np.argmax(fft_x))
    rows = int(np.argmax(fft_y))

    if cols == 0 or rows == 0:
        raise ValueError("Не удалось определить размер сетки.")

    # 4. Вычисляем размеры одной клетки
    size_w = w / cols
    size_h = h / rows

    # 5. Отрисовка отладочной сетки (для проверки)
    debug_img = board_img.copy()
    for r in range(rows):
        for c in range(cols):
            x1 = int(c * size_w)
            y1 = int(r * size_h)
            x2 = int((c + 1) * size_w)
            y2 = int((r + 1) * size_h)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imwrite("debug_grid.png", debug_img)

    # Возвращаем x, y (глобальные координаты на скрине), w, h (размер поля) и размеры клетки
    return x, y, w, h, size_w, size_h


if __name__ == "__main__":
    image_file = "pictures/empty1.png"

    if not os.path.exists(image_file):
        print(f"Файл {image_file} не найден.")
    else:
        try:
            # Распаковываем 6 значений
            board_x, board_y, board_w, board_h, cell_w, cell_h = get_board_metrics(image_file)

            print(f"\nРезультаты анализа:")
            print(f"Левый верхний угол: ({board_x}, {board_y})")
            print(f"Размер поля: {board_w}x{board_h} px")
            print(f"Размер клетки: {cell_w:.2f}x{cell_h:.2f} px")

            # Пример: как найти центр клетки на 3-й колонке, 5-м ряду (индексы 2 и 4)
            target_col, target_row = 2, 4
            center_x = int(board_x + target_col * cell_w + cell_w / 2)
            center_y = int(board_y + target_row * cell_h + cell_h / 2)

            print(f"\nКоординаты центра клетки (col={target_col}, row={target_row}): ({center_x}, {center_y})")

        except Exception as e:
            print(f"Ошибка: {e}")