import tkinter as tk
import numpy as np
import figure as fg





SizeX, SizeY = 800, 600
scale = SizeX / SizeY

# Настройки окна и холста
window = tk.Tk()
canvas = tk.Canvas(window, width=SizeX, height=SizeY)
canvas.pack()
img = tk.PhotoImage(width=SizeX, height=SizeY)
canvas.create_image((SizeX / 2, SizeY / 2), image=img)



# Гамма-корректировка
def gamma(color):
    gamma = 2.0
    factor = 255.0
    return np.power(color / factor, 1.0 / gamma) * factor


def is_in_shadow(hit_point, scene, current_obj):
    # Луч из точки попадания в сторону источника света
    shadow_ray_origin = hit_point
    shadow_ray_dir = light_dir  # Вектор направления на свет (нормализован)

    # Смещение для предотвращения самозатенения
    epsilon = 0.001

    for obj in scene:
        # пересечение вторичного луча с объектами сцены
        t = obj.hit(shadow_ray_origin, shadow_ray_dir)
        if epsilon < t < float('inf'):
            return True  # Если есть препятствие -> точка в тени
    return False


def calculate_lit_color(obj, hit_point, camera_pos, scene):
    n = obj.get_normal(hit_point)

    n_len = np.linalg.norm(n)
    if n_len > 1e-8:
        n = n / n_len

    # Проверка на нахождение точка в тени
    if is_in_shadow(hit_point, scene, obj):
        int0 = I_amb
        ispec = 0
    else:
        # не в тени -> считаем полное освещение по Фонгу
        # Вектор на камеру V
        view_dir = camera_pos - hit_point
        v_len = np.linalg.norm(view_dir)
        if v_len > 1e-8:
            view_dir /= v_len

        cos_ln = np.dot(n, light_dir)

        if cos_ln < 0:
            int0 = I_amb
            ispec = 0
        else:
            int0 = I_amb + I_diff * cos_ln
            # Отраженный вектор R = 2*(N·L)*N - L
            reflect_dir = 2.0 * cos_ln * n - light_dir
            cos_rv = np.dot(reflect_dir, view_dir)

            ispec = I_diff * (max(0, cos_rv) ** 8)

    # итоговый цвет: R := Int0*near.Kr + near.Ks*Ispec
    # Белый для блика
    final_color = obj.colour * int0 + np.array([255, 255, 255]) * (ispec * obj.ks)

    return np.clip(final_color, 0, 255).astype(np.uint8)


def lighting(obj, hit_point, camera_pos):
    # 1. нормаль в точке попадания
    n = obj.get_normal(hit_point)

    # 2. Вектор направления на камеру (V)
    view_dir = camera_pos - hit_point
    view_dir_len = np.linalg.norm(view_dir)
    if view_dir_len > 1e-8:
        view_dir /= view_dir_len

    # 3. Диффузная составляющая (cosLN)
    cos_ln = np.dot(n, light_dir)

    if cos_ln < 0:
        #  свет падает сзади
        i_total = I_amb
        i_spec = 0
    else:
        # Рассеянный свет (Фон + Диффуз)
        i_total = I_amb + I_diff * cos_ln

        # 4. Зеркальный блик (Specular)
        reflect_dir = 2.0 * cos_ln * n - light_dir

        # Скалярное произведение вектора отражения и вектора на камеру (cosRV)
        cos_rv = np.dot(reflect_dir, view_dir)

        if cos_rv > 0:
            i_spec = I_diff * (cos_rv ** 32)
        else:
            i_spec = 0

    # 5. Итоговый расчет цвета
    # obj.colour — массив [R, G, B], obj.ks — коэффициент зеркальности
    final_colour = obj.colour * i_total + White * (i_spec * obj.ks)

    return np.clip(final_colour, 0, 255).astype(np.uint8)


def render():
    scene_data = []
    for y in range(SizeY):
        # Преобразование экранных координат в локальные
        py = -(y - SizeY / 2) / SizeY
        row = []
        for x in range(SizeX):
            px = (x - SizeX / 2) / SizeX * scale

            # Направление луча из камеры
            ray_dir = np.array([px, py, 1.0])
            ray_dir /= np.linalg.norm(ray_dir)

            current_z = float('inf')
            closest_obj = None

            # Поиск ближайшей фигуры
            for obj in scene:
                z = obj.hit(camera_pos, ray_dir)
                if z < current_z:
                    current_z = z
                    closest_obj = obj

            if closest_obj:
                hit_point = camera_pos + current_z * ray_dir

                # Расчет освещения с учетом теней
                rgb = calculate_lit_color(closest_obj, hit_point, camera_pos, scene)

                # Применяем гамма-коррекцию
                rgb = gamma(rgb)

                pixel_colour = '#{:02x}{:02x}{:02x}'.format(*rgb.astype(int))

            else:
                # Фоновый цвет
                pixel_colour = '#{:02x}{:02x}{:02x}'.format(*background)

            row.append(pixel_colour)
        scene_data.append(row)

    # Блиттинг
    img.put(scene_data)

# https://colorscheme.ru/html-colors.html
Black      = np.array([0, 0, 0])
White      = np.array([255, 255, 255])
Red        = np.array([255, 0, 0])
Green      = np.array([0, 255, 0])
Blue       = np.array([0, 0, 255])
Yellow     = np.array([255, 255, 0])
Silver     = np.array([192, 192, 192])
Violet     = np.array([238, 130, 238])
Coral      = np.array([255, 127, 80])
Gray       = np.array([128, 128, 128])
SlateGrey  = np.array([112, 128, 144])
Brown      = np.array([165, 42, 42])

background = Black


camera_pos = np.array([0, 0, -135])

# Направление света
light_dir = np.array([0, 20, 0])
light_dir = light_dir / np.linalg.norm(light_dir)

# Интенсивности
I_amb = 55 / 255  # Фоновое освещение
I_diff = 200 / 255  # Диффузное освещение


scene = [
    fg.Plane([0, -50, 0], [0, 1, 0], Brown),  # Пол
    fg.Plane([0, 0, 50], [0, 0, -1], Silver),  # Задняя стена
    fg.Plane([-50, 0, 0], [1, 0, 0], SlateGrey),  # Левая стена

    fg.Sphere(np.array([0, 0, 0]), 4, Black),

    fg.Cylinder(np.array([-50, 0, 0]), np.array([50, 0, 0]), 1, Red),
    fg.Cylinder(np.array([0, 50, 0]), np.array([0, -50, 0]), 1, Red),

    fg.Cone(np.array([60, 0, 0]), np.array([50, 0, 0]), 3, Green),
    fg.Cone(np.array([0, 60, 0]), np.array([0, 50, 0]), 3, Green),

    fg.Ellipsoid(np.array([-20, 0, 10]), np.array([20, 0, 10]), 5, Violet),
    fg.Ellipsoid(np.array([20, 0, 10]), np.array([20, 30, 10]), 10, Coral),

    fg.Pipe(np.array([-35, -35, -15]), np.array([-40, -50, -25]), 4, 2, Yellow)

    # fg.Sphere(np.array([0, -25, 5]), 15, Red),
    # fg.Sphere(np.array([-25, 30, 25]), 7, Green),
    # fg.Sphere(np.array([-40, 0, -5]), 9, Blue),
    # fg.Sphere(np.array([0, 0, 0]), 11, B),
    #
    # fg.Cylinder(np.array([-40, 0, 5]), np.array([0, -25, 5]), 5, Violet),
    # fg.Cylinder(np.array([15, 20, -10]), np.array([25, -50, 50]), 10, Coral),
    #
    # fg.Cone(np.array([-50, 10, 20]), np.array([0, -25, 20]), 15, Black),
    # fg.Cone(np.array([10, 10, 20]), np.array([0, -25, 20]), 20, Green),
    #
    # fg.Ellipsoid(np.array([-20, 0, 10]), np.array([20, 0, 10]), 5, Violet),
    # fg.Ellipsoid(np.array([20, 0, 10]), np.array([20, 30, 10]), 10, Coral),
    #
    # fg.Pipe(np.array([-25, 30, 25]), np.array([0, 0, 0]), 4, 3, Silver),
    # fg.Pipe(np.array([10, -40, -20]), np.array([0, 0, 150]), 5, 3, Coral),
]

render()
window.mainloop()
