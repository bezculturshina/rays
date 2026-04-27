import numpy as np


class Shape():
    def hit(self):  # z-буффер (для каждой фигуры переопределён)
        return float('inf')


# сфера
class Sphere(Shape):
    def __init__(self, center, radius, colour, ks=0.5):
        self.center = center
        self.radius = radius
        self.colour = colour
        self.ks = ks  # Коэффициент зеркальности

    def hit(self, ray_origin, ray_dir):
        oc = ray_origin - self.center  # вектор от цента в камеру
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - self.radius ** 2
        discriminant = b ** 2 - 4 * c
        if discriminant > 0:
            t = (-b - np.sqrt(discriminant)) / 2
            if t > 0: return t
        return float('inf')

    def get_normal(self, hit_point):
        # Нормаль сферы: (Точка_попадания - Центр) / Радиус
        # Это дает единичный вектор, направленный наружу от центра
        n = (hit_point - self.center) / self.radius
        return n



# плоскость
class Plane(Shape):
    def __init__(self, point, normal, colour, ks=0.5):
        self.point = np.array(point)
        self.normal = np.array(normal) / np.linalg.norm(normal)
        self.colour = colour
        self.ks = ks  # Коэффициент зеркальности

    def hit(self, ray_origin, ray_dir):
        D = np.dot(ray_dir, self.normal)  # скалярное произведение луча и нормали
        if abs(D) > 1e-6:
            t = np.dot(self.point - ray_origin, self.normal) / D
            if t > 0:
                return t
        return float('inf')

    def get_normal(self, hit_point=None):
        return self.normal


# цилиндр
class Cylinder(Shape):
    def __init__(self, p1, p2, radius, colour, ks=0.5):
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.r = radius
        self.colour = colour
        self.ks = ks  # Коэффициент зеркальности (от 0 до 1)

        # Вектор оси цилиндра и его высота
        v = self.p2 - self.p1
        self.h = np.linalg.norm(v)
        if self.h < 1e-6:  # Защита от нулевого цилиндра
            v = np.array([0, 1, 0])
        else:
            v = v / self.h

        # Строим матрицы (SetTransform)
        self.M1, self.M2 = self._build_matrices(v, self.p1)

        # Служебные поля для хранения состояния пересечения
        self.loc = "None"
        self.save_point = None

    def _build_matrices(self, v, p1):
        # v — это нормализованный вектор от p1 к p2
        # Нам нужно построить матрицу поворота, где ось X совпадает с v

        # 1. Смещение в p1
        T = np.eye(4)
        T[:3, 3] = -p1

        # 2. Строим ортонормированный базис
        ax = v  # Новая ось X

        # Выбираем вспомогательный вектор для векторного произведения
        # Если цилиндр направлен вдоль Y, возьмем X, иначе Y

        if abs(v[1]) > 0.9:
            temp_vec = np.array([1, 0, 0])  # проверка на параллельность света стенкам
        else:
            temp_vec = np.array([0, 1, 0])

        az = np.cross(ax, temp_vec)
        az /= np.linalg.norm(az)

        ay = np.cross(az, ax)
        ay /= np.linalg.norm(ay)

        # Матрица вращения (собираем из новых осей)
        R = np.eye(4)
        R[0, :3] = ax
        R[1, :3] = ay
        R[2, :3] = az

        # M1: сначала переносим в p1, потом вращаем
        M1 = R @ T
        # M2: обратная матрица (для нормалей)
        M2 = np.linalg.inv(M1)

        return M1, M2

    def hit(self, ray_origin, ray_dir):
        # Переносим луч в локальную систему координат цилиндра
        # p_loc = M1 * p_world
        origin_4 = np.append(ray_origin, 1.0)
        local_origin = (self.M1 @ origin_4)[:3]

        # Направление только вращаем (без смещения)
        local_dir = (self.M1[:3, :3] @ ray_dir)

        t_min = float('inf')
        self.loc = "None"

        # ось цилиндра — X, уравнение: y^2 + z^2 = r^2
        a = local_dir[1] ** 2 + local_dir[2] ** 2
        b = 2 * (local_dir[1] * local_origin[1] + local_dir[2] * local_origin[2])
        c = local_origin[1] ** 2 + local_origin[2] ** 2 - self.r ** 2


        # Внутри метода hit для боковой поверхности:
        discriminant = b ** 2 - 4 * a * c
        if discriminant >= 0:
            if abs(a) < 1e-10:
                t1 = t2 = float('inf')
            else:
                sqrt_d = np.sqrt(discriminant)
                t1 = (-b - sqrt_d) / (2 * a)
                t2 = (-b + sqrt_d) / (2 * a)

            # Проверяем оба корня по очереди
            for t in [t1, t2]:
                if t > 0.001:  # Берем первый же положительный корень
                    x_hit = local_origin[0] + t * local_dir[0]
                    if 0 <= x_hit <= self.h:
                        if t < t_min:
                            t_min = t
                            self.loc = "Side"
                            break  # Нашли ближайшую точку на боку — выходим из цикла t1, t2

        # --- 2. Пересечение с крышками (плоскости x=0 и x=h) ---
        if abs(local_dir[0]) > 1e-8:
            # Крышка p2 (x = h)
            t_top = (self.h - local_origin[0]) / local_dir[0]
            if 0 < t_top < t_min:
                y = local_origin[1] + t_top * local_dir[1]
                z = local_origin[2] + t_top * local_dir[2]
                if y ** 2 + z ** 2 <= self.r ** 2:
                    t_min = t_top
                    self.loc = "Top"

            # Крышка p1 (x = 0)
            t_bottom = -local_origin[0] / local_dir[0]
            if 0 < t_bottom < t_min:
                y = local_origin[1] + t_bottom * local_dir[1]
                z = local_origin[2] + t_bottom * local_dir[2]
                if y ** 2 + z ** 2 <= self.r ** 2:
                    t_min = t_bottom
                    self.loc = "Bottom"

        if t_min < float('inf'):
            self.save_point = local_origin + t_min * local_dir
            return t_min
        return float('inf')

    def get_normal(self, point):
        # Вычисляем нормаль в локальной системе координат
        if self.loc == "Side":
            n = np.array([0, self.save_point[1] / self.r, self.save_point[2] / self.r])
        elif self.loc == "Top":
            n = np.array([1, 0, 0])
        elif self.loc == "Bottom":
            n = np.array([-1, 0, 0])
        else:
            n = np.array([0, 1, 0])

        # Поворачиваем нормаль обратно помощью M2
        world_normal = self.M2[:3, :3] @ n
        norm_len = np.linalg.norm(world_normal)
        return world_normal / norm_len if norm_len > 1e-8 else world_normal

# конус
class Cone(Shape):
    def __init__(self, p1, p2, r, colour, ks=0.5):
        self.p1 = np.array(p1)  # Вершина
        self.p2 = np.array(p2)  # Центр основания
        self.r = float(r)
        self.colour = colour
        self.ks = ks  # Коэффициент зеркальности (от 0 до 1)

        # Вычисляем высоту и матрицы трансформации
        # h — расстояние между p1 и p2
        diff = self.p2 - self.p1
        self.h = np.linalg.norm(diff)

        self.M1, self.M2 = self._build_matrices(diff / self.h if self.h > 1e-6 else np.array([0, 1, 0]))

    def _build_matrices(self, v):
        # v нормализованный вектор направления (ось Y)
        ay = v

        # 1. Выбираем вспомогательный вектор.
        # Если v направлен почти вдоль Y, берем X. В противном случае берем Y.
        if abs(v[1]) > 0.9:
            temp = np.array([1.0, 0.0, 0.0])
        else:
            temp = np.array([0.0, 1.0, 0.0])

        # 2. Строим ось Z
        az = np.cross(ay, temp)
        az_len = np.linalg.norm(az)

        if az_len < 1e-9:
            temp = np.array([0.0, 0.0, 1.0])
            az = np.cross(ay, temp)
            az_len = np.linalg.norm(az)

        az = az / az_len

        # 3. Строим ось X
        ax = np.cross(ay, az)
        ax_len = np.linalg.norm(ax)
        if ax_len > 1e-9:
            ax = ax / ax_len

        # Собираем матрицу вращения R
        R = np.eye(4)
        R[0, :3] = ax
        R[1, :3] = ay
        R[2, :3] = az

        # M1: Перенос в p1 и поворот (Мир -> Объект)
        T = np.eye(4)
        T[:3, 3] = -self.p1

        M1 = R @ T
        # M2: Обратная матрица (Объект -> Мир)
        M2 = np.linalg.inv(M1)

        return M1, M2

    def hit(self, ray_p0, ray_dir):
        # 1. Трансформируем луч в локальное пространство конуса
        p0_4 = np.append(ray_p0, 1.0)
        p0_loc = (self.M1 @ p0_4)[:3]
        dir_loc = (self.M1[:3, :3] @ ray_dir)

        # Коэффициент k = (r / h)^2
        h_safe = self.h if self.h > 1e-6 else 1e-6
        k = (self.r / h_safe) ** 2

        a = dir_loc[0] ** 2 + dir_loc[2] ** 2 - k * dir_loc[1] ** 2
        b = 2 * (dir_loc[0] * p0_loc[0] + dir_loc[2] * p0_loc[2] - k * dir_loc[1] * p0_loc[1])
        c = p0_loc[0] ** 2 + p0_loc[2] ** 2 - k * p0_loc[1] ** 2

        t_min = float('inf')
        self.hit_type = "None"

        # Решаем для боковой поверхности (если a не ноль)
        if abs(a) > 1e-10:
            det = b ** 2 - 4 * a * c
            if det >= 0:
                sqrt_det = np.sqrt(det)
                for t in [(-b - sqrt_det) / (2 * a), (-b + sqrt_det) / (2 * a)]:
                    if t > 0.001 and t < t_min:
                        y_hit = p0_loc[1] + t * dir_loc[1]
                        if 0 <= y_hit <= self.h:
                            t_min = t
                            self.hit_type = "Side"
                            self.save_point = p0_loc + t * dir_loc

        # Решаем для основания (крышки) — плоскость y = h
        if abs(dir_loc[1]) > 1e-8:
            t_cap = (self.h - p0_loc[1]) / dir_loc[1]
            if 0.001 < t_cap < t_min:
                p_cap = p0_loc + t_cap * dir_loc
                # Проверка: точка внутри круга радиуса r (x^2 + z^2 <= r^2)
                if p_cap[0] ** 2 + p_cap[2] ** 2 <= self.r ** 2 + 1e-6:
                    t_min = t_cap
                    self.hit_type = "Top"
                    self.save_point = p_cap

        return t_min

    def get_normal(self, hit_point):
        # Вычисляем нормаль в локальных координатах
        if self.hit_type == "Side":
            # Математика нормали конуса
            # y_normal = -r / sqrt(h^2 + r^2)
            # x, z нормали пропорциональны координатам точки
            hypot = np.sqrt(self.h ** 2 + self.r ** 2)
            ny = -self.r / hypot
            nxz_factor = self.h / hypot
            nx = self.save_point[0] / self.r * nxz_factor
            nz = self.save_point[2] / self.r * nxz_factor
            n_loc = np.array([nx, ny, nz])
        elif self.hit_type == "Top":
            n_loc = np.array([0, 1, 0])  # Основание смотрит "вверх" по оси Y
        else:
            n_loc = np.array([0, 0, 0])

        # Поворачиваем нормаль обратно в мировые координаты
        n_world = self.M2[:3, :3] @ n_loc
        return n_world / np.linalg.norm(n_world)

# эллипс
class Ellipsoid(Shape):
    def __init__(self, p1, p2, radius, colour, ks=0.5):
        super().__init__()
        self.p1 = np.array(p1)#.astype(float)
        self.p2 = np.array(p2)#.astype(float)
        self.r = float(radius)
        self.colour = colour
        self.ks = ks  # Коэффициент зеркальности (от 0 до 1)

        diff = self.p2 - self.p1
        self.h = np.linalg.norm(diff)

        # чтобы избежать деления на 0
        v = diff / self.h if self.h > 1e-6 else np.array([0.0, 1.0, 0.0])
        if self.h < 1e-6:
            self.h = 1e-6

        self.M1, self.M2 = self._build_matrices(v)
        self.save_point = None

    def _build_matrices(self, v):
        ay = v

        # Выбираем вспомогательный вектор
        if abs(v[0]) < 0.9:
            temp = np.array([1.0, 0.0, 0.0])
        else:
            temp = np.array([0.0, 1.0, 0.0])

        az = np.cross(ay, temp)
        mag_z = np.linalg.norm(az)

        if mag_z < 1e-9:
            temp = np.array([0.0, 0.0, 1.0])
            az = np.cross(ay, temp)
            mag_z = np.linalg.norm(az)

        az = az / mag_z
        ax = np.cross(ay, az)

        # Матрица трансформации
        R = np.eye(4)
        R[0, :3] = ax
        R[1, :3] = ay
        R[2, :3] = az

        T = np.eye(4)
        T[:3, 3] = -self.p1

        M1 = R @ T
        M2 = np.linalg.inv(M1)
        return M1, M2

    def hit(self, ray_p0, ray_dir):
        # Трансформируем луч
        p0_loc = (self.M1 @ np.append(ray_p0, 1.0))[:3]
        dir_loc = (self.M1[:3, :3] @ ray_dir)

        h2 = self.h ** 2
        r2 = self.r ** 2

        # Уравнение эллипсоида
        a = (dir_loc[0] * self.h) ** 2 + (dir_loc[1] * self.r) ** 2 + (dir_loc[2] * self.h) ** 2
        b = 2 * (dir_loc[0] * p0_loc[0] * h2 + dir_loc[1] * p0_loc[1] * r2 + dir_loc[2] * p0_loc[2] * h2)
        c = (p0_loc[0] * self.h) ** 2 + (p0_loc[1] * self.r) ** 2 + (p0_loc[2] * self.h) ** 2 - (self.r * self.h) ** 2

        det = b ** 2 - 4 * a * c
        if det < 0:
            return float('inf')

        t = (-b - np.sqrt(det)) / (2 * a)

        if t > 0.001:
            self.save_point = p0_loc + t * dir_loc
            return t
        return float('inf')

    def get_normal(self, hit_point):
        # Нормаль эллипсоида требует масштабирования координат
        nx = self.save_point[0] / (self.r ** 2)
        ny = self.save_point[1] / (self.h ** 2)
        nz = self.save_point[2] / (self.r ** 2)

        n_loc = np.array([nx, ny, nz])
        n_world = self.M2[:3, :3] @ n_loc

        norm = np.linalg.norm(n_world)
        return n_world / norm if norm > 1e-8 else n_world

# трубка
class Pipe(Shape):
    def __init__(self, p1, p2, r_out, r_in, colour, ks=0.5):
        super().__init__()
        self.p1 = np.array(p1)#.astype(float)
        self.p2 = np.array(p2)#.astype(float)
        self.r_out = float(r_out)
        self.r_in = float(r_in)
        self.colour = colour
        self.ks = ks  # Коэффициент зеркальности (от 0 до 1)

        diff = self.p2 - self.p1
        self.h = np.linalg.norm(diff)
        v = diff / self.h if self.h > 1e-6 else np.array([1.0, 0.0, 0.0])

        # матрицы для оси X
        self.M1, self.M2 = self._build_matrices(v, self.p1)
        self.loc = "None"
        self.save_point = None

    def _build_matrices(self, v, p1):
        ax = v
        temp_vec = np.array([0, 1, 0]) if abs(v[0]) < 0.9 else np.array([1, 0, 0])
        az = np.cross(ax, temp_vec)
        az = az / np.linalg.norm(az)
        ay = np.cross(az, ax)
        ay = ay / np.linalg.norm(ay)

        T = np.eye(4)
        T[:3, 3] = -p1
        R = np.eye(4)
        R[0, :3], R[1, :3], R[2, :3] = ax, ay, az
        M1 = R @ T
        return M1, np.linalg.inv(M1)

    def hit(self, ray_origin, ray_dir):
        origin_loc = (self.M1 @ np.append(ray_origin, 1.0))[:3]
        dir_loc = (self.M1[:3, :3] @ ray_dir)

        t_min = float('inf')
        self.loc = "None"

        # 1. Стенки (Внешняя и Внутренняя)
        a = dir_loc[1] ** 2 + dir_loc[2] ** 2
        if a > 1e-10:
            b_base = 2 * (dir_loc[1] * origin_loc[1] + dir_loc[2] * origin_loc[2])
            c_base = origin_loc[1] ** 2 + origin_loc[2] ** 2

            for r in [self.r_out, self.r_in]:
                c = c_base - r ** 2
                discriminant = b_base ** 2 - 4 * a * c
                if discriminant >= 0:
                    sqrt_d = np.sqrt(discriminant)
                    for t in [(-b_base - sqrt_d) / (2 * a), (-b_base + sqrt_d) / (2 * a)]:
                        if 0.001 < t < t_min:
                            x_hit = origin_loc[0] + t * dir_loc[0]
                            if 0 <= x_hit <= self.h:
                                t_min = t
                                self.loc = "Side" if r == self.r_out else "Inside"

        # 2. Торцы (Кольца на концах)
        if abs(dir_loc[0]) > 1e-8:
            for x_plane, label in [(self.h, "Top"), (0, "Bottom")]:
                t = (x_plane - origin_loc[0]) / dir_loc[0]
                if 0.001 < t < t_min:
                    y = origin_loc[1] + t * dir_loc[1]
                    z = origin_loc[2] + t * dir_loc[2]
                    dist_sq = y ** 2 + z ** 2
                    # Проверка попадания в кольцо между r_in и r_out
                    if self.r_in ** 2 <= dist_sq <= self.r_out ** 2:
                        t_min = t
                        self.loc = label

        if t_min < float('inf'):
            self.save_point = origin_loc + t_min * dir_loc
            return t_min
        return float('inf')

    def get_normal(self, point):
        if self.loc == "Side":
            n = np.array([0, self.save_point[1] / self.r_out, self.save_point[2] / self.r_out])
        elif self.loc == "Inside":
            # Нормаль внутренней стенки смотрит К ЦЕНТРУ оси
            n = np.array([0, -self.save_point[1] / self.r_in, -self.save_point[2] / self.r_in])
        elif self.loc == "Top":
            n = np.array([1, 0, 0])
        elif self.loc == "Bottom":
            n = np.array([-1, 0, 0])
        else:
            n = np.array([0, 1, 0])

        world_normal = self.M2[:3, :3] @ n
        return world_normal / np.linalg.norm(world_normal)