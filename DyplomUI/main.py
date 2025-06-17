import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.special import gamma
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import io
import os


# ===== МОДЕЛЬ U-NET =====
class DownConv(nn.Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(DownConv, self).__init__()
        self.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_feat, momentum=bn_momentum)
        self.conv1_drop = nn.Dropout2d(drop_rate)

        self.conv2 = nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_feat, momentum=bn_momentum)
        self.conv2_drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)

        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(UpConv, self).__init__()
        self.up1 = nn.functional.interpolate
        self.downconv = DownConv(in_feat, out_feat, drop_rate, bn_momentum)

    def forward(self, x, y):
        x = self.up1(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, y], dim=1)
        x = self.downconv(x)
        return x


class Unet(nn.Module):
    def __init__(self, drop_rate=0.4, bn_momentum=0.1):
        super(Unet, self).__init__()

        # Downsampling path
        self.conv1 = DownConv(1, 64, drop_rate, bn_momentum)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = DownConv(64, 128, drop_rate, bn_momentum)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = DownConv(128, 256, drop_rate, bn_momentum)
        self.mp3 = nn.MaxPool2d(2)

        # Bottom
        self.conv4 = DownConv(256, 256, drop_rate, bn_momentum)

        # Upsampling path
        self.up1 = UpConv(512, 256, drop_rate, bn_momentum)
        self.up2 = UpConv(384, 128, drop_rate, bn_momentum)
        self.up3 = UpConv(192, 64, drop_rate, bn_momentum)

        self.conv9 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.mp1(x1)

        x3 = self.conv2(x2)
        x4 = self.mp2(x3)

        x5 = self.conv3(x4)
        x6 = self.mp3(x5)

        # Bottom
        x7 = self.conv4(x6)

        # Up-sampling
        x8 = self.up1(x7, x5)
        x9 = self.up2(x8, x3)
        x10 = self.up3(x9, x1)

        x11 = self.conv9(x10)
        preds = torch.sigmoid(x11)

        return preds


# ===== ПОКРАЩЕНИЙ ОПЕРАТОР АТАНГАНА-БЕЛЕНАУ =====
class AtanganaBaleanuOperator:
    def __init__(self, alpha=0.8, window_size=7):
        self.alpha = alpha
        self.window_size = window_size
        self.kernel = self._create_enhanced_mittag_leffler_kernel()

    def _mittag_leffler_function(self, z, alpha, beta=1.0):
        """Покращена апроксимація функції Міттаг-Леффлера"""
        terms = 15  # Зменшено для стабільності
        result = 0
        for k in range(terms):
            try:
                # Додаємо експоненціальне згасання для стабільності
                term = (z ** k) / gamma(alpha * k + beta)
                if abs(term) < 1e-10:  # Зупиняємося при дуже малих значеннях
                    break
                result += term * np.exp(-k * 0.1)  # Експоненціальне згасання
            except (OverflowError, ZeroDivisionError):
                break
        return result

    def _create_enhanced_mittag_leffler_kernel(self):
        """Створення покращеного ядра з кращою видимістю результатів"""
        center = self.window_size // 2
        y, x = np.ogrid[-center:center + 1, -center:center + 1]
        dist = np.sqrt(x * x + y * y)

        if np.max(dist) == 0:
            normalized_dist = dist
        else:
            normalized_dist = dist / np.max(dist)

        kernel = np.zeros((self.window_size, self.window_size))

        # Створюємо більш контрастне ядро
        for i in range(self.window_size):
            for j in range(self.window_size):
                z = normalized_dist[i, j]
                # Масштабуємо входне значення для кращого ефекту
                scaled_z = -z * 2.0
                ml_value = self._mittag_leffler_function(scaled_z, self.alpha)
                kernel[i, j] = ml_value

        # Нормалізуємо ядро
        kernel_sum = np.sum(np.abs(kernel))
        if kernel_sum > 0:
            kernel = kernel / kernel_sum

        # Додаємо центральний негативний вплив для контрасту
        kernel[center, center] -= 0.5

        return kernel

    def enhance_image(self, image):
        """Покращене застосування оператора з більш помітними результатами"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Нормалізуємо до [0, 1]
        if image.max() > 1:
            image = image.astype(np.float64) / 255.0

        # Застосовуємо згортку
        filtered = cv2.filter2D(image, cv2.CV_64F, self.kernel)

        # Обчислюємо дробову похідну з посиленням
        fractional_derivative = image - filtered

        # Посилюємо ефект залежно від альфа
        enhancement_factor = self.alpha * 2.0  # Подвоюємо для кращої видимості
        enhanced = image + enhancement_factor * fractional_derivative

        # Застосовуємо адаптивне контрастування
        enhanced = self._adaptive_contrast_enhancement(enhanced)

        # Обмежуємо значення
        enhanced = np.clip(enhanced, 0, 1)

        return enhanced

    def _adaptive_contrast_enhancement(self, image):
        """Адаптивне покращення контрасту"""
        # Обчислюємо локальну середню та стандартне відхилення
        kernel_size = max(3, self.window_size // 3)
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

        local_mean = cv2.filter2D(image, cv2.CV_64F, kernel)
        local_var = cv2.filter2D(image ** 2, cv2.CV_64F, kernel) - local_mean ** 2
        local_std = np.sqrt(np.maximum(local_var, 0))

        # Адаптивне покращення
        enhanced = local_mean + self.alpha * (image - local_mean) * (1 + local_std)

        return enhanced


# ===== ПОКРАЩЕНІ ДРОБОВІ ПОХІДНІ РІССА =====
def fractional_differential_mask(img, nu=0.7, m=3):
    """Покращена реалізація дробових похідних Рісса"""
    n = 2 * m
    mask_size = (2 * m + 1, 2 * m + 1)
    h = 1

    k = np.arange(0, n + 1)
    w = np.empty(n + 1)

    try:
        w[0] = - gamma(1 - nu / 2) / (nu * gamma(1 + nu / 2) * gamma(-nu))
        w[1:] = ((-1) ** (k[1:] + 1) * gamma(nu / 2) * gamma(1 - nu / 2)) \
                / (gamma(nu / 2 - k[1:] + 1) * gamma(nu / 2 + k[1:] + 1) * gamma(-nu))
    except (OverflowError, ZeroDivisionError):
        # Альтернативний розрахунок при проблемах з гамма-функцією
        w[0] = -1.0
        for i in range(1, len(w)):
            w[i] = w[i - 1] * (nu / 2 - i + 1) / i

    Cs = -w / (2 * np.cos(np.pi * nu / 2) * h ** nu)

    def compute_sl(l, Wl, s):
        center = [(2 * m, m), (0, m), (m, 2 * m), (m, 0),
                  (0, 2 * m), (2 * m, 0), (2 * m, 2 * m), (0, 0)]

        cx, cy = center[l]
        anchor = (cy, cx)

        if 0 <= l < 4:
            sl = cv2.filter2D(s, cv2.CV_64F, Wl, anchor=anchor)
        elif 4 <= l < 8:
            factor = 2 ** (-nu / 2)
            sl = factor * cv2.filter2D(s, cv2.CV_64F, Wl, anchor=anchor)
            sl += (1 - factor) * Wl[0 + cx, 0 + cy] * s
        else:
            raise ValueError("Wrong value of parameter 'l', must be 0 <= l < 8")

        return sl

    W = []
    center = mask_size[0] // 2

    # Створюємо 8 масок для різних напрямків
    Wl = np.zeros(mask_size)
    Wl[:, center] = Cs[::-1]
    W.append(Wl)

    Wl = np.zeros(mask_size)
    Wl[:, center] = Cs
    W.append(Wl)

    Wl = np.zeros(mask_size)
    Wl[center] = Cs[::-1]
    W.append(Wl)

    Wl = np.zeros(mask_size)
    Wl[center] = Cs
    W.append(Wl)

    Wl = np.zeros(mask_size)
    np.fill_diagonal(np.fliplr(Wl), Cs)
    W.append(Wl)

    Wl = np.zeros(mask_size)
    np.fill_diagonal(np.flipud(Wl), Cs)
    W.append(Wl)

    Wl = np.zeros(mask_size)
    np.fill_diagonal(Wl, Cs[::-1])
    W.append(Wl)

    Wl = np.zeros(mask_size)
    np.fill_diagonal(Wl, Cs)
    W.append(Wl)

    s = []
    for l in range(8):
        sl = compute_sl(l, W[l], img)
        s.append(sl)

    # Посилений розрахунок фінального результату
    denominator = 4 * (sum(Cs) + sum(2 ** (-nu / 2) * c for c in Cs[1:]) + Cs[0])
    if abs(denominator) < 1e-10:
        denominator = 1.0

    S = np.sum(s, axis=0) / denominator

    # Додаткове підсилення контрасту
    S = S * (1 + nu)  # Посилюємо ефект

    return np.clip(S, 0, 1)


# ===== ГОЛОВНИЙ ДОДАТОК =====
class ICHDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Детекція крововиливів та покращення зображень")
        self.root.geometry("1400x900")

        # Ініціалізуємо змінні
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_image = None
        self.original_image = None

        # Створюємо інтерфейс
        self.create_widgets()

        # Ініціалізуємо модель
        self.init_model()

    def init_model(self):
        """Ініціалізація моделі U-Net"""
        try:
            self.model = Unet().float().to(self.device)
            self.model.eval()
            self.status_var.set("Модель завантажена успішно")
        except Exception as e:
            self.status_var.set(f"Помилка завантаження моделі: {str(e)}")

    def create_widgets(self):
        """Створення елементів інтерфейсу"""
        # Фрейм для кнопок
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)

        # Кнопки
        ttk.Button(button_frame, text="Завантажити зображення",
                   command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Завантажити модель",
                   command=self.load_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Скинути до оригіналу",
                   command=self.reset_image).pack(side=tk.LEFT, padx=5)

        # Фрейм для функцій
        func_frame = ttk.Frame(self.root)
        func_frame.pack(pady=10)

        ttk.Button(func_frame, text="Детекція крововиливу",
                   command=self.detect_hemorrhage).pack(side=tk.LEFT, padx=5)
        ttk.Button(func_frame, text="Покращення А-Б",
                   command=self.enhance_atangana_baleanu).pack(side=tk.LEFT, padx=5)
        ttk.Button(func_frame, text="Покращення Рісса",
                   command=self.enhance_riesz).pack(side=tk.LEFT, padx=5)
        ttk.Button(func_frame, text="Зберегти результат",
                   command=self.save_result).pack(side=tk.LEFT, padx=5)

        # ПОКРАЩЕНІ ПАРАМЕТРИ З ЦИФРОВИМ ВВЕДЕННЯМ
        param_frame = ttk.LabelFrame(self.root, text="Параметри покращення", padding=10)
        param_frame.pack(pady=10, padx=10, fill=tk.X)

        # Альфа параметр
        alpha_frame = ttk.Frame(param_frame)
        alpha_frame.pack(fill=tk.X, pady=5)

        ttk.Label(alpha_frame, text="Alpha (α/ν):").pack(side=tk.LEFT)
        self.alpha_var = tk.DoubleVar(value=0.8)
        alpha_entry = ttk.Entry(alpha_frame, textvariable=self.alpha_var, width=10)
        alpha_entry.pack(side=tk.LEFT, padx=5)

        # Кнопки швидкого доступу для альфа
        ttk.Button(alpha_frame, text="0.3", width=4,
                   command=lambda: self.alpha_var.set(0.3)).pack(side=tk.LEFT, padx=2)
        ttk.Button(alpha_frame, text="0.5", width=4,
                   command=lambda: self.alpha_var.set(0.5)).pack(side=tk.LEFT, padx=2)
        ttk.Button(alpha_frame, text="0.7", width=4,
                   command=lambda: self.alpha_var.set(0.7)).pack(side=tk.LEFT, padx=2)
        ttk.Button(alpha_frame, text="0.9", width=4,
                   command=lambda: self.alpha_var.set(0.9)).pack(side=tk.LEFT, padx=2)

        # Розмір вікна
        window_frame = ttk.Frame(param_frame)
        window_frame.pack(fill=tk.X, pady=5)

        ttk.Label(window_frame, text="Розмір вікна:").pack(side=tk.LEFT)
        self.window_var = tk.IntVar(value=7)
        window_entry = ttk.Entry(window_frame, textvariable=self.window_var, width=10)
        window_entry.pack(side=tk.LEFT, padx=5)

        # Кнопки швидкого доступу для розміру вікна
        ttk.Button(window_frame, text="3", width=4,
                   command=lambda: self.window_var.set(3)).pack(side=tk.LEFT, padx=2)
        ttk.Button(window_frame, text="5", width=4,
                   command=lambda: self.window_var.set(5)).pack(side=tk.LEFT, padx=2)
        ttk.Button(window_frame, text="7", width=4,
                   command=lambda: self.window_var.set(7)).pack(side=tk.LEFT, padx=2)
        ttk.Button(window_frame, text="9", width=4,
                   command=lambda: self.window_var.set(9)).pack(side=tk.LEFT, padx=2)
        ttk.Button(window_frame, text="11", width=4,
                   command=lambda: self.window_var.set(11)).pack(side=tk.LEFT, padx=2)

        # РЕКОМЕНДОВАНІ НАЛАШТУВАННЯ
        presets_frame = ttk.LabelFrame(self.root, text="Рекомендовані налаштування", padding=10)
        presets_frame.pack(pady=5, padx=10, fill=tk.X)

        # Фрейми для різних типів
        ab_frame = ttk.Frame(presets_frame)
        ab_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(ab_frame, text="Атангана-Беленау:", font=('Arial', 10, 'bold')).pack()

        ttk.Button(ab_frame, text="Слабкий (α=0.5, вікно=5)",
                   command=lambda: self.set_preset(0.5, 5)).pack(fill=tk.X, pady=1)
        ttk.Button(ab_frame, text="Помірний (α=0.7, вікно=7)",
                   command=lambda: self.set_preset(0.7, 7)).pack(fill=tk.X, pady=1)
        ttk.Button(ab_frame, text="Сильний (α=0.9, вікно=9)",
                   command=lambda: self.set_preset(0.9, 9)).pack(fill=tk.X, pady=1)

        riesz_frame = ttk.Frame(presets_frame)
        riesz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        ttk.Label(riesz_frame, text="Рісса:", font=('Arial', 10, 'bold')).pack()

        ttk.Button(riesz_frame, text="Слабкий (ν=0.3, m=2)",
                   command=lambda: self.set_preset(0.3, 5)).pack(fill=tk.X, pady=1)
        ttk.Button(riesz_frame, text="Помірний (ν=0.6, m=3)",
                   command=lambda: self.set_preset(0.6, 7)).pack(fill=tk.X, pady=1)
        ttk.Button(riesz_frame, text="Сильний (ν=0.8, m=4)",
                   command=lambda: self.set_preset(0.8, 9)).pack(fill=tk.X, pady=1)

        # Фрейм для відображення зображень
        image_frame = ttk.Frame(self.root)
        image_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Створюємо matplotlib фігуру
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.suptitle("Результати обробки", fontsize=14)

        for ax in self.axes:
            ax.axis('off')

        self.canvas = FigureCanvasTkAgg(self.fig, image_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

        # Статус бар
        self.status_var = tk.StringVar(value="Готово до роботи")
        ttk.Label(self.root, textvariable=self.status_var,
                  relief=tk.SUNKEN).pack(side=tk.BOTTOM, fill=tk.X)

    def set_preset(self, alpha, window):
        """Встановити передустановки параметрів"""
        self.alpha_var.set(alpha)
        self.window_var.set(window)

    def load_image(self):
        """Завантаження зображення"""
        file_path = filedialog.askopenfilename(
            title="Виберіть зображення",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )

        if file_path:
            try:
                # Завантажуємо зображення
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError("Не вдалося завантажити зображення")

                # Нормалізуємо
                self.original_image = image.astype(np.float64) / 255.0
                self.current_image = self.original_image.copy()

                # Відображаємо
                self.display_image(self.original_image, 0, "Оригінал")
                self.clear_other_axes()
                self.status_var.set(f"Зображення завантажено: {os.path.basename(file_path)}")

            except Exception as e:
                messagebox.showerror("Помилка", f"Помилка завантаження зображення: {str(e)}")

    def load_model(self):
        """Завантаження натренованої моделі"""
        file_path = filedialog.askopenfilename(
            title="Виберіть файл моделі",
            filetypes=[("Model files", "*.tar *.pth *.pt")]
        )

        if file_path:
            try:
                checkpoint = torch.load(file_path, map_location=self.device)

                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)

                self.model.eval()
                self.status_var.set(f"Модель завантажена: {os.path.basename(file_path)}")

            except Exception as e:
                messagebox.showerror("Помилка", f"Помилка завантаження моделі: {str(e)}")

    def detect_hemorrhage(self):
        """Детекція крововиливу"""
        if self.current_image is None:
            messagebox.showwarning("Попередження", "Спочатку завантажте зображення")
            return

        if self.model is None:
            messagebox.showwarning("Попередження", "Спочатку завантажте модель")
            return

        try:
            # Підготовка зображення для моделі
            image_tensor = torch.FloatTensor(self.current_image).unsqueeze(0).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)

            # Прогноз
            with torch.no_grad():
                prediction = self.model(image_tensor)
                prediction = prediction.cpu().numpy()[0, 0]

            # Створюємо бінарну маску
            binary_mask = (prediction > 0.5).astype(np.float64)

            # Обчислюємо статистику
            hemorrhage_pixels = np.sum(binary_mask)
            total_pixels = binary_mask.size
            percentage = (hemorrhage_pixels / total_pixels) * 100

            # Відображаємо результати
            self.display_image(self.current_image, 0, "Вхід")
            self.display_image(prediction, 1, "Сегментація")
            self.display_image(binary_mask, 2, "Бінарна маска")

            self.status_var.set(f"Детекція завершена. Крововилив: {percentage:.2f}% площі")

        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка детекції: {str(e)}")

    def enhance_atangana_baleanu(self):
        """Покращення через оператор Атангана-Беленау"""
        if self.current_image is None:
            messagebox.showwarning("Попередження", "Спочатку завантажте зображення")
            return

        try:
            alpha = float(self.alpha_var.get())
            window_size = int(self.window_var.get())

            # Перевірка діапазонів
            if not (0.1 <= alpha <= 1.0):
                raise ValueError("Alpha повинен бути між 0.1 та 1.0")
            if not (3 <= window_size <= 15):
                raise ValueError("Розмір вікна повинен бути між 3 та 15")

            # Застосовуємо оператор
            ab_operator = AtanganaBaleanuOperator(alpha=alpha, window_size=window_size)
            enhanced = ab_operator.enhance_image(self.current_image.copy())

            # Відображаємо результати
            self.display_image(self.current_image, 0, "Оригінал")
            self.display_image(enhanced, 1, f"А-Б α={alpha:.2f}")

            # Показуємо різницю (посилену для кращої видимості)
            difference = np.abs(enhanced - self.current_image) * 5.0  # Посилюємо різницю
            difference = np.clip(difference, 0, 1)
            self.display_image(difference, 2, "Різниця (×5)")

            self.current_image = enhanced
            self.status_var.set(f"Покращення А-Б завершено (α={alpha:.2f}, вікно={window_size})")

        except ValueError as e:
            messagebox.showerror("Помилка", f"Неправильні параметри: {str(e)}")
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка покращення А-Б: {str(e)}")

    def enhance_riesz(self):
        """Покращення через дробові похідні Рісса"""
        if self.current_image is None:
            messagebox.showwarning("Попередження", "Спочатку завантажте зображення")
            return

        try:
            nu = float(self.alpha_var.get())
            window_size = int(self.window_var.get())
            m = max(1, window_size // 2)  # Забезпечуємо мінімальне значення m

            # Перевірка діапазонів
            if not (0.1 <= nu <= 1.0):
                raise ValueError("Nu повинен бути між 0.1 та 1.0")
            if not (3 <= window_size <= 15):
                raise ValueError("Розмір вікна повинен бути між 3 та 15")

            # Застосовуємо дробові похідні Рісса
            enhanced = fractional_differential_mask(self.current_image.copy(), nu=nu, m=m)

            # Відображаємо результати
            self.display_image(self.current_image, 0, "Оригінал")
            self.display_image(enhanced, 1, f"Рісса ν={nu:.2f}")

            # Показуємо різницю
            difference = np.abs(enhanced - self.current_image) * 3.0  # Посилюємо різницю
            difference = np.clip(difference, 0, 1)
            self.display_image(difference, 2, "Різниця (×3)")

            self.current_image = enhanced
            self.status_var.set(f"Покращення Рісса завершено (ν={nu:.2f}, m={m})")

        except ValueError as e:
            messagebox.showerror("Помилка", f"Неправильні параметри: {str(e)}")
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка покращення Рісса: {str(e)}")

    def display_image(self, image, ax_index, title):
        """Відображення зображення на вказаній осі"""
        if ax_index < len(self.axes):
            self.axes[ax_index].clear()
            self.axes[ax_index].imshow(image, cmap='gray', vmin=0, vmax=1)
            self.axes[ax_index].set_title(title, fontsize=12)
            self.axes[ax_index].axis('off')
            self.canvas.draw()

    def clear_other_axes(self):
        """Очищення інших осей"""
        for i in range(1, len(self.axes)):
            self.axes[i].clear()
            self.axes[i].axis('off')
        self.canvas.draw()

    def save_result(self):
        """Збереження поточного результату"""
        if self.current_image is None:
            messagebox.showwarning("Попередження", "Немає зображення для збереження")
            return

        file_path = filedialog.asksaveasfilename(
            title="Зберегти результат",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")]
        )

        if file_path:
            try:
                # Конвертуємо в uint8
                image_uint8 = (self.current_image * 255).astype(np.uint8)
                cv2.imwrite(file_path, image_uint8)
                self.status_var.set(f"Результат збережено: {os.path.basename(file_path)}")

            except Exception as e:
                messagebox.showerror("Помилка", f"Помилка збереження: {str(e)}")

    def reset_image(self):
        """Повернення до оригінального зображення"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.display_image(self.current_image, 0, "Оригінал")
            self.clear_other_axes()
            self.status_var.set("Зображення скинуто до оригіналу")
        else:
            messagebox.showwarning("Попередження", "Немає оригінального зображення")


# ===== ЗАПУСК ПРОГРАМИ =====
if __name__ == "__main__":
    root = tk.Tk()
    app = ICHDetectionApp(root)
    root.mainloop()