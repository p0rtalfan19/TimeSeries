import os
os.environ['TCL_LIBRARY'] = "C:/Program Files/Python313/tcl/tcl8.6"
os.environ['TK_LIBRARY'] = "C:/Program Files/Python313/tcl/tk8.6"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pytz
import warnings
warnings.filterwarnings('ignore')

# Статистические тесты
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# Настройка отображения
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

class TimeSeriesAnalyzer:
    def __init__(self, csv_file):
        """Инициализация анализатора временных рядов"""
        self.df = pd.read_csv(csv_file)
        self.processed_df = None
        
    def load_and_explore_data(self):
        """Этап 1: Загрузка и первичное исследование данных"""
        print("=" * 60)
        print("ЭТАП 1: ЗАГРУЗКА И ПЕРВИЧНОЕ ИССЛЕДОВАНИЕ ДАННЫХ")
        print("=" * 60)
        
        print(f"Размер датасета: {self.df.shape}")
        print(f"\nИнформация о данных:")
        print(self.df.info())
        
        print(f"\nПервые 5 строк:")
        print(self.df.head())
        
        print(f"\nОписательная статистика:")
        print(self.df.describe())
        
        print(f"\nПропущенные значения:")
        print(self.df.isnull().sum())
        
        return self.df
    
    def clean_and_preprocess_data(self):
        """Этап 2: Предварительная очистка и предобработка данных"""
        print("\n" + "=" * 60)
        print("ЭТАП 2: ПРЕДВАРИТЕЛЬНАЯ ОЧИСТКА И ПРЕДОБРАБОТКА ДАННЫХ")
        print("=" * 60)
        
        # Создаем копию для обработки
        df_clean = self.df.copy()
        
        # 1. Приведение временных меток к единому формату
        print("1. Обработка временных меток...")
        df_clean['Date'] = pd.to_datetime(df_clean['Date'])
        
        # Устанавливаем временную зону (предполагаем UTC, затем конвертируем в Moscow)
        df_clean['Date'] = df_clean['Date'].dt.tz_localize('UTC').dt.tz_convert('Europe/Moscow')
        
        # 2. Удаление дубликатов по времени
        print("2. Удаление дубликатов...")
        initial_size = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['Store', 'Dept', 'Date'], keep='first')
        duplicates_removed = initial_size - len(df_clean)
        print(f"Удалено дубликатов: {duplicates_removed}")
        
        # 3. Проверка монотонности временного ряда
        print("3. Проверка монотонности временного ряда...")
        df_clean = df_clean.sort_values(['Store', 'Dept', 'Date'])
        
        # Проверяем на "прыжки" во времени для каждого магазина и отдела
        time_gaps = []
        for (store, dept), group in df_clean.groupby(['Store', 'Dept']):
            time_diffs = group['Date'].diff().dt.days
            gaps = time_diffs[time_diffs > 7]  # Пропуски больше недели
            if not gaps.empty:
                time_gaps.append((store, dept, gaps.sum()))
        
        if time_gaps:
            print(f"Найдены временные пропуски в {len(time_gaps)} комбинациях Store-Dept")
        else:
            print("Временной ряд монотонен")
        
        # 4. Обработка пропусков
        print("4. Обработка пропусков...")
        missing_before = df_clean.isnull().sum().sum()
        
        # Для Weekly_Sales применяем интерполяцию
        df_clean['Weekly_Sales'] = df_clean.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
            lambda x: x.interpolate(method='linear')
        )
        
        # Заполняем оставшиеся пропуски скользящим средним
        df_clean['Weekly_Sales'] = df_clean.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
            lambda x: x.fillna(x.rolling(window=4, min_periods=1).mean())
        )
        
        missing_after = df_clean.isnull().sum().sum()
        print(f"Пропусков до обработки: {missing_before}")
        print(f"Пропусков после обработки: {missing_after}")
        
        # 5. Обнаружение и обработка выбросов
        print("5. Обнаружение выбросов...")
        
        # Метод IQR для Weekly_Sales
        Q1 = df_clean['Weekly_Sales'].quantile(0.25)
        Q3 = df_clean['Weekly_Sales'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_iqr = df_clean[(df_clean['Weekly_Sales'] < lower_bound) | 
                               (df_clean['Weekly_Sales'] > upper_bound)]
        print(f"Выбросы по методу IQR: {len(outliers_iqr)} ({len(outliers_iqr)/len(df_clean)*100:.2f}%)")
        
        # Z-score метод
        z_scores = np.abs((df_clean['Weekly_Sales'] - df_clean['Weekly_Sales'].mean()) / 
                         df_clean['Weekly_Sales'].std())
        outliers_z = df_clean[z_scores > 3]
        print(f"Выбросы по Z-score: {len(outliers_z)} ({len(outliers_z)/len(df_clean)*100:.2f}%)")
        
        # Визуализация выбросов
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Boxplot
        df_clean.boxplot(column='Weekly_Sales', ax=axes[0])
        axes[0].set_title('Boxplot для Weekly_Sales')
        
        # Scatter plot
        axes[1].scatter(range(len(df_clean)), df_clean['Weekly_Sales'], alpha=0.5)
        axes[1].axhline(y=upper_bound, color='r', linestyle='--', label='Верхняя граница IQR')
        axes[1].axhline(y=lower_bound, color='r', linestyle='--', label='Нижняя граница IQR')
        axes[1].set_title('Scatter plot Weekly_Sales')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
        
        # 6. Ресемплирование до единой частоты (еженедельно)
        print("6. Ресемплирование данных...")
        
        # Создаем агрегированные данные по неделям
        df_clean['Year'] = df_clean['Date'].dt.year
        df_clean['Week'] = df_clean['Date'].dt.isocalendar().week
        
        # Агрегируем по неделям
        weekly_data = df_clean.groupby(['Year', 'Week']).agg({
            'Weekly_Sales': 'sum',
            'IsHoliday': 'any',
            'Store': 'nunique',
            'Dept': 'nunique'
        }).reset_index()
        
        # Создаем дату для каждой недели
        weekly_data['Date'] = pd.to_datetime(weekly_data['Year'].astype(str) + '-' + 
                                           weekly_data['Week'].astype(str) + '-1', 
                                           format='%Y-%W-%w')
        
        self.processed_df = weekly_data
        print(f"Размер обработанных данных: {self.processed_df.shape}")
        
        return self.processed_df
    
    def descriptive_analysis_and_visualization(self):
        """Этап 3: Описательный статистический анализ и визуализация"""
        print("\n" + "=" * 60)
        print("ЭТАП 3: ОПИСАТЕЛЬНЫЙ СТАТИСТИЧЕСКИЙ АНАЛИЗ И ВИЗУАЛИЗАЦИЯ")
        print("=" * 60)
        
        df = self.processed_df.copy()
        
        # 1. Дескриптивная статистика
        print("1. Дескриптивная статистика:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        desc_stats = df[numeric_cols].describe()
        
        # Добавляем дополнительные статистики
        additional_stats = pd.DataFrame({
            'skewness': df[numeric_cols].skew(),
            'kurtosis': df[numeric_cols].kurtosis()
        })
        
        print(desc_stats)
        print("\nДополнительные статистики:")
        print(additional_stats)
        
        # 2. Визуализация
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Линейный график целевой переменной
        axes[0, 0].plot(df['Date'], df['Weekly_Sales'])
        axes[0, 0].set_title('Weekly Sales по времени')
        axes[0, 0].set_xlabel('Дата')
        axes[0, 0].set_ylabel('Weekly Sales')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Гистограмма Weekly_Sales
        axes[0, 1].hist(df['Weekly_Sales'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Распределение Weekly_Sales')
        axes[0, 1].set_xlabel('Weekly Sales')
        axes[0, 1].set_ylabel('Частота')
        
        # Boxplot для Weekly_Sales
        axes[1, 0].boxplot(df['Weekly_Sales'])
        axes[1, 0].set_title('Boxplot Weekly_Sales')
        axes[1, 0].set_ylabel('Weekly Sales')
        
        # Корреляционная матрица
        corr_matrix = df[numeric_cols].corr()
        im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 1].set_xticks(range(len(corr_matrix.columns)))
        axes[1, 1].set_yticks(range(len(corr_matrix.columns)))
        axes[1, 1].set_xticklabels(corr_matrix.columns, rotation=45)
        axes[1, 1].set_yticklabels(corr_matrix.columns)
        axes[1, 1].set_title('Корреляционная матрица')
        
        # Добавляем значения корреляции
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black")
        
        plt.tight_layout()
        plt.show()
        
        # 3. Анализ мультиколлинеарности
        print("\n3. Анализ мультиколлинеарности:")
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            print("Найдены высокие корреляции:")
            for col1, col2, corr in high_corr_pairs:
                print(f"  {col1} - {col2}: {corr:.3f}")
        else:
            print("Высоких корреляций не найдено")
        
        return df
    
    def stationarity_tests(self):
        """Этап 4: Проверка на стационарность и статистические тесты"""
        print("\n" + "=" * 60)
        print("ЭТАП 4: ПРОВЕРКА НА СТАЦИОНАРНОСТЬ И СТАТИСТИЧЕСКИЕ ТЕСТЫ")
        print("=" * 60)
        
        df = self.processed_df.copy()
        series = df['Weekly_Sales'].dropna()
        
        # 1. Визуальный анализ
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Исходный ряд
        axes[0, 0].plot(series)
        axes[0, 0].set_title('Исходный временной ряд')
        axes[0, 0].set_ylabel('Weekly Sales')
        
        # Скользящее среднее
        rolling_mean_30 = series.rolling(window=30).mean()
        rolling_mean_60 = series.rolling(window=60).mean()
        rolling_mean_90 = series.rolling(window=90).mean()
        
        axes[0, 1].plot(series, alpha=0.7, label='Исходный ряд')
        axes[0, 1].plot(rolling_mean_30, label='Скользящее среднее (30)')
        axes[0, 1].plot(rolling_mean_60, label='Скользящее среднее (60)')
        axes[0, 1].plot(rolling_mean_90, label='Скользящее среднее (90)')
        axes[0, 1].set_title('Скользящие средние')
        axes[0, 1].legend()
        
        # Скользящая дисперсия
        rolling_std_30 = series.rolling(window=30).std()
        rolling_std_60 = series.rolling(window=60).std()
        rolling_std_90 = series.rolling(window=90).std()
        
        axes[1, 0].plot(rolling_std_30, label='Скользящее стд. откл. (30)')
        axes[1, 0].plot(rolling_std_60, label='Скользящее стд. откл. (60)')
        axes[1, 0].plot(rolling_std_90, label='Скользящее стд. откл. (90)')
        axes[1, 0].set_title('Скользящие стандартные отклонения')
        axes[1, 0].legend()
        
        # Распределение
        axes[1, 1].hist(series, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Распределение Weekly_Sales')
        axes[1, 1].set_xlabel('Weekly Sales')
        axes[1, 1].set_ylabel('Частота')
        
        plt.tight_layout()
        plt.show()
        
        # 2. Статистические тесты
        print("\n2. Статистические тесты стационарности:")
        
        # Тест Дики-Фуллера
        adf_result = adfuller(series, autolag='AIC')
        print(f"Тест Дики-Фуллера:")
        print(f"  ADF статистика: {adf_result[0]:.6f}")
        print(f"  p-value: {adf_result[1]:.6f}")
        print(f"  Критические значения:")
        for key, value in adf_result[4].items():
            print(f"    {key}: {value:.6f}")
        
        if adf_result[1] < 0.05:
            print("  Результат: Ряд СТАЦИОНАРЕН (p < 0.05)")
        else:
            print("  Результат: Ряд НЕСТАЦИОНАРЕН (p >= 0.05)")
        
        # Тест KPSS
        try:
            kpss_result = kpss(series, regression='c')
            print(f"\nТест KPSS:")
            print(f"  KPSS статистика: {kpss_result[0]:.6f}")
            print(f"  p-value: {kpss_result[1]:.6f}")
            print(f"  Критические значения:")
            for key, value in kpss_result[3].items():
                print(f"    {key}: {value:.6f}")
            
            if kpss_result[1] > 0.05:
                print("  Результат: Ряд СТАЦИОНАРЕН (p > 0.05)")
            else:
                print("  Результат: Ряд НЕСТАЦИОНАРЕН (p <= 0.05)")
        except Exception as e:
            print(f"Ошибка при выполнении теста KPSS: {e}")
        
        # 3. Дифференцирование при необходимости
        if adf_result[1] >= 0.05:
            print("\n3. Применение дифференцирования...")
            diff_series = series.diff().dropna()
            
            # Повторные тесты после дифференцирования
            adf_diff = adfuller(diff_series, autolag='AIC')
            print(f"Тест Дики-Фуллера после дифференцирования:")
            print(f"  ADF статистика: {adf_diff[0]:.6f}")
            print(f"  p-value: {adf_diff[1]:.6f}")
            
            if adf_diff[1] < 0.05:
                print("  Результат: Ряд стал СТАЦИОНАРНЫМ после дифференцирования")
                return diff_series
            else:
                print("  Результат: Ряд остается НЕСТАЦИОНАРНЫМ")
        
        return series
    
    def create_lag_features(self):
        """Этап 5: Создание лаговых признаков и скользящих статистик"""
        print("\n" + "=" * 60)
        print("ЭТАП 5: СОЗДАНИЕ ЛАГОВЫХ ПРИЗНАКОВ И СКОЛЬЗЯЩИХ СТАТИСТИК")
        print("=" * 60)
        
        df = self.processed_df.copy()
        df = df.sort_values('Date').reset_index(drop=True)
        
        # 1. Создание лагов целевой переменной
        print("1. Создание лаговых признаков...")
        df['target_lag_1'] = df['Weekly_Sales'].shift(1)
        df['target_lag_7'] = df['Weekly_Sales'].shift(7)
        df['target_lag_30'] = df['Weekly_Sales'].shift(30)
        
        # 2. Создание скользящих статистик
        print("2. Создание скользящих статистик...")
        df['target_rolling_mean_7'] = df['Weekly_Sales'].rolling(window=7).mean()
        df['target_rolling_mean_30'] = df['Weekly_Sales'].rolling(window=30).mean()
        df['target_rolling_std_7'] = df['Weekly_Sales'].rolling(window=7).std()
        df['target_rolling_std_30'] = df['Weekly_Sales'].rolling(window=30).std()
        
        # 3. Анализ корреляций между лагами и целевой переменной
        print("\n3. Корреляции между лагами и целевой переменной:")
        lag_cols = ['target_lag_1', 'target_lag_7', 'target_lag_30']
        correlations = df[lag_cols + ['Weekly_Sales']].corr()['Weekly_Sales'].drop('Weekly_Sales')
        
        for col, corr in correlations.items():
            print(f"  {col}: {corr:.4f}")
        
        # 4. Проверка мультиколлинеарности после добавления лагов
        print("\n4. Проверка мультиколлинеарности:")
        all_features = ['Weekly_Sales', 'Store', 'Dept', 'IsHoliday'] + lag_cols + \
                      ['target_rolling_mean_7', 'target_rolling_mean_30', 
                       'target_rolling_std_7', 'target_rolling_std_30']
        
        # Убираем NaN значения для корреляционного анализа
        df_clean = df[all_features].dropna()
        corr_matrix = df_clean.corr()
        
        # Находим высокие корреляции
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            print("Найдены высокие корреляции между признаками:")
            for col1, col2, corr in high_corr_pairs:
                print(f"  {col1} - {col2}: {corr:.3f}")
        else:
            print("Высоких корреляций между признаками не найдено")
        
        # 5. Визуализация лагов
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Корреляция с лагами
        lag_corrs = correlations.values
        axes[0, 0].bar(range(len(lag_corrs)), lag_corrs)
        axes[0, 0].set_title('Корреляция с лагами')
        axes[0, 0].set_xticks(range(len(lag_corrs)))
        axes[0, 0].set_xticklabels([col.replace('target_lag_', 'Lag ') for col in lag_cols])
        axes[0, 0].set_ylabel('Корреляция')
        
        # Скользящие средние
        axes[0, 1].plot(df['Date'], df['Weekly_Sales'], alpha=0.7, label='Исходный ряд')
        axes[0, 1].plot(df['Date'], df['target_rolling_mean_7'], label='Скользящее среднее (7)')
        axes[0, 1].plot(df['Date'], df['target_rolling_mean_30'], label='Скользящее среднее (30)')
        axes[0, 1].set_title('Скользящие средние')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Скользящие стандартные отклонения
        axes[1, 0].plot(df['Date'], df['target_rolling_std_7'], label='Скользящее стд. откл. (7)')
        axes[1, 0].plot(df['Date'], df['target_rolling_std_30'], label='Скользящее стд. откл. (30)')
        axes[1, 0].set_title('Скользящие стандартные отклонения')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Scatter plot лага 1
        axes[1, 1].scatter(df['target_lag_1'], df['Weekly_Sales'], alpha=0.5)
        axes[1, 1].set_xlabel('Weekly_Sales (lag 1)')
        axes[1, 1].set_ylabel('Weekly_Sales')
        axes[1, 1].set_title('Scatter: Weekly_Sales vs Lag 1')
        
        plt.tight_layout()
        plt.show()
        
        self.processed_df = df
        return df
    
    def autocorrelation_analysis(self):
        """Этап 6: Анализ автокорреляции ACF и PACF"""
        print("\n" + "=" * 60)
        print("ЭТАП 6: АНАЛИЗ АВТОКОРРЕЛЯЦИИ ACF И PACF")
        print("=" * 60)
        
        df = self.processed_df.copy()
        series = df['Weekly_Sales'].dropna()
        
        # Построение ACF и PACF
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # ACF
        plot_acf(series, ax=axes[0], lags=50, alpha=0.05)
        axes[0].set_title('Автокорреляционная функция (ACF)')
        axes[0].grid(True)
        
        # PACF
        plot_pacf(series, ax=axes[1], lags=50, alpha=0.05)
        axes[1].set_title('Частичная автокорреляционная функция (PACF)')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Анализ значимых лагов
        print("\nАнализ значимых лагов:")
        
        # Вычисляем ACF и PACF вручную для анализа
        from statsmodels.tsa.stattools import acf, pacf
        
        acf_values = acf(series, nlags=50, alpha=0.05)
        pacf_values = pacf(series, nlags=50, alpha=0.05)
        
        # Значимые ACF лаги
        significant_acf = []
        for i, (acf_val, (lower, upper)) in enumerate(zip(acf_values[0][1:], acf_values[1])):
            if acf_val > upper or acf_val < lower:
                significant_acf.append(i + 1)
        
        # Значимые PACF лаги
        significant_pacf = []
        for i, (pacf_val, (lower, upper)) in enumerate(zip(pacf_values[0][1:], pacf_values[1])):
            if pacf_val > upper or pacf_val < lower:
                significant_pacf.append(i + 1)
        
        print(f"Значимые ACF лаги: {significant_acf[:10]}")  # Показываем первые 10
        print(f"Значимые PACF лаги: {significant_pacf[:10]}")  # Показываем первые 10
        
        # Интерпретация
        print("\nИнтерпретация:")
        if significant_pacf:
            print(f"Резкий обрыв в PACF на лаге {significant_pacf[0]} → возможный порядок AR({significant_pacf[0]})")
        else:
            print("Резкого обрыва в PACF не обнаружено")
        
        if len(significant_acf) > 5:
            print("Постепенное затухание в ACF → возможный порядок MA(q)")
        else:
            print("Быстрое затухание в ACF")
        
        return series
    
    def time_series_decomposition(self):
        """Этап 7: Декомпозиция временного ряда"""
        print("\n" + "=" * 60)
        print("ЭТАП 7: ДЕКОМПОЗИЦИЯ ВРЕМЕННОГО РЯДА")
        print("=" * 60)
        
        df = self.processed_df.copy()
        series = df['Weekly_Sales'].dropna()
        
        # Устанавливаем индекс как дату для декомпозиции
        series_indexed = series.copy()
        series_indexed.index = pd.DatetimeIndex(df.loc[series.index, 'Date'])
        
        # 1. Аддитивная декомпозиция
        print("1. Аддитивная декомпозиция...")
        try:
            additive_decomp = seasonal_decompose(series_indexed, model='additive', period=52)  # 52 недели в году
            
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            
            # Исходный ряд
            additive_decomp.observed.plot(ax=axes[0], title='Исходный ряд')
            axes[0].set_ylabel('Weekly Sales')
            
            # Тренд
            additive_decomp.trend.plot(ax=axes[1], title='Тренд')
            axes[1].set_ylabel('Тренд')
            
            # Сезонность
            additive_decomp.seasonal.plot(ax=axes[2], title='Сезонность')
            axes[2].set_ylabel('Сезонность')
            
            # Остатки
            additive_decomp.resid.plot(ax=axes[3], title='Остатки')
            axes[3].set_ylabel('Остатки')
            
            plt.tight_layout()
            plt.show()
            
            # Анализ компонент
            print("\nАнализ аддитивной декомпозиции:")
            print(f"Сила тренда: {additive_decomp.trend.std():.2f}")
            print(f"Сила сезонности: {additive_decomp.seasonal.std():.2f}")
            print(f"Сила остатков: {additive_decomp.resid.std():.2f}")
            
        except Exception as e:
            print(f"Ошибка при аддитивной декомпозиции: {e}")
        
        # 2. Мультипликативная декомпозиция
        print("\n2. Мультипликативная декомпозиция...")
        try:
            multiplicative_decomp = seasonal_decompose(series_indexed, model='multiplicative', period=52)
            
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            
            # Исходный ряд
            multiplicative_decomp.observed.plot(ax=axes[0], title='Исходный ряд')
            axes[0].set_ylabel('Weekly Sales')
            
            # Тренд
            multiplicative_decomp.trend.plot(ax=axes[1], title='Тренд')
            axes[1].set_ylabel('Тренд')
            
            # Сезонность
            multiplicative_decomp.seasonal.plot(ax=axes[2], title='Сезонность')
            axes[2].set_ylabel('Сезонность')
            
            # Остатки
            multiplicative_decomp.resid.plot(ax=axes[3], title='Остатки')
            axes[3].set_ylabel('Остатки')
            
            plt.tight_layout()
            plt.show()
            
            # Анализ компонент
            print("\nАнализ мультипликативной декомпозиции:")
            print(f"Сила тренда: {multiplicative_decomp.trend.std():.2f}")
            print(f"Сила сезонности: {multiplicative_decomp.seasonal.std():.2f}")
            print(f"Сила остатков: {multiplicative_decomp.resid.std():.2f}")
            
        except Exception as e:
            print(f"Ошибка при мультипликативной декомпозиции: {e}")
        
        # 3. Анализ остатков
        print("\n3. Анализ остатков:")
        if 'additive_decomp' in locals():
            residuals = additive_decomp.resid.dropna()
            
            # Тест на нормальность остатков
            from scipy import stats
            shapiro_stat, shapiro_p = stats.shapiro(residuals.sample(min(5000, len(residuals))))
            print(f"Тест Шапиро-Уилка на нормальность остатков:")
            print(f"  Статистика: {shapiro_stat:.6f}")
            print(f"  p-value: {shapiro_p:.6f}")
            
            if shapiro_p > 0.05:
                print("  Остатки распределены нормально")
            else:
                print("  Остатки НЕ распределены нормально")
            
            # Визуализация остатков
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Гистограмма остатков
            axes[0, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Распределение остатков')
            axes[0, 0].set_xlabel('Остатки')
            axes[0, 0].set_ylabel('Частота')
            
            # Q-Q plot
            stats.probplot(residuals, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q plot остатков')
            
            # Остатки по времени
            axes[1, 0].plot(residuals)
            axes[1, 0].set_title('Остатки по времени')
            axes[1, 0].set_ylabel('Остатки')
            
            # Автокорреляция остатков
            plot_acf(residuals, ax=axes[1, 1], lags=20, alpha=0.05)
            axes[1, 1].set_title('ACF остатков')
            
            plt.tight_layout()
            plt.show()
        
        return series
    
    def run_full_analysis(self):
        """Запуск полного анализа временных рядов"""
        print("НАЧАЛО АНАЛИЗА ВРЕМЕННЫХ РЯДОВ")
        print("=" * 60)
        
        # Этап 1: Загрузка и исследование
        self.load_and_explore_data()
        
        # Этап 2: Очистка и предобработка
        self.clean_and_preprocess_data()
        
        # Этап 3: Описательный анализ
        self.descriptive_analysis_and_visualization()
        
        # Этап 4: Тесты стационарности
        self.stationarity_tests()
        
        # Этап 5: Лаговые признаки
        self.create_lag_features()
        
        # Этап 6: Автокорреляция
        self.autocorrelation_analysis()
        
        # Этап 7: Декомпозиция
        self.time_series_decomposition()
        
        print("\n" + "=" * 60)
        print("АНАЛИЗ ЗАВЕРШЕН")
        print("=" * 60)

# Запуск анализа
if __name__ == "__main__":
    # Создаем экземпляр анализатора
    analyzer = TimeSeriesAnalyzer('train.csv')
    
    # Запускаем полный анализ
    analyzer.run_full_analysis()
