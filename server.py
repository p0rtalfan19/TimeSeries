import os
os.environ['TCL_LIBRARY'] = "C:/Program Files/Python313/tcl/tcl8.6"
os.environ['TK_LIBRARY'] = "C:/Program Files/Python313/tcl/tk8.6"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Добавляем логирование для отладки
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Статистические тесты и анализ
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Настройка страницы
st.set_page_config(
    page_title="Анализ временных рядов",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TimeSeriesWebApp:
    def __init__(self):
        self.df = None
        self.processed_df = None
        self.target_column = None
        self.date_column = None
        
    def load_data(self, uploaded_file=None, use_sample=False):
        """Загрузка данных"""
        if use_sample:
            # Используем встроенный пример данных
            try:
                # Пытаемся найти файл в текущей директории
                csv_path = 'train.csv'
                if not os.path.exists(csv_path):
                    # Если файл не найден, пытаемся найти в директории time_series
                    csv_path = os.path.join('time_series', 'train.csv')
                    if not os.path.exists(csv_path):
                        st.error("Файл train.csv не найден. Убедитесь, что файл находится в правильной директории.")
                        return False
                
                self.df = pd.read_csv(csv_path)
                logger.info(f"Загружен файл: {csv_path}, размер: {self.df.shape}")
                self.df['Date'] = pd.to_datetime(self.df['Date'])
                self.date_column = 'Date'
                self.target_column = 'Weekly_Sales'
                logger.info("Данные успешно загружены и настроены")
                return True
            except Exception as e:
                st.error(f"Ошибка при загрузке примера данных: {e}")
                return False
        elif uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    self.df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.parquet'):
                    self.df = pd.read_parquet(uploaded_file)
                else:
                    st.error("Поддерживаются только CSV и Parquet файлы")
                    return False
                
                # Автоматическое определение колонок
                date_cols = [col for col in self.df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_cols:
                    self.date_column = date_cols[0]
                    self.df[self.date_column] = pd.to_datetime(self.df[self.date_column])
                else:
                    st.warning("Не найдена колонка с датами. Выберите вручную в настройках.")
                
                return True
            except Exception as e:
                st.error(f"Ошибка при загрузке файла: {e}")
                return False
        return False
    
    def preprocess_data(self):
        """Предобработка данных"""
        if self.df is None:
            return None
            
        df_clean = self.df.copy()
        
        # Обработка временных меток
        if self.date_column:
            df_clean[self.date_column] = pd.to_datetime(df_clean[self.date_column])
            df_clean = df_clean.sort_values(self.date_column)
        
        # Удаление дубликатов
        df_clean = df_clean.drop_duplicates()
        
        # Обработка пропусков
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col] = df_clean[col].interpolate(method='linear')
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        
        self.processed_df = df_clean
        return df_clean
    
    def create_time_series_plot(self, window_size=30):
        """Создание графика временного ряда с трендом"""
        if self.processed_df is None or self.target_column is None:
            return None
            
        df = self.processed_df.copy()
        
        # Скользящее среднее
        df['rolling_mean'] = df[self.target_column].rolling(window=window_size).mean()
        
        fig = go.Figure()
        
        # Исходный ряд
        fig.add_trace(go.Scatter(
            x=df[self.date_column] if self.date_column else df.index,
            y=df[self.target_column],
            mode='lines',
            name='Исходный ряд',
            line=dict(color='blue', width=1)
        ))
        
        # Скользящее среднее
        fig.add_trace(go.Scatter(
            x=df[self.date_column] if self.date_column else df.index,
            y=df['rolling_mean'],
            mode='lines',
            name=f'Скользящее среднее ({window_size})',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f'Временной ряд: {self.target_column}',
            xaxis_title='Время',
            yaxis_title=self.target_column,
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def create_correlation_heatmap(self):
        """Создание heatmap корреляций"""
        if self.processed_df is None:
            return None
            
        numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.processed_df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Матрица корреляций"
        )
        
        fig.update_layout(height=600)
        return fig
    
    def create_acf_pacf_plots(self, max_lags=50):
        """Создание графиков ACF и PACF"""
        if self.processed_df is None or self.target_column is None:
            return None
            
        series = self.processed_df[self.target_column].dropna()
        
        # ACF
        acf_values = acf(series, nlags=max_lags, alpha=0.05)
        pacf_values = pacf(series, nlags=max_lags, alpha=0.05)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('ACF (Автокорреляционная функция)', 'PACF (Частичная автокорреляционная функция)'),
            vertical_spacing=0.1
        )
        
        # ACF
        lags = range(0, max_lags + 1)
        fig.add_trace(
            go.Bar(x=list(lags), y=acf_values[0], name='ACF', marker_color='blue'),
            row=1, col=1
        )
        
        # Доверительные интервалы для ACF
        fig.add_trace(
            go.Scatter(x=list(lags), y=acf_values[1][:, 0], mode='lines', 
                      line=dict(color='red', dash='dash'), name='Доверительный интервал', showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=list(lags), y=acf_values[1][:, 1], mode='lines', 
                      line=dict(color='red', dash='dash'), name='Доверительный интервал', showlegend=False),
            row=1, col=1
        )
        
        # PACF
        fig.add_trace(
            go.Bar(x=list(lags), y=pacf_values[0], name='PACF', marker_color='green'),
            row=2, col=1
        )
        
        # Доверительные интервалы для PACF
        fig.add_trace(
            go.Scatter(x=list(lags), y=pacf_values[1][:, 0], mode='lines', 
                      line=dict(color='red', dash='dash'), name='Доверительный интервал', showlegend=False),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=list(lags), y=pacf_values[1][:, 1], mode='lines', 
                      line=dict(color='red', dash='dash'), name='Доверительный интервал', showlegend=False),
            row=2, col=1
        )
        
        fig.update_layout(height=800, showlegend=True)
        fig.update_xaxes(title_text="Лаг", row=2, col=1)
        fig.update_yaxes(title_text="ACF", row=1, col=1)
        fig.update_yaxes(title_text="PACF", row=2, col=1)
        
        return fig
    
    def create_decomposition_plot(self, period=52, model='additive'):
        """Создание графика декомпозиции"""
        if self.processed_df is None or self.target_column is None:
            return None
            
        series = self.processed_df[self.target_column].dropna()
        
        try:
            if self.date_column:
                series.index = pd.DatetimeIndex(self.processed_df.loc[series.index, self.date_column])
            
            decomp = seasonal_decompose(series, model=model, period=period)
            
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Исходный ряд', 'Тренд', 'Сезонность', 'Остатки'),
                vertical_spacing=0.05
            )
            
            # Исходный ряд
            fig.add_trace(
                go.Scatter(x=series.index, y=decomp.observed, mode='lines', name='Исходный ряд'),
                row=1, col=1
            )
            
            # Тренд
            fig.add_trace(
                go.Scatter(x=series.index, y=decomp.trend, mode='lines', name='Тренд'),
                row=2, col=1
            )
            
            # Сезонность
            fig.add_trace(
                go.Scatter(x=series.index, y=decomp.seasonal, mode='lines', name='Сезонность'),
                row=3, col=1
            )
            
            # Остатки
            fig.add_trace(
                go.Scatter(x=series.index, y=decomp.resid, mode='lines', name='Остатки'),
                row=4, col=1
            )
            
            fig.update_layout(height=1000, showlegend=False)
            fig.update_xaxes(title_text="Время", row=4, col=1)
            
            return fig
            
        except Exception as e:
            st.error(f"Ошибка при декомпозиции: {e}")
            return None
    
    def perform_stationarity_tests(self):
        """Выполнение тестов стационарности"""
        if self.processed_df is None or self.target_column is None:
            return None, None
            
        series = self.processed_df[self.target_column].dropna()
        
        # Тест Дики-Фуллера
        adf_result = adfuller(series, autolag='AIC')
        
        # Тест KPSS
        try:
            kpss_result = kpss(series, regression='c')
        except:
            kpss_result = None
        
        return adf_result, kpss_result
    
    def create_lag_features(self, lags=[1, 7, 30]):
        """Создание лаговых признаков"""
        if self.processed_df is None or self.target_column is None:
            return None
            
        df = self.processed_df.copy()
        
        for lag in lags:
            df[f'{self.target_column}_lag_{lag}'] = df[self.target_column].shift(lag)
        
        return df
    
    def generate_report(self):
        """Генерация HTML отчета"""
        if self.processed_df is None:
            return None
            
        # Создаем HTML отчет
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Отчет анализа временных рядов</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #333; }}
                .section {{ margin-bottom: 30px; }}
                .stats {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Отчет анализа временных рядов</h1>
            <p><strong>Дата генерации:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Информация о данных</h2>
                <div class="stats">
                    <p><strong>Размер датасета:</strong> {self.processed_df.shape[0]} строк, {self.processed_df.shape[1]} столбцов</p>
                    <p><strong>Целевая переменная:</strong> {self.target_column}</p>
                    <p><strong>Временная колонка:</strong> {self.date_column}</p>
                </div>
            </div>
        """
        
        # Добавляем статистики
        if self.target_column and self.processed_df is not None:
            desc_stats = self.processed_df[self.target_column].describe()
            html_content += f"""
            <div class="section">
                <h2>Описательная статистика</h2>
                <table>
                    <tr><th>Метрика</th><th>Значение</th></tr>
            """
            for stat, value in desc_stats.items():
                html_content += f"<tr><td>{stat}</td><td>{value:.2f}</td></tr>"
            html_content += "</table></div>"
        
        # Тесты стационарности
        adf_result, kpss_result = self.perform_stationarity_tests()
        if adf_result:
            html_content += f"""
            <div class="section">
                <h2>Тесты стационарности</h2>
                <div class="stats">
                    <h3>Тест Дики-Фуллера</h3>
                    <p><strong>ADF статистика:</strong> {adf_result[0]:.6f}</p>
                    <p><strong>p-value:</strong> {adf_result[1]:.6f}</p>
                    <p><strong>Результат:</strong> {'Стационарен' if adf_result[1] < 0.05 else 'Нестационарен'}</p>
            """
            if kpss_result:
                html_content += f"""
                    <h3>Тест KPSS</h3>
                    <p><strong>KPSS статистика:</strong> {kpss_result[0]:.6f}</p>
                    <p><strong>p-value:</strong> {kpss_result[1]:.6f}</p>
                    <p><strong>Результат:</strong> {'Стационарен' if kpss_result[1] > 0.05 else 'Нестационарен'}</p>
                """
            html_content += "</div></div>"
        
        html_content += """
        </body>
        </html>
        """
        
        return html_content

def main():
    st.title("📈 Интерактивный анализ временных рядов")
    st.markdown("---")
    
    # Инициализация приложения
    if 'app' not in st.session_state:
        st.session_state.app = TimeSeriesWebApp()
    
    app = st.session_state.app
    
    # Боковая панель
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Загрузка данных
        st.subheader("📁 Загрузка данных")
        use_sample = st.checkbox("Использовать пример данных", value=True)
        
        if not use_sample:
            uploaded_file = st.file_uploader(
                "Загрузите CSV или Parquet файл",
                type=['csv', 'parquet']
            )
        else:
            uploaded_file = None
        
        # Загрузка данных
        if st.button("🔄 Загрузить данные") or (use_sample and app.df is None):
            with st.spinner("Загрузка данных..."):
                logger.info(f"Начинаем загрузку данных. use_sample={use_sample}, uploaded_file={uploaded_file}")
                success = app.load_data(uploaded_file, use_sample)
                logger.info(f"Результат загрузки: {success}")
                if success:
                    st.success("Данные загружены успешно!")
                    processed_data = app.preprocess_data()
                    if processed_data is not None:
                        st.success("Данные предобработаны успешно!")
                        logger.info("Предобработка завершена успешно")
                    else:
                        st.warning("Ошибка при предобработке данных")
                        logger.warning("Ошибка при предобработке данных")
                else:
                    st.error("Ошибка при загрузке данных")
                    logger.error("Ошибка при загрузке данных")
        
        if app.df is not None:
            st.subheader("🎯 Выбор переменных")
            
            # Выбор колонок
            all_columns = app.df.columns.tolist()
            numeric_columns = app.df.select_dtypes(include=[np.number]).columns.tolist()
            
            app.date_column = st.selectbox(
                "Колонка с датами",
                options=all_columns,
                index=all_columns.index(app.date_column) if app.date_column in all_columns else 0
            )
            
            app.target_column = st.selectbox(
                "Целевая переменная",
                options=numeric_columns,
                index=numeric_columns.index(app.target_column) if app.target_column in numeric_columns else 0
            )
            
            st.subheader("📊 Параметры анализа")
            
            # Параметры для графиков
            window_size = st.slider("Окно скользящего среднего", 1, 100, 30)
            max_lags = st.slider("Максимальное количество лагов для ACF/PACF", 10, 100, 50)
            
            # Параметры декомпозиции
            st.subheader("🔍 Декомпозиция")
            period = st.selectbox("Период сезонности", [7, 30, 52, 365], index=2)
            model_type = st.selectbox("Тип модели", ["additive", "multiplicative"])
            
            # Лаговые признаки
            st.subheader("⏰ Лаговые признаки")
            lag_options = st.multiselect(
                "Выберите лаги",
                [1, 3, 7, 14, 30, 60, 90],
                default=[1, 7, 30]
            )
    
    # Основной контент
    if app.df is None:
        st.info("👈 Загрузите данные в боковой панели для начала анализа")
    else:
        # Информация о данных
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Строк данных", f"{app.df.shape[0]:,}")
        with col2:
            st.metric("Столбцов", app.df.shape[1])
        with col3:
            st.metric("Пропусков", f"{app.df.isnull().sum().sum():,}")
        
        # Проверка предобработанных данных
        if app.processed_df is None:
            st.warning("⚠️ Данные не были предобработаны. Нажмите 'Загрузить данные' для предобработки.")
        
        # Вкладки
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Временной ряд", "🔥 Корреляции", "🔄 ACF/PACF", 
            "🧩 Декомпозиция", "📊 Статистика", "📄 Отчет"
        ])
        
        with tab1:
            st.header("График временного ряда")
            fig = app.create_time_series_plot(window_size)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Статистики
            if app.target_column and app.processed_df is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Основные статистики")
                    stats = app.processed_df[app.target_column].describe()
                    st.dataframe(stats)
                
                with col2:
                    st.subheader("Распределение")
                    fig_hist = px.histogram(
                        app.processed_df, 
                        x=app.target_column, 
                        title="Гистограмма распределения"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
        
        with tab2:
            st.header("Матрица корреляций")
            fig = app.create_correlation_heatmap()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("Анализ автокорреляции")
            fig = app.create_acf_pacf_plots(max_lags)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.header("Декомпозиция временного ряда")
            fig = app.create_decomposition_plot(period, model_type)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            st.header("Тесты стационарности")
            adf_result, kpss_result = app.perform_stationarity_tests()
            
            if adf_result:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Тест Дики-Фуллера")
                    st.write(f"**ADF статистика:** {adf_result[0]:.6f}")
                    st.write(f"**p-value:** {adf_result[1]:.6f}")
                    st.write(f"**Результат:** {'Стационарен' if adf_result[1] < 0.05 else 'Нестационарен'}")
                
                with col2:
                    if kpss_result:
                        st.subheader("Тест KPSS")
                        st.write(f"**KPSS статистика:** {kpss_result[0]:.6f}")
                        st.write(f"**p-value:** {kpss_result[1]:.6f}")
                        st.write(f"**Результат:** {'Стационарен' if kpss_result[1] > 0.05 else 'Нестационарен'}")
        
        with tab6:
            st.header("Генерация отчета")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Сгенерировать HTML отчет"):
                    html_report = app.generate_report()
                    if html_report:
                        st.download_button(
                            label="💾 Скачать HTML отчет",
                            data=html_report,
                            file_name=f"timeseries_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html"
                        )
            
            with col2:
                if st.button("💾 Сохранить обработанные данные"):
                    if app.processed_df is not None:
                        csv = app.processed_df.to_csv(index=False)
                        st.download_button(
                            label="📊 Скачать CSV",
                            data=csv,
                            file_name="final_dataset.csv",
                            mime="text/csv"
                        )
            
            # Предварительный просмотр отчета
            st.subheader("Предварительный просмотр отчета")
            if st.button("👁️ Показать отчет"):
                html_report = app.generate_report()
                if html_report:
                    st.components.v1.html(html_report, height=600, scrolling=True)

if __name__ == "__main__":
    main()
