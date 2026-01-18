import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# --- 頁面設定 ---
st.set_page_config(page_title="AI 股市多因子分析", layout="wide")
st.title("🧠 AI 股市多因子分析：個股 vs 大盤")

# --- 側邊欄 ---
st.sidebar.header("設定")
ticker = st.sidebar.text_input("股票代碼", value="2330.TW")
history_years = st.sidebar.slider("資料長度(年)", 1, 5, 2)

# --- 抓取資料函數 (同步抓個股與大盤) ---
@st.cache_data
def load_data_with_market(ticker, years):
    start_date = date.today() - timedelta(days=years*365)
    end_date = date.today()
    
    try:
        # 1. 抓個股
        stock_df = yf.download(ticker, start=start_date, end=end_date)
        stock_df.reset_index(inplace=True)
        if isinstance(stock_df.columns, pd.MultiIndex):
             stock_df.columns = stock_df.columns.get_level_values(0)
        
        # 2. 抓大盤 (加權指數 ^TWII)
        market_df = yf.download("^TWII", start=start_date, end=end_date)
        market_df.reset_index(inplace=True)
        if isinstance(market_df.columns, pd.MultiIndex):
             market_df.columns = market_df.columns.get_level_values(0)

        # 3. 資料整理
        stock_df = stock_df[['Date', 'Close', 'Volume']]
        stock_df.columns = ['Date', 'Stock_Close', 'Volume']
        
        market_df = market_df[['Date', 'Close']]
        market_df.columns = ['Date', 'Market_Close']

        # 4. 合併資料 (用日期對齊)
        merged_df = pd.merge(stock_df, market_df, on='Date', how='inner')
        return merged_df

    except Exception as e:
        return None

# --- 載入資料 ---
data = load_data_with_market(ticker, history_years)

if data is not None and not data.empty:
    
    # 建立分頁
    tab1, tab2, tab3 = st.tabs(["📊 個股與大盤關聯", "🤖 隨機森林預測", "🧠 AI 關注點分析"])

    # === Tab 1: 關聯性分析 ===
    with tab1:
        st.subheader(f"{ticker} 與 加權指數(大盤) 的走勢對比")
        
        # 雙軸圖表
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Stock_Close'], name=f"{ticker} 股價", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Market_Close'], name="加權指數 (大盤)", line=dict(color='red'), yaxis="y2"))
        
        fig.update_layout(
            yaxis=dict(title="個股價格"),
            yaxis2=dict(title="大盤指數", overlaying="y", side="right"),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # 計算相關係數
        correlation = data['Stock_Close'].corr(data['Market_Close'])
        
        st.write("### 🔗 關聯度分析")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("相關係數 (Correlation)", f"{correlation:.2f}")
        with col2:
            if correlation > 0.8:
                st.success("高度正相關！這支股票非常容易受大盤漲跌影響 (隨波逐流型)。")
            elif correlation > 0.5:
                st.info("中度正相關。股票走勢與大盤有一定連動。")
            elif correlation > -0.5:
                st.warning("低相關或脫鉤。這支股票走勢比較「做自己」，不太理會大盤。")
            else:
                st.error("負相關！大盤漲它反而跌，通常是避險股或反向ETF。")

    # === Tab 2: 隨機森林預測 (Random Forest) ===
    with tab2:
        st.subheader("🌲 隨機森林 (Random Forest) 多因子預測")
        st.write("這個模型會同時考慮 **「昨天的股價」**、**「昨天的成交量」** 與 **「昨天的大盤指數」** 來預測今天的股價。")

        # 特徵工程 (Feature Engineering)
        # 我們要用 "T-1 (昨天)" 的資料來預測 "T (今天)" 的收盤價
        df_ml = data.copy()
        df_ml['Prev_Close'] = df_ml['Stock_Close'].shift(1)
        df_ml['Prev_Volume'] = df_ml['Volume'].shift(1)
        df_ml['Prev_Market'] = df_ml['Market_Close'].shift(1)
        df_ml.dropna(inplace=True) # 移除第一筆空值

        # 設定 X (特徵) 與 y (目標)
        X = df_ml[['Prev_Close', 'Prev_Volume', 'Prev_Market']]
        y = df_ml['Stock_Close']

        # 切割訓練集與測試集 (最後 30 天當測試)
        split_idx = len(df_ml) - 30
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        test_dates = df_ml['Date'].iloc[split_idx:]

        # 訓練模型
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # 預測
        y_pred = rf_model.predict(X_test)
        
        # 計算誤差
        mae = mean_absolute_error(y_test, y_pred)

        # 繪圖
        fig_rf = go.Figure()
        fig_rf.add_trace(go.Scatter(x=test_dates, y=y_test, mode='lines', name='真實股價', line=dict(color='black', width=3)))
        fig_rf.add_trace(go.Scatter(x=test_dates, y=y_pred, mode='lines', name='AI 預測股價', line=dict(color='green', dash='dash')))
        st.plotly_chart(fig_rf, use_container_width=True)
        
        st.caption(f"模型誤差 (MAE): {mae:.2f} (代表平均預測誤差約為 {mae:.2f} 元)")

    # === Tab 3: 特徵重要性 (Feature Importance) ===
    with tab3:
        st.subheader("🧐 AI 到底看重什麼？")
        st.write("這是機器學習最有趣的地方：我們可以問模型，在預測股價時，哪個因素最重要？")

        # 提取重要性
        importance = rf_model.feature_importances_
        feature_names = ['前一日股價', '前一日成交量', '前一日大盤指數']
        
        # 繪製長條圖
        fig_imp = px.bar(x=importance, y=feature_names, orientation='h', 
                         labels={'x': '重要性分數', 'y': '影響因子'},
                         title="影響股價預測的關鍵因子權重",
                         color=importance, color_continuous_scale='Viridis')
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.info("""
        **如何解讀？**
        * 如果 **「前一日股價」** 分數最高：代表這支股票有很強的慣性（強者恆強）。
        * 如果 **「前一日大盤指數」** 分數很高：代表這支股票是標準的「權值股」，基本上跟著大盤走（如台積電）。
        * 如果 **「前一日成交量」** 分數高：代表這支股票可能是「量先價行」，主力進出對股價影響很大。
        """)

else:
    st.error("無法抓取資料，請確認網路連線或股票代碼。")