import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import feedparser
import io  # <--- 新增：用於處理檔案下載
from datetime import date, timedelta, datetime
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- 1. 全局設定 ---
st.set_page_config(page_title="AI 股市全能戰情室", layout="wide", page_icon="📈")

# --- 2. 側邊欄：主控台 ---
st.sidebar.title("🎛️ 戰情室控制台")
app_mode = st.sidebar.radio("選擇功能模組", ["🔮 未來 K 線推演 (90天)", "🔬 趨勢預測實驗室", "🎛️ 操盤手情境模擬"])
st.sidebar.markdown("---")
st.sidebar.header("股票參數設定")
ticker = st.sidebar.text_input("輸入股票代碼", value="2330.TW")
history_months = st.sidebar.slider("歷史資料長度 (月)", 3, 60, 6) 
history_years = history_months / 12 

# --- 新聞抓取函數 ---
def get_stock_news(stock_name):
    rss_url = f"https://news.google.com/rss/search?q={stock_name}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    feed = feedparser.parse(rss_url)
    return feed.entries[:5]

# --- 資料讀取函數 ---
@st.cache_data
def load_data(ticker, years):
    start_date = date.today() - timedelta(days=years*365)
    end_date = date.today()
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        df.reset_index(inplace=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        needed_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in needed_cols): return None
        return df
    except: return None

data = load_data(ticker, history_years)
if data is None or data.empty:
    st.error(f"❌ 找不到股票代碼 {ticker} 的資料。")
    st.stop()

# ==========================================
# 功能模組 1: 未來 K 線推演
# ==========================================
if app_mode == "🔮 未來 K 線推演 (90天)":
    st.title(f"🔮 {ticker} 未來 90 天 K 線推演")
    st.info("💡 說明：AI 預測「收盤價趨勢」，並結合歷史波動率生成「模擬 K 棒」。")
    
    with st.spinner("AI 運算中..."):
        df_prophet = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        m = Prophet(daily_seasonality=True)
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=90)
        forecast = m.predict(future)
        
        data['H_L'] = data['High'] - data['Low']
        avg_volatility = data['H_L'].mean()
        
        future_data = forecast[['ds', 'yhat']].tail(90).copy()
        future_data.columns = ['Date', 'Pred_Close']
        
        np.random.seed(42)
        future_opens, future_highs, future_lows = [], [], []
        future_closes = future_data['Pred_Close'].values
        last_close = data['Close'].iloc[-1]
        
        for i in range(90):
            current_close = future_closes[i]
            open_price = last_close if i == 0 else future_closes[i-1] * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, current_close) + abs(np.random.normal(avg_volatility * 0.5, avg_volatility * 0.2))
            low_price = min(open_price, current_close) - abs(np.random.normal(avg_volatility * 0.5, avg_volatility * 0.2))
            future_opens.append(open_price); future_highs.append(high_price); future_lows.append(low_price)

        future_data['Open'] = future_opens; future_data['High'] = future_highs; future_data['Low'] = future_lows; future_data['Close'] = future_closes

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=recent['Date'], open=recent['Open'], high=recent['High'], low=recent['Low'], close=recent['Close'], name='歷史'))
        fig.add_trace(go.Candlestick(x=future_data['Date'], open=future_data['Open'], high=future_data['High'], low=future_data['Low'], close=future_data['Close'], name='預測', increasing_line_color='cyan', decreasing_line_color='gray'))
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 功能模組 2: 趨勢預測實驗室
# ==========================================
elif app_mode == "🔬 趨勢預測實驗室":
    st.title("🔬 AI 預測實驗室")
    predict_days = st.slider("預測天數", 30, 180, 90)
    
    with st.spinner("AI 模型競賽中..."):
        # Prophet
        df_p = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        m = Prophet(daily_seasonality=True); m.fit(df_p)
        p1 = m.predict(m.make_future_dataframe(periods=predict_days))['yhat'].values[-predict_days:]
        
        # Linear Reg
        data['Ordinal'] = pd.to_datetime(data['Date']).map(pd.Timestamp.toordinal)
        lr = LinearRegression().fit(data[['Ordinal']], data['Close'])
        last_ord = data['Ordinal'].iloc[-1]
        p2 = lr.predict(np.array([last_ord + i for i in range(1, predict_days + 1)]).reshape(-1, 1))
        
        # Holt-Winters
        p3 = ExponentialSmoothing(data['Close'], trend='add', seasonal=None).fit().forecast(predict_days).values
        
        future_dates = [data['Date'].iloc[-1] + timedelta(days=x) for x in range(1, predict_days + 1)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'].tail(180), y=data['Close'].tail(180), name="歷史", line=dict(color='black')))
        fig.add_trace(go.Scatter(x=future_dates, y=p1, name="Prophet", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=future_dates, y=p2, name="Linear Reg", line=dict(color='green', dash='dot')))
        fig.add_trace(go.Scatter(x=future_dates, y=p3, name="Holt-Winters", line=dict(color='orange', dash='dash')))
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 功能模組 3: 操盤手模擬器 (含存檔功能)
# ==========================================
elif app_mode == "🎛️ 操盤手情境模擬":
    st.title("🎛️ 股價情境模擬器 (含報告下載)")
    
    @st.cache_data
    def load_market():
        try:
            m = yf.download("^TWII", start=data['Date'].iloc[0], end=date.today())
            m.reset_index(inplace=True)
            if isinstance(m.columns, pd.MultiIndex): m.columns = m.columns.get_level_values(0)
            return m[['Date', 'Close']].rename(columns={'Close': 'Market_Close'})
        except: return None

    market_df = load_market()
    if market_df is not None:
        df_sim = pd.merge(data[['Date', 'Close', 'Volume']], market_df, on='Date', how='inner')
        df_sim['Target'] = df_sim['Close']; df_sim['Prev_Close'] = df_sim['Close'].shift(1)
        df_sim['Prev_Vol'] = df_sim['Volume'].shift(1); df_sim['Prev_Market'] = df_sim['Market_Close'].shift(1)
        df_sim.dropna(inplace=True)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(df_sim[['Prev_Close', 'Prev_Vol', 'Prev_Market']], df_sim['Target'])
        
        last_close = df_sim['Prev_Close'].iloc[-1]
        last_vol = df_sim['Prev_Vol'].iloc[-1]
        last_market = df_sim['Prev_Market'].iloc[-1]
    else: st.stop()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🛠️ 參數模擬")
        sim_market = st.number_input("預測大盤", value=float(last_market), step=50.0)
        sim_vol = st.slider("預測成交量", int(last_vol*0.5), int(last_vol*3), int(last_vol))
        
        st.subheader("📰 消息面修正")
        news_input = st.text_area("貼上新聞標題", placeholder="AI 自動分析...")
        sentiment_adj = st.slider("手動調整衝擊 (%)", -10, 10, 0)
        
        sentiment_score = 0.0
        kw_list = []
        if news_input:
            keywords = {'撤資': -0.04, '賣出': -0.03, '看空': -0.03, '下修': -0.02, '大跌': -0.03, '買進': 0.03, '看多': 0.03, '上修': 0.02, '大漲': 0.03, '新高': 0.04}
            for kw, score in keywords.items():
                if kw in news_input: 
                    sentiment_score += score
                    kw_list.append(kw)
        
        final_sentiment = 1 + (sentiment_adj / 100) + sentiment_score

    with col2:
        st.subheader(f"📢 {ticker} 新聞")
        try: news_items = get_stock_news(ticker.replace(".TW", ""))
        except: news_items = []
        if news_items:
            for item in news_items:
                with st.expander(item.title):
                    st.write(item.get('published', '')); st.write(f"[閱讀]({item.link})")
        else: st.write("無新聞")

    ai_price = rf.predict([[last_close, sim_vol, sim_market]])[0]
    final_price = ai_price * final_sentiment
    final_chg = (final_price - last_close) / last_close * 100
    
    st.divider()
    st.metric("🔮 最終預測", f"{final_price:.2f}", f"{final_chg:.2f}%")
    
    # --- 新增：生成與下載報告功能 ---
    st.write("---")
    st.subheader("💾 存檔與記錄")
    
    # 準備報告內容 (文字格式)
    report_text = f"""
    【AI 股市戰情室 - 每日分析報告】
    --------------------------------
    日期: {date.today()}
    股票代碼: {ticker}
    --------------------------------
    [模擬參數]
    - 基準股價: {last_close:.2f}
    - 預測大盤: {sim_market:.2f}
    - 預測成交量: {sim_vol}
    
    [消息面分析]
    - 輸入新聞: {news_input if news_input else "無"}
    - 偵測關鍵字: {", ".join(kw_list) if kw_list else "無"}
    - 綜合情緒修正: {(final_sentiment - 1)*100:.1f}%
    
    [最終預測結果]
    - AI 原始預測: {ai_price:.2f}
    - 最終模擬股價: {final_price:.2f}
    - 預期漲跌幅: {final_chg:.2f}%
    --------------------------------
    (本報告由 AI 自動生成，僅供參考)
    """
    
    # 下載按鈕
    st.download_button(
        label="📥 下載今日分析報告 (TXT)",
        data=report_text,
        file_name=f"{ticker}_分析報告_{date.today()}.txt",
        mime="text/plain"
    )