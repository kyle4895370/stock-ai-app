import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.ensemble import RandomForestRegressor

# 1. 頁面設定
st.set_page_config(page_title="股價操盤模擬器", layout="centered") # 改用 centered 比較像計算機
st.title("🎛️ 股價情境模擬器：如果明天...")

# 2. 側邊欄：選擇股票
st.sidebar.header("基礎設定")
ticker = st.sidebar.text_input("股票代碼", value="2330.TW")

# 3. 抓取與整理資料 (個股 + 大盤)
@st.cache_data
def get_data(ticker):
    # 抓過去 1 年的資料來訓練
    start = date.today() - timedelta(days=365)
    end = date.today()
    
    try:
        # 抓個股
        stock = yf.download(ticker, start=start, end=end)
        stock.reset_index(inplace=True)
        # 處理 yfinance 可能的 MultiIndex 格式
        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = stock.columns.get_level_values(0)
            
        # 抓大盤 (^TWII)
        market = yf.download("^TWII", start=start, end=end)
        market.reset_index(inplace=True)
        if isinstance(market.columns, pd.MultiIndex):
            market.columns = market.columns.get_level_values(0)

        # 資料合併與清洗
        df = pd.merge(stock[['Date', 'Close', 'Volume']], 
                      market[['Date', 'Close']], 
                      on='Date', suffixes=('_Stock', '_Market'))
        
        # 特徵工程：我們用「昨天的數據」預測「今天的價格」
        # 下面這行意思是：把今天的收盤價，對齊昨天的成交量跟大盤
        df['Target_Price'] = df['Close_Stock'] # 目標是當日收盤價
        df['Prev_Close'] = df['Close_Stock'].shift(1) # 昨日收盤
        df['Prev_Volume'] = df['Volume'].shift(1)     # 昨日量
        df['Prev_Market'] = df['Close_Market'].shift(1) # 昨日大盤
        
        df.dropna(inplace=True) # 刪除第一筆空值
        return df
    except:
        return None

df = get_data(ticker)

if df is not None:
    # 4. 訓練 AI 模型 (隨機森林)
    X = df[['Prev_Close', 'Prev_Volume', 'Prev_Market']]
    y = df['Target_Price']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 取得「最近一天」的真實數據，當作基準
    last_close = df['Prev_Close'].iloc[-1]
    last_vol = df['Prev_Volume'].iloc[-1]
    last_market = df['Prev_Market'].iloc[-1]
    
    st.write(f"### 📍 目前 {ticker} 基準價格：{last_close:.2f}")
    st.write("請在下方調整參數，模擬明天的狀況：")
    
    # 5. 使用者輸入 (模擬器介面)
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("📊 模擬大盤環境")
        # 讓使用者輸入「大盤指數」
        market_input = st.number_input("預測大盤指數 (點)", 
                                       value=float(last_market), 
                                       step=50.0)
        # 計算大盤漲跌幅給使用者參考
        market_change = ((market_input - last_market) / last_market) * 100
        st.caption(f"大盤漲跌幅模擬: {market_change:+.2f}%")

    with col2:
        st.warning("💰 模擬資金動能 (外資/主力)")
        # 讓使用者輸入「成交量」
        # 預設值是昨天的量，範圍可以拉大到 3 倍
        vol_input = st.slider("預測成交量 (張/股)", 
                              min_value=int(last_vol * 0.5), 
                              max_value=int(last_vol * 3.0), 
                              value=int(last_vol))
        
        # 判斷是否爆量
        vol_ratio = vol_input / last_vol
        if vol_ratio > 1.5:
            st.caption("🔥 爆大量！模擬外資或主力積極進場")
        elif vol_ratio < 0.8:
            st.caption("🧊 量縮，市場觀望中")
        else:
            st.caption("☁️ 成交量持平")

    # 6. AI 進行預測
    # 這裡很關鍵：我們把「現在的價格」當作「昨日價格」餵給模型
    # 加上使用者設定的「模擬大盤」與「模擬成交量」
    prediction = model.predict([[last_close, vol_input, market_input]])[0]
    
    pred_change = prediction - last_close
    pred_pct = (pred_change / last_close) * 100

    # 7. 顯示結果
    st.divider()
    st.subheader("🔮 AI 模擬結果")
    
    # 用大字體顯示預測價格
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        st.metric(label="預測收盤價", 
                  value=f"{prediction:.2f}", 
                  delta=f"{pred_change:.2f} ({pred_pct:.2f}%)")
    
    with col_res2:
        if pred_pct > 2:
            st.success("🚀 AI 判斷：強勢上漲！這組參數對股價非常有利。")
        elif pred_pct < -2:
            st.error("📉 AI 判斷：賣壓沉重，小心下跌。")
        else:
            st.info("⚖️ AI 判斷：盤整波動，方向不明確。")

else:
    st.error("找不到資料，請確認代碼 (台股請加 .TW)")