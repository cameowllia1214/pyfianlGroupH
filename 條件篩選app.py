import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="啡你不可 — 台北咖啡廳推薦",
    layout="centered"
)
df = pd.read_csv('/workspaces/pyfianlGroupH/cafe_detail_with_scores_最終版.csv')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+TC:wght@300;400;600&family=Cormorant+Garamond:ital,wght@0,300;0,400;1,300&display=swap');

.stApp { background-color: #1a0f0a; font-family: 'Noto Serif TC', serif; }
.stApp, .stApp p, .stApp label, .stApp .stMarkdown, .stApp div { color: #f5ede0 !important; }
.stSelectbox > div > div, .stMultiSelect > div > div, .stTextInput > div > div > input {
    background-color: #1f1008 !important; border: 1px solid #3d2510 !important;
    color: #f5ede0 !important; border-radius: 3px !important;
}
.stCheckbox label { color: #f5ede0 !important; }
.stMultiSelect span[data-baseweb="tag"] { background-color: #3d2a10 !important; color: #f5ede0 !important; }
.stMultiSelect > div > div:focus-within, .stSelectbox > div > div:focus-within,
.stTextInput > div > div > input:focus { border-color: #e8d5a0 !important; box-shadow: 0 0 0 1px #e8d5a0 !important; }
.stButton > button {
    background-color: transparent !important; border: 1px solid #8b5e3c !important;
    color: #f5ede0 !important; font-family: 'Noto Serif TC', serif !important;
    letter-spacing: 0.2em !important; padding: 0.7rem 2rem !important;
    min-width: 200px !important; width: auto !important;
    display: block !important; margin: 1rem auto !important;
    transition: all 0.3s ease !important; border-radius: 2px !important;
}
.stButton > button:hover { background-color: #3d2510 !important; border-color: #c4916a !important; }
hr { border-color: #2d1a0e !important; margin: 1.5rem 0 !important; }
.cafe-card {
    background: linear-gradient(135deg, #1f1008 0%, #170d05 100%);
    border: 1px solid #2d1a0e; border-radius: 4px;
    padding: 1.4rem 1.6rem; margin-bottom: 1rem;
    position: relative; overflow: hidden;
}
.cafe-card::before {
    content: ''; position: absolute; top: 0; left: 0;
    width: 3px; height: 100%; background: linear-gradient(180deg, #8b5e3c, #3d2510);
}
.cafe-name { font-family:'Cormorant Garamond',serif; font-size:1.4rem; color:#f5ede0 !important; margin-bottom:0.3rem; }
.cafe-rating { font-size:0.8rem; color:#b09070 !important; letter-spacing:0.1em; }
.cafe-tags { display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.8rem; }
.tag-yes { background:#1a3020; color:#6aad80 !important; border:1px solid #2d5040; padding:0.2rem 0.7rem; border-radius:2px; font-size:0.75rem; }
.tag-no  { background:#1a1010; color:#9a5050 !important; border:1px solid #3d2020; padding:0.2rem 0.7rem; border-radius:2px; font-size:0.75rem; }
.tag-neutral { background:#1a1508; color:#9a8060 !important; border:1px solid #3d3010; padding:0.2rem 0.7rem; border-radius:2px; font-size:0.75rem; }
.cafe-info { margin-top:0.8rem; font-size:0.8rem; color:#9a7a5a !important; line-height:1.8; }
.cafe-hours { margin-top:0.6rem; font-size:0.78rem; color:#7a6040 !important; line-height:1.6; }
.cafe-link { display:inline-block; margin-top:0.6rem; font-size:0.75rem; color:#a07050 !important; letter-spacing:0.1em; text-decoration:none; border-bottom:1px solid #3d2510; }
.result-header { font-size:0.8rem; color:#9a7a5a !important; letter-spacing:0.2em; text-align:center; margin-bottom:1.5rem; padding-bottom:1rem; border-bottom:1px solid #2d1a0e; }
.section-label { font-size:0.8rem; color:#9a7a5a !important; letter-spacing:0.15em; margin-bottom:0.5rem; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ==================
# 讀取資料
# ==================
@st.cache_data
def load_data():
    df = pd.read_csv('/workspaces/pyfianlGroupH/cafe_detail_all.csv')
    def extract_district(address):
        if pd.isna(address):
            return None
        for part in address.split(','):
            part = part.strip()
            if 'District' in part:
                return part
        return None
    df['district'] = df['address'].apply(extract_district)
    return df

df = load_data()

district_map = {
    '不限': '不限', '松山區': 'Songshan District', '大安區': "Da'an District",
    '中正區': 'Zhongzheng District', '中山區': 'Zhongshan District',
    '萬華區': 'Wanhua District', '信義區': 'Xinyi District',
    '大同區': 'Datong District', '文山區': 'Wenshan District',
    '南港區': 'Nangang District', '內湖區': 'Neihu District',
    '士林區': 'Shilin District', '北投區': 'Beitou District',
}
price_labels = {1: '$ 100元以下', 2: '$$ 100-300元', 3: '$$$ 300元以上'}
day_col_map = {
    0: ('Mon_open','Mon_close'), 1: ('Tue_open','Tue_close'),
    2: ('Wed_open','Wed_close'), 3: ('Thur_open','Thur_close'),
    4: ('Fri_open','Fri_close'), 5: ('Sat_open','Sat_close'),
    6: ('Sun_open','Sun_close'),
}
day_zh_map = {
    "今天": datetime.now().weekday(),
    "週一": 0, "週二": 1, "週三": 2, "週四": 3,
    "週五": 4, "週六": 5, "週日": 6,
}
all_day_cols = [
    ("週日","Sun_open","Sun_close"), ("週一","Mon_open","Mon_close"),
    ("週二","Tue_open","Tue_close"), ("週三","Wed_open","Wed_close"),
    ("週四","Thur_open","Thur_close"), ("週五","Fri_open","Fri_close"),
    ("週六","Sat_open","Sat_close"),
]

# ==================
# 標題
# ==================
st.markdown("""
<div style="text-align:center; padding: 2.5rem 0 1.5rem 0;">
    <svg width="120" height="60" viewBox="0 0 120 60" fill="none" xmlns="http://www.w3.org/2000/svg" style="opacity:0.5; margin-bottom:1rem;">
        <ellipse cx="60" cy="30" rx="55" ry="20" stroke="#8b5e3c" stroke-width="0.5" fill="none"/>
        <ellipse cx="60" cy="30" rx="40" ry="14" stroke="#6b4a2c" stroke-width="0.5" fill="none"/>
        <ellipse cx="60" cy="30" rx="25" ry="9"  stroke="#4d3520" stroke-width="0.5" fill="none"/>
        <ellipse cx="60" cy="30" rx="12" ry="4"  stroke="#3d2510" stroke-width="0.5" fill="none"/>
        <path d="M60 10 Q75 20 60 30 Q45 20 60 10Z" stroke="#8b5e3c" stroke-width="0.5" fill="none"/>
        <line x1="60" y1="10" x2="60" y2="30" stroke="#5c3a20" stroke-width="0.3"/>
    </svg>
    <div style="font-family:'Cormorant Garamond',serif; font-size:3rem; font-weight:300; font-style:italic; color:#f5ede0; letter-spacing:0.05em;">啡你不可</div>
    <div style="width:60px; height:1px; background:linear-gradient(90deg,transparent,#8b5e3c,transparent); margin:1rem auto;"></div>
    <div style="font-size:0.8rem; color:#7a5c42; letter-spacing:0.3em;">臺 北 市 咖 啡 廳 推 薦 系 統</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ==================
# 第一階段：篩選條件
# ==================
st.markdown('<p class="section-label">行 政 區</p>', unsafe_allow_html=True)
selected_zh = st.selectbox("", options=list(district_map.keys()), label_visibility="collapsed")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<p class="section-label">需 求 條 件</p>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    want_outlet = st.checkbox("需要插座")
with col2:
    want_wifi = st.checkbox("需要 WiFi")
with col3:
    no_time_limit = st.checkbox("不要時間限制")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<p class="section-label">價 格 區 間</p>', unsafe_allow_html=True)
price_options = st.multiselect("", options=list(price_labels.keys()),
    format_func=lambda x: price_labels[x], default=[], label_visibility="collapsed")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<p class="section-label">營 業 時 間</p>', unsafe_allow_html=True)
time_options = ["不限"] + [f"{h:02d}:00" for h in range(24)]
col_d, col_t1, col_t2 = st.columns(3)
with col_d:
    selected_day = st.selectbox("星期", options=list(day_zh_map.keys()))
with col_t1:
    start_time = st.selectbox("抵達時間", options=time_options)
with col_t2:
    end_time = st.selectbox("離開時間", options=time_options)

search = st.button("搜 尋 咖 啡 廳")

# ==================
# 篩選 + 排序邏輯
# ==================
def render_card(row, open_col, close_col):
    outlet_tag    = '<span class="tag-yes">有插座</span>'    if row.get('outlet') == 'yes'    else '<span class="tag-no">無插座</span>'
    wifi_tag      = '<span class="tag-yes">有 WiFi</span>'   if row.get('wifi') == 'yes'      else '<span class="tag-no">無 WiFi</span>'
    timelimit_tag = '<span class="tag-no">有時間限制</span>'  if row.get('time_limit') == 'yes' else '<span class="tag-yes">無時間限制</span>'
    price         = price_labels.get(row.get('price_level'), '')
    price_tag     = f'<span class="tag-neutral">{price}</span>' if price else ''
    rating        = row.get('rating', '')
    rating_count  = int(row.get('rating_number', 0)) if not pd.isna(row.get('rating_number', 0)) else 0
    station = row.get('nearest_station', '')
    bus     = row.get('nearest_bus', '')
    try:
        sdist = f"{float(row.get('station_distance','')):.0f} 公尺"
    except:
        sdist = ''
    try:
        bdist = f"{float(row.get('bus_distance','')):.0f} 公尺"
    except:
        bdist = ''

    hours_lines = []
    for day_zh, oc, cc in all_day_cols:
        o = row.get(oc)
        c = row.get(cc)
        if not pd.isna(o) and not pd.isna(c):
            hours_lines.append(f"{day_zh}　{str(o)[:5]} – {str(c)[:5]}")
        else:
            hours_lines.append(f"{day_zh}　公休")
    hours_html = "<br>".join(hours_lines)

    url       = row.get('url', '')
    link_html = f'<a href="{url}" target="_blank" class="cafe-link">→ Google Maps</a>' if url else ''

    st.markdown(f"""
    <div class="cafe-card">
        <div class="cafe-name">{row['name']}</div>
        <div class="cafe-rating">★ {rating} &nbsp;·&nbsp; {rating_count} 則評論</div>
        <div class="cafe-tags">{outlet_tag}{wifi_tag}{timelimit_tag}{price_tag}</div>
        <div class="cafe-info">🚇 {station} &nbsp;{sdist}<br>🚌 {bus} &nbsp;{bdist}</div>
        <div class="cafe-hours">🕐 營業時間<br>{hours_html}</div>
        {link_html}
    </div>
    """, unsafe_allow_html=True)


if search:
    result = df.copy()

    if selected_zh != '不限':
        result = result[result['district'] == district_map[selected_zh]]
    if want_outlet:
        result = result[result['outlet'] == 'yes']
    if want_wifi:
        result = result[result['wifi'] == 'yes']
    if no_time_limit:
        result = result[result['time_limit'] != 'yes']
    if price_options:
        result = result[result['price_level'].isin(price_options)]

    today_idx = day_zh_map[selected_day]
    open_col, close_col = day_col_map[today_idx]

    if start_time != "不限" or end_time != "不限":
        def is_open_during(row):
            o = row.get(open_col)
            c = row.get(close_col)
            if pd.isna(o) or pd.isna(c):
                return False
            shop_open  = str(o)[:5]
            shop_close = str(c)[:5]
            if start_time != "不限" and shop_open > start_time:
                return False
            if end_time != "不限" and shop_close < end_time:
                return False
            return True
        result = result[result.apply(is_open_during, axis=1)]

    result = result.reset_index(drop=True)

    # 排出前5名
    result_sorted = result.sort_values(by='total_score', ascending=False).head(5).reset_index(drop=True)

    # 存進 session_state 供第二階段使用
    st.session_state['result']        = result
    st.session_state['result_sorted'] = result_sorted
    st.session_state['open_col']      = open_col
    st.session_state['close_col']     = close_col
    st.session_state['phase']         = 'like'

# ==================
# 第二階段：顯示前5名 + 讓使用者排喜好
# ==================
if st.session_state.get('phase') == 'like':
    result_sorted = st.session_state['result_sorted']
    open_col      = st.session_state['open_col']
    close_col     = st.session_state['close_col']

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f'<div class="result-header">為您推薦前 5 間最符合條件的咖啡廳</div>', unsafe_allow_html=True)

    if len(result_sorted) == 0:
        st.markdown('<p style="text-align:center; color:#7a5c42;">沒有完全符合的咖啡廳，試試放寬條件。</p>', unsafe_allow_html=True)
    else:
        for _, row in result_sorted.iterrows():
            render_card(row, open_col, close_col)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">告 訴 我 們 你 的 喜 好</p>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.8rem; color:#7a5c42;">從上方 5 間中，選出你最喜歡的 1～3 間（可不選）</p>', unsafe_allow_html=True)

        cafe_names = ["（不選）"] + result_sorted['name'].tolist()

        col_l1, col_l2, col_l3 = st.columns(3)
        with col_l1:
            like1 = st.selectbox("最喜歡", options=cafe_names)
        with col_l2:
            like2 = st.selectbox("第二喜歡", options=cafe_names)
        with col_l3:
            like3 = st.selectbox("第三喜歡", options=cafe_names)

        recommend = st.button("根 據 喜 好 再 推 薦 5 間")

        if recommend:
            result        = st.session_state['result']
            result_sorted = st.session_state['result_sorted'].copy()

            likes = [like1, like2, like3]
            result_sorted['new_score'] = np.nan

            for i in range(len(result_sorted)):
                name = result_sorted['name'].iloc[i]
                if name in likes:
                    idx = likes.index(name)
                    result_sorted.loc[result_sorted.index[i], 'new_score'] = 10 - idx
                else:
                    result_sorted.loc[result_sorted.index[i], 'new_score'] = 7

            # 訓練 Random Forest
            features = ['rating', 'score_outlet', 'score_wifi', 'score_time_limit', 'station_distance']
            X_train = result_sorted[features]
            y_train = result_sorted['new_score']
            model = RandomForestClassifier(random_state=30)
            model.fit(X_train, y_train)

            # 從剩下的店預測
            remaining = result[~result['name'].isin(result_sorted['name'])].copy()
            X_test = remaining[features]
            remaining['new_score'] = model.predict(X_test)
            new_result = remaining.sort_values(by='new_score', ascending=False).head(5).reset_index(drop=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<div class="result-header">根據您的喜好，為您推薦另外 5 間</div>', unsafe_allow_html=True)
            for _, row in new_result.iterrows():
                render_card(row, open_col, close_col)