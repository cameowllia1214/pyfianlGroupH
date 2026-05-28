import streamlit as st
import pandas as pd

st.title("☕ 台北咖啡廳推薦系統")
st.caption("請選擇您的需求，我們幫您找到最合適的咖啡廳")

# ==================
# 讀取資料
# ==================
@st.cache_data
def load_data():
    df = pd.read_csv('/workspaces/pyfianlGroupH/cafe_detail_all.csv')

    # 抓出行政區
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

# ==================
# 行政區中英對照
# ==================
district_map = {
    '不限':                    '不限',
    '松山區': 'Songshan District',
    '大安區': "Da'an District",
    '中正區': 'Zhongzheng District',
    '中山區': 'Zhongshan District',
    '萬華區': 'Wanhua District',
    '信義區': 'Xinyi District',
    '大同區': 'Datong District',
    '文山區': 'Wenshan District',
    '南港區': 'Nangang District',
    '內湖區': 'Neihu District',
    '士林區': 'Shilin District',
    '北投區': 'Beitou District',
}

# ==================
# 欄位中文對照
# ==================
column_labels = {
    'name':             '店名',
    'rating':           '評分',
    'rating_number':    '評論數',
    'price_level':      '價格區間',
    'outlet':           '插座',
    'wifi':             'WiFi',
    'time_limit':       '時間限制',
    'nearest_station':  '最近捷運站',
    'station_distance': '捷運站距離（公尺）',
    'nearest_bus':      '最近公車站',
    'bus_distance':     '公車站距離（公尺）',
    'url':              'Google Maps',
}

# ==================
# 價格區間對照
# ==================
price_labels = {
    1: '$ 便宜',
    2: '$$ 普通',
    3: '$$$ 較貴',
    4: '$$$$ 高級',
}

# ==================
# 側邊欄：使用者輸入條件
# ==================
with st.sidebar:
    st.header("🔍 搜尋條件")

    # 行政區下拉
    selected_zh = st.selectbox(
        "行政區",
        options=list(district_map.keys())
    )

    st.divider()
    st.subheader("需求條件")

    want_outlet    = st.checkbox("需要插座")
    want_wifi      = st.checkbox("需要 WiFi")
    no_time_limit  = st.checkbox("不要有時間限制")

    st.divider()
    st.subheader("價格區間")
    price_options = st.multiselect(
        "可接受的價格（可複選）",
        options=list(price_labels.keys()),
        format_func=lambda x: price_labels[x],
        default=[]
    )

    st.divider()
    st.subheader("營業時間")
    col1, col2 = st.columns(2)
    with col1:
        start_time = st.text_input("想去時間", placeholder="14:00")
    with col2:
        end_time = st.text_input("離開時間", placeholder="17:00")

    search_button = st.button("搜尋咖啡廳", type="primary", use_container_width=True)

# ==================
# 篩選邏輯
# ==================
if search_button:
    result = df.copy()

    # 行政區篩選
    if selected_zh != '不限':
        selected_en = district_map[selected_zh]
        result = result[result['district'] == selected_en]

    # 插座篩選
    if want_outlet:
        result = result[result['outlet'] == 'yes']

    # WiFi 篩選
    if want_wifi:
        result = result[result['wifi'] == 'yes']

    # 時間限制篩選
    if no_time_limit:
        result = result[result['time_limit'] != 'yes']

    # 價格篩選
    if price_options:
        result = result[result['price_level'].isin(price_options)]

    # 結果顯示
    st.subheader(f"找到 {len(result)} 間符合條件的咖啡廳")

    if len(result) == 0:
        st.warning("沒有符合所有條件的咖啡廳，試試放寬條件看看！")
    else:
        # 整理顯示欄位
        display_df = result[[
            'name', 'rating', 'rating_number', 'price_level',
            'outlet', 'wifi', 'time_limit',
            'nearest_station', 'station_distance',
            'nearest_bus', 'bus_distance',
            'url'
        ]].copy()

        # 價格數字轉文字
        display_df['price_level'] = display_df['price_level'].map(price_labels)

        # 欄位改成中文
        display_df.columns = [column_labels[c] for c in display_df.columns]

        # 距離四捨五入
        display_df['捷運站距離（公尺）'] = display_df['捷運站距離（公尺）'].round(0)
        display_df['公車站距離（公尺）'] = display_df['公車站距離（公尺）'].round(0)

        st.dataframe(
            display_df.reset_index(drop=True),
            use_container_width=True,
            column_config={
                "Google Maps": st.column_config.LinkColumn("Google Maps")
            }
        )

        # 傳給後面組員用的原始資料（含所有欄位）
        st.divider()
        with st.expander("📋 完整資料（給推薦排序用）"):
            st.dataframe(result.reset_index(drop=True), use_container_width=True)

else:
    st.info("請在左側選擇條件後按下「搜尋咖啡廳」")