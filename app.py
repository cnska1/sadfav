import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import plotly.express as px

# 페이지 설정
st.set_page_config(
    page_title="대학원 합격 가능성 예측 시스템",
    page_icon="🎓",
    layout="wide"
)

# 제목
st.title("🎓 대학원 합격 가능성 예측 시스템")
st.markdown("---")

# 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_csv('Admission_Predict_Ver1.1.csv')
    df = df.drop('Serial No.', axis=1)
    
    # GRE와 TOEFL 점수 정규화 (0-1 사이로)
    df['GRE Score'] = (df['GRE Score'] - df['GRE Score'].min()) / (df['GRE Score'].max() - df['GRE Score'].min())
    df['TOEFL Score'] = (df['TOEFL Score'] - df['TOEFL Score'].min()) / (df['TOEFL Score'].max() - df['TOEFL Score'].min())
    
    df['Admission'] = (df['Chance of Admit '] >= 0.75).astype(int)
    return df

df = load_data()

# 사이드바 - 사용자 입력
st.sidebar.header("📝 지원자 정보 입력")

# 점수 범위 설정
gre_min, gre_max = 260, 340  # GRE 점수 범위
toefl_min, toefl_max = 0, 550  # TOEFL 점수 범위 수정

gre = st.sidebar.slider("GRE 점수 (높을수록 합격 가능성 ↑)", 
                       float(gre_min), 
                       float(gre_max), 
                       320.0)

toefl = st.sidebar.slider("TOEFL 점수 (0~550점, 높을수록 합격 가능성 ↑)", 
                         float(toefl_min), 
                         float(toefl_max), 
                         250.0)  # 기본값 수정

# 점수 정규화 함수
def normalize_scores(gre_score, toefl_score):
    gre_normalized = (gre_score - gre_min) / (gre_max - gre_min)
    toefl_normalized = (toefl_score - toefl_min) / (toefl_max - toefl_min)
    return gre_normalized, toefl_normalized

gre_norm, toefl_norm = normalize_scores(gre, toefl)

univ_rating = st.sidebar.selectbox("대학 등급 (1등급이 합격 가능성 최고)", [1, 2, 3, 4, 5])
sop = st.sidebar.slider("자기소개서 평가 (높을수록 합격 가능성 ↑)", 1.0, 5.0, 3.5)
lor = st.sidebar.slider("추천서 평가 (높을수록 합격 가능성 ↑)", 1.0, 5.0, 3.5)
cgpa = st.sidebar.slider("학부 CGPA (높을수록 합격 가능성 ↑)", float(df['CGPA'].min()), float(df['CGPA'].max()), 8.5)
research = st.sidebar.selectbox("연구 경험 (있을 경우 합격 가능성 ↑)", [0, 1], format_func=lambda x: "있음" if x == 1 else "없음")

# 대학 등급 역변환
univ_rating_reversed = 6 - univ_rating

# 모델 학습
def train_model(df):
    X = df.drop(['Chance of Admit ', 'Admission'], axis=1)
    y = df['Admission']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(df)

# 예측
input_data = np.array([[gre_norm, toefl_norm, univ_rating_reversed, sop, lor, cgpa, research]])
prediction_proba = model.predict_proba(input_data)[0][1]

# 결과 표시
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 합격 가능성 분석")
    
    # 게이지 차트
    fig = px.pie(values=[prediction_proba, 1-prediction_proba], 
                 names=['합격 가능성', '보완 필요'],
                 hole=0.7,
                 color_discrete_sequence=['#00ff00' if prediction_proba >= 0.75 else '#ff0000', '#808080'])
    fig.update_layout(
        annotations=[dict(text=f'{prediction_proba:.1%}', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    st.plotly_chart(fig)

    if prediction_proba >= 0.75:
        st.success("🎉 합격 가능성이 높습니다!")
    else:
        st.warning("⚠️ 합격 가능성을 높이기 위한 보완이 필요합니다.")

with col2:
    st.subheader("📈 합격 가능성 영향 요소 분석")
    
    # 레이더 차트 데이터
    categories = ['GRE 점수', 'TOEFL 점수', '대학 등급', '자기소개서', '추천서', 'CGPA']
    max_values = [gre_max-gre_min, toefl_max-toefl_min, 5, 5, 5, 10]
    values = [gre-gre_min, toefl-toefl_min, univ_rating_reversed, sop, lor, cgpa]
    
    # 정규화
    normalized_values = [v/m for v, m in zip(values, max_values)]
    
    # 레이더 차트
    fig = px.line_polar(
        r=normalized_values + [normalized_values[0]],
        theta=categories + [categories[0]],
        line_close=True
    )
    fig.update_traces(fill='toself')
    st.plotly_chart(fig)

# 통계 정보
st.markdown("---")
st.subheader("📊 전체 지원자 합격 가능성 통계")
col3, col4 = st.columns(2)

with col3:
    fig = px.histogram(df, x='Chance of Admit ', 
                      title='전체 지원자 합격 가능성 분포',
                      labels={'Chance of Admit ': '합격 가능성'})
    st.plotly_chart(fig)

with col4:
    fig = px.scatter(df, x='CGPA', y='Chance of Admit ', 
                    title='학부 성적(CGPA)과 합격 가능성의 관계',
                    labels={'CGPA': 'CGPA', 'Chance of Admit ': '합격 가능성'})
    st.plotly_chart(fig)
