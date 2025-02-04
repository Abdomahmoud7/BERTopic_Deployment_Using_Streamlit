import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from bertopic import BERTopic
import plotly.express as px

nltk.download("punkt")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r"http\S+", " رابط ", text)
    text = re.sub(r"\d+", " رقم ", text)
    text = re.sub(r"([^\w\s])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    
    words = word_tokenize(text)
    cleaned_text = " ".join([w for w in words if len(w) > 1])
    
    return cleaned_text.lower().strip() if cleaned_text else "no_text"

st.title("Analyze topics using BERTopic")
st.write("Upload a CSV file containing a text column.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        column_name = st.selectbox("Select the column containing the texts", df.columns)
        
        if st.button("Topic analysis"):
            with st.spinner("Data processing and analysis is underway..."):
                try:
                    
                    texts = df[column_name].dropna().astype(str).apply(clean_text).tolist()
                    
                    
                    valid_texts = [t for t in texts if t != "no_text"]
                    if len(valid_texts) < 10:
                        st.error("⚠️ يجب أن يحتوي الملف على 10 نصوص صالحة على الأقل.")
                        st.stop()  
                    
                    
                    topic_model = BERTopic(
                        language="multilingual",
                        calculate_probabilities=False,
                        nr_topics=10
                    )
                    
                    
                    topics, _ = topic_model.fit_transform(valid_texts)
                    
                    
                    topics_df = topic_model.get_topic_info()
                    
                    if topics_df.empty:
                        st.warning("❗ لم يتم العثور على أي مواضيع في البيانات.")
                        st.stop()
                    
                    
                    st.success("Topics were successfully analyzed!✅")
                    
                    st.write("### The most important topics discovered:")
                    st.dataframe(topics_df)
                    
                    
                    topics_list = topics_df.Topic.tolist()
                    if len(topics_list) > 0:
                        st.write("### Distribution of topics")
                        try:
                            fig = topic_model.visualize_barchart(topics=topics_list)
                            st.plotly_chart(fig)
                        except Exception as e:
                            st.warning(f"تعذر عرض الرسم البياني: {str(e)}")
                    else:
                        st.warning("لا توجد مواضيع لعرضها.")
                        
                except ValueError as ve:
                    st.error(f"خطأ في القيم: {str(ve)}")
                    st.stop()
                except scipy.sparse.linalg.ArpackNoConvergence:
                    st.error("تعذر التقارب في خوارزمية التحليل.")
                    st.stop()
                except Exception as e:
                    st.error(f"خطأ غير متوقع: {str(e)}")
                    st.stop()
                    
    except pd.errors.EmptyDataError:
        st.error("الملف المرفوع فارغ أو غير صالح.")
        st.stop()
    except pd.errors.ParserError:
        st.error("تنسيق الملف غير مدعوم. يرجى رفع ملف CSV صالح.")
        st.stop()
    except Exception as e:
        st.error(f"خطأ في قراءة الملف: {str(e)}")
        st.stop()