import streamlit as st
import pandas as pd
import openai
import os
from dotenv import load_dotenv

load_dotenv()

# Load your dataset
df = pd.read_csv(r'data\F_DIS_DS_WHS_half.csv')

# Load HTML content from external file
with open('template.html', 'r') as f:
    html_content = f.read()

# Display HTML content in Streamlit
st.markdown(html_content, unsafe_allow_html=True)

st.subheader("Dataset Preview")
with st.expander("Show Dataset"):
    st.write(df.head(10).to_html(), unsafe_allow_html=True)

st.subheader("Ask a Question:")
user_question = st.text_input("Enter your question:")

if st.button("Submit"):
    try:
        api_key = os.environ["OPENAI_API_KEY"]
        openai.api_key = api_key

        dataset_summary = df.to_dict(orient='records')

        prompt = f"""Here's some data: {dataset_summary}

        Please answer the following question based on the provided data: 
        {user_question}
        """

        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        ai_response = completion.choices[0].message['content']
        st.write("AI Answer:")
        st.write(ai_response)
    except KeyError:
        st.error("Please set the environment variable 'OPENAI_API_KEY' with your API key")
    except Exception as e:
        st.error(f"An error occurred: {e}")
