

from openai import OpenAI

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import pandas as pd
import numpy as np
import yaml
import os



# ---------------------------
# 1. Setup
# ---------------------------

# OPENAI SETUP
os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"
os.environ['OPENAI_API_KEY'] = yaml.safe_load(open('../credentials.yml'))['openai']

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# DATASET 
df = pd.read_csv("temp/customer_churn_ai_ml/data/customer_churn.csv")

# ---------------------------
# 2. Generate Summaries with an LLM
# ---------------------------

# This will take a few minutes and will incur charges to your OpenAI account

def summarize_ticket(ticket_text):
    prompt = f"Summarize the following customer ticket focusing on the main complaint or request:\n\n{ticket_text}\n\nSummary:"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    # Use message.content for chat responses
    return response.choices[0].message.content.strip()

df['ticket_summary'] = df['ticket_notes'].apply(summarize_ticket)

df.to_csv("temp/customer_churn_ai_ml/data/customer_churn_summary.csv", index=False)

df = pd.read_csv("temp/customer_churn_ai_ml/data/customer_churn_summary.csv")
df

# ---------------------------
# 3. Get Embeddings for Summaries
# ---------------------------

# This will take a few minutes and will incur charges to your OpenAI account

def get_embeddings(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    # Access data as attributes instead of dict indexing:
    embedding = response.data[0].embedding
    return embedding

df['summary_embedding'] = df['ticket_summary'].apply(get_embeddings)

df.to_csv("temp/customer_churn_ai_ml/data/customer_churn_summary_embeddings.csv", index=False)

df = pd.read_csv("temp/customer_churn_ai_ml/data/customer_churn_summary_embeddings.csv")

df

# ---------------------------
# 4. Prepare Features
# ---------------------------
df['plan_type_encoded'] = LabelEncoder().fit_transform(df['plan_type'])

import ast

# If embeddings are stored in a pandas Series
df['summary_embedding'] = df['summary_embedding'].apply(ast.literal_eval)

X_base = df[['age','tenure','spend_rate','plan_type_encoded']].values
embeddings_matrix = np.vstack(df['summary_embedding'].values)

X_base = X_base.astype(np.float32)
embeddings_matrix = embeddings_matrix.astype(np.float32)

X = np.hstack([X_base, embeddings_matrix]).astype(np.float32)
y = df['churn'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# ---------------------------
# 5. Train an XGBoost Model
# ---------------------------
# We’ll start with a basic XGBClassifier. You can tune these parameters for better performance.
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_pred_proba))

# ---------------------------
# 6. Optional - Use LLM to Generate Insights
# ---------------------------
report_prompt = f"""
We analyzed customer churn using numeric features (age, tenure, spend_rate, plan_type) and LLM-based embeddings of ticket summaries.
We used an XGBoost model and got an AUC of {roc_auc_score(y_test, y_pred_proba):.2f}.

Summarize what this might mean for the business and how to leverage these insights.
"""

response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=report_prompt,
    max_tokens=200,
    temperature=0.0
)
print("\nLLM Summary of Insights:")
print(response.choices[0].text.strip())
