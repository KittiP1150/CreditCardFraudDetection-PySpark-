import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import pandas as pd
import os
import threading

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

RESULT_PATH = "result/fraud_clustering_results.parquet"
CSV_PATH = "result/fraud_only.csv"

# ---------------- TRAIN ----------------
def run_training():
    try:
        subprocess.run([sys.executable, "src/app.py"], check=True)
        messagebox.showinfo("Success", "Training completed")
        load_dashboard()
    except Exception as e:
        messagebox.showerror("Error", str(e))

def train_thread():
    threading.Thread(target=run_training).start()

# ---------------- LOAD DATA ----------------
def load_data():
    if os.path.exists(RESULT_PATH):
        return pd.read_parquet(RESULT_PATH)
    return None

# ---------------- SAVE FRAUD CSV ----------------
def save_fraud_csv(df):
    if "predicted_fraud_final" not in df.columns:
        print("No fraud column found, skip saving CSV")
        return

    fraud_df = df[df["predicted_fraud_final"] == 1]

    os.makedirs("result", exist_ok=True)
    fraud_df.to_csv(CSV_PATH, index=False)

    print(f"Saved fraud CSV: {CSV_PATH} ({len(fraud_df)} rows)")

# ---------------- DASHBOARD ----------------
def load_dashboard():
    df = load_data()
    if df is None:
        return

    # auto detect column
    if "predicted_fraud_final" in df.columns:
        fraud_col = "predicted_fraud_final"
    elif "predicted_fraud" in df.columns:
        fraud_col = "predicted_fraud"
    else:
        messagebox.showerror("Error", "No fraud column found")
        return

    # KPI
    total = len(df)
    fraud = int(df[fraud_col].sum())
    fraud_rate = fraud / total * 100

    total_label.config(text=f"Total: {total}")
    fraud_label.config(text=f"Fraud: {fraud}")
    rate_label.config(text=f"Fraud %: {fraud_rate:.2f}%")

    # TABLE
    for row in tree.get_children():
        tree.delete(row)

    df_sorted = df.sort_values("risk_score", ascending=False).head(30)

    for _, row in df_sorted.iterrows():
        tree.insert("", "end", values=(
            int(row["Time"]),
            round(row["Amount"], 2),
            row["risk_score"],
            row.get("reason", "-")
        ))

    # PLOT
    fig.clear()

    ax1 = fig.add_subplot(121)
    df[fraud_col].value_counts().plot(kind="bar", ax=ax1)
    ax1.set_title("Fraud vs Normal")

    ax2 = fig.add_subplot(122)
    df["prediction"].value_counts().plot(kind="bar", ax=ax2)
    ax2.set_title("Cluster Distribution")

    canvas.draw()

    # SAVE CSV
    save_fraud_csv(df)

# ---------------- FILTER FRAUD ONLY ----------------
def show_fraud_only():
    df = load_data()
    if df is None:
        return

    if "predicted_fraud_final" in df.columns:
        fraud_col = "predicted_fraud_final"
    elif "predicted_fraud" in df.columns:
        fraud_col = "predicted_fraud"
    else:
        return

    df = df[df[fraud_col] == 1]

    for row in tree.get_children():
        tree.delete(row)

    df = df.sort_values("risk_score", ascending=False).head(30)

    for _, row in df.iterrows():
        tree.insert("", "end", values=(
            int(row["Time"]),
            round(row["Amount"], 2),
            row["risk_score"],
            row.get("reason", "-")
        ))

# ---------------- UI ----------------
root = tk.Tk()
root.title("Fraud Detection App")
root.geometry("1100x700")

title = tk.Label(root, text="Fraud Detection Dashboard", font=("Arial", 18))
title.pack(pady=10)

# Buttons
btn_frame = tk.Frame(root)
btn_frame.pack()

train_btn = tk.Button(btn_frame, text="Train Model", command=train_thread)
train_btn.grid(row=0, column=0, padx=10)

filter_btn = tk.Button(btn_frame, text="Show Fraud Only", command=show_fraud_only)
filter_btn.grid(row=0, column=1, padx=10)

refresh_btn = tk.Button(btn_frame, text="Refresh", command=load_dashboard)
refresh_btn.grid(row=0, column=2, padx=10)

# KPI
kpi_frame = tk.Frame(root)
kpi_frame.pack(pady=10)

total_label = tk.Label(kpi_frame, text="Total: -", font=("Arial", 12))
total_label.grid(row=0, column=0, padx=20)

fraud_label = tk.Label(kpi_frame, text="Fraud: -", font=("Arial", 12))
fraud_label.grid(row=0, column=1, padx=20)

rate_label = tk.Label(kpi_frame, text="Fraud %: -", font=("Arial", 12))
rate_label.grid(row=0, column=2, padx=20)

# Table
columns = ("Time", "Amount", "Risk %", "Reason")
tree = ttk.Treeview(root, columns=columns, show="headings", height=10)

for col in columns:
    tree.heading(col, text=col)
    tree.column(col, anchor="center", width=150)

tree.pack(pady=10)

# Plot
fig = plt.Figure(figsize=(10, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Load initial
load_dashboard()

root.mainloop()