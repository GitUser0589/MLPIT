import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class MembershipPredictionGUI:
    def __init__(self, master):
        self.master = master
        master.title("Membership Prediction")

        self.dataset_label = tk.Label(master, text="Dataset")
        self.dataset_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.csv_path_label = tk.Label(master, text="CSV File Path:")
        self.csv_path_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.csv_path_entry = tk.Entry(master, width=50)
        self.csv_path_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self.browse_button = tk.Button(master, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=1, column=2, padx=5, pady=5)

        self.train_button = tk.Button(master, text="Train Model", command=self.train_model)
        self.train_button.grid(row=2, column=0, columnspan=3, pady=10)

        self.training_results_label = tk.Label(master, text="Training Results")
        self.training_results_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")

        self.results_text = tk.Text(master, height=5, width=60)
        self.results_text.grid(row=4, column=0, columnspan=3, padx=5, pady=5)
        self.results_text.config(state=tk.DISABLED)

        self.predict_label = tk.Label(master, text="Predict Membership")
        self.predict_label.grid(row=5, column=0, columnspan=3, pady=10)

        self.age_label = tk.Label(master, text="Are you above the age of 25?")
        self.age_label.grid(row=6, column=0, padx=10, pady=5, sticky="w")
        self.job_label = tk.Label(master, text="Do you have a job?")
        self.job_label.grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.income_label = tk.Label(master, text="Is your income above $1000?")
        self.income_label.grid(row=8, column=0, padx=10, pady=5, sticky="w")

        self.options = ["Yes", "No"]
        self.age_var = tk.StringVar(master, value="Yes")
        self.job_var = tk.StringVar(master, value="Yes")
        self.income_var = tk.StringVar(master, value="Yes")

        self.age_dropdown = tk.OptionMenu(master, self.age_var, *self.options)
        self.age_dropdown.grid(row=6, column=1, padx=10, pady=5, sticky="ew")
        self.job_dropdown = tk.OptionMenu(master, self.job_var, *self.options)
        self.job_dropdown.grid(row=7, column=1, padx=10, pady=5, sticky="ew")
        self.income_dropdown = tk.OptionMenu(master, self.income_var, *self.options)
        self.income_dropdown.grid(row=8, column=1, padx=10, pady=5, sticky="ew")

        self.predict_button = tk.Button(master, text="Predict", command=self.predict_membership)
        self.predict_button.grid(row=9, column=0, columnspan=3, pady=10)

        self.model = None

    def browse_file(self):
        filepath = filedialog.askopenfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        self.csv_path_entry.delete(0, tk.END)
        self.csv_path_entry.insert(0, filepath)

    def train_model(self):
        filepath = self.csv_path_entry.get()
        if not filepath:
            self.update_results("Please select a CSV file.")
            return

        try:
            df = pd.read_csv(filepath)

            if "ID" not in df.columns or "Avails_Membership" not in df.columns:
                self.update_results("CSV file must contain 'ID' and 'Avails_Membership' columns.")
                return

            X = df.drop(columns=["ID", "Avails_Membership"])
            y = df["Avails_Membership"].map({"Yes": 1, "No": 0})

            # Ensure all features are present (Age, Job, Income)
            expected_columns = ["Age", "Job", "Income"]
            for col in expected_columns:
                if col not in X.columns:
                    self.update_results(f"CSV file must contain '{col}' column.")
                    return

            # Convert Yes/No to binary for training
            X_processed = X.copy()
            for col in ["Age", "Job", "Income"]:
                if X_processed[col].dtype == 'object':
                    X_processed[col] = X_processed[col].map({"Yes": 1, "No": 0})
                elif X_processed[col].dtype != 'int64' and X_processed[col].dtype != 'float64':
                    self.update_results(f"Column '{col}' should contain 'Yes'/'No' or numeric data.")
                    return

            X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

            self.model = LogisticRegression()
            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=["No", "Yes"])

            results = f"Model trained successfully!\n"
            results += f"Accuracy: {accuracy:.4f}\n"
            results += f"Classification Report:\n{report}"
            self.update_results(results)

        except FileNotFoundError:
            self.update_results("Error: CSV file not found.")
        except Exception as e:
            self.update_results(f"An error occurred: {e}")

    def update_results(self, text):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.config(state=tk.DISABLED)

    def yes_no_to_binary(self, value):
        return 1 if value == "Yes" else 0

    def predict_membership(self):
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first.")
            return

        try:
            age = self.yes_no_to_binary(self.age_var.get())
            job = self.yes_no_to_binary(self.job_var.get())
            income = self.yes_no_to_binary(self.income_var.get())

            input_data = np.array([[age, job, income]])
            prediction = self.model.predict(input_data)[0]
            prob = self.model.predict_proba(input_data)[0][1]

            result = "YES" if prediction == 1 else "NO"
            messagebox.showinfo("Prediction", f"Will avail membership? {result}\nProbability: {prob:.2f}")

        except Exception as e:
            messagebox.showerror("Error", f"Something went wrong during prediction:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    gui = MembershipPredictionGUI(root)
    root.mainloop()