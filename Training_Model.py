# **AI MediCare System – Disease Prediction (Beginner-Friendly Explanation)**
# Yeh system ek **machine learning model** banata hai jo **patient ke symptoms ko analyze** kar ke **disease predict** karta hai. Isko develop karne ke liye **6 major steps** hain:  

# ✅ **Step 1: Medical Dataset Load Karna**  
# ✅ **Step 2: Data Ko Samajhna Aur Process Karna**  
# ✅ **Step 3: Machine Learning Model Train Karna**  
# ✅ **Step 4: Model Se Predictions Lena**  
# ✅ **Step 5: Model Save Aur Load Karna**  
# ✅ **Step 6: Medical Recommendations Dena**  

## **Step 1: Medical Dataset Load Karna**
# Sabse pehle, ek **medical dataset** use kiya jata hai jisme **patients ke symptoms aur unki diseases ka record hota hai**.  
# Yeh dataset ek **CSV (Excel file)** ki tarah hota hai jo AI model ko **training data** provide karta hai.  

# Python mein is file ko load karne ke liye `pandas` library ka use kiya jata hai:  

import pandas as pd  
dataset = pd.read_csv('Training.csv')  # CSV file load ki  
print(dataset.shape)  # Check karo ke kitni rows aur columns hain  

# 🔹 **4920 rows aur 133 columns hain:**  
# - **Pehle 132 columns** – Different **symptoms** ko represent karte hain.  
# - **Aakhri column (`prognosis`)** – Disease ka naam show karta hai.  

# 🔹 **Example:**  
# | Fever  | Cough  | Headache  | Weakness  | Prognosis (Disease)|  
# |--------|--------|-----------|-----------|--------------------|  
# | 1      | 0      | 1         | 0         | Malaria            |  
# | 0      | 1      | 0         | 1         | Cold               |  
# | 1      | 1      | 1         | 1         | Dengue             |  

# 💡 **1 ka matlab symptom present hai, 0 ka matlab nahi hai.**  


## **Step 2: Data Ko Samajhna Aur Process Karna**  
# Machine Learning Model ko train karne ke liye:  
# 1️⃣ **Symptoms ko (`X`) features bana diya jata hai.**  
# 2️⃣ **Diseases ko (`y`) labels bana diya jata hai.**  

X = dataset.drop('prognosis', axis=1)  # Symptoms ko alag kiya  
y = dataset['prognosis']  # Diseases ko alag kiya  

# 🔹 **Ab AI model sirf symptoms ko dekh kar disease predict karega.**  

# 💡 **Problem:** AI text format nahi samajhta, is liye diseases ko **numbers** me convert karna hoga.  

from sklearn.preprocessing import LabelEncoder  
le = LabelEncoder()  
y = le.fit_transform(y)  # Disease names → Numbers me convert  

# 🔹 **Example:**  
# - `"Fungal Infection"` → `0`  
# - `"Malaria"` → `5`  
# - `"Dengue"` → `2`  

# Ab AI model diseases ko **numbers** ke form me samajhne lagega. 🚀  

## **Step 3: Machine Learning Model Train Karna**
# Ab **Machine Learning Model** train kiya jayega jo symptoms ko analyze kar ke disease predict karega.  

### **Konsa Model Best Hai?**
# Hum **different algorithms** use kar sakte hain:  
# ✅ **SVM (Support Vector Machine)** – Best accuracy ke liye  
# ✅ **Random Forest** – Decision Trees ka advanced version  
# ✅ **Naïve Bayes** – Simple aur fast model  

### **Training Ke Liye Steps:**
# 1️⃣ **Dataset ko Train aur Test me split karo (70%-30%)**  
# 2️⃣ **Different models train karo**  
# 3️⃣ **Best accuracy wala model select karo**  


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  



from sklearn.svm import SVC  
model = SVC(kernel='linear')  # SVM Model use kiya  
model.fit(X_train, y_train)  # Model ko train kiya  

# 🤖 **Model ab seekh gaya hai ke symptoms ke basis par diseases kaise predict karni hain.**  

## **Step 4: Model Se Predictions Lena**  
# Model ko test karne ke liye **ek patient ke symptoms input** diye jate hain aur dekha jata hai ke woh sahi disease predict karta hai ya nahi.  

sample_input = X_test.iloc[0].values.reshape(1, -1)  # Ek patient ke symptoms  
predicted_disease = model.predict(sample_input)  # AI model ki prediction  
print("Predicted Disease:", le.inverse_transform(predicted_disease))  

# 🔹 **Example:**  
# Agar model ka output `5` aata hai, toh `le.inverse_transform([5])` `"Malaria"` return karega. 🎯  

## **Step 5: Model Save Aur Load Karna**  
# AI model ko save karna zaroori hota hai taki baad me dobara train na karna pade.  


import pickle  
pickle.dump(model, open('disease_predictor.pkl', 'wb'))  # Save model  

# Baad me use karne ke liye:  


model = pickle.load(open('disease_predictor.pkl', 'rb'))  # Load model  

## **Step 6: Medical Recommendations Dena**  
# Agar kisi patient ka disease predict ho jata hai, toh usko **medicines, diet, precautions aur exercises** recommend ki ja sakti hain.  

# 🔹 **Data files load karo:**  

precautions = pd.read_csv("precautions_df.csv")  
medications = pd.read_csv("medications.csv")  
diets = pd.read_csv("diets.csv")  

# 🔹 **Disease ke basis par recommendations lene ka function:**  

def get_recommendations(disease):  
    pre = precautions[precautions['Disease'] == disease].values.flatten().tolist()  
    med = medications[medications['Disease'] == disease]['Medication'].values.tolist()  
    die = diets[diets['Disease'] == disease]['Diet'].values.tolist()  
    
    return pre, med, die  

# 🔹 **Example:**  
# Agar model **"Malaria"** detect kare, toh yeh function **precautions, medicines aur diet plan** return karega.  


precautions, medicines, diet = get_recommendations("Malaria")  
print("Precautions:", precautions)  
print("Medicines:", medicines)  
print("Diet Plan:", diet)  

# 💡 **Output Example:**  

# Precautions: ["Use mosquito net", "Wear full sleeves", "Use mosquito repellent"]  
# Medicines: ["Chloroquine", "Quinine", "Mefloquine"]  
# Diet Plan: ["Hydration drinks", "Fruits", "High protein diet"]  

## **Summary (Aasan Lafzon Mein)**
# ✅ **AI ko medical dataset par train kiya** symptoms ke basis par disease predict karne ke liye.  
# ✅ **SVM model ka use kiya** jo best accuracy deta hai.  
# ✅ **Predictions ko real names me convert kiya** taake AI readable results de.  
# ✅ **Medical recommendations system banaya** jo **precautions, medicines aur diet** suggest karta hai.  
# ✅ **AI model ko save kiya** taake dobara train na karna pade.  

## **Next Steps (Agar Aap Isko Improve Karna Chahte Hain)**
# 1️⃣ **Overfitting fix karo** – Zyada realistic accuracy ke liye hyperparameter tuning karo.  
# 2️⃣ **Deep Learning Models Try Karo** – CNN aur LSTMs ko medical diagnosis ke liye implement karo.  
# 3️⃣ **Web Interface Banao** – React.js aur Django REST API ka use karo.  