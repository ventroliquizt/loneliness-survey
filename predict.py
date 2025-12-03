import pandas as pd
import numpy as np
import pickle


MODEL_FILENAME = '/home/vent/vscode/vscode./ML/project/comprehensive_predictor_model.pkl'
PREPROCESSOR_FILENAME = '/home/vent/vscode/vscode./ML/project/comprehensive_preprocessor.pkl'


# LOAD
with open(MODEL_FILENAME, 'rb') as file:
    model = pickle.load(file)
with open(PREPROCESSOR_FILENAME, 'rb') as file:
    preprocessor = pickle.load(file)
print("Loaded success")



def get_prediction(new_user_data: dict) -> tuple:  

    df_new = pd.DataFrame([new_user_data])

    if 'Meaningful_Rels' in df_new.columns:
        df_new['Meaningful_Rels_R'] = 6 - df_new['Meaningful_Rels']
        df_new.drop('Meaningful_Rels', axis=1, inplace=True)
        
    #! Transform data!
    X_new_processed = preprocessor.transform(df_new)
    
    prediction = model.predict(X_new_processed)[0]
    probability = model.predict_proba(X_new_processed)[0][1]
    
    return int(prediction), float(probability)







example_user_input = {
        'Year': 1,                         
        'Living': 'With family',           
        'Lack_Comp': 5,                    #H
        'Left_Out': 5,                     #H
        'Isolated': 5,                     #H
        'Talk_Friends': 1,                 #l
        'Understood': 1,                   #l
        'Rely_On': 1,                      #l
        'Unknown_Me': 5,                   #H
        'Meaningful_Rels': 1,              # L
        'Avoid_Money': 1,                  #H
        'In_Person_Meetings': 1,           
        'Time_Outside_Home': 1,            
        'Weekend_Social': 1,               
        'Phone_Hours': 8,                  #H
        'Phone_Late_Night': 5,             
        'Sleep_Hours': 2,                  
        'Sleep_Restful': 0,                
        'Physical_Activity': 0,            
        'Depressed': 5,                    #H
        'Little_Interest': 5,              #H
        'Ask_Help': 1,                     #L
}

print("\n Prediction: ")
level, prob = get_prediction(example_user_input)
    
level_text = "HIGH Loneliness" if level == 1 else "LOW Loneliness"
    
print(f"Predicted Loneliness Level: {level_text}")
print(f"Probability of HIGH Loneliness: {prob * 100:.2f}%")