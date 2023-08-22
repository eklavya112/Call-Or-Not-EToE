from modules import *
df = get_data("src/Bank.csv",";")
final_df = ftr_eng_sel(df)

trained_model(final_df,[0,1,2,3,5,6])