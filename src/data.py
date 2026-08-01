import pandas as pd

def load_and_split(file_path: str, oos_months: int = 60):
    price_data = pd.read_csv(file_path)
    
    if 'Unnamed: 0' in price_data.columns:
        price_data = price_data.drop(columns=['Unnamed: 0'])
        
    price_data.columns = [col.lower().replace(' ', '_') for col in price_data.columns]
    
    price_data['date'] = pd.to_datetime(price_data['date'])
    price_data = price_data.sort_values('date').set_index('date')
    
    max_date = price_data.index.max()
    oos_start_date = max_date - pd.DateOffset(months=oos_months)
    
    in_sample_df = price_data[price_data.index < oos_start_date].copy()
    out_of_sample_df = price_data[price_data.index >= oos_start_date].copy()
    
    if len(out_of_sample_df) == 0 or len(in_sample_df) == 0:
        raise ValueError("Insufficient data history for dynamic split.")
        
    return in_sample_df, out_of_sample_df