import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import find_peaks
import yfinance as yf
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

def buy_operation(cash, current_price):
    try:
        n_shares = (cash // current_price)
    except ZeroDivisionError:
        # Handle the case where current_price is zero
        return 0, cash, 0
    n_shares_adjusted = n_shares - (n_shares % 100)  # Adjust n_shares to be divisible by 100
    traded_money = n_shares_adjusted * current_price
    rest_money = cash - traded_money
    return traded_money, rest_money, n_shares_adjusted

def designPortfolio(resulted_df,company_name,types):
    if resulted_df.empty:
        print('The DataFrame is empty')
        return pd.DataFrame()
    df = resulted_df.copy() 
    Current_point = list(df['Option_price'])
    predicted_future_point = list(df['predicted_future_point'])
    Actual_future_point = list(df['future_option_one_day'])
    Stock_price = list(df['Stock_price'])
    Strike = list(df['Strike_Price'])
    option_type = list(df['Option_type'])
    
    model_decision = []
    resulted_op_money =[]
    resulted_op_shares = []
    rest_op_money = []
    N_shares = []
    model_corrector = []
    sp_portfolio = []
    commission = []

    teade_yet = False
    starting = True
    starting_cashs = 50000
    
    wrong_decision = 0
    right_decision = 0
    
    for c, p, a, s, o in zip(Current_point, predicted_future_point, Actual_future_point, Strike, option_type):

        
        # Handling --------------- Stock movenemt----------------
        if starting :

            starting_cash, _, starting_shares = buy_operation(starting_cashs, c)
#             sp_portfolio.append(starting_shares*c)
            starting = False
        else:
#             sp_portfolio.append(starting_shares*c)
            starting = False
        # Checked .........True

        # Handling --------------- Model Decision----------------- 
        if teade_yet == True : # you already traded before
            last_decision = model_decision[-1]
            
            if ((p > c)):
                if last_decision == f'BUY_{o}' or last_decision == f'HOLD_{o}': # you already have the S&P

                    model_decision.append(f'HOLD_{o}')
                    prev_shares = N_shares[-1]
                    prev_rest = rest_op_money[-1]

                    commission.append(0)
                    resulted_op_money.append(prev_shares*c+prev_rest)
                    resulted_op_shares.append(prev_shares*c)
                    rest_op_money.append(prev_rest)
                    N_shares.append(prev_shares)
                    sp_portfolio.append(prev_shares*c)


                else: # last_decision, 'SELL', 'SKIP'
                    model_decision.append(f'BUY_{o}')

                    # Buying your previous resulted op money

                    prev_resulted_op_money = resulted_op_money[-1]
                    traded_money, rest_money, n_shares = buy_operation(prev_resulted_op_money, c)

                    resulted_op_money.append(n_shares*c+rest_money)
                    resulted_op_shares.append(n_shares*c)
                    rest_op_money.append(rest_money)
                    N_shares.append(n_shares)
                    sp_portfolio.append(n_shares*c)
                    if ((n_shares*0.01)<3.5):
                        x = 3.5
                    else:
                        x = n_shares*0.01
                    commission.append(x)


            elif p < c: # in case of equality 
                if last_decision == f'SELL_{o}' or last_decision == f'SKIP_{o}': # you already have the S&P
                    model_decision.append(f'SKIP_{o}')

                    # everething Stay as they was
                    commission.append(0)
                    resulted_op_money.append(resulted_op_money[-1])
                    resulted_op_shares.append(resulted_op_shares[-1])
                    rest_op_money.append(rest_op_money[-1])
                    N_shares.append(N_shares[-1])
                    sp_portfolio.append(sp_portfolio[-1])

                else: # last_decision, 'BUY', 'HOLD'
                    model_decision.append(f'SELL_{o}')
                    prev_shares = N_shares[-1]
                    prev_rest = rest_op_money[-1]
                    resulted_op_money.append(prev_shares*c+prev_rest)
                    resulted_op_shares.append(prev_shares*c)
                    rest_op_money.append(prev_rest)
                    N_shares.append(0)
                    sp_portfolio.append(prev_shares*c)
                    if ((n_shares*0.01)<3.5):
                        x = 3.5
                    else:
                        x = n_shares*0.01
                    commission.append(x)
            
            elif p == c:
                if last_decision == f'SELL_{o}' or last_decision == f'SKIP_{o}': # you already have the S&P
                    model_decision.append(f'SKIP_{o}')

                    # everething Stay as they was
                    commission.append(0)
                    resulted_op_money.append(resulted_op_money[-1])
                    resulted_op_shares.append(resulted_op_shares[-1])
                    rest_op_money.append(rest_op_money[-1])
                    N_shares.append(N_shares[-1])
                    sp_portfolio.append(sp_portfolio[-1])
                    
                else: # last_decision, 'BUY', 'HOLD'
                    model_decision.append(f'HOLD_{o}')
                    prev_shares = N_shares[-1]
                    prev_rest = rest_op_money[-1]

                    commission.append(0)
                    resulted_op_money.append(prev_shares*a+prev_rest)
                    resulted_op_shares.append(prev_shares*a)
                    rest_op_money.append(prev_rest)
                    N_shares.append(prev_shares)
                    sp_portfolio.append(prev_shares*c)
                    
                    
                

        else: # 1st time to trade
            if ((p > c)):
                teade_yet = True
                model_decision.append(f'BUY_{o}')
                # executing by operation by starting cash 
                traded_money, rest_money, n_shares = buy_operation(starting_cashs, c)

                resulted_op_money.append(n_shares*c+rest_money)
                resulted_op_shares.append(n_shares*c)
                rest_op_money.append(rest_money)
                N_shares.append(n_shares)
                sp_portfolio.append(n_shares*c)
                if ((n_shares*0.01)<3.5):
                    x = 3.5
                else:
                    x = n_shares*0.01
                commission.append(x)

            else:
                model_decision.append(f'SKIP_{o}')
                resulted_op_money.append(starting_cashs)
                commission.append(0)
                resulted_op_shares.append(0)
                rest_op_money.append(0)
                N_shares.append(0)
                sp_portfolio.append(starting_cashs)

                #Rest_money.append(starting_cash)
                #portfolio_money.append(0)

        # Handle -------------------------------Model Corrector--------------------
        last_decision = model_decision[-1]
        if (p > c and a > c) or (p<c and a<c) or (p == c and a >= c) : 
            model_corrector.append('Right--'+last_decision)
            right_decision += 1
        else:
            model_corrector.append('Wrong--'+last_decision)
            wrong_decision += 1
            



    data = {
            'Date':df['Date'],
            'Day': pd.to_datetime(df['Date']).dt.day_name(),
            'C_point':Current_point, 
            'PN_point': predicted_future_point, 
            'AN_point': Actual_future_point,
            'Strike': Strike,
            'options': option_type,
            'Model-Decision': model_decision,
            'Model-Corrector':model_corrector ,
            'resulted_op_shares':resulted_op_shares,
            'rest_op_money':rest_op_money,
            'resulted_op_money': np.round(resulted_op_money,2),
            'N_shares':N_shares,
            'commission': commission,
            'portfolio':sp_portfolio
            }
    
    porto_df = pd.DataFrame(data)
#     com = round(commission,3)
    commissions = round(sum(commission), 3)
    m_return = ((Stock_price[-1]-Stock_price[0])/Stock_price[0])*100
    print(f'Stock_Return: {round(m_return,3)} %')
    sp_return = (((sp_portfolio[-1]+rest_op_money[-1])-starting_cashs)-commissions)/starting_cashs*100
    print(f'portfolio_Return: {round(sp_return,3)} %')
    rw_ratio = (right_decision)/(right_decision+wrong_decision)*100
    print(f'R/W Ratio: {round(rw_ratio,3)} %')
    #mae_test = mean_absolute_error(Actual_future_point, predicted_future_point)
    #print(f'MAE : ' , mae_test , 'dollar')
    metrics_df = pd.DataFrame({
        'portfolio_Return': [round(sp_return, 3)],
        'S&P return':[round(m_return,3)],
        'R/W_Ratio': [round(rw_ratio, 3)],
        'commission': commissions,
        'symbol': company_name,
        'type': types
    })
    return porto_df,metrics_df


def draw_actual_vs_predict(date, actual, predicted, str_, marker):
    # Check the lengths of the input lists
    len_date = len(date)
    len_actual = len(actual)
    len_predicted = len(predicted)

    print(f"Length of date list: {len_date}")
    print(f"Length of actual list: {len_actual}")
    print(f"Length of predicted list: {len_predicted}")

    if len_date != len_actual or len_date != len_predicted:
        print("Error: The lengths of the input lists do not match.")
        return

    # Create the DataFrame
    df = pd.DataFrame({
        'Date': date,
        'actual': actual,
        'predicted': predicted
    })

    # Sort the DataFrame by date
    df = df.sort_values(by='Date')

    # Plot the data
    fig = px.line(df, x='Date', y=['actual', 'predicted'], markers=marker)

    fig.update_layout(
        title_text=str_,
        plot_bgcolor='white',
        font_size=15,
        font_color='black',
        legend_title_text=''
    )

    fig.update_xaxes(title_text="Date", zeroline=False, showgrid=False)
    fig.update_yaxes(title_text='actual', secondary_y=False, zeroline=False, showgrid=False)
    fig.update_yaxes(title_text='predicted', secondary_y=True, zeroline=False, showgrid=False)

    fig.show()
    
    
def process_and_sort_data(df_put_OUT):
    if df_put_OUT.empty:
        #print(f'The DataFrame is empty.')
        return df_put_OUT
    
    random_index = np.random.choice(df_put_OUT.index)
    df_call1_filtered = df_put_OUT.loc[[random_index]]
    strike = df_call1_filtered['Strike_Price'].iloc[0]
    condition2 = df_put_OUT['Strike_Price'] == strike
    df_put_OUT = df_put_OUT.drop(df_put_OUT[condition2].index)

    return df_put_OUT

def test_portfolio(df, company_name):
    df['Symbol'] = company_name
    df['difference'] = df['future_option_one_day'] - df['Option_price']
    df.replace({'Option_type': {0: 'call', 1: 'put'}}, inplace=True)
    
    Date_counts = df['Strike_Price'].value_counts().to_frame()
    Date_counts.rename(columns={'Strike_Price': 'value_counts'}, inplace=True)
    Date_counts.index.name = 'Strike_Price'
    Date_counts_sorted = Date_counts.sort_index()
    strike_prices = Date_counts_sorted.index.tolist()
    
    df_call = df[df['Option_type'] == "call"]
#     df_put = df[df['Option_type'] == "put"]
    
    df_call_IN = df_call[(df_call['Strike_Price'] == strike_prices[0]) | (df_call['Strike_Price'] == strike_prices[1])]
    df_call_OUT = df_call[(df_call['Strike_Price'] == strike_prices[4]) | (df_call['Strike_Price'] == strike_prices[5])]
    df_call_NEAR = df_call[(df_call['Strike_Price'] == strike_prices[2]) | (df_call['Strike_Price'] == strike_prices[3])]
    
    final_df_call_OUT = process_and_sort_data(df_call_OUT)
    final_df_call_IN = process_and_sort_data(df_call_IN)
    final_df_call_NEAR = process_and_sort_data(df_call_NEAR)
    
    print("Final DataFrame (Call OUT):")
    final_df_call_OUT, metrics_call_OUT = designPortfolio(final_df_call_OUT,company_name,'call_OUT')
    print('________________________________________________________________________________________________________________')
#     display(final_df_call_OUT)
    print("Final DataFrame (Call IN):")
    final_df_call_IN, metrics_call_IN = designPortfolio(final_df_call_IN,company_name,'call_IN')
    
    print('________________________________________________________________________________________________________________')
#     display(final_df_call_IN)
    print("Final DataFrame (Call NEAR):")
    final_df_call_NEAR, metrics_call_NEAR = designPortfolio(final_df_call_NEAR,company_name,'call_NEAR')
    print('________________________________________________________________________________________________________________')
#     display(final_df_call_NEAR)
    
    final_df_test = pd.concat([final_df_call_OUT, final_df_call_IN, final_df_call_NEAR], axis=0)
    final_metrics = pd.concat([metrics_call_OUT, metrics_call_IN, metrics_call_NEAR], axis=0)
    return final_df_test,final_metrics


def get_window_series(start_date, bucket_size, df):
    date_obj = datetime.strptime(start_date, '%m/%d/%Y')
    final_date = pd.Timestamp(date_obj)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    comparison_result = df['Date'].iloc[0] <= final_date
    if (comparison_result):
        df_test = df[df['Date'] >= final_date]
        date_list =df_test['Date']
    else:
        date_list = df[-512*12:]['Date']
    n_trading_days = date_list.count()
    n_trading_days = n_trading_days
    bucket_size = bucket_size*12
    num_buckets = int(n_trading_days / bucket_size)

    df = df.copy()
    #total_rows = len(df)

    # Step 3: Create the buckets
    buckets = []
    for i in range(num_buckets, 0, -1):
        train_data = df.iloc[0:-i*bucket_size]
        if i > 1:
            valid_data = df.iloc[-i*bucket_size: (-i*bucket_size)+bucket_size]
        else:
            valid_data = df.iloc[-i*bucket_size:]

        buckets.append((train_data, valid_data))
    
    return buckets, date_list

def my_standard_model_exp2_multiperiod(df, n_trading_days, included_features, buckets, date_list, current_point, buying_point, target, rf_model, company_name, future_test_data=None, load_model=False, model_filename=None):
    # Check if rf_model is an object; if so, use it directly without loading from a file
    if load_model and isinstance(model_filename, str):
        # Load the pre-trained model from the pickle file
        with open(model_filename, 'rb') as file:
            rf_model = pickle.load(file)
        print(f'Model loaded from {model_filename}')
    elif load_model and model_filename is not None:
        # If model_filename is already a model object, use it directly
        rf_model = model_filename

    print('Prediction From ', date_list.iloc[0], ' To ', date_list.iloc[-1])
    valid_prediction_list = []
    valid_y_list = []
    train_prediction_list = []
    train_y_list = []
    i = 0
    models_list = []

    for bucket in buckets:
        i += 1
        train, valid = bucket
        start_date = valid['Date'].iloc[0]
        end_date = valid['Date'].iloc[-1]
        X_train, y_train = train[included_features], train[target]
        X_test, y_test = valid[included_features], valid[target]

        if not load_model:
            rf_model.fit(X_train, y_train)
        models_list.append(rf_model)

        train_predictions = rf_model.predict(X_train)
        test_predictions = rf_model.predict(X_test)

        # Calculate error
        mae_test = mean_absolute_error(y_test, test_predictions)
        r2_test = r2_score(y_test, test_predictions)
        print('Bucket ', i, " Test-- MAE: ", mae_test, 'R2 Score: ', r2_test)
        print("Start date: ", start_date, 'End date: ', end_date)

        train_prediction_list.append(list(train_predictions))
        train_y_list.append(list(y_train))
        valid_prediction_list.append(list(test_predictions))
        valid_y_list.append(list(y_test))

    valid_y_list = [item for sublist in valid_y_list for item in sublist]
    valid_prediction_list = [item for sublist in valid_prediction_list for item in sublist]
    train_y_list = [item for sublist in train_y_list for item in sublist]
    train_prediction_list = [item for sublist in train_prediction_list for item in sublist]

    #mae_train = mean_absolute_error(train_y_list, train_prediction_list)
    #r2_train = r2_score(train_y_list, train_prediction_list)
    mae_test = mean_absolute_error(valid_y_list, valid_prediction_list)
    r2_test = r2_score(valid_y_list, valid_prediction_list)
    print("Total-Test-- MAE: ", mae_test, 'R2 Score: ', r2_test)

    # Truncate date_list to match the length of valid_y_list and valid_prediction_list
    if len(date_list) > len(valid_y_list):
        date_list = date_list[:len(valid_y_list)]

    #draw_actual_vs_predict(date_list, valid_y_list, valid_prediction_list, 'Model performance in Testing', True)

    test_data = df[-n_trading_days:]

    # Truncate test_data to match the length of valid_prediction_list
    min_len = min(len(test_data), len(valid_prediction_list))
    test_data = test_data.iloc[:min_len]
    valid_prediction_list = valid_prediction_list[:min_len]
    valid_y_list = valid_y_list[:min_len]

    result_df = pd.DataFrame(columns=['Date', 'Option_price', 'predicted_future_point', 'future_option_one_day',
                                      'Stock_price', 'Strike_Price', 'Option_type'])
    result_df['Date'] = test_data['Date']
    result_df['Option_price'] = test_data[current_point]
    result_df['future_option_one_day'] = test_data[target]
    result_df['predicted_future_point'] = valid_prediction_list
    result_df['Stock_price'] = test_data['Stock_price']
    result_df['Strike_Price'] = test_data['Strike_Price']
    result_df['Option_type'] = test_data['Option_type']

    n = len(buckets)
    rows_per_part = len(result_df) // n
    df_parts = [result_df.iloc[i * rows_per_part: (i + 1) * rows_per_part] for i in range(n)]
    list_of_portfolio = []
    list_of_portfolio.append(0)
    list_of_matric = []

    for j, df_part in enumerate(df_parts, start=1):
        print('Bucket ', j)
        start_date = df_part['Date'].iloc[0].date()
        end_date = df_part['Date'].iloc[-1].date()
        portfolio_df, final_metrics = test_portfolio(df_part, company_name)
        final_metrics = final_metrics.assign(Start_Date=start_date)
        final_metrics = final_metrics.assign(End_Date=end_date)
        list_of_portfolio.append(portfolio_df)
        list_of_matric.append(final_metrics)
        print('------------------------------------------------------------')

    print('----Total Portfolio----')
    start_date2 = result_df['Date'].iloc[0].date()
    end_date2 = result_df['Date'].iloc[-1].date()
    total_portfolio_df, total_final_metrics = test_portfolio(result_df, company_name)
    total_final_metrics = total_final_metrics.assign(Start_Date=start_date2)
    total_final_metrics = total_final_metrics.assign(End_Date=end_date2)
    list_of_portfolio.append(total_portfolio_df)
    list_of_matric.append(total_final_metrics)

    # Predict future option prices if future_test_data is provided
    if future_test_data is not None:
        X_future = future_test_data[included_features]
        future_predictions = rf_model.predict(X_future)
        future_test_data['predicted_future_point'] = future_predictions
        print("Future predictions added to the test data.")
        return models_list, result_df, list_of_portfolio, list_of_matric, future_test_data

    return models_list, result_df, list_of_portfolio, list_of_matric



ML_algo_dict = {
     'AAPL': RandomForestRegressor(),
     'ADBE': Ridge(alpha=0.8),
     'AEYE': RandomForestRegressor(),
     'AMD': DecisionTreeRegressor(),
     'AMZN': DecisionTreeRegressor(),
     'ANET': RandomForestRegressor(),
     'ARKG': Ridge(alpha=0.8),
     'ARKK': Lasso(alpha=0.1),
     'ASML': RandomForestRegressor(),
     'BILL': RandomForestRegressor(),
     'CELH': RandomForestRegressor(),
     'CMG': RandomForestRegressor(),
     'COIN': DecisionTreeRegressor(),
     'COST': RandomForestRegressor(),
     'CRM': RandomForestRegressor(),
     'CRWD': Lasso(alpha=0.1),
     'CYBR': Lasso(alpha=0.1),
     'DDOG': Lasso(alpha=0.1),
     'DKNG': Ridge(alpha=0.8),
     'DT': RandomForestRegressor(),
     'ELF': Lasso(alpha=0.1),
     'FTI': Lasso(alpha=0.1),
     'FTNT': Lasso(alpha=0.1),
     'GOOGL': Ridge(alpha=0.8),
     'GTEK': RandomForestRegressor(),
     'HUBS': RandomForestRegressor(),
     'INTC': Lasso(alpha=0.1),
     'KLAC': RandomForestRegressor(),
     'LCID': Ridge(alpha=0.8),
     'LLY': RandomForestRegressor(),
     'LPLA': DecisionTreeRegressor(),
     'MA': DecisionTreeRegressor(),
     'MELI': LinearRegression(),
     'META': RandomForestRegressor(),
     'MLTX': Lasso(alpha=0.01),
     'MRVL': RandomForestRegressor(),
     'MSFT': LinearRegression(),
     'MSI': Lasso(alpha=0.01),
     'NIO': DecisionTreeRegressor(),
     'NVDA': DecisionTreeRegressor(),
     'ORCL': DecisionTreeRegressor(),
     'OXY': RandomForestRegressor(),
     'PANW': RandomForestRegressor(),
     'PATH': LinearRegression(),
     'RBLX': Lasso(alpha=0.1),
     'RIVN': Lasso(alpha=0.1),
     'ROIV': RandomForestRegressor(),
     'ROKU': LinearRegression(),
     'SMCI': RandomForestRegressor(),
     'SMH': LinearRegression(),
     'SOUN': DecisionTreeRegressor(),
     'SPCE': Lasso(alpha=0.1),
     'SQ': RandomForestRegressor(),
     'SYM': Lasso(alpha=0.1),
     'TEAM': DecisionTreeRegressor(),##
     'TSLA': RandomForestRegressor(),
     'TSM': RandomForestRegressor(),
     'TWLO': RandomForestRegressor(),
     'U': Lasso(alpha=0.1),
     'UBER': Lasso(alpha=0.1),
     'UNH': RandomForestRegressor(),
     'V': DecisionTreeRegressor(),
     'VKTX': Lasso(alpha=0.1),
     'VRT':DecisionTreeRegressor(),
     'WDAY': LinearRegression(),
     'XLE': RandomForestRegressor(),
     'XLF': LinearRegression(),
     'ZM': DecisionTreeRegressor(),
     'ARM': Ridge(alpha=0.8),
     'AVGO': Lasso(alpha=0.1),
     'LRCX': DecisionTreeRegressor(),
     'MARA': DecisionTreeRegressor(),
     'MSTR': LinearRegression(),
     'MU':DecisionTreeRegressor(),
     'NFLX': Lasso(alpha=0.1),
     'NOW': RandomForestRegressor(),
     'QCOM': Ridge(alpha=0.8),
     'RIOT': DecisionTreeRegressor()
}
features = ["Strike_Price", "Stock_price",'Option_price', "Rate",
                      'Option_type', "implied_volatility",
                      "Vega", "theta", "rho",'delta', 'Inflation_Rate',
                      'CCI','Assets Management Beta',"CPI"]

def track_record_10y(company,pickle_file,data_file,start_date):
    #model = pickle.load(pickle_file)
    data_file.replace({'Option_type':{'call':0,'put':1,'Call':0,'Put':1}},inplace=True)
    bucket_size = 20
    buckets, date_list = get_window_series(start_date, bucket_size, data_file)
    n_trading_days = date_list.count()
    models_list, result_df, list_of_portfolio,total_final_metrics = my_standard_model_exp2_multiperiod(
    data_file, 
    n_trading_days, 
    features,
    buckets, 
    date_list, 
    'Option_price', 
    'Option_price', 
    'future_option_one_day', 
    ML_algo_dict[company],
    company,
#     df2,
    load_model=True,
    model_filename=pickle_file
    )
    combined_df = pd.concat(total_final_metrics, ignore_index=True)
    combined_df['R/W_Ratio'] = combined_df['R/W_Ratio'].clip(lower=50)
    combined_df['portfolio_Return'] = combined_df['portfolio_Return'].clip(lower=-20, upper=25000)
    return combined_df

def filter_by_type(df,type1,):
    df_filtered = df[df['type'].str.contains(f'{type1}')]
    return df_filtered

def filter_by_quarter(df, year, quarter):
    df = df.copy()  # Ensure you're working with a copy of the DataFrame
    df['Start_Date'] = df['Start_Date'].astype(str)  # Convert 'Start_Date' to string safely
    month_map = {
        1: ['01'],
        2: ['02'],
        3: ['03'],
        4: ['04'],
        5: ['05'],
        6: ['06'],
        7: ['07'],
        8: ['08'],
        9: ['09'],
        10: ['10'],
        11: ['11'],
        12: ['12'],
    }
    months = month_map.get(quarter)
    if months:
        # Create a pattern to match dates in the specified year and quarter
        pattern = '|'.join([f'{year}-{month}' for month in months])
        
        # Filter the dataframe for rows that match the pattern
        df_quarter = df[df['Start_Date'].str.contains(pattern)]
        
        # Drop duplicate companies, keeping the first occurrence
        df_quarter_unique = df_quarter.drop_duplicates(subset='symbol', keep='first')
        
        return df_quarter_unique
    else:
        raise ValueError("Invalid quarter. Quarter should be between 1 and 4.")


def filter_and_store(df, prefix, years, quarters):
    df_dict = {}

    for year in years:
        for quarter in quarters:
            # if year == '2024' and quarter != 1:
            #     continue
            
            key = f'{year}_M{quarter}'
            df_dict[key] = filter_by_quarter(df, year, quarter)
            globals()[f'{prefix}_{year}_M{quarter}'] = df_dict[key]
    
    return df_dict

def wights(df):
    # Use .loc to modify DataFrame columns in place, avoiding SettingWithCopyWarning
    df.loc[:, 'R/W_Ratio'] = df['R/W_Ratio'].clip(lower=50)
    df.loc[:, 'portfolio_Return'] = df['portfolio_Return'].clip(lower=-20, upper=25000)

    df_negative_return = df[df['portfolio_Return'] <= 0]
    df_positive_return = df[df['portfolio_Return'] > 0].copy()  # Use .copy() to ensure it's a new DataFrame

    total_portfolio_return = df_positive_return['portfolio_Return'].sum()
    df_positive_return['proportion'] = df_positive_return['portfolio_Return'] / total_portfolio_return

    total_amount = 50000
    df_positive_return['allocation'] = df_positive_return['proportion'] * total_amount
    df_positive_return['Returns'] = (df_positive_return['portfolio_Return'] / 100) * df_positive_return['allocation']

    return df_positive_return, df_negative_return

def momentum_wights(df1, df2, df3):
    df_final = pd.DataFrame()
    df_final['period'] = df1['period']
    df_final['Accuracy_out'] = df1['Accuracy']
    df_final['Accuracy_in'] = df2['Accuracy']
    df_final['Accuracy_near'] = df3['Accuracy']
    df_final['Aaccuracy'] = df_final[['Accuracy_out', 'Accuracy_in', 'Accuracy_near']].max(axis=1)
    df_final = df_final.drop(['Accuracy_out', 'Accuracy_near', 'Accuracy_in'], axis=1)
    df_final['options_return_out'] = df1['portfolio_return']
    df_final['options_return_in'] = df2['portfolio_return']
    df_final['options_return_near'] = df3['portfolio_return']
    df_final['weight_out'] = df_final['options_return_out'] / (df_final['options_return_out'] + df_final['options_return_in'] + df_final['options_return_near'])
    df_final['weight_in'] = df_final['options_return_in'] / (df_final['options_return_out'] + df_final['options_return_in'] + df_final['options_return_near'])
    df_final['weight_near'] = df_final['options_return_near'] / (df_final['options_return_out'] + df_final['options_return_in'] + df_final['options_return_near'])
    df_final['options_total_return'] = ((df_final['weight_out']*df_final['options_return_out'])+
                               (df_final['weight_in']*df_final['options_return_in'])+
                               (df_final['weight_near']*df_final['options_return_near']))
    df_final['stocks_return'] = df1['quantum_stocks']
    return df_final


st.title('Options Track Record')
st.header('This is back test for option price')

link_dict = ML_algo_dict
available_companies = list(link_dict.keys())

# Multi-select dropdown for company choices
selected_companies = st.multiselect("Select Companies (choose one or more)", options=available_companies)

# Date input with formatted output
selected_date = st.date_input("Select a date")
start_date = selected_date.strftime('%d/%m/%Y')  # Format the date as 'dd/mm/yyyy'

# Upload models for each selected company
for company in selected_companies:
    model_key = f"loaded_model_{company}"  # Use a different key to avoid conflict
    st.write(f"Upload model file for {company}")
    model_file = st.file_uploader(f"Upload model for {company}", type=["pkl"], key=f"model_{company}")

    # Load and store each model in session state with a non-conflicting key
    if model_file is not None:
        st.session_state[model_key] = pickle.load(model_file)

# Upload data files for each selected company
for company in selected_companies:
    data_key = f"loaded_data_{company}"  # Use a different key to avoid conflict
    st.write(f"Upload data file for {company}")
    data_file = st.file_uploader(f"Upload data for {company}", type=["csv"], key=f"data_{company}")

    # Load and store each data file in session state
    if data_file is not None:
        st.session_state[data_key] = pd.read_csv(data_file)

# Check if models and data are loaded for each selected company
for company in selected_companies:
    model_key = f"loaded_model_{company}"
    data_key = f"loaded_data_{company}"
    
    if model_key in st.session_state and data_key in st.session_state:
        test_data = st.session_state[data_key]
        
        if not test_data.empty:
            st.write(f"Results for {company}:")
            st.session_state[f'result_df_{company}'] = track_record_10y(
                company,
                st.session_state[model_key], 
                test_data,
                start_date
            )
            st.write(st.session_state[f'result_df_{company}'])
        else:
            st.write(f"Data file for {company} is empty.")
    else:
        st.write(f"Please ensure both model and test data are loaded for {company}.")

scaler = MinMaxScaler(feature_range=(-1, 1))

for company in selected_companies:
    data_key = f"loaded_data_{company}"
    
    if data_key in st.session_state:
        test_data = st.session_state[data_key]
        
        if not test_data.empty:
            # Filter the data from the selected date onwards
            test_data = test_data[test_data['Date'] >= start_date]
            
            # Scale the specified columns to [-1, 1] range
            columns_to_scale = ['Vega', 'theta', 'delta', 'gamma', 'rho']
            test_data[columns_to_scale] = scaler.fit_transform(test_data[columns_to_scale])
            
            st.write(f"Greeks for {company} starting from {start_date}:")
            fig_greek_df = px.line(test_data, x='Date', y=columns_to_scale, title="Greeks")
            st.plotly_chart(fig_greek_df, use_container_width=True)
        else:
            st.write(f"Data file for {company} is empty.")
    else:
        st.write(f"Please ensure both model and test data are loaded for {company}.")

if st.button("Portfolio"):
    dfs1 = []
    dfs2 = []
    
    for company in selected_companies:
        result_key = f'result_df_{company}'
        
        # Retrieve the DataFrame from session state
        if result_key in st.session_state:
            df = st.session_state[result_key]
            df2 = df.iloc[:-3]
            df = df.iloc[-3:]
            dfs1.append(df)
            dfs2.append(df2)    
        else:
            st.write(f"Results for {company} not available.")
        
    # Concatenate the DataFrames if there is data to display
    if dfs1:
        df1 = pd.concat(dfs1, axis=0)
        df1['Start_Date'] = pd.to_datetime(df1['Start_Date'])
        df1['End_Date'] = pd.to_datetime(df1['End_Date'])
        df1 = df1.dropna()
        df1['Start_Date'] = df1['Start_Date'].astype(str)
    
    if dfs2:
        df2 = pd.concat(dfs2, axis=0)
        df2['Start_Date'] = pd.to_datetime(df2['Start_Date'])
        df2['End_Date'] = pd.to_datetime(df2['End_Date'])
        df2 = df2.dropna()
        df2['Start_Date'] = df2['Start_Date'].astype(str)
    
    df_call_OUT_total = filter_by_type(df1,'call_OUT')
    df_call_OUT = filter_by_type(df2,'call_OUT')
    
    df_call_IN_total = filter_by_type(df1,"call_IN")
    df_call_IN = filter_by_type(df2,"call_IN")
    
    df_call_NEAR_total = filter_by_type(df1,"call_NEAR")
    df_call_NEAR = filter_by_type(df2,"call_NEAR")
    
    df_IN = df_call_IN_total
    df_OUT = df_call_OUT_total
    df_NEAR = df_call_NEAR_total
    df_positive_return_in, df_negative_return = wights(df_IN)
    df_positive_return_out, df_negative_return = wights(df_OUT)
    df_positive_return_near, df_negative_return = wights(df_NEAR)
    total_return_call_IN = df_positive_return_in['Returns'].sum().round(4)
    total_return_call_OUT = df_positive_return_out['Returns'].sum().round(4)
    total_return_call_NEAR = df_positive_return_near['Returns'].sum().round(4)
    x1 = (df_positive_return_in['S&P return'])*df_positive_return_in['proportion']
    x2 = (df_positive_return_out['S&P return'])*df_positive_return_out['proportion']
    x3 = (df_positive_return_near['S&P return'])*df_positive_return_near['proportion']
    quantum_stocks_call_IN = x1.sum().round(4)
    quantum_stocks_call_OUT = x2.sum().round(4)
    quantum_stocks_call_NEAR = x3.sum().round(4)
    accuracy_call_IN = df_positive_return_in['R/W_Ratio'].mean()
    accuracy_call_OUT = df_positive_return_out['R/W_Ratio'].mean()
    accuracy_call_NEAR = df_positive_return_near['R/W_Ratio'].mean()
    total_amount = 50000
    per_total_return_call_IN = (total_return_call_IN/total_amount)*100
    per_total_return_call_OUT = (total_return_call_OUT/total_amount)*100
    per_total_return_call_NEAR = (total_return_call_NEAR/total_amount)*100
    column = ['type', 'quantum_stocks', 'portfolio_return', 'Accuracy']
    df_final = pd.DataFrame(columns=column)
    new_row = {'type': 'IN', 'quantum_stocks': quantum_stocks_call_IN, 'portfolio_return': per_total_return_call_IN, 'Accuracy': accuracy_call_IN}
    df_final = pd.concat([df_final, pd.DataFrame([new_row])], ignore_index=True)
    new_row = {'type': 'OUT', 'quantum_stocks': quantum_stocks_call_OUT, 'portfolio_return': per_total_return_call_OUT, 'Accuracy': accuracy_call_OUT}
    df_final = pd.concat([df_final, pd.DataFrame([new_row])], ignore_index=True)
    new_row = {'type': 'NEAR', 'quantum_stocks': quantum_stocks_call_NEAR, 'portfolio_return': per_total_return_call_NEAR, 'Accuracy': accuracy_call_NEAR}
    df_final = pd.concat([df_final, pd.DataFrame([new_row])], ignore_index=True)
    df_OUT = df_OUT.reset_index(drop=True)
    df_IN = df_IN.reset_index(drop=True)
    df_NEAR = df_NEAR.reset_index(drop=True)
    
    # Create a DataFrame for correlation analysis
    df_correlation = pd.DataFrame({
        'OUT': df_OUT['portfolio_Return'],
        'IN': df_IN['portfolio_Return'],
        'NEAR': df_NEAR['portfolio_Return']
    })
    
    # Calculate correlation matrix
    corr = df_correlation.corr()
    
    # Display results
    st.write('Return for three portfolios for this period:')
    st.write(df_final)
    
    st.write('Correlation for three portfolios for this period:')
    fig, ax = plt.subplots(figsize=(8, 6))  # Set figure size for better readability
    
    # Customize the heatmap
    sns.heatmap(
        corr, 
        annot=True,            # Display correlation values on the heatmap
        fmt=".2f",             # Format correlation values to 2 decimal places
        cmap="coolwarm",       # Color scheme, adjust to 'coolwarm', 'viridis', etc. as needed
        center=0,              # Center the color map at 0
        linewidths=0.5,        # Add thin lines between cells
        cbar_kws={"shrink": 0.8}  # Shrink color bar slightly for better layout
    )
    
    # Set title and labels for clarity
    ax.set_title("Correlation Heatmap of Portfolio Returns", fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Display the figure in Streamlit
    st.write(fig)
    st.write("weights Portfolio for Near the Money:")
    fig_near_per = px.bar(df_positive_return_near, x='symbol', y='proportion', title="weights")
    st.plotly_chart(fig_near_per, use_container_width=True)
    st.write("weights Portfolio for IN the Money:")
    fig_in_per = px.bar(df_positive_return_in, x='symbol', y='proportion', title="weights")
    st.plotly_chart(fig_in_per, use_container_width=True)
    st.write("weights Portfolio for OUT of the Money:")
    fig_out_per = px.bar(df_positive_return_out, x='symbol', y='proportion', title="weights")
    st.plotly_chart(fig_out_per, use_container_width=True)
    start_year = selected_date.year
    current_year = datetime.now().year
    years = [str(year) for year in range(start_year, current_year + 1)]
    quarters = [1, 2, 3, 4,5,6,7,8,9,10,11,12]
    df_NEAR_ = filter_and_store(df_call_NEAR, 'df_NEAR', years, quarters)
    df_IN_ = filter_and_store(df_call_IN, 'df_IN', years, quarters)
    df_OUT_ = filter_and_store(df_call_OUT, 'df_OUT', years, quarters)
    columns = ['period', 'quantum_stocks', 'portfolio_return', 'Accuracy']
    df_returns_OUT = pd.DataFrame(columns=columns)
    df_returns_IN = pd.DataFrame(columns=columns)
    df_returns_NEAR = pd.DataFrame(columns=columns)
    near_rows = []
    dict_near = {}
    dict_in = {}
    dict_out = {}
    df_dict_near = pd.DataFrame()
    df_dict_in = pd.DataFrame()
    df_dict_out = pd.DataFrame()
    for year in years:
        for quarter in quarters:
            df = globals().get(f'df_NEAR_{year}_M{quarter}')
            if df is not None and not df.empty:
                df_positive_return_near, df_negative_return = wights(df)
                df_positive_return_near = df_positive_return_near.sort_values(by='proportion', ascending=False)
                accuracy = df_positive_return_near['R/W_Ratio'].mean()
                total_return = df_positive_return_near['Returns'].sum().round(4)
                quantum_stocks = (df_positive_return_near['S&P return'] * df_positive_return_near['proportion']).sum().round(4)
                percentage = ((total_return / 50000) * 100).round(2)
                period = f'NEAR_{year}_M{quarter}'
                df_dict_near = df_positive_return_near[['Start_Date', 'symbol', 'portfolio_Return', 'R/W_Ratio', 'proportion']].copy()
                df_dict_near = df_dict_near.rename(columns={'symbol':'Company','R/W_Ratio':'Accuracy','proportion':'Weights'})
                dict_near[f"{year}_M{quarter}"] = df_dict_near
                near_rows.append({'period': period, 'quantum_stocks': quantum_stocks, 'portfolio_return': percentage, 'Accuracy': accuracy})
    
    df_returns_NEAR = pd.DataFrame(near_rows, columns=columns)
       
    in_rows = []
    for year in years:
        for quarter in quarters:
            df = globals().get(f'df_IN_{year}_M{quarter}')
            if df is not None and not df.empty:
                df_positive_return_in, df_negative_return = wights(df)
                df_positive_return_in = df_positive_return_in.sort_values(by='proportion', ascending=False)
                accuracy = df_positive_return_in['R/W_Ratio'].mean()
                total_return = df_positive_return_in['Returns'].sum().round(4)
                quantum_stocks = (df_positive_return_in['S&P return'] * df_positive_return_in['proportion']).sum().round(4)
                percentage = ((total_return / 50000) * 100).round(2)
                period = f'IN_{year}_M{quarter}'
                df_dict_in = df_positive_return_in[['Start_Date', 'symbol', 'portfolio_Return', 'R/W_Ratio', 'proportion']].copy()
                df_dict_in = df_dict_in.rename(columns={'symbol':'Company','R/W_Ratio':'Accuracy','proportion':'Weights'})
                dict_in[f"{year}_M{quarter}"] = df_dict_in
                in_rows.append({'period': period, 'quantum_stocks': quantum_stocks, 'portfolio_return': percentage, 'Accuracy': accuracy})
    
    df_returns_IN = pd.DataFrame(in_rows, columns=columns)
    
    out_rows = []  # Collect rows here
    
    for year in years:
        for quarter in quarters:
            df = globals().get(f'df_OUT_{year}_M{quarter}')
            if df is not None and not df.empty:
                df_positive_return_out, df_negative_return = wights(df)
                df_positive_return_out = df_positive_return_out.sort_values(by='proportion', ascending=False)
                accuracy = df_positive_return_out['R/W_Ratio'].mean()
                total_return = df_positive_return_out['Returns'].sum().round(4)
                quantum_stocks = (df_positive_return_out['S&P return'] * df_positive_return_out['proportion']).sum().round(4)
                percentage = ((total_return / 50000) * 100).round(2)
                period = f'OUT_{year}_M{quarter}'
                df_dict_out = df_positive_return_out[['Start_Date', 'symbol', 'portfolio_Return', 'R/W_Ratio', 'proportion']].copy()
                df_dict_out = df_dict_out.rename(columns={'symbol':'Company','R/W_Ratio':'Accuracy','proportion':'Weights'})
                dict_out[f"{year}_M{quarter}"] = df_dict_out
                out_rows.append({'period': period,'quantum_stocks': quantum_stocks, 'portfolio_return': percentage, 'Accuracy': accuracy})
    
    df_returns_OUT = pd.DataFrame(out_rows, columns=columns)
    
    df_returns_OUT['portfolio_return'] = df_returns_OUT['portfolio_return'].clip(lower=-20, upper=400)
    df_returns_IN['portfolio_return'] = df_returns_IN['portfolio_return'].clip(lower=-20, upper=400)
    df_returns_NEAR['portfolio_return'] = df_returns_NEAR['portfolio_return'].clip(lower=-20, upper=400)
    df_returns_OUT['quantum_stocks'] = df_returns_OUT['quantum_stocks'].clip(lower=-20, upper=200)
    df_returns_IN['quantum_stocks'] = df_returns_IN['quantum_stocks'].clip(lower=-20, upper=200)
    df_returns_NEAR['quantum_stocks'] = df_returns_NEAR['quantum_stocks'].clip(lower=-20, upper=200)
    st.write('Return for out portfolio for this period monthly:')
    st.write(df_returns_OUT)
    st.write('Return for in portfolio for this period monthly:')
    st.write(df_returns_IN)
    st.write('Return for near portfolio for this period monthly:')
    st.write(df_returns_NEAR)
    
    df_momentum = momentum_wights(df_returns_OUT, df_returns_IN, df_returns_NEAR)
    df_momentum['period'] = df_momentum['period'].str.replace('OUT_', '', regex=False)
    st.write('Returns over time for three portfolios:')
    fig_returns_df = px.line(df_momentum, x='period', y=['options_return_out', 'options_return_in', 'options_return_near'], 
                             title="Returns Over Time")
    st.plotly_chart(fig_returns_df, use_container_width=True)
    st.write('Weights over time for three portfolios:')
    fig_weight_df = px.line(df_momentum, x='period', y=['weight_out', 'weight_in', 'weight_near'], 
                            title="Weights Over Time")
    st.plotly_chart(fig_weight_df, use_container_width=True)
    st.write('Returns over time for options portfolio and stocks portfolio:')
    fig_return_df = px.line(df_momentum, x='period', y=['options_total_return', 'stocks_return'], 
                            title="Return Over Time")
    st.plotly_chart(fig_return_df, use_container_width=True)    
    st.write('Accuracy over time for options portfolio:')
    fig_accuracy_df = px.line(df_momentum, x='period', y='Aaccuracy', title="Accuracy Over Time")
    st.plotly_chart(fig_accuracy_df, use_container_width=True)
    
    ticker_symbol = '^GSPC'

    # Define the start and end dates for the data extraction
    end_date = '2024-10-16'
    current_date = datetime.strptime(start_date, '%d/%m/%Y')
    start_date = current_date.strftime('%Y-%m-%d')
    
    # Fetch the historical data for the S&P 500
    sp500_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval='1d')
    
    # Convert the index to datetime
    sp500_data.index = pd.to_datetime(sp500_data.index)
    
    # Automatically generate the start dates at 3-month intervals
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    start_dates = []
    
    while current_date <= end_date:
        start_dates.append(current_date)
        current_date += pd.DateOffset(months=1)
    
    # Initialize the list of periods
    periods = []
    
    # Adjust start dates and calculate end dates for the periods
    for i in range(len(start_dates)):
        start_date = start_dates[i]
        if i < len(start_dates) - 1:
            # The end date for each period is the nearest trading day before the next period starts
            next_start_date = start_dates[i + 1]
            nearest_end_date = max([date for date in sp500_data.index if date < next_start_date], default=None)
        else:
            # For the last period, use the final provided end date
            nearest_end_date = max([date for date in sp500_data.index if date <= end_date], default=None)
    
        if nearest_end_date is None:
            continue  # Skip if no valid end date is found
    
        # Find the nearest available trading day for the start date
        nearest_start_date = min(sp500_data.index, key=lambda x: abs(x - start_date), default=None)
    
        # Handle the case where no start date is found
        if nearest_start_date is None:
            continue
    
        # If this is not the first period, adjust the start date to the day after the previous period's end date
        if periods:
            previous_end_date = periods[-1][1]
            possible_start_dates = [date for date in sp500_data.index if date > previous_end_date]
            if possible_start_dates:
                nearest_start_date = min(possible_start_dates)
            else:
                continue  # Skip if no valid start date is found
    
        periods.append((nearest_start_date, nearest_end_date))
    # Initialize investment variables
    investment_data = []
    prev_invested_cash = 10000  # Example initial investment amount
    
    def calculate_investment_value(initial_investment, initial_price, final_price):
        num_shares = initial_investment / initial_price
        return num_shares * final_price
    
    # Calculate investment value for each period
    for i, (start, end) in enumerate(periods):
        # Get the start and end prices for the period
        start_price = sp500_data.loc[start, 'Close']
        end_price = sp500_data.loc[end, 'Close']
    
        # Calculate investment value
        investment_value = calculate_investment_value(prev_invested_cash, start_price, end_price)
    
        # Calculate revenue and returns
        revenue = investment_value - prev_invested_cash
        returns = revenue / prev_invested_cash * 100
        
        # Append data to list
        investment_data.append({
            'Start Date': start.strftime('%Y-%m-%d'),
            'End Date': end.strftime('%Y-%m-%d'),
            'Invested Cash': np.round(prev_invested_cash, 2),
            'Returns': np.round(returns, 2),
            'Revenue': np.round(revenue, 2),
            'Total Cash': np.round(investment_value, 2)
        })
    
        # Update prev_invested_cash for the next period
        prev_invested_cash = investment_value
    
    # Create a DataFrame from the investment data
    df_sp = pd.DataFrame(investment_data)
    
    df_options2 = pd.DataFrame()
    df_options2[['Start Date','End Date']] = df_sp[['Start Date','End Date']]
    df_options2['Returns'] = df_momentum['options_total_return']
    
    def quality(df):
        investment = 10000
        df_f = pd.DataFrame()
        df_f[['Start Date','End Date','Returns']] = df[['Start Date','End Date','Returns']]
        df_f['Total Cash'] = 0.0
    
        for index, row in df.iterrows():
            if index == 0:
                df_f.loc[index, 'Invested Cash'] = (investment)
                df_f.loc[index, 'Revenue'] = ((df.loc[index, 'Returns']/100) * investment)
                df_f.loc[index, 'Total Cash'] = ((df.loc[index, 'Returns']/100) * investment) + investment
                
            else:
                previous_value4 = df_f.loc[index - 1, 'Total Cash']
                df_f.loc[index, 'Invested Cash'] = (previous_value4)
                df_f.loc[index, 'Revenue'] = ((df.loc[index, 'Returns']/100) * previous_value4)                
                df_f.loc[index, 'Total Cash'] = ((df.loc[index, 'Returns']/100) * previous_value4) + previous_value4
    
        return df_f
    
    df_qualified = quality(df_options2)
    df_qualified['Total Cash sp'] = df_sp['Total Cash']
    fig_performance_df = px.line(df_qualified, x='Start Date', y=['Total Cash', 'Total Cash sp'], title="performance Over Time")
    st.plotly_chart(fig_performance_df, use_container_width=True)
    peaks, _ = find_peaks(df_momentum['options_total_return'])
    bottoms, _ = find_peaks(-df_momentum['options_total_return'])
    df_peaks =  pd.DataFrame()
    df_bottoms =  pd.DataFrame()
    df_peaks['Period'] = df_momentum['period'][peaks]
    df_peaks['Return'] = df_momentum['options_total_return'][peaks]
    df_bottoms['Period'] = df_momentum['period'][bottoms]
    df_bottoms['Return'] = df_momentum['options_total_return'][bottoms]
    def filter_rows_peaks(df, col, threshold):
        return df[df[col] > threshold]
        
    def filter_rows_bottoms(df, col, threshold):
        return df[df[col] < threshold]
    average_peaks = df_peaks['Return'].sum()/df_peaks['Return'].count()
    half_peaks = average_peaks/2
    threshold_peaks = average_peaks-half_peaks
    average_bottoms = df_bottoms['Return'].sum()/df_bottoms['Return'].count()
    half_bottoms = average_bottoms/2
    threshold_bottoms = average_bottoms-half_bottoms
    df_bottoms = filter_rows_bottoms(df_bottoms, 'Return', threshold_bottoms)
    df_peaks = filter_rows_peaks(df_peaks, 'Return', threshold_peaks)
    def get_dict_data(df, dicti):
        return {period: dicti[period] for period in df['Period'] if period in dicti}
    dict_peaks_near = get_dict_data(df_peaks, dict_near)
    dict_bottoms_near = get_dict_data(df_bottoms, dict_near)
    dict_peaks_in = get_dict_data(df_peaks, dict_in)
    dict_bottoms_in = get_dict_data(df_bottoms, dict_in)
    dict_peaks_out = get_dict_data(df_peaks, dict_out)
    dict_bottoms_out = get_dict_data(df_bottoms, dict_out)
    st.write('Data Frame for Peaks:')
    st.write(df_peaks)
    st.write('Data Frame for Bottoms:')
    st.write(df_bottoms)
    
    fig = go.Figure()

        # Add the main line plot
    fig.add_trace(go.Scatter(
        x=df_momentum['period'],
        y=df_momentum['options_total_return'],
        mode='lines+markers',
        name='Total Return',
        line=dict(color='blue'),
        marker=dict(symbol='circle')
        ))
        
        # Add peaks
    fig.add_trace(go.Scatter(
        x=df_momentum['period'][peaks],
        y=df_momentum['options_total_return'][peaks],
        mode='markers',
        marker=dict(color='green', size=10, symbol='triangle-up'),
        name='Peaks'
        ))
        
        # Add bottoms
    fig.add_trace(go.Scatter(
            x=df_momentum['period'][bottoms],
            y=df_momentum['options_total_return'][bottoms],
            mode='markers',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            name='Bottoms'
        ))
        
        # Customize layout
    fig.update_layout(
            title="Peaks and Bottoms in Options Total Return",
            xaxis_title='Period',
            yaxis_title='Options Total Return',
            xaxis_tickangle=45,
            legend=dict(title='Legend'),
            showlegend=True,
            template='plotly_white'
        )
        
        # Display the figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)
        
        
    st.write("Analyze for peaks in near the money:")
    for period, data in dict_peaks_near.items():
        st.write(f"Data for {period}")
        st.write(data)
        st.write(f"weights Portfolio for {period}:")
        fig_data_per = px.bar(data, x='Company', y='Weights', title="weights")
        st.plotly_chart(fig_data_per, use_container_width=True)
        
    st.write("Analyze for bottoms in near the money:")
    for period, data in dict_bottoms_near.items():
        st.write(f"Data for {period}")
        st.write(data)
        st.write(f"weights Portfolio for {period}:")
        fig_data_per = px.bar(data, x='Company', y='Weights', title="weights")
        st.plotly_chart(fig_data_per, use_container_width=True)
            
            
    st.write("Analyze for peaks in in the money:")
    for period, data in dict_peaks_in.items():
        st.write(f"Data for {period}")
        st.write(data)
        st.write(f"weights Portfolio for {period}:")
        fig_data_per = px.bar(data, x='Company', y='Weights', title="weights")
        st.plotly_chart(fig_data_per, use_container_width=True)
            
    st.write("Analyze for bottoms in in the money:")
    for period, data in dict_bottoms_in.items():
        st.write(f"Data for {period}")
        st.write(data)
        st.write(f"weights Portfolio for {period}:")
        fig_data_per = px.bar(data, x='Company', y='Weights', title="weights")
        st.plotly_chart(fig_data_per, use_container_width=True)
            
    st.write("Analyze for peaks in out of the money:")
    for period, data in dict_peaks_out.items():
        st.write(f"Data for {period}")
        st.write(data)
        st.write(f"weights Portfolio for {period}:")
        fig_data_per = px.bar(data, x='Company', y='Weights', title="weights")
        st.plotly_chart(fig_data_per, use_container_width=True)
            
    st.write("Analyze for bottoms in out of the money:")
    for period, data in dict_bottoms_out.items():
        st.write(f"Data for {period}")
        st.write(data)
        st.write(f"weights Portfolio for {period}:")
        fig_data_per = px.bar(data, x='Company', y='Weights', title="weights")
        st.plotly_chart(fig_data_per, use_container_width=True)            
            
            
            
            
            
            