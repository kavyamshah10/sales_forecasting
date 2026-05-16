import pandas as pd

def forecast_2013(model, data, weeks=52):

    predictions = []
    data = data.copy()

    for i in range(weeks):

        new_rows = []

        for store in data['Store'].unique():

            store_data = data[data['Store'] == store].sort_values(['Year', 'week'])

            # Safety check (important)
            if len(store_data) < 7:
                continue

            lag_1 = store_data.iloc[-1]['Weekly_Sales']
            lag_7 = store_data.iloc[-7]['Weekly_Sales']
            rolling_mean = store_data['Weekly_Sales'].tail(7).mean()

            last_row = store_data.iloc[-1]

            new_row = {
                'Store': store,
                'Holiday_Flag': 0,

                'Temperature': last_row['Temperature'],
                'Fuel_Price': last_row['Fuel_Price'],
                'CPI': last_row['CPI'],
                'Unemployment': last_row['Unemployment'],

                'Year': 2013,
                'Month': (i // 4) + 1,
                'week': i + 1,
                'Day': 1,
                'Day_of_week': 0,

                'Lag_1': lag_1,
                'Lag_7': lag_7,
                'Rolling_mean_7': rolling_mean
            }

            new_rows.append(new_row)

        new_df = pd.DataFrame(new_rows)
        new_df=new_df.drop(columns=['Date'],errors='ignore')

        preds = model.predict(new_df)
        new_df['Weekly_Sales'] = preds

        predictions.append(new_df)

        # VERY IMPORTANT → update data
        data = pd.concat([data, new_df], ignore_index=True)

    return pd.concat(predictions, ignore_index=True)