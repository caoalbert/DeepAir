import pandas as pd

# od_demand_path = '../data/1993_2023_CA.csv'
# throughput_path = '../data/throughput_1993_2023.csv'

# for debugging
od_demand_path = 'data/1993_2023_CA.csv'
throughput_path = 'data/throughput_1993_2023.csv'

def _pre_process_throughput():
    throughput = pd.read_csv(throughput_path).iloc[3:,:].reset_index(drop=True)
    throughput.dropna(inplace=True)
    throughput['month'] = throughput['Date'].apply(lambda x: int(str(x)[:2]))
    throughput['year'] = throughput['Date'].apply(lambda x: int(str(x)[3:]))
    throughput['quarter'] = throughput['month'].apply(lambda x: int((x-1)/3)+1)
    throughput['quarter_id'] = (throughput['year']-1993)*4 + throughput['quarter']
    throughput['Departures'] = throughput['Departures'].apply(lambda x: int(x.replace(',','')))
    throughput['Arrivals'] = throughput['Arrivals'].apply(lambda x: int(x.replace(',','')))
    throughput['total_throughput'] = throughput['Departures'] + throughput['Arrivals']
    throughput_grouped = throughput.groupby(['Facility', 'quarter_id'])['total_throughput'].sum().reset_index()

    return throughput_grouped


def _pre_process_od(aspm77):
    df = pd.read_csv(od_demand_path)
    df['quarter_id'] = (df['Year']-1993)*4 + df['Quarter'] 
    ca_airport = df[df['OriginState'] == 'CA']['Origin'].unique()

    df = df[['Origin', 'Dest', 'Passengers','quarter_id']]
    df['Passengers'] = (df['Passengers']*10).astype(int)

    number_of_quarters = df.groupby('Origin')['quarter_id'].nunique().sort_values(ascending=False).reset_index()
    selected_airports_from_od = number_of_quarters[number_of_quarters['quarter_id'] >= 124]['Origin'].values

    ca_airport = set(ca_airport).intersection(aspm77.values()).intersection(set(selected_airports_from_od))
    selected_airports = set(selected_airports_from_od).intersection(aspm77.values())

    df = df[(df['Origin'].isin(selected_airports)) & (df['Dest'].isin(selected_airports))].reset_index(drop=True)

    return df, ca_airport, selected_airports