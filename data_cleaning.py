import pandas as pd
import duckdb

# 1. load a manageable chuck of the dataset
# using nrows=500000 to avoid memory crashes while building the pipeline
file_path ="2019-Oct.csv"
df=pd.read_csv(file_path,nrows=500000)

print(f"Original shape: {df.shape}")

# 2. Clean the timestamps
# convert the event_time into proper datetime formate 
df['event_time'] = pd.to_datetime(df['event_time'])

# 3.extract the day of the week and hour of the day from the event_time
df['hour_of_day'] = df['event_time'].dt.hour
df['day_of_week'] = df['event_time'].dt.day_name()
df['is_weekend'] = df['event_time'].dt.dayofweek >=5

# 4. clean up the missing data and sessionoze
# drop rows where user_session is missing
df= df.dropna(subset=['user_session'])

# 5. sort the data so every users's action are in chronnological order
df=df.sort_values(by=['user_session','event_time'])

# Downcast the user_id
df['user_id']=pd.to_numeric(df['user_id'], downcast='integer')

# saving  the cleaned data to a new csv file 
df.to_csv('data_cleaned.csv', index= False)


print(df[['event_time', 'event_type','product_id','user_session','hour_of_day']].head())


ecommerce_events=df
query = """
-- 1.  flag each session if it contains at least one view, cart and purchase
with SessionFlags as(
	select
		user_session,
		max(case when event_type ='view' then 1 else 0 end) as has_view,
		max(case when event_type='cart' then 1 else 0 end) as has_cart,
		max(case when event_type='purchase' then 1 else 0 end) as has_purchase
	from ecommerce_events
	group by user_session
),

-- 2. sum the flag to get the absolute number of sessions that reached each stage
FunnelCounts as(
select 
	sum(has_view) as total_view_sessions,
	sum(has_cart) as total_cart_sessions,
	sum(has_purchase) as total_purchase_sessions
from SessionFlags
)

-- 3 Calculate the drop-off and conversion percentage
select 
	total_view_sessions,
	total_cart_sessions,
	total_purchase_sessions,
	round((total_cart_sessions*100.0)/nullif(total_view_sessions,0),2) as view_to_cart_pct,
	round((total_purchase_sessions*100.0)/nullif(total_cart_sessions,0),2) as cart_to_purchase_pct,
	round((total_purchase_sessions*100.0)/nullif(total_view_sessions,0),2) as overall_conversion_pct
from funnelCounts
"""

# run the query and output the result as a new pandas dataframe
funnel_results = duckdb.query(query).df()
print(funnel_results)






